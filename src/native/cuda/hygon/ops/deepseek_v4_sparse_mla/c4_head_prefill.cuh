#pragma once

constexpr unsigned kSharedMlaHeadsPerWave = 4;
constexpr unsigned kSharedMlaWavesPerBlock = 4;

template <unsigned Step>
static __device__ __forceinline__ float mla_score_partner(float value) {
#if defined(__gfx936__)
  // Only these steps read partners within a 16-lane row on the path to lane 0.
  if constexpr (Step < 16) {
    constexpr unsigned kDppRowShiftLeft = 0x100;
    return __uint_as_float(__builtin_amdgcn_mov_dpp(
        __float_as_uint(value), kDppRowShiftLeft + Step, 0xf, 0xf, true));
  }
#endif
  return __shfl_down_sync(kLongPrefillWaveMask, value, Step, kLongPrefillWaveSize);
}

template <unsigned Step, unsigned Heads>
static __device__ __forceinline__ void mla_reduce_dot(float (&dot)[Heads]) {
#pragma unroll
  for (unsigned h = 0; h < Heads; ++h)
    dot[h] = __fadd_rn(dot[h], mla_score_partner<Step>(dot[h]));
  if constexpr (Step > 1) mla_reduce_dot<Step / 2>(dot);
}

static __device__ __forceinline__ void load_aligned_mla_key(
    const Pages& table, unsigned page_shift, int32_t id, unsigned lane,
    float (&lo)[4], float (&hi)[4]) {
  const auto position = static_cast<unsigned>(id);
  const auto offset = position & (table.page_tokens - 1);
  const auto* page = table.data + (position >> page_shift) * table.page_stride;
  const auto* values = page + offset * kValueBytes;
  const auto* exponents = page + table.page_tokens * kValueBytes + offset * 8;
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned channel = j * kThreads + lane;
    lo[j] = value_at(values, exponents, channel);
    hi[j] = value_at(values, exponents, channel + kLongPrefillWaveSize);
  }
}

// Bound decoded pages by both absolute workspace and expected reuse.
// Long single requests can amortize decoding without needing a batch.
static size_t decoded_mla_workspace(size_t swa_pages, size_t extra_pages, size_t extra_tokens,
                                    size_t query_rows, bool expand) {
  constexpr size_t kSwaPageTokens = 256;
  constexpr size_t kCompactTokens = 20480;
  constexpr size_t kMaxDecodedTokens = 65536;
  size_t max_tokens = query_rows >= 16384 && query_rows <= 32768
                          ? kMaxDecodedTokens : kCompactTokens;
  if (expand &&
      query_rows > kFlashMlaMaxRows && query_rows < 16384) {
    // The numerator is packed tokens, and each unpacked token is FP32[512].
    max_tokens = std::max(max_tokens, std::min(kMaxDecodedTokens, query_rows * 16));
  }
  if (swa_pages > max_tokens / kSwaPageTokens) return 0;
  const size_t swa_tokens = swa_pages * kSwaPageTokens;
  if (extra_tokens && extra_pages > (max_tokens - swa_tokens) / extra_tokens) return 0;
  return (swa_tokens + extra_pages * extra_tokens) * kDim * sizeof(float);
}

// Decode each physical packed token once for both QK and weighted V.
static __global__ void DecodeMlaPages(Pages table, float* decoded) {
  const size_t count = table.page_count * table.page_tokens;
  for (size_t token = blockIdx.x; token < count; token += gridDim.x) {
    const size_t offset = token % table.page_tokens;
    const auto* page = table.data + token / table.page_tokens * table.page_stride;
    const auto* key = page + offset * kValueBytes;
    const auto* exponents = page + table.page_tokens * kValueBytes + offset * 8;
    auto* target = decoded + token * kDim;
    for (unsigned col = threadIdx.x; col < kDim; col += blockDim.x)
      target[col] = value_at(key, exponents, col);
  }
}

template <bool Predecode>
static __device__ __forceinline__ void load_reused_mla_key(
    const Pages& table, const float* decoded, unsigned page_shift, int32_t id,
    unsigned lane, float (&lo)[4], float (&hi)[4]) {
  if constexpr (Predecode) {
    const auto* key = decoded + static_cast<size_t>(id) * kDim + lane;
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      lo[j] = key[j * kThreads];
      hi[j] = key[j * kThreads + kLongPrefillWaveSize];
    }
  } else {
    load_aligned_mla_key(table, page_shift, id, lane, lo, hi);
  }
}

template <unsigned Step, unsigned Keys, unsigned Heads>
static __device__ __forceinline__ void reduce_mla_tile(float (&dot)[Keys][Heads]) {
#pragma unroll
  for (unsigned c = 0; c < Keys; ++c) {
#pragma unroll
    for (unsigned h = 0; h < Heads; ++h)
      dot[c][h] = __fadd_rn(dot[c][h], mla_score_partner<Step>(dot[c][h]));
  }
  if constexpr (Step > 1) reduce_mla_tile<Step / 2>(dot);
}

using MlaF32Pair = float __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ MlaF32Pair mla_pair_fma(MlaF32Pair a, MlaF32Pair b, MlaF32Pair c) {
#if defined(__gfx936__)
  MlaF32Pair result;
  // Fuse two independent FP32 lanes without changing their accumulation order.
  asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(result) : "v"(a), "v"(b), "v"(c));
  return result;
#else
  return {__fmaf_rn(a[0],b[0],c[0]),__fmaf_rn(a[1],b[1],c[1])};
#endif
}

// Preserve the original candidate order, softmax tile shape and FP32 rounding.
template <unsigned ExtraPageShift, bool Predecode>
static __global__ void AttendLongSharedPrefill(
    const __nv_bfloat16* q, const float* sink, Pages first, Pages second, __nv_bfloat16* out,
    size_t rows, size_t heads, float scale, int parallel_queries,
    const float* first_decoded, const float* second_decoded) {
  using Convert = Caster<Device::Type::kHygon>;
  constexpr unsigned Group = kSharedMlaHeadsPerWave;
  constexpr unsigned Waves = kSharedMlaWavesPerBlock;
  constexpr unsigned kTile = ExtraPageShift == 0 ? 128 : 64;
  constexpr bool kMatrixTile = Predecode && ExtraPageShift == 6;
  __shared__ __align__(16) volatile float tile_values[Waves][Group * kTile];
  const auto tile_offset = [](size_t head, size_t key) {
    return kMatrixTile ? key * Group + head : head * kTile + key;
  };
  const auto wave = threadIdx.x / kLongPrefillWaveSize;
  const auto lane = threadIdx.x % kLongPrefillWaveSize;
  const bool online_tiles = ExtraPageShift != 0 && rows >= static_cast<size_t>(parallel_queries);
  const auto groups = heads / Group;
  for (size_t job = blockIdx.x * Waves + wave; job < rows * groups; job += gridDim.x * Waves) {
    const size_t row = job / groups;
    const size_t head = job % groups * Group;
    const auto* q_row = q + (row * heads + head) * kDim;
    auto* out_row = out + (row * heads + head) * kDim;
    float query_lo[Group][4], query_hi[Group][4];
    MlaF32Pair matrix_query[Group][4];
    float accum_lo[Group][4]{}, accum_hi[Group][4]{};
    float maximum = -INFINITY, denominator = 0.0f;
#pragma unroll
    for (unsigned h = 0; h < Group; ++h) {
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) {
        const auto col = h * kDim + j * kThreads + lane;
        const float lo = Convert::template Cast<float>(q_row[col]);
        const float hi = Convert::template Cast<float>(q_row[col + kLongPrefillWaveSize]);
        if constexpr (kMatrixTile) {
          matrix_query[h][j] = {lo, hi};
        } else {
          query_lo[h][j] = lo;
          query_hi[h][j] = hi;
        }
      }
    }
    for (unsigned source = 0; source < (ExtraPageShift == 0 ? 1u : 2u); ++source) {
      const Pages table = source == 0 ? first : second;
      const unsigned page_shift = source == 0 ? 8 : ExtraPageShift;
      const float* decoded = source == 0 ? first_decoded : second_decoded;
      const int32_t length = table.lengths[row];
      const size_t count = length > 0 ? min(static_cast<size_t>(length), table.index_width) : 0;
      const auto* indices = table.indices + row * table.index_width;
      for (size_t start = 0; start < count; start += kTile) {
        const auto tile_size = min(count - start, size_t(kTile));
        if constexpr (kMatrixTile) {
          static_assert(Group == 4);
          // Each microtile has four heads and two keys. Shuffle stages for
          // the two keys are interleaved and each dot keeps its own sum tree.
          constexpr unsigned ScoreKeys = 2;
          for (size_t i = 0; i < tile_size; i += ScoreKeys) {
            MlaF32Pair keys[ScoreKeys][4];
            bool valid[ScoreKeys];
#pragma unroll
            for (unsigned c = 0; c < ScoreKeys; ++c) {
              const int32_t id = i + c < tile_size ? indices[start + i + c] : -1;
              valid[c] = id >= 0 && (static_cast<unsigned>(id) >> page_shift) < table.page_count;
              float lo[4], hi[4];
              // Invalid candidates still receive -inf. Load token zero to
              // keep both independent dot products in flight.
              load_reused_mla_key<true>(table, decoded, page_shift, valid[c] ? id : 0, lane, lo, hi);
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) keys[c][j] = {lo[j],hi[j]};
            }
            float dot[ScoreKeys][Group]{};
#pragma unroll
            for (unsigned c = 0; c < ScoreKeys; ++c) {
#pragma unroll
              for (unsigned h = 0; h < Group; ++h) {
                MlaF32Pair acc{};
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                  acc = mla_pair_fma(matrix_query[h][j], keys[c][j], acc);
                dot[c][h] = __fadd_rn(acc[0], acc[1]);
              }
            }
            reduce_mla_tile<32>(dot);
            if (lane == 0) {
#pragma unroll
              for (unsigned c = 0; c < ScoreKeys; ++c) {
                if (i + c >= tile_size) continue;
#pragma unroll
                for (unsigned h = 0; h < Group; ++h)
                  tile_values[wave][tile_offset(h, i + c)] = valid[c] ? dot[c][h] * scale : -INFINITY;
              }
            }
          }
        } else {
          for (size_t i = 0; i < tile_size; ++i) {
            const int32_t id = indices[start + i];
            float dot[Group];
            if (id >= 0 && (static_cast<unsigned>(id) >> page_shift) < table.page_count) {
              float kv_lo[4], kv_hi[4];
              load_reused_mla_key<Predecode>(table, decoded, page_shift, id, lane, kv_lo, kv_hi);
#pragma unroll
              for (unsigned h = 0; h < Group; ++h) {
                float lo = 0.0f, hi = 0.0f;
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                  lo = __fmaf_rn(query_lo[h][j], kv_lo[j], lo);
                  hi = __fmaf_rn(query_hi[h][j], kv_hi[j], hi);
                }
                dot[h] = __fadd_rn(lo, hi);
              }
              mla_reduce_dot<32>(dot);
#pragma unroll
              for (unsigned h = 0; h < Group; ++h)
                dot[h] *= scale;
            } else {
#pragma unroll
              for (unsigned h = 0; h < Group; ++h) dot[h] = -INFINITY;
            }
            if (lane == 0) {
#pragma unroll
              for (unsigned h = 0; h < Group; ++h) tile_values[wave][tile_offset(h, i)] = dot[h];
            }
          }
        }
        __syncwarp();
        float prior_weight = 0.0f, tile_weight = 0.0f;
        // The first four lanes compute one head's softmax serially in key order.
        if (lane < Group) {
          float tile_max = -INFINITY;
          for (size_t i = 0; i < tile_size; ++i)
            tile_max = fmaxf(tile_max, tile_values[wave][tile_offset(lane, i)]);
          const float next = fmaxf(maximum, tile_max);
          const float shift = online_tiles ? next : tile_max;
          float mass = 0.0f;
          for (size_t i = 0; i < tile_size; ++i) {
            const float score = tile_values[wave][tile_offset(lane, i)];
            const float prob = score == -INFINITY ? 0.0f : (score == shift ? 1.0f : expf(score - shift));
            mass += prob;
            // The tile's scores now become BF16-rounded weights.
            tile_values[wave][tile_offset(lane, i)] = Convert::template Cast<float>(Convert::template Cast<__nv_bfloat16>(prob));
          }
          prior_weight = maximum == next ? 1.0f : expf(maximum - next);
          tile_weight = shift == next ? 1.0f : expf(shift - next);
          maximum = next;
          denominator = __fmaf_rn(mass, tile_weight, __fmul_rn(denominator, prior_weight));
        }
        __syncwarp();
        float weighted_lo[Group][4]{}, weighted_hi[Group][4]{};
        for (size_t i = 0; i < tile_size; ++i) {
          float weight[Group];
          bool needed = false;
          if constexpr (kMatrixTile) {
            // Key-major LDS places this key's four head weights in one vector.
            using WeightRow = float __attribute__((ext_vector_type(4), may_alias));
            const auto coefficients = *reinterpret_cast<const volatile WeightRow*>(
                &tile_values[wave][tile_offset(0, i)]);
#pragma unroll
            for (unsigned h = 0; h < Group; ++h) weight[h] = coefficients[h];
          } else {
#pragma unroll
            for (unsigned h = 0; h < Group; ++h)
              weight[h] = tile_values[wave][tile_offset(h, i)];
          }
#pragma unroll
          for (unsigned h = 0; h < Group; ++h) needed |= weight[h] != 0.0f;
          if (!needed) continue;
          float kv_lo[4], kv_hi[4];
          load_reused_mla_key<Predecode>(table, decoded, page_shift, indices[start + i], lane, kv_lo, kv_hi);
#pragma unroll
          for (unsigned h = 0; h < Group; ++h) {
            if (weight[h] == 0.0f) continue;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
              weighted_lo[h][j] = __fmaf_rn(weight[h], kv_lo[j], weighted_lo[h][j]);
              weighted_hi[h][j] = __fmaf_rn(weight[h], kv_hi[j], weighted_hi[h][j]);
            }
          }
        }
#pragma unroll
        for (unsigned h = 0; h < Group; ++h) {
          const float prior = __shfl_sync(kLongPrefillWaveMask, prior_weight, h, kLongPrefillWaveSize);
          const float weight = __shfl_sync(kLongPrefillWaveMask, tile_weight, h, kLongPrefillWaveSize);
#pragma unroll
          for (unsigned j = 0; j < 4; ++j) {
            accum_lo[h][j] = __fmaf_rn(accum_lo[h][j], prior, __fmul_rn(weighted_lo[h][j], weight));
            accum_hi[h][j] = __fmaf_rn(accum_hi[h][j], prior, __fmul_rn(weighted_hi[h][j], weight));
          }
        }
        __syncwarp();
      }
    }
    if (lane < Group && denominator != 0.0f)
      denominator += expf(sink[head + lane] - maximum);
#pragma unroll
    for (unsigned h = 0; h < Group; ++h) {
      const float norm = __shfl_sync(kLongPrefillWaveMask, denominator, h, kLongPrefillWaveSize);
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) {
        const auto col = h * kDim + j * kThreads + lane;
        out_row[col] = Convert::template Cast<__nv_bfloat16>(norm ? accum_lo[h][j] / norm : 0.0f);
        out_row[col + kLongPrefillWaveSize] =
            Convert::template Cast<__nv_bfloat16>(norm ? accum_hi[h][j] / norm : 0.0f);
      }
    }
  }
}
