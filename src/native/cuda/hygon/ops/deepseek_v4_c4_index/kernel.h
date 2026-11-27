#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "base/deepseek_v4_c4_index.h"

namespace infini::ops {
namespace v4_c4 {
constexpr size_t kWidth = 128, kPage = 64, kPageBytes = kPage*(kWidth+sizeof(float));
constexpr size_t kTop = 512;

static __device__ float decode(uint8_t bits) {
  const int exp = (bits >> 3) & 15, mantissa = bits & 7;
  const float mag = exp == 0 ? ldexpf(float(mantissa),-9)
                            : ldexpf(1.0f + float(mantissa)/8,exp-7);
  return (bits & 128) ? -mag : mag;
}

static __global__ void transform(const __nv_bfloat16* in, const int32_t* slots, uint8_t* dst,
                                 float* scales, size_t count, size_t pages, size_t stride) {
  __shared__ float lane_value[kWidth];
  const unsigned lane = threadIdx.x;
  for (size_t row=blockIdx.x; row<count; row+=gridDim.x) {
    const int32_t slot = slots ? slots[row] : 0;
    if (slots && (slot < 0 || size_t(slot)/kPage >= pages)) continue;
    float value = __bfloat162float(in[row*kWidth+lane]);
    for (unsigned bit=1; bit<kWidth; bit*=2) {
      lane_value[lane] = value;
      __syncthreads();
      const float other = lane_value[lane ^ bit];
      value = (lane & bit) ? other-value : value+other;
      __syncthreads();
    }
    value *= 0.08838834764831845f;
    lane_value[lane] = fabsf(value);
    __syncthreads();
    for (unsigned bit=kWidth/2; bit; bit/=2) {
      if (lane < bit) lane_value[lane] = fmaxf(lane_value[lane],lane_value[lane+bit]);
      __syncthreads();
    }
    const float scale = fmaxf(lane_value[0],1.0e-4f) / 448.f;
    auto* page = slots ? dst+(size_t(slot)/kPage)*stride : nullptr;
    auto* bytes = slots ? page+(size_t(slot)%kPage)*kWidth : dst+row*kWidth;
    bytes[lane] = __nv_cvt_float_to_fp8(value/scale,__NV_SATFINITE,__NV_E4M3);
    if (lane == 0) {
      if (slots) reinterpret_cast<float*>(page+kPage*kWidth)[size_t(slot)%kPage] = scale;
      else scales[row] = scale;
    }
    __syncthreads();
  }
}

static __device__ int32_t physical(const int32_t* pages, size_t pos, size_t page_count) {
  const auto page = pages[pos/kPage];
  return page < 0 || size_t(page) >= page_count ? -1 : page*int(kPage)+int(pos%kPage);
}

static __global__ void score(const uint8_t* query, const float* qs, const __nv_bfloat16* weights,
                             const int32_t* table, const int32_t* lengths, const uint8_t* cache,
                             float* scores, size_t rows, size_t heads, size_t width,
                             size_t cache_pages, size_t stride,
                             size_t first_query = 0) {
  __shared__ float dot[kWidth];
  const unsigned lane=threadIdx.x;
  const size_t span=width*kPage;
  for (size_t flat=blockIdx.x; flat<(rows-first_query)*span; flat+=gridDim.x) {
    const size_t query_row=flat/span, row=first_query+query_row, pos=flat%span;
    if (lengths[row] <= int(kTop) || pos >= size_t(lengths[row])) continue;
    const auto id = physical(table+row*width,pos,cache_pages);
    if (id < 0) {
      if (lane == 0) scores[flat]=-INFINITY;
      continue;
    }
    const auto* page = cache+(size_t(id)/kPage)*stride;
    const auto offset=size_t(id)%kPage;
    const float key=decode(page[offset*kWidth+lane]);
    const float key_scale=reinterpret_cast<const float*>(page+kPage*kWidth)[offset];
    float total=0;
    for (size_t h=0;h<heads;++h) {
      const size_t head=query_row*heads+h;
      dot[lane]=key*decode(query[head*kWidth+lane]);
      __syncthreads();
      for (unsigned step=kWidth/2;step;step/=2) {
        if (lane<step) dot[lane]+=dot[lane+step];
        __syncthreads();
      }
      if (lane == 0)
        total+=fmaxf(dot[0]*key_scale*qs[head],0.f)*__bfloat162float(weights[head]);
      __syncthreads();
    }
    if (lane == 0) scores[flat]=total;
  }
}


static __device__ __forceinline__ float decode_e4m3(uint8_t bits) {
  const uint32_t exponent = (bits >> 3) & 15u;
  const uint32_t mantissa = bits & 7u;
  const float magnitude = exponent
      ? __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20))
      : static_cast<float>(mantissa) * 0x1p-9f;
  return __uint_as_float(__float_as_uint(magnitude) | (uint32_t(bits & 128u) << 24));
}

constexpr unsigned kScoreWave = 64, kScoreThreads = 256;
constexpr size_t kWideScoreMinQueries = 1024;

template <unsigned Step>
static __device__ __forceinline__ float score_partner(float value) {
#if defined(__gfx936__)
  // At these steps the lanes that contribute to lane zero stay in the
  // same 16-lane row. DPP reads the same partner as the wave shuffle.
  if constexpr (Step < 16) {
    constexpr unsigned kDppRowShiftLeft = 0x100;
    return __uint_as_float(__builtin_amdgcn_mov_dpp(
        __float_as_uint(value), kDppRowShiftLeft + Step, 0xf, 0xf, true));
  }
#endif
  return __shfl_down_sync(0xffffffffffffffffull, value, Step, kScoreWave);
}

template <unsigned Step, unsigned Candidates>
static __device__ __forceinline__ void reduce_score(float (&dot)[Candidates]) {
#pragma unroll
  for (unsigned c = 0; c < Candidates; ++c)
    dot[c] = __fadd_rn(dot[c], score_partner<Step>(dot[c]));
  if constexpr (Step > 1) reduce_score<Step / 2>(dot);
}

// A wave's aligned score group belongs to one logical cache page. Reuse that
// page lookup and accumulate each head in the scalar score's FP32 order.
template <unsigned CandidatesPerWave>
static __global__ void score_tiled(
    const uint8_t* query, const float* qs, const __nv_bfloat16* weights,
    const int32_t* table, const int32_t* lengths, const uint8_t* cache,
    float* scores, size_t rows, size_t heads, size_t width,
    size_t cache_pages, size_t stride, size_t first_query) {
  static_assert(kWidth == 2 * kScoreWave);
  constexpr unsigned kTile = kScoreThreads / kScoreWave * CandidatesPerWave;
  static_assert(kPage % kTile == 0);
  const unsigned lane = threadIdx.x % kScoreWave;
  const unsigned wave = threadIdx.x / kScoreWave;
  const size_t span = width * kPage;
  const size_t tiles = span / kTile;
  const size_t jobs = (rows - first_query) * tiles;

  for (size_t job = blockIdx.x; job < jobs; job += gridDim.x) {
    const size_t query_row = job / tiles;
    const size_t row = first_query + query_row;
    const int32_t length = lengths[row];
    const size_t base = (job % tiles) * kTile + wave * CandidatesPerWave;
    if (length <= int(kTop) || base >= size_t(length)) continue;

    const unsigned count = min(size_t(CandidatesPerWave), size_t(length) - base);
    const int32_t page_index = table[row * width + base / kPage];
    float* const output = scores + query_row * span + base;
    if (page_index < 0 || size_t(page_index) >= cache_pages) {
      if (lane == 0) {
#pragma unroll
        for (unsigned c = 0; c < CandidatesPerWave; ++c)
          if (c < count) output[c] = -INFINITY;
      }
      continue;
    }

    const auto* page = cache + size_t(page_index) * stride;
    const unsigned offset = base % kPage;
    const auto* keys = page + offset * kWidth;
    const auto* scales = reinterpret_cast<const float*>(page + kPage * kWidth) + offset;
    float key_lo[CandidatesPerWave], key_hi[CandidatesPerWave], key_scale[CandidatesPerWave];
    float total[CandidatesPerWave]{};

#pragma unroll
    for (unsigned c = 0; c < CandidatesPerWave; ++c) {
      // Repeat one valid key in partial groups so unfilled slots are never read.
      const unsigned key = c < count ? c : 0;
      key_lo[c] = decode_e4m3(keys[key * kWidth + lane]);
      key_hi[c] = decode_e4m3(keys[key * kWidth + lane + kScoreWave]);
      key_scale[c] = scales[key];
    }

    for (size_t h = 0; h < heads; ++h) {
      const size_t head = query_row * heads + h;
      const auto* q = query + head * kWidth;
      const float q_lo = decode_e4m3(q[lane]);
      const float q_hi = decode_e4m3(q[lane + kScoreWave]);
      float dot[CandidatesPerWave];
#pragma unroll
      for (unsigned c = 0; c < CandidatesPerWave; ++c)
        dot[c] = __fadd_rn(__fmul_rn(key_lo[c], q_lo), __fmul_rn(key_hi[c], q_hi));
      reduce_score<kScoreWave / 2>(dot);
      if (lane == 0) {
        const float query_scale = qs[head];
        const float weight = __bfloat162float(weights[head]);
#pragma unroll
        for (unsigned c = 0; c < CandidatesPerWave; ++c) {
          const float relu = fmaxf(__fmul_rn(__fmul_rn(dot[c], key_scale[c]), query_scale), 0.0f);
          total[c] = __fmaf_rn(relu, weight, total[c]);
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (unsigned c = 0; c < CandidatesPerWave; ++c)
        if (c < count) output[c] = total[c];
    }
  }
}

static __global__ void select_serial(float* scores, const int32_t* table, const int32_t* lengths,
                              int32_t* out, size_t rows, size_t width, size_t cache_pages, size_t first_query = 0) {
  const size_t span=width*kPage;
  for (size_t row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=gridDim.x*blockDim.x) {
    const size_t length=lengths[row] > 0 ? min(size_t(lengths[row]),span) : 0;
    for (size_t k=0;k<kTop;++k) {
      int32_t idx=-1;
      if (length<=kTop) {
        if (k<length) idx=physical(table+row*width,k,cache_pages);
      } else if (row >= first_query) {
        float best=-INFINITY;
        size_t chosen=0;
        for (size_t j=0;j<length;++j) {
          const float value=scores[(row-first_query)*span+j];
          if (value>best) { best=value; chosen=j; }
        }
        if (best!=-INFINITY) {
          idx=physical(table+row*width,chosen,cache_pages);
          scores[(row-first_query)*span+chosen]=-INFINITY;
        }
      }
      out[row*kTop+k]=idx;
    }
  }
}

static __device__ void write_sequential(const int32_t* table, int32_t* out,
                                         size_t length, size_t cache_pages) {
  for (size_t i = threadIdx.x; i < kTop; i += blockDim.x)
    out[i] = i < length ? physical(table, i, cache_pages) : -1;
}

static __global__ void select_sequential(const int32_t* table, const int32_t* lengths,
                                          int32_t* out, size_t rows, size_t width,
                                          size_t cache_pages) {
  const size_t span = width * kPage;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const size_t length = lengths[row] > 0 ? min(size_t(lengths[row]), span) : 0;
    write_sequential(table + row * width, out + row * kTop, length, cache_pages);
  }
}

// The legacy selector ignores NaNs and -infinity, and keeps the first
// logical candidate on a tie. The low 32 bits encode that same tie break.
static __device__ uint64_t sort_key(float value, uint32_t index) {
  if (!(value > -INFINITY)) return 0;
  const uint32_t bits = value == 0.0f ? 0 : __float_as_uint(value);
  const uint32_t ordered = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
  return (uint64_t(ordered) << 32) | uint64_t(~index);
}

constexpr unsigned kSelectThreads = 128;

template <unsigned Items>
static __global__ void select_parallel(const float* scores, const int32_t* table,
                                        const int32_t* lengths, int32_t* out,
                                        size_t rows, size_t width, size_t cache_pages,
                                        size_t first_query = 0) {
  using Sort = cub::BlockRadixSort<uint64_t, kSelectThreads, Items>;
  __shared__ typename Sort::TempStorage scratch;
  const size_t span = width * kPage;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const size_t length = lengths[row] > 0 ? min(size_t(lengths[row]), span) : 0;
    const auto* row_table = table + row * width;
    auto* row_out = out + row * kTop;
    if (length <= kTop || row < first_query) {
      write_sequential(row_table, row_out, length <= kTop ? length : 0, cache_pages);
      continue;
    }

    const auto* row_scores = scores + (row - first_query) * span;
    uint64_t keys[Items];
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      const unsigned pos = threadIdx.x * Items + i;
      keys[i] = pos < length ? sort_key(row_scores[pos], pos) : 0;
    }
    Sort(scratch).SortDescending(keys);
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      const unsigned pos = threadIdx.x * Items + i;
      if (pos < kTop)
        row_out[pos] = keys[i] ? physical(row_table, ~uint32_t(keys[i]), cache_pages) : -1;
    }
    __syncthreads();
  }
}

inline unsigned grid(size_t n) {
  return static_cast<unsigned>(std::min<size_t>(n, 65535));
}

inline bool supports_cache(Tensor key, Tensor slots, Tensor table, Tensor lengths,
                           Tensor cache, Tensor out) {
  if (key.device().type() != Device::Type::kHygon ||
      key.dtype() != DataType::kBFloat16 || key.ndim() != 2 || key.size(1) != kWidth ||
      table.ndim() != 2 || !table.size(0) ||
      !table.size(1) || table.size(1) > INT32_MAX / kPage ||
      cache.ndim() != 2 || !cache.size(0) || cache.size(0) > INT32_MAX / kPage ||
      cache.size(1) < kPageBytes || cache.size(1) % sizeof(float)) return false;
  const auto rows = table.size(0);
  if (slots.shape() != Tensor::Shape{key.size(0)} || lengths.shape() != Tensor::Shape{rows} ||
      out.shape() != Tensor::Shape{rows, kTop} || slots.dtype() != DataType::kInt32 ||
      lengths.dtype() != DataType::kInt32 || table.dtype() != DataType::kInt32 ||
      cache.dtype() != DataType::kUInt8 || out.dtype() != DataType::kInt32) return false;
  for (const auto& tensor : {key, slots, table, lengths, cache, out})
    if (tensor.device() != key.device() || !tensor.IsContiguous()) return false;
  return true;
}

inline void write_key_cache(Tensor key, Tensor slots, Tensor cache, cudaStream_t stream) {
  if (!key.size(0)) return;
  transform<<<grid(key.size(0)), kWidth, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(key.data()), static_cast<const int32_t*>(slots.data()),
      static_cast<uint8_t*>(cache.data()), nullptr, key.size(0), cache.size(0), cache.size(1));
}

inline void sequential_indices(Tensor table, Tensor lengths, Tensor cache, Tensor out,
                               cudaStream_t stream) {
  select_sequential<<<grid(out.size(0)), kSelectThreads, 0, stream>>>(
      static_cast<const int32_t*>(table.data()), static_cast<const int32_t*>(lengths.data()),
      static_cast<int32_t*>(out.data()), out.size(0), table.size(1), cache.size(0));
}
} // namespace v4_c4

template <>
class Operator<DeepseekV4C4Index, Device::Type::kHygon> : public DeepseekV4C4Index {
 public:
  static bool supports(Tensor k, Tensor slots, Tensor q, Tensor w, Tensor table,
                       Tensor lens, Tensor cache, Tensor out) {
    using namespace v4_c4;
    if (!supports_cache(k, slots, table, lens, cache, out) ||
        q.ndim() != 3 || !q.size(0) || q.size(0) > table.size(0) || !q.size(1) || q.size(2) != kWidth ||
        w.shape() != Tensor::Shape{q.size(0), q.size(1)} ||
        q.dtype() != DataType::kBFloat16 || w.dtype() != DataType::kBFloat16 ||
        q.device() != k.device() || w.device() != k.device() ||
        !q.IsContiguous() || !w.IsContiguous()) return false;
    const auto heads = q.size(1);
    return heads <= SIZE_MAX / (kWidth + sizeof(float)) &&
           q.size(0) <= SIZE_MAX / 2 / (heads * (kWidth + sizeof(float))) &&
           table.size(0) <= SIZE_MAX / 2 / (table.size(1) * kPage * sizeof(float));
  }
  Operator(Tensor k,Tensor slots,Tensor q,Tensor w,Tensor table,Tensor lens,Tensor cache,Tensor out)
      : DeepseekV4C4Index(k,slots,q,w,table,lens,cache,out) {
    if (!supports(k,slots,q,w,table,lens,cache,out))
      throw std::invalid_argument("Invalid DeepSeek V4 C4 index geometry");
    bytes_ = workspace_bytes(q, w, table);
    if (q.size(1) == 64 && (q.size(0) >= v4_c4::kWideScoreMinQueries ||
                              (q.size(0) >= 2 && q.size(0) <= 16))) {
      cudaDeviceProp properties{};
      wide_score_ = cudaGetDeviceProperties(&properties, q.device().index()) == cudaSuccess &&
                    properties.warpSize == v4_c4::kScoreWave &&
                    std::strstr(properties.name, "gfx936") != nullptr;
    }
  }
  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override { return bytes_; }
  void operator()(Tensor k,Tensor slots,Tensor q,Tensor w,Tensor table,Tensor lens,Tensor cache,Tensor out) const override {
    using namespace v4_c4;
    if (!supports(k, slots, q, w, table, lens, cache, out))
      throw std::invalid_argument("Invalid DeepSeek V4 C4 index execution");
    const auto required = workspace_bytes(q, w, table);
    if (required && (!workspace_ || workspace_size_in_bytes_ < required))
      throw std::invalid_argument("Insufficient DeepSeek V4 C4 index workspace");
    const auto rows = table.size(0), query_rows = q.size(0), heads = q.size(1), width = table.size(1);
    const size_t first_query = rows - query_rows;
    const auto span = width * kPage;
    const auto stream = static_cast<cudaStream_t>(stream_);
    write_key_cache(k, slots, cache, stream);
    if (span <= kTop) {
      sequential_indices(table, lens, cache, out, stream);
    } else {
      auto* scores = static_cast<float*>(workspace_);
      auto* scales = scores + query_rows * span;
      auto* query = reinterpret_cast<uint8_t*>(scales + query_rows * heads);
      const auto* pages = static_cast<const int32_t*>(table.data());
      const auto* lengths = static_cast<const int32_t*>(lens.data());
      auto* output = static_cast<int32_t*>(out.data());
      const auto* weight_data = static_cast<const __nv_bfloat16*>(w.data());
      const auto* cache_data = static_cast<const uint8_t*>(cache.data());
      transform<<<grid(query_rows * heads), kWidth, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(q.data()), nullptr, query, scales, query_rows * heads, 0, 0);
      const char* tile_mode = std::getenv("INFINIOPS_DSV4_C4_DECODE_TILE");
      const bool use_decode_tile = wide_score_ && tile_mode && tile_mode[0] == '1' &&
          tile_mode[1] == '\0' && query_rows >= 2 && query_rows <= 16;
      if (heads == 64 && (query_rows >= 64 || use_decode_tile)) {
        const auto launch_score = [&](auto width_tag) {
          constexpr unsigned kCandidates = decltype(width_tag)::value;
          constexpr unsigned kTile = kCandidates * (kScoreThreads / kScoreWave);
          score_tiled<kCandidates><<<grid(query_rows * (span / kTile)), kScoreThreads, 0, stream>>>(
              query, scales, weight_data, pages, lengths, cache_data, scores,
              rows, heads, width, cache.size(0), cache.size(1), first_query);
        };
        if (wide_score_ && query_rows >= kWideScoreMinQueries)
          launch_score(std::integral_constant<unsigned, 8>{});
        else
          launch_score(std::integral_constant<unsigned, 4>{});
      } else {
        score<<<grid(query_rows * span), kWidth, 0, stream>>>(
            query, scales, weight_data, pages, lengths, cache_data, scores,
            rows, heads, width, cache.size(0), cache.size(1), first_query);
      }

      if (span <= 1024)
        select_parallel<8><<<grid(rows), kSelectThreads, 0, stream>>>(
            scores, pages, lengths, output, rows, width, cache.size(0), first_query);
      else if (span <= 2048)
        select_parallel<16><<<grid(rows), kSelectThreads, 0, stream>>>(
            scores, pages, lengths, output, rows, width, cache.size(0), first_query);
      else if (span <= 4096)
        select_parallel<32><<<grid(rows), kSelectThreads, 0, stream>>>(
            scores, pages, lengths, output, rows, width, cache.size(0), first_query);
      else
        select_serial<<<grid((rows + 63) / 64), 64, 0, stream>>>(
            scores, pages, lengths, output, rows, width, cache.size(0), first_query);
    }
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 C4 index launch failed");
  }
 private:
  static size_t workspace_bytes(Tensor query, Tensor weights, Tensor table) {
    const size_t span = table.size(1) * v4_c4::kPage;
    return span > v4_c4::kTop
        ? query.numel() + sizeof(float) * (weights.numel() + query.size(0) * span)
        : 0;
  }

  size_t bytes_{};
  bool wide_score_{};
};
} // namespace infini::ops
