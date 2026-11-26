#pragma once

constexpr unsigned kLongPrefillWaveSize = 64;
constexpr unsigned kLongPrefillWaves = 4;
constexpr unsigned long long kLongPrefillWaveMask = 0xffffffffffffffffull;

template <class T>
__global__ void AttendLongPrefill(const T* q, const float* sink, Pages first, Pages second,
                                  T* out, size_t rows, size_t heads, float scale,
                                  int parallel_queries) {
  using Convert = Caster<Device::Type::kHygon>;
  __shared__ volatile float scores[kLongPrefillWaves][kMaxKeyTile];
  __shared__ volatile float weights[kLongPrefillWaves][kMaxKeyTile];
  const unsigned wave = threadIdx.x / kLongPrefillWaveSize;
  const unsigned lane = threadIdx.x % kLongPrefillWaveSize;
  const bool online_tiles = rows >= static_cast<size_t>(parallel_queries) && second.data;
  const size_t keys_per_tile = rows > 1 && !second.data ? 128 : 64;
  for (size_t block = blockIdx.x * kLongPrefillWaves + wave; block < rows * heads;
       block += gridDim.x * kLongPrefillWaves) {
    const auto row = block / heads, head = block % heads;
    float accum_lo[4] = {}, accum_hi[4] = {};
    float query_lo[4], query_hi[4];
    for (size_t j = 0; j < 4; ++j) {
      query_lo[j] = Convert::template Cast<float>(q[block * kDim + j * kThreads + lane]);
      query_hi[j] = Convert::template Cast<float>(
          q[block * kDim + j * kThreads + lane + kLongPrefillWaveSize]);
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (int source = 0; source < 2; ++source) {
      const Pages table = source == 0 ? first : second;
      if (!table.data) continue;
      const int32_t length = table.lengths[row];
      const size_t positive = length <= 0 ? 0 : static_cast<size_t>(length);
      const size_t count = positive < table.index_width ? positive : table.index_width;
      const auto* indices = table.indices + row * table.index_width;
      for (size_t start = 0; start < count; start += keys_per_tile) {
        const size_t remaining = count - start;
        const size_t tile_size = remaining < keys_per_tile ? remaining : keys_per_tile;
        for (size_t i = 0; i < tile_size; ++i) {
          const int32_t id = indices[start + i];
          float score = -INFINITY;
          if (valid_key(table, id)) {
            float kv_lo[4], kv_hi[4], dot_lo = 0.0f, dot_hi = 0.0f;
            load_key(table, id, lane, kv_lo);
            load_key(table, id, lane + kLongPrefillWaveSize, kv_hi);
            for (size_t j = 0; j < 4; ++j) {
              dot_lo += query_lo[j] * kv_lo[j];
              dot_hi += query_hi[j] * kv_hi[j];
            }
            // Match the original 128-thread tree: combine the two logical lanes
            // first, then reduce widths 32, 16, 8, 4, 2, 1 within this wave.
            float dot = dot_lo + dot_hi;
            for (unsigned width = kLongPrefillWaveSize / 2; width; width /= 2) {
              dot += __shfl_down_sync(kLongPrefillWaveMask, dot, width, kLongPrefillWaveSize);
            }
            score = dot * scale;
          }
          if (lane == 0) scores[wave][i] = score;
        }
        __syncwarp();
        float prior_weight = 0.0f, tile_weight = 0.0f;
        if (lane == 0) {
          float tile_max = -INFINITY;
          for (size_t i = 0; i < tile_size; ++i) tile_max = fmaxf(tile_max, scores[wave][i]);
          const float next = fmaxf(maximum, tile_max);
          const float shift = online_tiles ? next : tile_max;
          float mass = 0.0f;
          for (size_t i = 0; i < tile_size; ++i) {
            const float score = scores[wave][i];
            const float prob = score == -INFINITY ? 0.0f :
                (score == shift ? 1.0f : expf(score - shift));
            mass += prob;
            weights[wave][i] = Convert::template Cast<float>(Convert::template Cast<T>(prob));
          }
          prior_weight = maximum == next ? 1.0f : expf(maximum - next);
          tile_weight = shift == next ? 1.0f : expf(shift - next);
          maximum = next;
          // Match the legacy FMA order and the separately rounded product.
          denominator = __fmaf_rn(mass, tile_weight, __fmul_rn(denominator, prior_weight));
        }
        __syncwarp();
        prior_weight = __shfl_sync(kLongPrefillWaveMask, prior_weight, 0, kLongPrefillWaveSize);
        tile_weight = __shfl_sync(kLongPrefillWaveMask, tile_weight, 0, kLongPrefillWaveSize);
        float weighted_lo[4] = {}, weighted_hi[4] = {};
        for (size_t i = 0; i < tile_size; ++i) {
          const float weight = weights[wave][i];
          if (weight == 0.0f) continue;
          float kv_lo[4], kv_hi[4];
          const int32_t id = indices[start + i];
          load_key(table, id, lane, kv_lo);
          load_key(table, id, lane + kLongPrefillWaveSize, kv_hi);
          for (size_t j = 0; j < 4; ++j) {
            weighted_lo[j] += weight * kv_lo[j];
            weighted_hi[j] += weight * kv_hi[j];
          }
        }
        for (size_t j = 0; j < 4; ++j) {
          accum_lo[j] = __fmaf_rn(accum_lo[j], prior_weight, __fmul_rn(weighted_lo[j], tile_weight));
          accum_hi[j] = __fmaf_rn(accum_hi[j], prior_weight, __fmul_rn(weighted_hi[j], tile_weight));
        }
        __syncwarp();
      }
    }
    if (lane == 0 && denominator != 0.0f) {
      denominator += expf(sink[head] - maximum);
    }
    denominator = __shfl_sync(kLongPrefillWaveMask, denominator, 0, kLongPrefillWaveSize);
    for (size_t j = 0; j < 4; ++j) {
      out[block * kDim + j * kThreads + lane] = Convert::template Cast<T>(
          denominator ? accum_lo[j] / denominator : 0.0f);
      out[block * kDim + j * kThreads + lane + kLongPrefillWaveSize] = Convert::template Cast<T>(
          denominator ? accum_hi[j] / denominator : 0.0f);
    }
  }
}
