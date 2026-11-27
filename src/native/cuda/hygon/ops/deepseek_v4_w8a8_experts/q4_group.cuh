#pragma once

#include "math.cuh"
#include "q4_group_map.h"

namespace infini::ops::deepseek_v4_w8a8_detail {

constexpr unsigned kGroupWave = 64;
constexpr unsigned kGroupWaves = 4;
constexpr unsigned kGroupHidden = 4096;
constexpr unsigned kGroupInner = 256;
constexpr unsigned kM12GroupTokens = 3 * kGroupTokens;

// Split a small batch into tiles without a single-row tail when possible.
struct ExpertTiles {
  unsigned count;
  unsigned width;

  explicit ExpertTiles(unsigned tokens)
      : count((tokens + kGroupTokens - 1) / kGroupTokens),
        width((tokens + count - 1) / count) {}
};

struct ExpertTile {
  unsigned first;
  unsigned rows;
};

static __device__ __forceinline__ ExpertTile MakeExpertTile(
    unsigned tile, unsigned width, unsigned tokens) {
  const unsigned first = tile * width;
  const unsigned remaining = tokens - first;
  return {first, remaining < width ? remaining : width};
}

template <class T>
static __device__ float FinishGateUp(int32_t gate, int32_t up,
                                     float token_scale, float gate_scale,
                                     float up_scale, float limit, bool routed,
                                     bool joint_scale) {
  using Convert = Caster<Device::Type::kHygon>;
  const auto cast_gate = Convert::template Cast<T>(
      ScaledDot(gate, token_scale, gate_scale, joint_scale));
  const auto cast_up = Convert::template Cast<T>(
      ScaledDot(up, token_scale, up_scale, joint_scale));
  float clipped_gate = fminf(Convert::template Cast<float>(cast_gate), limit);
  if (routed) clipped_gate = fmaxf(clipped_gate, -limit);
  const float clipped_up =
      fminf(fmaxf(Convert::template Cast<float>(cast_up), -limit), limit);
  float gate_activation = clipped_gate / (1.0f + expf(-clipped_gate));
  if (routed)
    gate_activation = Convert::template Cast<float>(
        Convert::template Cast<T>(gate_activation));
  return Convert::template Cast<float>(
      Convert::template Cast<T>(gate_activation * clipped_up));
}

template <unsigned TopK>
__global__ void BuildQ4ExpertGroups(const int64_t* ids, size_t experts,
                                    float* activated, Q4ExpertGroup* groups,
                                    unsigned tokens, unsigned tile_width) {
  const auto tile = MakeExpertTile(blockIdx.x, tile_width, tokens);
  const unsigned first_route = tile.first * TopK;
  const unsigned routes = tile.rows * TopK;
  ids += first_route;
  activated += first_route * kGroupInner;
  groups += first_route;
  const unsigned slot = threadIdx.x;
  if (slot < routes)
    groups[slot] = FindQ4ExpertGroup<TopK>(ids, slot, experts, tile.rows);
  // Invalid expert slots must have the same activation as the fallback.
  // Valid slots are fully written by GateUpQ4 during this replay.
  for (unsigned row = 0; row < routes; ++row) {
    const auto expert = ids[row];
    if (expert < 0 || static_cast<size_t>(expert) >= experts) {
      for (unsigned col = slot; col < kGroupInner; col += kGroupWave)
        activated[row * kGroupInner + col] = 0.0f;
    }
  }
}

static __device__ __forceinline__ int32_t LoadPackedInt8(const int8_t* ptr) {
  int32_t value;
  __builtin_memcpy(&value, __builtin_assume_aligned(ptr, alignof(int32_t)),
                   sizeof(value));
  return value;
}

static __device__ __forceinline__ int32_t Dot4(int32_t a, int32_t b,
                                               int32_t accumulator) {
#if defined(__AMDGCN__) && __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(a, b, accumulator, false);
#else
  // Keep the operator buildable with CUDA-compat host/parser passes.
  auto x = static_cast<uint32_t>(a), y = static_cast<uint32_t>(b);
  for (unsigned byte = 0; byte < 4; ++byte) {
    const int lhs = static_cast<int>((x & 0xff) ^ 0x80) - 128;
    const int rhs = static_cast<int>((y & 0xff) ^ 0x80) - 128;
    accumulator += lhs * rhs;
    x >>= 8;
    y >>= 8;
  }
  return accumulator;
#endif
}

static __device__ __forceinline__ int32_t Q4WaveSum(int32_t value) {
  for (unsigned offset = kGroupWave / 2; offset; offset /= 2)
    value += __shfl_down(value, offset, kGroupWave);
  return value;
}

template <class T, unsigned TopK>
__global__ void GateUpQ4(const int8_t* input, const float* input_scale,
                         const Q4ExpertGroup* groups, const int8_t* weights,
                         const float* scales, float* activated, float limit,
                         unsigned tokens, unsigned tile_width) {
  constexpr bool routed = TopK != 1;
  const unsigned wave = threadIdx.x / kGroupWave;
  const unsigned lane = threadIdx.x % kGroupWave;
  const unsigned index = blockIdx.x * kGroupWaves + wave;
  const auto tile = MakeExpertTile(blockIdx.y, tile_width, tokens);
  const unsigned group_index = index / kGroupInner;
  if (group_index >= tile.rows * TopK) return;
  const unsigned first_route = tile.first * TopK;
  input += tile.first * kGroupHidden;
  input_scale += tile.first;
  activated += first_route * kGroupInner;
  const auto group = groups[first_route + group_index];
  if (group.expert < 0) return;

  const unsigned channel = index % kGroupInner;
  const unsigned gate_channel = group.expert * (2 * kGroupInner) + channel;
  const unsigned up_channel = gate_channel + kGroupInner;
  const auto* gate_row = weights + gate_channel * kGroupHidden;
  const auto* up_row = weights + up_channel * kGroupHidden;
  int32_t gate[kGroupTokens] = {}, up[kGroupTokens] = {};
#pragma unroll 2
  for (unsigned i = lane * 4; i < kGroupHidden; i += kGroupWave * 4) {
    const int32_t gate_weights = LoadPackedInt8(gate_row + i);
    const int32_t up_weights = LoadPackedInt8(up_row + i);
#pragma unroll
    for (unsigned token = 0; token < kGroupTokens; ++token) {
      if (group.slots[token] < 0) continue;
      const int32_t value = LoadPackedInt8(input + token * kGroupHidden + i);
      gate[token] = Dot4(value, gate_weights, gate[token]);
      up[token] = Dot4(value, up_weights, up[token]);
    }
  }
#pragma unroll
  for (unsigned token = 0; token < kGroupTokens; ++token) {
    const int slot = group.slots[token];
    if (slot < 0) continue;
    const int32_t g = Q4WaveSum(gate[token]), u = Q4WaveSum(up[token]);
    if (lane == 0) {
      activated[slot * kGroupInner + channel] =
          FinishGateUp<T>(g, u, input_scale[token], scales[gate_channel],
                          scales[up_channel], limit, routed, routed);
    }
  }
}

template <class T>
__global__ void RoutedDownQ4(const int8_t* activated,
                             const float* activation_scale, const int64_t* ids,
                             const Q4ExpertGroup* groups,
                             const float* route_weights, const int8_t* weights,
                             const float* scales, T* out, size_t experts,
                             float routed_scale, unsigned tokens,
                             unsigned tile_width) {
  using Convert = Caster<Device::Type::kHygon>;
  const auto tile = MakeExpertTile(blockIdx.y, tile_width, tokens);
  const unsigned first_route = tile.first * kGroupTopK;
  const unsigned routes = tile.rows * kGroupTopK;
  activated += first_route * kGroupInner;
  activation_scale += first_route;
  ids += first_route;
  groups += first_route;
  route_weights += first_route;
  out += tile.first * kGroupHidden;
  __shared__ float contributions[kGroupTokens * kGroupTopK];
  const unsigned wave = threadIdx.x / kGroupWave;
  const unsigned lane = threadIdx.x % kGroupWave;
  const unsigned channel = blockIdx.x;
  for (unsigned index = wave; index < routes; index += kGroupWaves) {
    const auto group = groups[index];
    if (group.expert < 0) continue;
    const unsigned weight_channel = group.expert * kGroupHidden + channel;
    const int32_t packed_weight =
        LoadPackedInt8(weights + weight_channel * kGroupInner + lane * 4);
#pragma unroll
    for (unsigned token = 0; token < kGroupTokens; ++token) {
      const int slot = group.slots[token];
      if (slot < 0) continue;
      const int32_t value =
          LoadPackedInt8(activated + slot * kGroupInner + lane * 4);
      const int32_t sum = Q4WaveSum(Dot4(value, packed_weight, 0));
      if (lane == 0) {
        const float weighted = ScaledDot(sum, activation_scale[slot],
                                         scales[weight_channel], true) *
                               route_weights[slot];
        contributions[slot] =
            Convert::template Cast<float>(Convert::template Cast<T>(weighted));
      }
    }
  }
  __syncthreads();
  const unsigned token = threadIdx.x;
  if (token < tile.rows) {
    float output = 0.0f;
    for (unsigned k = 0; k < kGroupTopK; ++k) {
      const unsigned slot = token * kGroupTopK + k;
      const auto expert = ids[slot];
      if (expert >= 0 && static_cast<size_t>(expert) < experts)
        output += contributions[slot];
    }
    out[token * kGroupHidden + channel] =
        Convert::template Cast<T>(output * routed_scale);
  }
}

// Each wave owns one gate/up channel and loads its weights once for all rows.
template <class T, unsigned Rows>
__global__ void SharedGateUpDecode(const int8_t* input, const float* input_scale,
                                   const int64_t* ids, const int8_t* weights,
                                   const float* scales, float* activated,
                                   float limit, unsigned rows) {
  const unsigned lane = threadIdx.x % kGroupWave;
  const unsigned channel = blockIdx.x * kGroupWaves + threadIdx.x / kGroupWave;
  const unsigned up_channel = channel + kGroupInner;
  const auto* gate_row = weights + channel * kGroupHidden;
  const auto* up_row = weights + up_channel * kGroupHidden;
  int32_t gate[Rows] = {}, up[Rows] = {};
  unsigned valid_mask = 0;
#pragma unroll
  for (unsigned row = 0; row < Rows; ++row)
    if (row < rows && ids[row] == 0) valid_mask |= 1u << row;
#pragma unroll 2
  for (unsigned i = lane * 4; valid_mask && i < kGroupHidden; i += kGroupWave * 4) {
    const int32_t gate_weights = LoadPackedInt8(gate_row + i);
    const int32_t up_weights = LoadPackedInt8(up_row + i);
#pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
      if ((valid_mask & (1u << row)) == 0) continue;
      const int32_t value = LoadPackedInt8(input + row * kGroupHidden + i);
      gate[row] = Dot4(value, gate_weights, gate[row]);
      up[row] = Dot4(value, up_weights, up[row]);
    }
  }
#pragma unroll
  for (unsigned row = 0; row < Rows; ++row) {
    if (row >= rows) break;
    const int32_t g = Q4WaveSum(gate[row]), u = Q4WaveSum(up[row]);
    if (lane == 0) {
      activated[row * kGroupInner + channel] = (valid_mask & (1u << row))
          ? FinishGateUp<T>(g, u, input_scale[row], scales[channel],
                            scales[up_channel], limit, false, false)
          : 0.0f;
    }
  }
}

// The shared path has one expert. All valid rows use the same packed weight.
template <class T, unsigned Rows>
__global__ void SharedDownRows(const int8_t* activated,
                             const float* activation_scale, const int64_t* ids,
                             const float* route_weights, const int8_t* weights,
                             const float* scales, T* out, unsigned tokens) {
  using Convert = Caster<Device::Type::kHygon>;
  const auto tile = MakeExpertTile(blockIdx.y, Rows, tokens);
  activated += tile.first * kGroupInner;
  activation_scale += tile.first;
  ids += tile.first;
  route_weights += tile.first;
  out += tile.first * kGroupHidden;
  const unsigned channel = blockIdx.x * kGroupWaves + threadIdx.x / kGroupWave;
  const unsigned lane = threadIdx.x % kGroupWave;
  const int32_t packed_weight =
      LoadPackedInt8(weights + channel * kGroupInner + lane * 4);
#pragma unroll
  for (unsigned token = 0; token < Rows; ++token) {
    if (token >= tile.rows) break;
    float output = 0.0f;
    if (ids[token] == 0) {
      const int32_t value =
          LoadPackedInt8(activated + token * kGroupInner + lane * 4);
      const int32_t sum = Q4WaveSum(Dot4(value, packed_weight, 0));
      if (lane == 0)
        output +=
            ScaledDot(sum, activation_scale[token], scales[channel], false) *
            route_weights[token];
    }
    if (lane == 0)
      out[token * kGroupHidden + channel] =
          Convert::template Cast<T>(output * 1.0f);
  }
}

template <class T, unsigned TopK>
static cudaError_t LaunchGroupedGateUp(
    const int8_t* input, const float* input_scale, const int64_t* ids,
    const int8_t* weights, const float* scales, float* activated,
    Q4ExpertGroup* groups, size_t tokens, size_t experts, float limit,
    cudaStream_t stream) {
  constexpr auto block = kGroupWave * kGroupWaves;
  const ExpertTiles tiles(tokens);
  BuildQ4ExpertGroups<TopK><<<tiles.count, kGroupWave, 0, stream>>>(
      ids, experts, activated, groups, tokens, tiles.width);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const dim3 grid(tiles.width * TopK * kGroupInner / kGroupWaves, tiles.count);
  GateUpQ4<T, TopK><<<grid, block, 0, stream>>>(
      input, input_scale, groups, weights, scales, activated, limit, tokens,
      tiles.width);
  return cudaGetLastError();
}

template <class T, unsigned TopK>
static cudaError_t LaunchGroupedDown(
    const int8_t* activated, const float* activation_scale, const int64_t* ids,
    const float* route_weights, const int8_t* weights, const float* scales,
    T* out, const Q4ExpertGroup* groups, size_t tokens, size_t experts,
    float routed_scale, cudaStream_t stream) {
  constexpr auto block = kGroupWave * kGroupWaves;
  const ExpertTiles tiles(tokens);
  if constexpr (TopK == 1) {
    const dim3 grid(kGroupHidden / kGroupWaves, tiles.count);
    SharedDownRows<T, kGroupTokens><<<grid, block, 0, stream>>>(
        activated, activation_scale, ids, route_weights, weights, scales, out,
        tokens);
  } else {
    const dim3 grid(kGroupHidden, tiles.count);
    RoutedDownQ4<T><<<grid, block, 0, stream>>>(
        activated, activation_scale, ids, groups, route_weights, weights,
        scales, out, experts, routed_scale, tokens, tiles.width);
  }
  return cudaGetLastError();
}

}  // namespace infini::ops::deepseek_v4_w8a8_detail
