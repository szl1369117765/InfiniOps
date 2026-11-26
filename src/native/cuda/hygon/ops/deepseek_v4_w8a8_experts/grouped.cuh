#pragma once

#include "kernel_v2.cuh"
#include "native/cuda/hygon/caster.cuh"

#include <cstring>
#include <stdexcept>

namespace infini::ops::expert_grouped {

constexpr unsigned kM = 16, kN = 16, kK = 32;
constexpr unsigned kThreads = 256, kNTile = 64;
constexpr unsigned kDownColumns = 512;
static_assert(kDownColumns * sizeof(__nv_bfloat16) == 256 * sizeof(float));

struct Tile {
  uint32_t expert, start, count;
};

struct Plan {
  uint32_t* counts;
  uint32_t* cursors;
  uint32_t* routes;
  uint32_t* tile_count;
  Tile* tiles;
};

inline bool supported(size_t tokens, size_t experts, size_t hidden, size_t inner,
                      size_t top_k, bool routed) {
  return tokens >= (routed ? 256 : 128) && tokens <= 8192 &&
         hidden == 4096 && inner == 256 &&
         (routed ? experts == 256 && top_k == 6 : experts == 1 && top_k == 1);
}

inline bool available(int device) {
  cudaDeviceProp properties{};
  return cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.warpSize == 64 && std::strstr(properties.name, "gfx936") != nullptr;
}

inline size_t max_tiles(size_t routes, size_t experts) {
  return routes / kM + experts;
}

inline size_t workspace_bytes(size_t routes, size_t experts) {
  return (2 * experts + routes + 1) * sizeof(uint32_t) +
         max_tiles(routes, experts) * sizeof(Tile);
}

inline Plan workspace(void* storage, size_t routes, size_t experts) {
  auto* counts = static_cast<uint32_t*>(storage);
  auto* cursors = counts + experts;
  auto* order = cursors + experts;
  auto* tile_count = order + routes;
  return {counts, cursors, order, tile_count, reinterpret_cast<Tile*>(tile_count + 1)};
}

static __global__ void Count(const int64_t* ids, unsigned routes, unsigned experts,
                             uint32_t* counts) {
  const unsigned route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) return;
  const auto expert = ids[route];
  if (expert >= 0 && size_t(expert) < experts) atomicAdd(counts + expert, 1);
}

static __global__ void Build(Plan plan, unsigned experts) {
  uint32_t start = 0, tile = 0;
  for (unsigned expert = 0; expert < experts; ++expert) {
    plan.cursors[expert] = start;
    const uint32_t count = plan.counts[expert];
    for (uint32_t i = 0; i < count; i += kM)
      plan.tiles[tile++] = {expert, start + i, min(kM, count - i)};
    start += count;
  }
  *plan.tile_count = tile;
}

static __global__ void Scatter(const int64_t* ids, unsigned routes, unsigned experts,
                               Plan plan) {
  const unsigned route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) return;
  const auto expert = ids[route];
  if (expert >= 0 && size_t(expert) < experts)
    plan.routes[atomicAdd(plan.cursors + expert, 1)] = route;
}

inline cudaError_t make_plan(cudaStream_t stream, const int64_t* ids, unsigned routes,
                             unsigned experts, Plan plan) {
  const auto status = cudaMemsetAsync(plan.counts, 0, experts * sizeof(uint32_t), stream);
  if (status != cudaSuccess) return status;
  const auto blocks = (routes + kThreads - 1) / kThreads;
  Count<<<blocks, kThreads, 0, stream>>>(ids, routes, experts, plan.counts);
  Build<<<1,1,0,stream>>>(plan, experts);
  Scatter<<<blocks,kThreads,0,stream>>>(ids, routes, experts, plan);
  return cudaGetLastError();
}

using Int2 = int __attribute__((ext_vector_type(2)));
using Int4 = int __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ Int2 load8(const int8_t* row, unsigned offset) {
  return *reinterpret_cast<const Int2*>(row + offset);
}

static __device__ __forceinline__ Int4 mmac(Int2 a, Int2 b, Int4 c) {
#if defined(__gfx936__)
  return __builtin_hcu_mmac_i32_16x16x32_i8(a, b, c);
#else
  // Preserve the tile layout in compatible device binaries without the intrinsic.
  const unsigned row = threadIdx.x % 16, group = threadIdx.x / 16 % 4;
  constexpr uint64_t mask = 0xffffffffffffffffull;
  for (unsigned k = 0; k < 4; ++k) {
    const Int2 aa{__shfl_sync(mask, a[0], row + 16 * k, 64),
                  __shfl_sync(mask, a[1], row + 16 * k, 64)};
    for (unsigned output = 0; output < 4; ++output) {
      const unsigned lane = group + 4 * output + 16 * k;
      const Int2 bb{__shfl_sync(mask, b[0], lane, 64),
                    __shfl_sync(mask, b[1], lane, 64)};
      c[output] = expert_wave::DotWord(aa[0], bb[0], c[output]);
      c[output] = expert_wave::DotWord(aa[1], bb[1], c[output]);
    }
  }
  return c;
#endif
}

template <bool Routed>
static __device__ __forceinline__ float Activate(
    int gate, int up, float token_scale, float gate_scale, float up_scale, float limit) {
  using Convert = Caster<Device::Type::kHygon>;
  using deepseek_v4_w8a8_detail::ScaledDot;
  const auto cast_gate = Convert::template Cast<__nv_bfloat16>(
      ScaledDot(gate, token_scale, gate_scale, true));
  const auto cast_up = Convert::template Cast<__nv_bfloat16>(
      ScaledDot(up, token_scale, up_scale, true));
  float clipped_gate = fminf(Convert::template Cast<float>(cast_gate), limit);
  if constexpr (Routed) clipped_gate = fmaxf(clipped_gate, -limit);
  const float clipped_up = fminf(fmaxf(Convert::template Cast<float>(cast_up), -limit), limit);
  float gate_activation = clipped_gate / (1.0f + expf(-clipped_gate));
  if constexpr (Routed)
    gate_activation = Convert::template Cast<float>(Convert::template Cast<__nv_bfloat16>(gate_activation));
  return Convert::template Cast<float>(Convert::template Cast<__nv_bfloat16>(gate_activation * clipped_up));
}

template <bool Routed>
static __global__ void Gate(const int8_t* x, const float* input_scale, const int8_t* weights,
                            const float* scales, float* activated, Plan plan, unsigned top_k,
                            unsigned hidden, unsigned inner, float limit) {
  if (blockIdx.x >= *plan.tile_count) return;
  const auto tile = plan.tiles[blockIdx.x];
  const unsigned lane_row = threadIdx.x % kM, lane_k = threadIdx.x / kM % 4;
  const unsigned col = blockIdx.y * kNTile + threadIdx.x / 64 * kN;
  const bool valid = lane_row < tile.count;
  const auto route = valid ? plan.routes[tile.start + lane_row] : 0;
  const auto* input = x + size_t(route / top_k) * hidden;
  const auto gate_row = size_t(tile.expert) * 2 * inner + col;
  const auto* gate = weights + (gate_row + lane_row) * hidden;
  const auto* up = gate + size_t(inner) * hidden;
  Int4 g{}, u{};
  for (unsigned k = 0; k < hidden; k += kK) {
    const unsigned offset = k + lane_k * 8;
    const Int2 value = valid ? load8(input, offset) : Int2{};
    g = mmac(value, load8(gate, offset), g);
    u = mmac(value, load8(up, offset), u);
  }
  if (!valid) return;
  const float token_scale = input_scale[route / top_k];
#pragma unroll
  for (unsigned o = 0; o < 4; ++o) {
    const unsigned channel = lane_k + 4 * o;
    const auto scale = scales + gate_row + channel;
    activated[size_t(route) * inner + col + channel] =
        Activate<Routed>(g[o], u[o], token_scale, *scale, scale[inner], limit);
  }
}

template <bool Routed>
static __global__ void GateQuant(
    const int8_t* x, const float* input_scale, const int8_t* weights, const float* scales,
    int8_t* q_activated, float* activation_scale, Plan plan, unsigned top_k,
    unsigned hidden, unsigned inner, float limit) {
  if (blockIdx.x >= *plan.tile_count) return;
  const auto tile = plan.tiles[blockIdx.x];
  const unsigned lane_row = threadIdx.x % kM;
  const unsigned lane_k = threadIdx.x / kM % 4;
  const unsigned wave = threadIdx.x / 64;
  const bool valid = lane_row < tile.count;
  const auto route = valid ? plan.routes[tile.start + lane_row] : 0;
  const auto* input = x + size_t(route / top_k) * hidden;
  const auto gate_row = size_t(tile.expert) * 2 * inner + wave * kN;
  const auto* gate = weights + (gate_row + lane_row) * hidden;
  const auto* up = gate + size_t(inner) * hidden;
  // Interleave four output tiles while keeping the input fragment in registers.
  constexpr unsigned slices = 256 / kNTile;
  Int4 g[slices]{}, u[slices]{};
  for (unsigned k = 0; k < hidden; k += kK) {
    const unsigned offset = k + lane_k * 8;
    const Int2 value = valid ? load8(input, offset) : Int2{};
#pragma unroll
    for (unsigned t = 0; t < slices; ++t) {
      const auto step = size_t(t) * kNTile * hidden;
      g[t] = mmac(value, load8(gate + step, offset), g[t]);
      u[t] = mmac(value, load8(up + step, offset), u[t]);
    }
  }
  float activation[slices][4];
  float maximum = 0.0f;
  const float token_scale = input_scale[route / top_k];
#pragma unroll
  for (unsigned t = 0; t < slices; ++t) {
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) {
      const unsigned channel = t * kNTile + lane_k + 4 * o;
      const auto* scale = scales + gate_row + channel;
      activation[t][o] = Activate<Routed>(g[t][o], u[t][o], token_scale, *scale, scale[inner], limit);
      maximum = fmaxf(maximum, fabsf(activation[t][o]));
    }
  }
  // Each MMAC row occupies the same lane index in four wave64 segments.
  constexpr uint64_t mask = 0xffffffffffffffffull;
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 32, 64));
  maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 16, 64));
  __shared__ float maxima[4][kM];
  if (lane_k == 0) maxima[wave][lane_row] = maximum;
  __syncthreads();
  if (!valid) return;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) maximum = fmaxf(maximum, maxima[i][lane_row]);
  maximum = fmaxf(maximum, 1.0e-10f);
  if (threadIdx.x < kM) activation_scale[route] = maximum / 127.0f;
  const float inverse = 127.0f / maximum;
  auto* result = q_activated + size_t(route) * inner + wave * kN;
#pragma unroll
  for (unsigned t = 0; t < slices; ++t) {
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) {
      const int value = __float2int_rn(activation[t][o] * inverse);
      result[t * kNTile + lane_k + 4 * o] = static_cast<int8_t>(min(max(value, -127), 127));
    }
  }
}

template <bool Routed, unsigned Slices = 1>
static __global__ void Down(const int8_t* activated, const float* activation_scale,
                            const float* weights, const int8_t* down, const float* scales,
                            __nv_bfloat16* output, Plan plan, unsigned hidden, unsigned inner,
                            unsigned first_col) {
  if (blockIdx.x >= *plan.tile_count) return;
  const auto tile = plan.tiles[blockIdx.x];
  const unsigned lane_row = threadIdx.x % kM, lane_k = threadIdx.x / kM % 4;
  const unsigned col = blockIdx.y * Slices * kNTile + threadIdx.x / 64 * kN;
  const bool valid = lane_row < tile.count;
  const auto route = valid ? plan.routes[tile.start + lane_row] : 0;
  const auto* input = activated + size_t(route) * inner;
  const auto weight_row = size_t(tile.expert) * hidden + first_col + col;
  const auto* weight = down + (weight_row + lane_row) * inner;
  Int4 dots[Slices]{};
  for (unsigned k = 0; k < inner; k += kK) {
    const unsigned offset = k + lane_k * 8;
    const Int2 value = valid ? load8(input, offset) : Int2{};
#pragma unroll
    for (unsigned t = 0; t < Slices; ++t)
      dots[t] = mmac(value, load8(weight, offset + t * kNTile * inner), dots[t]);
  }
  if (!valid) return;
  const float token_scale = activation_scale[route];
  const float routing_weight = weights[route];
  using Convert = Caster<Device::Type::kHygon>;
  using deepseek_v4_w8a8_detail::ScaledDot;
#pragma unroll
  for (unsigned t = 0; t < Slices; ++t) {
#pragma unroll
    for (unsigned o = 0; o < 4; ++o) {
      const unsigned channel = t * kNTile + lane_k + 4 * o;
      const float scaled = ScaledDot(dots[t][o], token_scale, scales[weight_row + channel], true);
      if constexpr (Routed) {
        output[size_t(route) * kDownColumns + col + channel] =
            Convert::template Cast<__nv_bfloat16>(scaled * routing_weight);
      } else {
        float value = 0.0f;
        value += scaled * routing_weight;
        output[size_t(route) * hidden + first_col + col + channel] =
            Convert::template Cast<__nv_bfloat16>(value);
      }
    }
  }
}

static __global__ void Combine(const __nv_bfloat16* partial, const int64_t* ids,
                               __nv_bfloat16* output, unsigned tokens, unsigned top_k,
                               unsigned experts, unsigned hidden, unsigned first_col,
                               float routed_scale) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens * kDownColumns) return;
  const unsigned token = i / kDownColumns, col = i % kDownColumns;
  float value = 0.0f;
  using Convert = Caster<Device::Type::kHygon>;
  for (unsigned k = 0; k < top_k; ++k) {
    const size_t route = size_t(token) * top_k + k;
    const auto expert = ids[route];
    if (expert >= 0 && size_t(expert) < experts)
      value += Convert::template Cast<float>(partial[route * kDownColumns + col]);
  }
  output[size_t(token) * hidden + first_col + col] =
      Convert::template Cast<__nv_bfloat16>(value * routed_scale);
}

// The stream owns the grouping buffers. The FP32 activation range is
// reused for 512 BF16 down values per route after gate and quantization.
inline void run(cudaStream_t stream, const __nv_bfloat16* input,
                const int64_t* ids, const float* route_weights,
                const int8_t* gate, const float* gate_scales,
                const int8_t* down, const float* down_scales,
                __nv_bfloat16* output, float* activations,
                float* input_scale, float* activation_scale,
                int8_t* q_input, int8_t* q_activated, Plan plan,
                unsigned tokens, unsigned top_k, unsigned experts,
                unsigned hidden, unsigned inner, float limit,
                bool routed, float routed_scale) {
  const auto routes = tokens * top_k;
  if (make_plan(stream, ids, routes, experts, plan) != cudaSuccess)
    throw std::runtime_error("DeepSeek V4 grouped expert planning failed");

  expert_wave::Quant<__nv_bfloat16><<<tokens,kThreads,0,stream>>>(
      input, q_input, input_scale, tokens, hidden);
  const auto tiles = static_cast<unsigned>(max_tiles(routes, experts));
  // Large batches fill the GPU even with four times fewer thread blocks.
  const bool wide = tokens >= 2048;
  if (wide) {
    if (routed)
      GateQuant<true><<<tiles,kThreads,0,stream>>>(q_input, input_scale, gate,
          gate_scales, q_activated, activation_scale, plan, top_k, hidden, inner, limit);
    else
      GateQuant<false><<<tiles,kThreads,0,stream>>>(q_input, input_scale, gate,
          gate_scales, q_activated, activation_scale, plan, top_k, hidden, inner, limit);
  } else {
    if (cudaMemsetAsync(activations, 0, size_t(routes) * inner * sizeof(float), stream) != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 grouped expert activation reset failed");
    const dim3 gate_grid(tiles, inner / kNTile);
    if (routed)
      Gate<true><<<gate_grid,kThreads,0,stream>>>(q_input, input_scale, gate,
          gate_scales, activations, plan, top_k, hidden, inner, limit);
    else
      Gate<false><<<gate_grid,kThreads,0,stream>>>(q_input, input_scale, gate,
          gate_scales, activations, plan, top_k, hidden, inner, limit);
    expert_wave::Quant<float><<<routes,kThreads,0,stream>>>(
        activations, q_activated, activation_scale, routes, inner);
  }
  constexpr unsigned wide_tiles = 4;
  const auto down_tile_width = kNTile * (wide ? wide_tiles : 1);
  if (routed) {
    const dim3 down_grid(tiles, kDownColumns / down_tile_width);
    auto* partial = reinterpret_cast<__nv_bfloat16*>(activations);
    const auto combine_grid = (tokens * kDownColumns + kThreads - 1) / kThreads;
    for (unsigned first = 0; first < hidden; first += kDownColumns) {
      if (wide)
        Down<true, wide_tiles><<<down_grid,kThreads,0,stream>>>(q_activated, activation_scale,
            route_weights, down, down_scales, partial, plan, hidden, inner, first);
      else
        Down<true><<<down_grid,kThreads,0,stream>>>(q_activated, activation_scale,
            route_weights, down, down_scales, partial, plan, hidden, inner, first);
      Combine<<<combine_grid,kThreads,0,stream>>>(partial, ids, output,
          tokens, top_k, experts, hidden, first, routed_scale);
    }
  } else {
    if (cudaMemsetAsync(output, 0, size_t(tokens) * hidden * sizeof(__nv_bfloat16), stream) != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 shared expert output reset failed");
    const dim3 down_grid(tiles, hidden / down_tile_width);
    if (wide)
      Down<false, wide_tiles><<<down_grid,kThreads,0,stream>>>(q_activated, activation_scale,
          route_weights, down, down_scales, output, plan, hidden, inner, 0);
    else
      Down<false><<<down_grid,kThreads,0,stream>>>(q_activated, activation_scale,
          route_weights, down, down_scales, output, plan, hidden, inner, 0);
  }
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("DeepSeek V4 grouped expert launch failed");
}

} // namespace infini::ops::expert_grouped
