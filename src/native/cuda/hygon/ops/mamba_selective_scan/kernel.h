// Hygon-specific Mamba selective scan kernels migrated from the
// InfiniCore indicator-5 submission (zxk branch, 2026-10-31):
// src/infiniop/ops/mamba_selective_scan/hygon/mamba_selective_scan_hygon.cu.
// Ported to the InfiniOps operator-specialization architecture.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
#ifndef INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_
#define INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "base/mamba_selective_scan.h"
#include "caster.h"
#include "dispatcher.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"

namespace infini::ops {
namespace hygon_mamba_scan_detail {
namespace {
template <typename T>
__device__ float to_float(T v) {
  return static_cast<float>(v);
}
template <>
__device__ float to_float<half>(half v) {
  return __half2float(v);
}
template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T>
__device__ T from_float(float v) {
  return static_cast<T>(v);
}
template <>
__device__ half from_float<half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ float softplusf_stable(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}
__device__ float siluf(float x) { return x / (1.0f + expf(-x)); }
// Only the final output store converts to the activation dtype.
__device__ __forceinline__ float finish_mamba_output(float state_dot, float x,
                                                     float d, float gate) {
  return (state_dot + x * d) * siluf(gate);
}

constexpr int kWavefrontSize = 64;
constexpr int kWavesPerBlock = 4;
constexpr unsigned long long kWavefrontMask = 0xffffffffffffffffull;

struct AffinePair {
  float a;
  float u;
};

__device__ __forceinline__ AffinePair compose_affine(AffinePair prefix,
                                                     AffinePair next) {
  return {next.a * prefix.a, next.a * prefix.u + next.u};
}

__device__ __forceinline__ AffinePair warp_inclusive_affine(AffinePair value) {
  for (int offset = 1; offset < kWavefrontSize; offset <<= 1) {
    const float previous_a = __shfl_up_sync(kWavefrontMask, value.a, offset);
    const float previous_u = __shfl_up_sync(kWavefrontMask, value.u, offset);
    if ((threadIdx.x & (kWavefrontSize - 1)) >= offset) {
      value = compose_affine({previous_a, previous_u}, value);
    }
  }
  return value;
}

template <typename T>
__global__ void mamba_selective_scan_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate, size_t state_size) {
  size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = batch * intermediate;
  if (linear >= total) {
    return;
  }
  size_t ch = linear % intermediate;
  size_t batch_idx = linear / intermediate;
  size_t state_base = (batch_idx * intermediate + ch) * state_size;
  const float dt_bias_ch = to_float<T>(dt_bias[ch]);
  const float d_ch = to_float<T>(d[ch]);
  const T *a_ch = a_log + ch * state_size;
  for (size_t t = 0; t < seq_len; ++t) {
    size_t x_idx = (batch_idx * seq_len + t) * intermediate + ch;
    float xt = to_float<T>(x[x_idx]);
    float dtv = (softplusf_stable(to_float<T>(dt[x_idx]) + dt_bias_ch));
    float y = 0.0f;
    const size_t token_base = (batch_idx * seq_len + t) * state_size;
    const T *b_t = b + token_base;
    const T *c_t = c + token_base;
#pragma unroll 4
    for (size_t n = 0; n < state_size; ++n) {
      float a = -expf(to_float<T>(a_ch[n]));
      float discrete_a = expf(a * dtv);
      float bn = to_float<T>(b_t[n]);
      float cn = to_float<T>(c_t[n]);
      float s = discrete_a * state[state_base + n] + dtv * bn * xt;
      state[state_base + n] = s;
      y += (s)*cn;
    }
    y = finish_mamba_output(y, xt, d_ch, to_float<T>(gate[x_idx]));
    out[x_idx] = from_float<T>(y);
  }
}

template <typename T, int kItems, int kGroupWaves = kWavesPerBlock,
          bool kCanonical = false>
__global__
__launch_bounds__(kWavefrontSize *kGroupWaves) void mamba_selective_scan_grouped_shared_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state, size_t batch,
    size_t seq_len, size_t intermediate_arg, size_t state_size_arg) {
  constexpr bool kPrefetchState = kGroupWaves <= 4;
  const size_t intermediate = kCanonical ? 4096 : intermediate_arg;
  const size_t state_size = kCanonical ? 16 : state_size_arg;
  const size_t total = batch * intermediate;

  const size_t block_linear = blockIdx.x;
  const size_t base_linear = block_linear * kGroupWaves;
  if (base_linear >= total) {
    return;
  }
  const size_t wave = threadIdx.x / kWavefrontSize;
  const size_t lane = threadIdx.x & (kWavefrontSize - 1);
  const size_t linear = base_linear + wave;
  const size_t batch_idx = base_linear / intermediate;

  extern __shared__ unsigned char smem[];
  const size_t tile_tokens = static_cast<size_t>(kWavefrontSize * kItems);
  const size_t shared_stride = tile_tokens + (kCanonical ? 4 : 0);
  T *b_shared = reinterpret_cast<T *>(smem);
  T *c_shared = b_shared + shared_stride * state_size;
  const size_t ch = linear % intermediate;
  const size_t state_base = linear * state_size;
  const T *a_ch = a_log + ch * state_size;
  const float d_ch = to_float<T>(d[ch]);
  const float dt_bias_ch = to_float<T>(dt_bias[ch]);
  const float coefficient_lane =
      lane < state_size ? -expf(to_float<T>(a_ch[lane])) : 0.0f;
  float carry_lane =
      kPrefetchState && lane < state_size ? state[state_base + lane] : 0.0f;
  for (size_t tile_begin = 0; tile_begin < seq_len; tile_begin += tile_tokens) {
    const size_t remaining = seq_len - tile_begin;
    const size_t valid_tokens =
        remaining < tile_tokens ? remaining : tile_tokens;
    const size_t tile_elements = valid_tokens * state_size;
    for (size_t index = threadIdx.x; index < tile_elements;
         index += blockDim.x) {
      const size_t token = index / state_size;
      const size_t state_idx = index - token * state_size;
      const size_t global_index =
          (batch_idx * seq_len + tile_begin + token) * state_size + state_idx;
      b_shared[state_idx * shared_stride + token] = b[global_index];
      c_shared[state_idx * shared_stride + token] = c[global_index];
    }
    __syncthreads();
    float x_vals[kItems];
    float dt_vals[kItems];
    float gate_vals[kItems];
    size_t token_indices[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const size_t t = lane * kItems + i;
      token_indices[i] = t;
      if (t < valid_tokens) {
        const size_t x_idx =
            (batch_idx * seq_len + tile_begin + t) * intermediate + ch;
        x_vals[i] = to_float<T>(x[x_idx]);
        dt_vals[i] = (softplusf_stable(to_float<T>(dt[x_idx]) + dt_bias_ch));
        gate_vals[i] = to_float<T>(gate[x_idx]);
      } else {
        x_vals[i] = 0.0f;
        dt_vals[i] = 0.0f;
        gate_vals[i] = 0.0f;
      }
    }

    float y_vals[kItems] = {};
    for (size_t n = 0; n < state_size; ++n) {
      const float a_coeff = __shfl_sync(kWavefrontMask, coefficient_lane, n);
      AffinePair local_prefix[kItems];
#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        local_prefix[i] = {1.0f, 0.0f};
        const size_t t = token_indices[i];
        if (t < valid_tokens) {
          const float b_value = to_float<T>(b_shared[n * shared_stride + t]);
          const AffinePair current = {
              kItems == 2 ? __expf(a_coeff * dt_vals[i])
                          : expf(a_coeff * dt_vals[i]),
              dt_vals[i] * b_value * x_vals[i],
          };
          local_prefix[i] =
              i == 0 ? current : compose_affine(local_prefix[i - 1], current);
        } else if (i > 0) {
          local_prefix[i] = local_prefix[i - 1];
        }
      }

      const AffinePair lane_prefix =
          warp_inclusive_affine(local_prefix[kItems - 1]);
      const float previous_a = __shfl_up_sync(kWavefrontMask, lane_prefix.a, 1);
      const float previous_u = __shfl_up_sync(kWavefrontMask, lane_prefix.u, 1);
      const AffinePair prefix_before_lane =
          lane == 0 ? AffinePair{1.0f, 0.0f}
                    : AffinePair{previous_a, previous_u};
      float initial_state;
      if constexpr (kPrefetchState) {
        initial_state = __shfl_sync(kWavefrontMask, carry_lane, n);
      } else {
        initial_state = lane == 0 ? state[state_base + n] : 0.0f;
        initial_state = __shfl_sync(kWavefrontMask, initial_state, 0);
      }
      float final_state = initial_state;

#pragma unroll
      for (int i = 0; i < kItems; ++i) {
        const size_t t = token_indices[i];
        if (t < valid_tokens) {
          const AffinePair prefix =
              compose_affine(prefix_before_lane, local_prefix[i]);
          const float current_state = prefix.a * initial_state + prefix.u;
          const float c_value = to_float<T>(c_shared[n * shared_stride + t]);
          y_vals[i] += (current_state)*c_value;
          final_state = current_state;
        }
      }

      if constexpr (kPrefetchState) {
        const float next_state = __shfl_sync(kWavefrontMask, final_state,
                                             (valid_tokens - 1) / kItems);
        if (lane == n) carry_lane = next_state;
      } else {
        if (lane * kItems <= valid_tokens - 1 &&
            (lane + 1) * kItems > valid_tokens - 1)
          state[state_base + n] = final_state;
      }
    }

#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const size_t t = token_indices[i];
      if (t < valid_tokens) {
        const size_t x_idx =
            (batch_idx * seq_len + tile_begin + t) * intermediate + ch;
        float y = finish_mamba_output(y_vals[i], x_vals[i], d_ch, gate_vals[i]);
        out[x_idx] = from_float<T>(y);
      }
    }
    __syncthreads();
  }
  if constexpr (kPrefetchState) {
    if (lane < state_size) state[state_base + lane] = carry_lane;
  }
}

__device__ __forceinline__ float decode_expf(float value) {
  return __builtin_amdgcn_exp2f(value * 1.4426950408889634f);
}

__device__ __forceinline__ float decode_logf(float value) {
  return __builtin_amdgcn_logf(value) * 0.6931471805599453f;
}

__device__ __forceinline__ float decode_softplusf(float value) {
  if (value > 20.0f) {
    return value;
  }
  const float exponential = decode_expf(value);
  if (value < -8.0f) {
    return exponential * (1.0f - 0.5f * exponential);
  }
  return decode_logf(1.0f + exponential);
}

template <int kPattern>
__device__ __forceinline__ float decode_quad(float value) {
  return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(value),
                                                 kPattern, 0xf, 0xf, false));
}

template <typename T>
__device__ __forceinline__ T decode_from_float(float value) {
  return from_float<T>(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
decode_from_float<__nv_bfloat16>(float value) {
  const unsigned int bits = __float_as_uint(value);
  const unsigned int rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  const unsigned short result =
      __builtin_amdgcn_classf(value, 3)
          ? 0x7fffu
          : static_cast<unsigned short>(rounded >> 16);
  return __ushort_as_bfloat16(result);
}

template <typename T, int kCount>
struct alignas(sizeof(T) * kCount) DecodeValues {
  T values[kCount];
};

template <typename T>
__device__ __forceinline__ T *decode_offset(T *base, ptrdiff_t index) {
  using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
  const ptrdiff_t offset = index * static_cast<ptrdiff_t>(sizeof(T));
  return reinterpret_cast<T *>(reinterpret_cast<Byte *>(base) + offset);
}

template <bool kCompact, typename T, typename Index>
__device__ __forceinline__ T *decode_indexed_offset(T *base, Index index) {
  if constexpr (kCompact) {
    using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
    const uint32_t bytes = static_cast<uint32_t>(index) * sizeof(T);
    return reinterpret_cast<T *>(reinterpret_cast<Byte *>(base) + bytes);
  } else {
    return decode_offset(base, static_cast<ptrdiff_t>(index));
  }
}

template <typename T, int kThreads, bool kAligned, int kLanes = 4,
          bool kPreserveQuadSum = false, bool kCompact = false>
__global__
__launch_bounds__(kThreads) void mamba_selective_scan_decode_s16_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state) {
  constexpr int kStatesPerLane = 16 / kLanes;
  using Index = std::conditional_t<kCompact, uint32_t, ptrdiff_t>;
  using StateVector = DecodeValues<float, kStatesPerLane>;
  using InputVector = DecodeValues<T, kStatesPerLane>;
  const int channel = blockIdx.x * (kThreads / kLanes) + threadIdx.x / kLanes;
  const int lane = threadIdx.x % kLanes;
  const Index batch_index = blockIdx.y;
  if constexpr (kCompact) {
    __builtin_assume(channel >= 0 && channel < 4096);
    __builtin_assume(batch_index < 8192);
  }
  const Index linear = batch_index * 4096 + channel;
  const Index state_base = linear * 16 + lane * kStatesPerLane;
  StateVector previous;
  if constexpr (kAligned) {
    previous = *reinterpret_cast<const StateVector *>(
        decode_indexed_offset<kCompact>(state, state_base));
  } else {
    __builtin_memcpy(&previous,
                     decode_indexed_offset<kCompact>(state, state_base),
                     sizeof(StateVector));
  }
  InputVector log_values;
  InputVector b_values;
  InputVector c_values;
  if constexpr (kAligned) {
    log_values =
        *reinterpret_cast<const InputVector *>(decode_indexed_offset<kCompact>(
            a_log, channel * 16 + lane * kStatesPerLane));
    b_values =
        *reinterpret_cast<const InputVector *>(decode_indexed_offset<kCompact>(
            b, batch_index * 16 + lane * kStatesPerLane));
    c_values =
        *reinterpret_cast<const InputVector *>(decode_indexed_offset<kCompact>(
            c, batch_index * 16 + lane * kStatesPerLane));
  }
  const float x_value =
      static_cast<float>(*decode_indexed_offset<kCompact>(x, linear));
  const float dt_value =
      static_cast<float>(*decode_indexed_offset<kCompact>(dt, linear)) +
      static_cast<float>(*decode_indexed_offset<kCompact>(dt_bias, channel));
  const float delta = decode_softplusf(dt_value);
  const float raw_gate =
      static_cast<float>(*decode_indexed_offset<kCompact>(gate, linear));
  const float gate_value =
      raw_gate * __builtin_amdgcn_rcpf(1.0f + decode_expf(-raw_gate));
  const float scaled_input = delta * x_value;
  const float scaled_delta = delta * 1.4426950408889634f;
  float residual = 0.0f;
  if constexpr (kLanes == 4) {
    residual = static_cast<float>(*decode_indexed_offset<kCompact>(d, channel));
  }
  StateVector updated;
  float result = 0.0f;
  // Keep the original four-lane (sum0 + sum2) + (sum1 + sum3) reduction order.
  float second_result = 0.0f;
#pragma unroll
  for (int index = 0; index < kStatesPerLane; ++index) {
    const int state_index = lane * kStatesPerLane + index;
    const float log_value = kAligned
                                ? to_float<T>(log_values.values[index])
                                : to_float<T>(*decode_indexed_offset<kCompact>(
                                      a_log, channel * 16 + state_index));
    const float coefficient = -decode_expf(log_value);
    const float b_value = kAligned
                              ? to_float<T>(b_values.values[index])
                              : to_float<T>(*decode_indexed_offset<kCompact>(
                                    b, batch_index * 16 + state_index));
    const float c_value = kAligned
                              ? to_float<T>(c_values.values[index])
                              : to_float<T>(*decode_indexed_offset<kCompact>(
                                    c, batch_index * 16 + state_index));
    const float current = __builtin_amdgcn_exp2f(coefficient * scaled_delta) *
                              previous.values[index] +
                          scaled_input * b_value;
    updated.values[index] = current;
    if constexpr (kPreserveQuadSum) {
      if (index < 4) {
        result += current * c_value;
      } else {
        second_result += current * c_value;
      }
    } else {
      result += current * c_value;
    }
  }
  // All batches use the same early store, shortening register liveness.
  if constexpr (kAligned)
    *reinterpret_cast<StateVector *>(
        decode_indexed_offset<kCompact>(state, state_base)) = updated;
  else
    __builtin_memcpy(decode_indexed_offset<kCompact>(state, state_base),
                     &updated, sizeof(StateVector));
  if constexpr (kPreserveQuadSum) {
    result += decode_quad<0xb1>(result);
    second_result += decode_quad<0xb1>(second_result);
    result += second_result;
  } else {
    if constexpr (kLanes >= 4) {
      result += decode_quad<0x4e>(result);
    }
    result += decode_quad<0xb1>(result);
  }
  if (lane == 0) {
    if constexpr (kLanes != 4) {
      residual =
          static_cast<float>(*decode_indexed_offset<kCompact>(d, channel));
    }
    *decode_indexed_offset<kCompact>(out, linear) =
        decode_from_float<T>((result + x_value * residual) * gate_value);
  }
}

template <typename T, bool kAligned, int kThreads = 256, bool kStagedBC = false>
__global__
__launch_bounds__(kThreads) void mamba_selective_scan_register_s16_kernel(
    T *out, const T *x, const T *dt, const T *b, const T *c, const T *a_log,
    const T *d, const T *gate, const T *dt_bias, float *state,
    std::conditional_t<kStagedBC, uint32_t, size_t> batch,
    std::conditional_t<kStagedBC, uint32_t, size_t> length,
    std::conditional_t<kStagedBC, uint32_t, size_t> channels_arg) {
  using Index = std::conditional_t<kStagedBC, uint32_t, size_t>;
  const Index channels = kStagedBC ? 4096 : channels_arg;
  constexpr int kLanes = 4;
  constexpr int kStatesPerLane = 16 / kLanes;
  const Index linear = static_cast<Index>(blockIdx.x) * (kThreads / kLanes) +
                       threadIdx.x / kLanes;
  if (linear >= batch * channels) return;
  const int lane = threadIdx.x % kLanes;
  const Index channel = linear % channels, row = linear / channels;
  const Index state_base = linear * 16 + lane * kStatesPerLane;
  using StateValues = DecodeValues<float, kStatesPerLane>;
  using InputValues = DecodeValues<T, kStatesPerLane>;
  StateValues current;
  InputValues logs;
  if constexpr (kAligned) {
    current = *reinterpret_cast<const StateValues *>(state + state_base);
    logs = *reinterpret_cast<const InputValues *>(a_log + channel * 16 +
                                                  lane * kStatesPerLane);
  } else {
#pragma unroll
    for (int i = 0; i < kStatesPerLane; ++i) {
      current.values[i] = state[state_base + i];
      logs.values[i] = a_log[channel * 16 + lane * kStatesPerLane + i];
    }
  }
  // Staged blocks contain only complete channel groups from one request.
  // Eight KiB keeps multiple register-recursion blocks resident per CU.
  extern __shared__ unsigned char shared_storage[];
  T *shared_b = reinterpret_cast<T *>(shared_storage);
  T *shared_c = shared_b + length * 16;
  if constexpr (kStagedBC) {
    for (Index index = threadIdx.x; index < length * 16; index += kThreads) {
      shared_b[index] = b[row * length * 16 + index];
      shared_c[index] = c[row * length * 16 + index];
    }
    __syncthreads();
  }
  float coefficients[kStatesPerLane];
#pragma unroll
  for (int i = 0; i < kStatesPerLane; ++i)
    coefficients[i] = -decode_expf(to_float<T>(logs.values[i]));
  const float residual = lane == 0 ? to_float<T>(d[channel]) : 0.0f;
  const float bias = lane == 0 ? to_float<T>(dt_bias[channel]) : 0.0f;
#pragma unroll 1
  for (Index token = 0; token < length; ++token) {
    const Index offset = (row * length + token) * channels + channel;
    float value = 0.0f, delta = 0.0f, activated_gate = 0.0f;
    if (lane == 0) {
      value = to_float<T>(x[offset]);
      delta = decode_softplusf(to_float<T>(dt[offset]) + bias);
      const float raw_gate = to_float<T>(gate[offset]);
      activated_gate =
          raw_gate * __builtin_amdgcn_rcpf(1.0f + decode_expf(-raw_gate));
    }
    value = __shfl_sync(kWavefrontMask, value, 0, kLanes);
    delta = __shfl_sync(kWavefrontMask, delta, 0, kLanes);
    activated_gate = __shfl_sync(kWavefrontMask, activated_gate, 0, kLanes);
    const Index bc_offset = (row * length + token) * 16 + lane * kStatesPerLane;
    const T *token_b = kStagedBC ? shared_b + token * 16 + lane * kStatesPerLane
                                 : b + bc_offset;
    const T *token_c = kStagedBC ? shared_c + token * 16 + lane * kStatesPerLane
                                 : c + bc_offset;
    InputValues b_values, c_values;
    if constexpr (kAligned) {
      b_values = *reinterpret_cast<const InputValues *>(token_b);
      c_values = *reinterpret_cast<const InputValues *>(token_c);
    } else {
#pragma unroll
      for (int i = 0; i < kStatesPerLane; ++i) {
        b_values.values[i] = token_b[i];
        c_values.values[i] = token_c[i];
      }
    }
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kStatesPerLane; ++i) {
      current.values[i] =
          decode_expf(coefficients[i] * delta) * current.values[i] +
          delta * to_float<T>(b_values.values[i]) * value;
      dot += current.values[i] * to_float<T>(c_values.values[i]);
    }
    dot += decode_quad<0x4e>(dot);
    dot += decode_quad<0xb1>(dot);
    if (lane == 0)
      out[offset] =
          decode_from_float<T>((dot + value * residual) * activated_gate);
  }
  if constexpr (kAligned)
    *reinterpret_cast<StateValues *>(state + state_base) = current;
  else {
#pragma unroll
    for (int i = 0; i < kStatesPerLane; ++i)
      state[state_base + i] = current.values[i];
  }
}

template <typename T, int kThreads, bool kAligned, int kLanes = 4,
          bool kPreserveQuadSum = false>
void launch_decode_s16_impl(const MambaScanLayout &p, T *out, const T *x,
                            const T *dt, const T *b, const T *c, const T *a_log,
                            const T *d, const T *gate, const T *dt_bias,
                            float *state, cudaStream_t stream) {
  const dim3 blocks(4096 / (kThreads / kLanes),
                    static_cast<unsigned int>(p.batch));
  const auto launch_indexed = [&](auto compact) {
    constexpr bool kCompact = decltype(compact)::value;
    mamba_selective_scan_decode_s16_kernel<T, kThreads, kAligned, kLanes,
                                           kPreserveQuadSum, kCompact>
        <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state);
  };
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    constexpr size_t kMaxBatch = 2147483647ULL / (4096 * 16 * sizeof(float));
    if (p.batch <= kMaxBatch) {
      launch_indexed(std::true_type{});
      return;
    }
  }
  launch_indexed(std::false_type{});
}

template <typename T>
void launch_decode_s16(const MambaScanLayout &p, T *out, const T *x,
                       const T *dt, const T *b, const T *c, const T *a_log,
                       const T *d, const T *gate, const T *dt_bias,
                       float *state, cudaStream_t stream, int compute_units) {
  const bool aligned = (reinterpret_cast<uintptr_t>(state) & 15) == 0 &&
                       (reinterpret_cast<uintptr_t>(a_log) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(b) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(c) & 7) == 0;
  const bool wide_aligned = (reinterpret_cast<uintptr_t>(state) & 31) == 0 &&
                            (reinterpret_cast<uintptr_t>(a_log) & 15) == 0 &&
                            (reinterpret_cast<uintptr_t>(b) & 15) == 0 &&
                            (reinterpret_cast<uintptr_t>(c) & 15) == 0;
  const size_t wide_blocks = p.batch * (4096 / (512 / 2));
  if (wide_aligned &&
      wide_blocks >= static_cast<size_t>(std::max(1, compute_units / 2)))
    return launch_decode_s16_impl<T, 512, true, 2,
                                  std::is_same_v<T, __nv_bfloat16>>(
        p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
  if (aligned)
    return launch_decode_s16_impl<T, 256, true>(p, out, x, dt, b, c, a_log, d,
                                                gate, dt_bias, state, stream);
  return launch_decode_s16_impl<T, 256, false>(p, out, x, dt, b, c, a_log, d,
                                               gate, dt_bias, state, stream);
}

// ---------------------------------------------------------------------------
// Host-side launcher: shape-based dispatch migrated from the old Hygon
// descriptor. T=1 decode launches the standalone optimized scan.
// ---------------------------------------------------------------------------
template <typename T, int kItems, int kGroupWaves, bool kCanonical>
void launch_grouped_scan_impl(const MambaScanLayout &p, T *out, const T *x,
                              const T *dt, const T *b, const T *c,
                              const T *a_log, const T *d, const T *gate,
                              const T *dt_bias, float *state,
                              cudaStream_t stream) {
  const size_t states = kCanonical ? 16 : p.states;
  const size_t stride = kWavefrontSize * kItems + (kCanonical ? 4 : 0);
  const size_t shared_bytes = 2 * stride * states * sizeof(T);
  const dim3 blocks((p.batch * p.channels + kGroupWaves - 1) / kGroupWaves);
  mamba_selective_scan_grouped_shared_kernel<T, kItems, kGroupWaves, kCanonical>
      <<<blocks, kWavefrontSize * kGroupWaves, shared_bytes, stream>>>(
          out, x, dt, b, c, a_log, d, gate, dt_bias, state, p.batch, p.length,
          p.channels, p.states);
}

template <typename T, int kItems, int kGroupWaves>
void launch_grouped_scan(const MambaScanLayout &p, T *out, const T *x,
                         const T *dt, const T *b, const T *c, const T *a_log,
                         const T *d, const T *gate, const T *dt_bias,
                         float *state, cudaStream_t stream) {
  if (p.channels == 4096 && p.states == 16)
    return launch_grouped_scan_impl<T, kItems, kGroupWaves, true>(
        p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
  return launch_grouped_scan_impl<T, kItems, kGroupWaves, false>(
      p, out, x, dt, b, c, a_log, d, gate, dt_bias, state, stream);
}

// Cap shared B/C storage at eight KiB per block; larger inputs retain
// the original register recurrence. These are storage/resource bounds.
constexpr size_t kRegisterBCBytes = 8 * 1024;
inline bool can_stage_register_bc(const MambaScanLayout &p) {
  constexpr size_t kByteLimit = 2147483647ULL;
  const size_t total = p.batch * p.channels;
  return p.channels == 4096 && total && p.length &&
         total <= kByteLimit / (16 * sizeof(float)) &&
         p.length <= kByteLimit / (total * sizeof(uint16_t)) &&
         p.length <= kRegisterBCBytes / (2 * 16 * sizeof(uint16_t));
}

inline int register_scan_threads(const MambaScanLayout &p, int compute_units,
                                 int fallback_threads) {
  if (!can_stage_register_bc(p)) return fallback_threads;
  // Spread small workloads across more blocks; use wider blocks once there
  // are enough independent channels to amortize the shared tile per block.
  return p.batch * p.channels >= static_cast<size_t>(compute_units) * 256 ? 256
                                                                          : 128;
}

// Shape/resource decisions are independent of tensor storage addresses.
// Alignment remains an invocation-time choice in the family launchers.
enum class ScanFamily { kDecodeS16, kRegisterS16, kPrefix, kGeneric };

struct ScanLaunchPlan {
  ScanFamily family;
  int threads = 0;
  int items = 0;
  int waves = 0;
};

inline ScanLaunchPlan select_scan_plan(const MambaScanLayout &p,
                                       int compute_units) {
  const size_t total = p.batch * p.channels;
  if (p.length == 1 && p.states == 16 && p.channels == 4096 && p.batch <= 65535)
    return {ScanFamily::kDecodeS16};
  if (p.states == 16) {
    if (p.length <= 64)
      return {ScanFamily::kRegisterS16,
              register_scan_threads(p, compute_units, 256)};
    // Two independent channel blocks per compute unit hide the recurrence
    // without paying the affine-prefix overhead.
    const size_t independent_blocks = (total + 128 / 4 - 1) / (128 / 4);
    if (independent_blocks >= static_cast<size_t>(compute_units) * 2)
      return {ScanFamily::kRegisterS16,
              register_scan_threads(p, compute_units, 128)};
  }
  if (p.channels % 4 != 0 || p.states > 64) return {ScanFamily::kGeneric, 256};
  if (p.length <= 64) return {ScanFamily::kPrefix, 0, 1, 4};
  const size_t block_budget = static_cast<size_t>(compute_units) * 8;
  if (p.channels % 16 == 0 && total / 16 >= block_budget)
    return {ScanFamily::kPrefix, 0, 2, 16};
  if (p.channels % 8 == 0 && total / 8 >= block_budget)
    return {ScanFamily::kPrefix, 0, 2, 8};
  return {ScanFamily::kPrefix, 0, 2, 4};
}

template <typename T, int kThreads>
void launch_register_s16(const MambaScanLayout &p, T *out, const T *x,
                         const T *dt, const T *b, const T *c, const T *a_log,
                         const T *d, const T *gate, const T *dt_bias,
                         float *state, cudaStream_t stream) {
  const size_t total = p.batch * p.channels;
  const dim3 blocks((total + kThreads / 4 - 1) / (kThreads / 4));
  const bool aligned = (reinterpret_cast<uintptr_t>(state) & 15) == 0 &&
                       (reinterpret_cast<uintptr_t>(a_log) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(b) & 7) == 0 &&
                       (reinterpret_cast<uintptr_t>(c) & 7) == 0;
  if (aligned && can_stage_register_bc(p)) {
    const size_t shared_bytes = 2 * p.length * 16 * sizeof(T);
    mamba_selective_scan_register_s16_kernel<T, true, kThreads, true>
        <<<blocks, kThreads, shared_bytes, stream>>>(
            out, x, dt, b, c, a_log, d, gate, dt_bias, state,
            static_cast<uint32_t>(p.batch), static_cast<uint32_t>(p.length),
            static_cast<uint32_t>(p.channels));
  } else if (aligned) {
    mamba_selective_scan_register_s16_kernel<T, true, kThreads>
        <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state, p.batch, p.length,
                                          p.channels);
  } else {
    mamba_selective_scan_register_s16_kernel<T, false, kThreads>
        <<<blocks, kThreads, 0, stream>>>(out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state, p.batch, p.length,
                                          p.channels);
  }
}

template <typename T>
void launch_scan_dispatch(const MambaScanLayout &p, T *out, const T *x,
                          const T *dt, const T *b, const T *c, const T *a_log,
                          const T *d, const T *gate, const T *dt_bias,
                          float *state, cudaStream_t stream,
                          int compute_units) {
  const auto plan = select_scan_plan(p, compute_units);
  switch (plan.family) {
    case ScanFamily::kDecodeS16:
      return launch_decode_s16<T>(p, out, x, dt, b, c, a_log, d, gate, dt_bias,
                                  state, stream, compute_units);
    case ScanFamily::kRegisterS16:
      if (plan.threads == 256)
        return launch_register_s16<T, 256>(p, out, x, dt, b, c, a_log, d, gate,
                                           dt_bias, state, stream);
      return launch_register_s16<T, 128>(p, out, x, dt, b, c, a_log, d, gate,
                                         dt_bias, state, stream);
    case ScanFamily::kPrefix:
      if (plan.items == 1)
        return launch_grouped_scan<T, 1, 4>(p, out, x, dt, b, c, a_log, d, gate,
                                            dt_bias, state, stream);
      if (plan.waves == 16)
        return launch_grouped_scan<T, 2, 16>(p, out, x, dt, b, c, a_log, d,
                                             gate, dt_bias, state, stream);
      if (plan.waves == 8)
        return launch_grouped_scan<T, 2, 8>(p, out, x, dt, b, c, a_log, d, gate,
                                            dt_bias, state, stream);
      return launch_grouped_scan<T, 2, 4>(p, out, x, dt, b, c, a_log, d, gate,
                                          dt_bias, state, stream);
    case ScanFamily::kGeneric:
      break;
  }
  constexpr unsigned threads = 256;
  const dim3 blocks((p.batch * p.channels + threads - 1) / threads);
  mamba_selective_scan_kernel<T><<<blocks, threads, 0, stream>>>(
      out, x, dt, b, c, a_log, d, gate, dt_bias, state, p.batch, p.length,
      p.channels, p.states);
}
}  // anonymous namespace
}  // namespace hygon_mamba_scan_detail

struct MambaScanHygonBackend : Runtime<Device::Type::kHygon> {
  static void CheckLaunch() {
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon MambaSelectiveScan CUDA launch failed");
  }
};

template <>
class Operator<MambaSelectiveScan, Device::Type::kHygon>
    : public CudaMambaSelectiveScan<MambaScanHygonBackend> {
 public:
  Operator(Tensor out, const Tensor x, const Tensor dt, const Tensor b,
           const Tensor c, const Tensor a_log, const Tensor d,
           const Tensor gate, const Tensor dt_bias, Tensor state)
      : CudaMambaSelectiveScan<MambaScanHygonBackend>(out, x, dt, b, c, a_log,
                                                      d, gate, dt_bias, state),
        prepared_{out, x, dt, b, c, a_log, d, gate, dt_bias, state} {
    if (device_.type() != Device::Type::kHygon)
      throw std::invalid_argument("Hygon mamba scan requires Hygon device");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device_.index()) != cudaSuccess)
      throw std::runtime_error(
          "Hygon mamba scan device properties unavailable");
    compute_units_ = std::max(1, properties.multiProcessorCount);
    max_grouped_elements_ = static_cast<size_t>(properties.maxGridSize[0]) * 4;
    for (size_t i = 0; i < prepared_.size(); ++i) {
      const auto &t = prepared_[i];
      if (!t.numel()) continue;
      size_t span = 1;
      for (size_t axis = 0; axis < t.ndim(); ++axis)
        span += (t.size(axis) - 1) * t.stride(axis);
      spans_[i] = span * (t.dtype() == DataType::kFloat32 ? 4 : 2);
    }
  }

  void operator()(Tensor out, const Tensor x, const Tensor dt, const Tensor b,
                  const Tensor c, const Tensor a_log, const Tensor d,
                  const Tensor gate, const Tensor dt_bias,
                  Tensor state) const override {
    const std::array<Tensor, 10> tensors{out,   x, dt,   b,       c,
                                         a_log, d, gate, dt_bias, state};
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto &t = tensors[i];
      if (!std::equal_to<Tensor>{}(t, prepared_[i]) || (t.numel() && !t.data()))
        throw std::invalid_argument(
            "Hygon mamba scan metadata changed or nonempty storage is null");
    }
    check_aliases_(tensors);
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != device_.index())
      throw std::invalid_argument("Hygon mamba scan owner device mismatch");
    if (!layout_.batch || !layout_.length || !layout_.channels) return;
    // Optimized paths require contiguous low-precision data and matching
    // parameter dtypes. Other supported layouts/dtypes use the generic path.
    if (layout_.contiguous && a_log.dtype() == dtype_ && d.dtype() == dtype_ &&
        dtype_ != DataType::kFloat32 &&
        layout_.batch * layout_.channels <= max_grouped_elements_) {
      const auto stream = static_cast<cudaStream_t>(stream_);
      DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
          dtype_,
          [&](auto tag) {
            using T = TypeMapType<Device::Type::kHygon, decltype(tag)::value>;
            hygon_mamba_scan_detail::launch_scan_dispatch<T>(
                layout_, static_cast<T *>(out.data()),
                static_cast<const T *>(x.data()),
                static_cast<const T *>(dt.data()),
                static_cast<const T *>(b.data()),
                static_cast<const T *>(c.data()),
                static_cast<const T *>(a_log.data()),
                static_cast<const T *>(d.data()),
                static_cast<const T *>(gate.data()),
                static_cast<const T *>(dt_bias.data()),
                static_cast<float *>(state.data()), stream, compute_units_);
          },
          "HygonMambaSelectiveScan");
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon mamba scan launch failed");
      return;
    }
    CudaMambaSelectiveScan<MambaScanHygonBackend>::operator()(
        out, x, dt, b, c, a_log, d, gate, dt_bias, state);
  }

  bool capture_safe() const override { return true; }

 private:
  std::array<Tensor, 10> prepared_;
  std::array<size_t, 10> spans_{};
  void check_aliases_(const std::array<Tensor, 10> &tensors) const {
    for (size_t dst : {size_t{0}, size_t{9}})
      for (size_t src = 0; src < tensors.size(); ++src) {
        if (src == dst || !spans_[dst] || !spans_[src]) continue;
        const auto x = reinterpret_cast<uintptr_t>(tensors[dst].data());
        const auto y = reinterpret_cast<uintptr_t>(tensors[src].data());
        if (!(x <= y ? y - x < spans_[dst] : x - y < spans_[src])) continue;
        if (dst == 0 && (src == 1 || src == 2 || src == 7) && x == y &&
            tensors[dst].strides() == tensors[src].strides())
          continue;
        throw std::invalid_argument(
            "Hygon mamba scan unsupported tensor overlap");
      }
  }
  int compute_units_ = 1;
  size_t max_grouped_elements_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_MAMBA_SELECTIVE_SCAN_KERNEL_H_
