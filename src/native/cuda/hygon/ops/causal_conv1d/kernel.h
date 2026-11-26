// Hygon-specific causal_conv1d kernels migrated from the InfiniCore
// indicator-5 submission (zxk branch, 2026-10-31):
// src/infiniop/ops/causal_conv1d/hygon/causal_conv1d_hygon.cu.
// Decode, tiled prefill, and general indexed/strided paths share the
// InfiniOps operator-specialization interface.
// Copyright 2025 InfiniTensor; MIT notice: LICENSES/InfiniCore-MIT.txt.
#ifndef INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "base/causal_conv1d.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops {
namespace hygon_conv_detail {

struct ConvInfo {
  size_t request_count, T, C, pool_size, total_tokens;
  int compute_units = 1;
  int max_threads_per_block = 1024;
  const void *history = nullptr;  // Per-request snapshot, owned by workspace.
  bool has_bias, has_cu_seqlens, indexed_state_pool;
  int cu_seqlens_kind, initial_indices_kind,
      final_indices_kind;  // 0=none 1=i32 2=i64
  ptrdiff_t out_strides[3]{}, conv_state_strides[3]{}, qkv_strides[3]{};
  ptrdiff_t weight_strides[3]{};
  std::vector<ptrdiff_t> bias_strides;
  std::vector<ptrdiff_t> final_conv_state_strides;
};

// The contiguous specialization is a layout property, not a benchmark shape
// whitelist. Batch and sequence length remain runtime dimensions.
inline bool has_indexed_k4_layout(const ConvInfo &info) {
  return info.indexed_state_pool && info.T > 0 && info.C == 4096 &&
         info.initial_indices_kind != 2 && info.final_indices_kind == 1 &&
         info.has_bias &&
         info.qkv_strides[0] == static_cast<ptrdiff_t>(info.T * 4096) &&
         info.qkv_strides[1] == 4096 && info.qkv_strides[2] == 1 &&
         info.out_strides[0] == static_cast<ptrdiff_t>(info.T * 4096) &&
         info.out_strides[1] == 4096 && info.out_strides[2] == 1 &&
         info.conv_state_strides[0] == 4096 * 3 &&
         info.conv_state_strides[1] == 3 && info.conv_state_strides[2] == 1 &&
         info.weight_strides[0] == 4 && info.weight_strides[2] == 1 &&
         !info.bias_strides.empty() && info.bias_strides[0] == 1;
}

// Public metadata validation must remain active in release builds. The base
// key's legacy assert checks do not protect malformed ranks under NDEBUG.
inline Tensor checked_input(
    const Tensor qkv, const Tensor weight, const Tensor state,
    const std::optional<Tensor> &bias, const std::optional<Tensor> &cu,
    const std::optional<Tensor> &initial, const std::optional<Tensor> &final,
    const Tensor out, const std::optional<Tensor> &final_state) {
  if (qkv.ndim() != 3 || weight.ndim() != 3 || state.ndim() != 3 ||
      out.ndim() != 3)
    throw std::invalid_argument(
        "Hygon causal conv requires rank-three data tensors");
  const auto dtype = qkv.dtype();
  if (dtype != DataType::kFloat32 && dtype != DataType::kFloat16 &&
      dtype != DataType::kBFloat16)
    throw std::invalid_argument("Hygon causal conv dtype unsupported");
  const size_t channels = qkv.size(2);
  if (out.shape() != qkv.shape() ||
      weight.shape() != Tensor::Shape{channels, 1, 4} ||
      state.size(1) != channels || state.size(2) != 3 ||
      final.has_value() == final_state.has_value())
    throw std::invalid_argument(
        "Hygon causal conv weight/state/output metadata mismatch");
  size_t requests = qkv.size(0);
  if (cu) {
    if (qkv.size(0) != 1 || cu->ndim() != 1 || cu->numel() < 2)
      throw std::invalid_argument(
          "Hygon causal conv packed sequence metadata mismatch");
    requests = cu->numel() - 1;
  }
  if ((!initial && state.size(0) < requests) ||
      (final_state &&
       final_state->shape() != Tensor::Shape{requests, channels, 3}) ||
      (bias && bias->shape() != Tensor::Shape{channels}))
    throw std::invalid_argument(
        "Hygon causal conv state pool/bias metadata mismatch");
  const std::array<std::optional<Tensor>, 9> views{
      qkv, weight, state, bias, cu, initial, final, out, final_state};
  for (size_t i = 0; i < views.size(); ++i) {
    if (!views[i]) continue;
    const auto &t = *views[i];
    if (t.device() != qkv.device() || t.ndim() != t.strides().size())
      throw std::invalid_argument(
          "Hygon causal conv device or stride rank mismatch");
    const bool index = i >= 4 && i <= 6;
    if (index ? (t.ndim() != 1 || !t.IsContiguous() ||
                 (t.dtype() != DataType::kInt32 &&
                  t.dtype() != DataType::kInt64) ||
                 (i != 4 && t.numel() != requests))
              : t.dtype() != dtype)
      throw std::invalid_argument(
          "Hygon causal conv tensor dtype or index metadata mismatch");
    size_t span = 1;
    const size_t limit =
        static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    for (size_t axis = 0; axis < t.ndim(); ++axis) {
      if (t.stride(axis) < 0)
        throw std::invalid_argument(
            "Hygon causal conv negative stride unsupported");
      const auto stride = static_cast<size_t>(t.stride(axis));
      if (t.size(axis) && stride && t.size(axis) - 1 > (limit - span) / stride)
        throw std::overflow_error("Hygon causal conv tensor span overflow");
      if (t.size(axis)) span += (t.size(axis) - 1) * stride;
    }
    const size_t width =
        t.dtype() == DataType::kInt64
            ? 8
            : (t.dtype() == DataType::kFloat32 || t.dtype() == DataType::kInt32
                   ? 4
                   : 2);
    if (span > limit / width)
      throw std::overflow_error("Hygon causal conv tensor byte span overflow");
    if (t.numel() && ((i == 2 && final) || i == 7 || i == 8)) {
      size_t extent = 1;
      for (int axis = static_cast<int>(t.ndim()) - 1; axis >= 0; --axis) {
        if (t.size(axis) > 1 && static_cast<size_t>(t.stride(axis)) < extent)
          throw std::invalid_argument(
              "Hygon causal conv overlapping writable layout");
        if (t.size(axis))
          extent += (t.size(axis) - 1) * static_cast<size_t>(t.stride(axis));
      }
    }
  }
  return qkv;
}

namespace {
struct __align__(16) half8 { half lane[8]; };

struct __align__(16) bfloat168 { __nv_bfloat16 lane[8]; };

template <typename T>
struct causal_vec16;

template <>
struct causal_vec16<float> {
  using type = float4;
};

template <>
struct causal_vec16<half> {
  using type = half8;
};

template <>
struct causal_vec16<__nv_bfloat16> {
  using type = bfloat168;
};

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, ptrdiff_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half *ptr,
                                                     ptrdiff_t offset) {
  return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16 *ptr, ptrdiff_t offset) {
  return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ T cast_from_float(float x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half cast_from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16
cast_from_float<__nv_bfloat16>(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  const uint16_t result = (bits & 0x7fffffffu) > 0x7f800000u
                              ? 0x7fffu
                              : static_cast<uint16_t>(rounded >> 16);
  return __ushort_as_bfloat16(result);
}

template <typename T>
__device__ __forceinline__ float conv4_with_bias(float w0, float h0, float w1,
                                                 float h1, float w2, float h2,
                                                 float w3, float h3,
                                                 float bias) {
  float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3;
  if constexpr (std::is_same_v<T, half>) {
    T rounded = cast_from_float<T>(acc);
    acc = load_as_float(&rounded, 0);
  }
  return acc + bias;
}

template <typename T, int kVec>
__device__ __forceinline__ void load_vec16(const T *ptr, ptrdiff_t offset,
                                           float *values) {
  using Vec = typename causal_vec16<T>::type;
  const Vec value = *reinterpret_cast<const Vec *>(ptr + offset);
  const T *lanes = reinterpret_cast<const T *>(&value);
#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    values[lane] = load_as_float(lanes, lane);
  }
}

template <typename T, int kVec>
__device__ __forceinline__ void store_vec16(T *ptr, ptrdiff_t offset,
                                            const float *values) {
  using Vec = typename causal_vec16<T>::type;
  Vec value;
  T *lanes = reinterpret_cast<T *>(&value);
#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    lanes[lane] = cast_from_float<T>(values[lane]);
  }
  *reinterpret_cast<Vec *>(ptr + offset) = value;
}

__device__ __forceinline__ int64_t load_optional_index(const void *indices,
                                                       bool is_i64, int idx,
                                                       int fallback) {
  if (indices == nullptr) {
    return static_cast<int64_t>(fallback);
  }
  return is_i64
             ? static_cast<const int64_t *>(indices)[idx]
             : static_cast<int64_t>(static_cast<const int32_t *>(indices)[idx]);
}

template <typename T, int kThreads>
__global__ __launch_bounds__(kThreads) void causal_conv1d_k4_decode_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t pool_size, const T *history_snapshot) {
  constexpr int kChannels = 4096;
  const int request = blockIdx.y;
  const int channel = blockIdx.x * kThreads + threadIdx.x;
  const int read_slot = initial_state_indices == nullptr
                            ? request
                            : initial_state_indices[request];
  const int write_slot = final_state_indices[request];
  if (read_slot < 0 || write_slot < 0 ||
      static_cast<size_t>(read_slot) >= pool_size ||
      static_cast<size_t>(write_slot) >= pool_size) {
    return;
  }
  const ptrdiff_t read_base =
      (static_cast<ptrdiff_t>(history_snapshot ? request : read_slot) *
           kChannels +
       channel) *
      3;
  const ptrdiff_t write_base =
      (static_cast<ptrdiff_t>(write_slot) * kChannels + channel) * 3;
  const T *history_source = history_snapshot ? history_snapshot : conv_state;
  const T history0 = history_source[read_base];
  const T history1 = history_source[read_base + 1];
  const T history2 = history_source[read_base + 2];
  const ptrdiff_t input_index =
      static_cast<ptrdiff_t>(request) * kChannels + channel;
  const T current = qkv[input_index];
  float weights[4];
  if ((reinterpret_cast<uintptr_t>(weight) & 7) == 0) {
    const uint2 packed = *reinterpret_cast<const uint2 *>(weight + channel * 4);
    const T *values = reinterpret_cast<const T *>(&packed);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      weights[index] = load_as_float(values, index);
    }
  } else {
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      weights[index] = load_as_float(weight, channel * 4 + index);
    }
  }
  const float result = conv4_with_bias<T>(
      weights[0], load_as_float(&history0, 0), weights[1],
      load_as_float(&history1, 0), weights[2], load_as_float(&history2, 0),
      weights[3], load_as_float(&current, 0), load_as_float(bias, channel));
  out[input_index] = cast_from_float<T>(result);
  conv_state[write_base] = history1;
  conv_state[write_base + 1] = history2;
  conv_state[write_base + 2] = current;
}

template <typename T>
__device__ __forceinline__ float causal_decode_bits(unsigned int bits) {
  const unsigned short raw = static_cast<unsigned short>(bits);
  T value;
  __builtin_memcpy(&value, &raw, sizeof(T));
  return load_as_float<T>(&value, 0);
}

template <typename T>
__device__ __forceinline__ unsigned short causal_encode_bits(float value) {
  const T converted = cast_from_float<T>(value);
  unsigned short bits;
  __builtin_memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

// Compact kernels are selected only when every addressed byte fits in
// signed 32 bits. Larger state pools retain the original size_t path.
template <bool kCompact, typename T, typename Index>
__device__ __forceinline__ T *causal_decode_indexed_offset(T *base,
                                                           Index index) {
  if constexpr (kCompact) {
    using Byte = std::conditional_t<std::is_const_v<T>, const char, char>;
    const uint32_t bytes = static_cast<uint32_t>(index) * sizeof(T);
    return reinterpret_cast<T *>(reinterpret_cast<Byte *>(base) + bytes);
  } else {
    return base + index;
  }
}

template <>
__device__ __forceinline__ unsigned short causal_encode_bits<__nv_bfloat16>(
    float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return __builtin_amdgcn_classf(value, 3)
             ? 0x7fffu
             : static_cast<unsigned short>(rounded >> 16);
}

struct alignas(4) CausalHistoryPair {
  unsigned int values[3];
};

__device__ __forceinline__ float causal_conv1d_fp16_accumulate_pair(
    unsigned int w01, unsigned int w23, unsigned int h01, unsigned int h2x) {
  return causal_decode_bits<half>(w01) * causal_decode_bits<half>(h01) +
         causal_decode_bits<half>(w01 >> 16) *
             causal_decode_bits<half>(h01 >> 16) +
         causal_decode_bits<half>(w23) * causal_decode_bits<half>(h2x) +
         causal_decode_bits<half>(w23 >> 16) *
             causal_decode_bits<half>(h2x >> 16);
}

// Round the accumulation to FP16 before adding the FP16 bias, as in
// conv4_with_bias<half>. Packed addition avoids the intermediate float
// conversion while retaining both rounding points.
__device__ __forceinline__ unsigned int causal_conv1d_fp16_pair_output(
    const CausalHistoryPair &history, const uint4 &weights,
    unsigned int current, unsigned int biases) {
  const float first = causal_conv1d_fp16_accumulate_pair(
      weights.x, weights.y, history.values[0],
      (history.values[1] & 0xffffu) | (current << 16));
  const float second = causal_conv1d_fp16_accumulate_pair(
      weights.z, weights.w,
      (history.values[1] >> 16) | (history.values[2] << 16),
      (history.values[2] >> 16) | (current & 0xffff0000u));
  const __half2 rounded =
      __halves2half2(__float2half_rn(first), __float2half_rn(second));
  __half2 bias_pair;
  __builtin_memcpy(&bias_pair, &biases, sizeof(bias_pair));
  const __half2 output = __hadd2(rounded, bias_pair);
  unsigned int packed;
  __builtin_memcpy(&packed, &output, sizeof(packed));
  return packed;
}

template <typename T, int kThreads, bool kCompact = false>
__global__ __launch_bounds__(kThreads) void causal_conv1d_k4_decode_pair_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    std::conditional_t<kCompact, uint32_t, size_t> pool_size,
    const T *history_snapshot) {
  constexpr int kChannels = 4096;
  using Index = std::conditional_t<kCompact, uint32_t, ptrdiff_t>;
  const int request = blockIdx.y;
  const int channel = (blockIdx.x * kThreads + threadIdx.x) * 2;
  const int read_slot = initial_state_indices == nullptr
                            ? request
                            : initial_state_indices[request];
  const int write_slot = final_state_indices[request];
  if constexpr (kCompact) {
    if (static_cast<uint32_t>(read_slot) >= pool_size ||
        static_cast<uint32_t>(write_slot) >= pool_size)
      return;
    __builtin_assume(channel >= 0 && channel < 4096);
    __builtin_assume(request >= 0 && request < 65535);
  } else {
    if (read_slot < 0 || write_slot < 0 ||
        static_cast<size_t>(read_slot) >= pool_size ||
        static_cast<size_t>(write_slot) >= pool_size)
      return;
  }
  const Index read_base =
      (static_cast<Index>(history_snapshot ? request : read_slot) * kChannels +
       channel) *
      3;
  const Index write_base =
      (static_cast<Index>(write_slot) * kChannels + channel) * 3;
  const Index input_base = static_cast<Index>(request) * kChannels + channel;
  const CausalHistoryPair history =
      *reinterpret_cast<const CausalHistoryPair *>(
          causal_decode_indexed_offset<kCompact>(
              (history_snapshot ? history_snapshot : conv_state), read_base));
  const uint4 weights = *reinterpret_cast<const uint4 *>(
      causal_decode_indexed_offset<kCompact>(weight, channel * 4));
  const unsigned int current = *reinterpret_cast<const unsigned int *>(
      causal_decode_indexed_offset<kCompact>(qkv, input_base));
  const unsigned int biases = *reinterpret_cast<const unsigned int *>(
      causal_decode_indexed_offset<kCompact>(bias, channel));
  if constexpr (std::is_same_v<T, half>) {
    *reinterpret_cast<unsigned int *>(
        causal_decode_indexed_offset<kCompact>(out, input_base)) =
        causal_conv1d_fp16_pair_output(history, weights, current, biases);
  } else {
    const unsigned int packed_weights[4] = {weights.x, weights.y, weights.z,
                                            weights.w};
    const unsigned int raw_history[6] = {
        history.values[0], history.values[0] >> 16,
        history.values[1], history.values[1] >> 16,
        history.values[2], history.values[2] >> 16};
    unsigned short encoded[2];
#pragma unroll
    for (int offset = 0; offset < 2; ++offset) {
      const unsigned int first = packed_weights[offset * 2];
      const unsigned int second = packed_weights[offset * 2 + 1];
      const float result =
          conv4_with_bias<T>(causal_decode_bits<T>(first),
                             causal_decode_bits<T>(raw_history[offset * 3]),
                             causal_decode_bits<T>(first >> 16),
                             causal_decode_bits<T>(raw_history[offset * 3 + 1]),
                             causal_decode_bits<T>(second),
                             causal_decode_bits<T>(raw_history[offset * 3 + 2]),
                             causal_decode_bits<T>(second >> 16),
                             causal_decode_bits<T>(current >> (offset * 16)),
                             causal_decode_bits<T>(biases >> (offset * 16)));
      encoded[offset] = causal_encode_bits<T>(result);
    }
    *reinterpret_cast<unsigned int *>(causal_decode_indexed_offset<kCompact>(
        out, input_base)) = static_cast<unsigned int>(encoded[0]) |
                            (static_cast<unsigned int>(encoded[1]) << 16);
  }
  CausalHistoryPair updated;
  updated.values[0] = (history.values[0] >> 16) | (history.values[1] << 16);
  updated.values[1] = (current & 0xffffu) | (history.values[2] << 16);
  updated.values[2] = (history.values[2] >> 16) | (current & 0xffff0000u);
  *reinterpret_cast<CausalHistoryPair *>(
      causal_decode_indexed_offset<kCompact>(conv_state, write_base)) = updated;
}

template <typename T, int kTimeTile, int kChannelsPerThread,
          bool kHasInitialState>
__global__ void causal_conv1d_k4_contiguous_tiled_kernel(
    T *out, T *conv_state, const T *qkv, const T *weight, const T *bias,
    const int32_t *initial_state_indices, const int32_t *final_state_indices,
    size_t token_count, size_t pool_size, const T *history_snapshot) {
  constexpr int kChannels = 4096;
  const int request = blockIdx.x;
  const int chunk = blockIdx.y;
  const int first_channel =
      blockIdx.z * blockDim.x * kChannelsPerThread + threadIdx.x;
  const int read_slot =
      kHasInitialState ? initial_state_indices[request] : request;
  const int write_slot = final_state_indices[request];
  if (read_slot < 0 || write_slot < 0 ||
      read_slot >= static_cast<int64_t>(pool_size) ||
      write_slot >= static_cast<int64_t>(pool_size)) {
    return;
  }

#pragma unroll
  for (int offset = 0; offset < kChannelsPerThread; ++offset) {
    const int channel = first_channel + offset * blockDim.x;
    if (channel >= kChannels) continue;
    const ptrdiff_t weight_offset = static_cast<ptrdiff_t>(channel) * 4;
    const float weight0 = load_as_float(weight, weight_offset);
    const float weight1 = load_as_float(weight, weight_offset + 1);
    const float weight2 = load_as_float(weight, weight_offset + 2);
    const float weight3 = load_as_float(weight, weight_offset + 3);
    const float bias_value = load_as_float(bias, channel);
    const ptrdiff_t batch_base =
        static_cast<ptrdiff_t>(request) * token_count * kChannels + channel;

    if (chunk == 0) {
      const ptrdiff_t read_base =
          (static_cast<ptrdiff_t>(history_snapshot ? request : read_slot) *
               kChannels +
           channel) *
          3;
      const T *history_source =
          history_snapshot ? history_snapshot : conv_state;
      float history0 = load_as_float(history_source, read_base);
      float history1 = load_as_float(history_source, read_base + 1);
      float history2 = load_as_float(history_source, read_base + 2);
      for (int token = 0; token < 3; ++token) {
        if (token >= token_count) {
          break;
        }
        const ptrdiff_t offset = batch_base + token * kChannels;
        const float current = load_as_float(qkv, offset);
        const float result =
            conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                               history2, weight3, current, bias_value);
        out[offset] = cast_from_float<T>(result);
        history0 = history1;
        history1 = history2;
        history2 = current;
      }

      if (token_count >= 4) {
        const ptrdiff_t final_offset =
            batch_base + (token_count - 1) * kChannels;
        const float previous0 =
            load_as_float(qkv, final_offset - 3 * kChannels);
        const float previous1 =
            load_as_float(qkv, final_offset - 2 * kChannels);
        const float previous2 = load_as_float(qkv, final_offset - kChannels);
        const float current = load_as_float(qkv, final_offset);
        const float result =
            conv4_with_bias<T>(weight0, previous0, weight1, previous1, weight2,
                               previous2, weight3, current, bias_value);
        out[final_offset] = cast_from_float<T>(result);
        history0 = previous1;
        history1 = previous2;
        history2 = current;
      }

      const ptrdiff_t write_base =
          (static_cast<ptrdiff_t>(write_slot) * kChannels + channel) * 3;
      conv_state[write_base] = cast_from_float<T>(history0);
      conv_state[write_base + 1] = cast_from_float<T>(history1);
      conv_state[write_base + 2] = cast_from_float<T>(history2);
    } else {
      const int64_t first_token =
          3 + static_cast<int64_t>(chunk - 1) * kTimeTile;
      const ptrdiff_t first_offset = batch_base + first_token * kChannels;
      float history0 = load_as_float(qkv, first_offset - 3 * kChannels);
      float history1 = load_as_float(qkv, first_offset - 2 * kChannels);
      float history2 = load_as_float(qkv, first_offset - kChannels);
#pragma unroll
      for (int tile_token = 0; tile_token < kTimeTile; ++tile_token) {
        const int64_t token = first_token + tile_token;
        if (token < static_cast<int64_t>(token_count) - 1) {
          const ptrdiff_t offset = first_offset + tile_token * kChannels;
          const float current = load_as_float(qkv, offset);
          const float result =
              conv4_with_bias<T>(weight0, history0, weight1, history1, weight2,
                                 history2, weight3, current, bias_value);
          out[offset] = cast_from_float<T>(result);
          history0 = history1;
          history1 = history2;
          history2 = current;
        }
      }
    }
  }
}

template <typename T, int kVec, bool kVectorIO, int kTimeTile, int kThreads,
          bool kIndexed>
__global__ __launch_bounds__(256) void causal_conv1d_k4_time_tiled_vec_kernel(
    T *out, T *conv_state, T *final_conv_state, const T *qkv, const T *weight,
    const T *bias, const void *initial_state_indices,
    const void *final_state_indices, bool has_bias,
    bool initial_state_indices_i64, bool final_state_indices_i64,
    size_t T_tokens, size_t C, size_t pool_size, ptrdiff_t out_s0,
    ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t state_s0, ptrdiff_t state_s1,
    ptrdiff_t state_s2, ptrdiff_t final_s0, ptrdiff_t final_s1,
    ptrdiff_t final_s2, ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0,
    const T *history_snapshot) {
  constexpr int kChannelsPerBlock = kThreads * kVec;
  constexpr bool kSingleChunk = kIndexed && kTimeTile == 16;

  const int64_t request = static_cast<int64_t>(blockIdx.x);
  const int64_t chunk_start =
      kSingleChunk ? 0 : static_cast<int64_t>(blockIdx.y) * kTimeTile;
  const int64_t token_count = static_cast<int64_t>(T_tokens);
  if (chunk_start >= token_count && !(token_count == 0 && chunk_start == 0)) {
    return;
  }

  const int64_t chunk_end =
      kSingleChunk ? token_count : min(chunk_start + kTimeTile, token_count);
  const int64_t channel_start =
      static_cast<int64_t>(blockIdx.z) * kChannelsPerBlock +
      static_cast<int64_t>(threadIdx.x) * kVec;
  if (channel_start >= static_cast<int64_t>(C)) {
    return;
  }

  int64_t read_slot = request;
  int64_t write_slot = request;
  if constexpr (kIndexed) {
    read_slot = load_optional_index(
        initial_state_indices, initial_state_indices_i64, request, request);
    if (final_state_indices != nullptr) {
      write_slot = load_optional_index(
          final_state_indices, final_state_indices_i64, request, request);
    }
    if (read_slot < 0 || write_slot < 0 ||
        read_slot >= static_cast<int64_t>(pool_size) ||
        (final_state_indices != nullptr &&
         write_slot >= static_cast<int64_t>(pool_size))) {
      return;
    }
  }

  float w0[kVec];
  float w1[kVec];
  float w2[kVec];
  float w3[kVec];
  float bias_value[kVec];
  float history0[kVec];
  float history1[kVec];
  float history2[kVec];

  const T *history_source = history_snapshot ? history_snapshot : conv_state;
  const ptrdiff_t history_s1 = history_snapshot ? 3 : state_s1;
  const ptrdiff_t history_s2 = history_snapshot ? 1 : state_s2;
  const ptrdiff_t request_state_base =
      history_snapshot ? static_cast<ptrdiff_t>(request) * C * 3
                       : static_cast<ptrdiff_t>(read_slot) * state_s0;
  const ptrdiff_t request_qkv_base = static_cast<ptrdiff_t>(request) * qkv_s0;
  const ptrdiff_t request_out_base = static_cast<ptrdiff_t>(request) * out_s0;

#pragma unroll
  for (int lane = 0; lane < kVec; ++lane) {
    const int64_t channel = channel_start + lane;
    if (channel >= static_cast<int64_t>(C)) {
      w0[lane] = 0.0f;
      w1[lane] = 0.0f;
      w2[lane] = 0.0f;
      w3[lane] = 0.0f;
      bias_value[lane] = 0.0f;
      history0[lane] = 0.0f;
      history1[lane] = 0.0f;
      history2[lane] = 0.0f;
      continue;
    }

    const ptrdiff_t state_base =
        request_state_base + static_cast<ptrdiff_t>(channel) * history_s1;
    const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
    w0[lane] = load_as_float(weight, weight_base);
    w1[lane] = load_as_float(weight, weight_base + weight_s2);
    w2[lane] = load_as_float(weight, weight_base + 2 * weight_s2);
    w3[lane] = load_as_float(weight, weight_base + 3 * weight_s2);
    bias_value[lane] =
        has_bias
            ? load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0)
            : 0.0f;

    if (chunk_start == 0) {
      history0[lane] = load_as_float(history_source, state_base);
      history1[lane] = load_as_float(history_source, state_base + history_s2);
      history2[lane] =
          load_as_float(history_source, state_base + 2 * history_s2);
    } else if (chunk_start == 1) {
      history0[lane] = load_as_float(history_source, state_base + history_s2);
      history1[lane] =
          load_as_float(history_source, state_base + 2 * history_s2);
      history2[lane] = load_as_float(
          qkv, request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2);
    } else if (chunk_start == 2) {
      history0[lane] =
          load_as_float(history_source, state_base + 2 * history_s2);
      history1[lane] = load_as_float(
          qkv, request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2);
      history2[lane] =
          load_as_float(qkv, request_qkv_base + qkv_s1 +
                                 static_cast<ptrdiff_t>(channel) * qkv_s2);
    } else {
      const ptrdiff_t history_base =
          request_qkv_base + static_cast<ptrdiff_t>(channel) * qkv_s2;
      history0[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 3) * qkv_s1);
      history1[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 2) * qkv_s1);
      history2[lane] = load_as_float(
          qkv, history_base + static_cast<ptrdiff_t>(chunk_start - 1) * qkv_s1);
    }
  }

  ptrdiff_t qkv_offset = request_qkv_base + chunk_start * qkv_s1 +
                         static_cast<ptrdiff_t>(channel_start) * qkv_s2;
  ptrdiff_t out_offset = request_out_base + chunk_start * out_s1 +
                         static_cast<ptrdiff_t>(channel_start) * out_s2;

  const bool full_vector =
      channel_start + static_cast<int64_t>(kVec) <= static_cast<int64_t>(C);
  for (int64_t token = chunk_start; token < chunk_end; ++token) {
    float current[kVec];
    if constexpr (kVectorIO) {
      if (full_vector) {
        float result[kVec];
        load_vec16<T, kVec>(qkv, qkv_offset, current);
#pragma unroll
        for (int lane = 0; lane < kVec; ++lane) {
          result[lane] = conv4_with_bias<T>(
              w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
              history2[lane], w3[lane], current[lane], bias_value[lane]);
        }
        store_vec16<T, kVec>(out, out_offset, result);
      } else {
#pragma unroll
        for (int lane = 0; lane < kVec; ++lane) {
          const int64_t channel = channel_start + lane;
          if (channel < static_cast<int64_t>(C)) {
            current[lane] = load_as_float(
                qkv, qkv_offset + static_cast<ptrdiff_t>(lane) * qkv_s2);
            const float acc = conv4_with_bias<T>(
                w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
                history2[lane], w3[lane], current[lane], bias_value[lane]);
            out[out_offset + static_cast<ptrdiff_t>(lane) * out_s2] =
                cast_from_float<T>(acc);
          } else {
            current[lane] = 0.0f;
          }
        }
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < kVec; ++lane) {
        const int64_t channel = channel_start + lane;
        if (channel < static_cast<int64_t>(C)) {
          current[lane] = load_as_float(
              qkv, qkv_offset + static_cast<ptrdiff_t>(lane) * qkv_s2);
          const float acc = conv4_with_bias<T>(
              w0[lane], history0[lane], w1[lane], history1[lane], w2[lane],
              history2[lane], w3[lane], current[lane], bias_value[lane]);
          out[out_offset + static_cast<ptrdiff_t>(lane) * out_s2] =
              cast_from_float<T>(acc);
        } else {
          current[lane] = 0.0f;
        }
      }
    }

#pragma unroll
    for (int lane = 0; lane < kVec; ++lane) {
      const int64_t channel = channel_start + lane;
      if (channel < static_cast<int64_t>(C)) {
        history0[lane] = history1[lane];
        history1[lane] = history2[lane];
        history2[lane] = current[lane];
      }
    }
    qkv_offset += qkv_s1;
    out_offset += out_s1;
  }

  if (chunk_end == token_count) {
    T *final_target = final_conv_state;
    ptrdiff_t final_request_base = static_cast<ptrdiff_t>(request) * final_s0;
    ptrdiff_t final_channel_stride = final_s1;
    ptrdiff_t final_tap_stride = final_s2;
    if constexpr (kIndexed) {
      if (final_state_indices != nullptr) {
        final_target = conv_state;
        final_request_base = static_cast<ptrdiff_t>(write_slot) * state_s0;
        final_channel_stride = state_s1;
        final_tap_stride = state_s2;
      }
    }
#pragma unroll
    for (int lane = 0; lane < kVec; ++lane) {
      const int64_t channel = channel_start + lane;
      if (channel < static_cast<int64_t>(C)) {
        const ptrdiff_t final_base =
            final_request_base +
            static_cast<ptrdiff_t>(channel) * final_channel_stride;
        final_target[final_base] = cast_from_float<T>(history0[lane]);
        final_target[final_base + final_tap_stride] =
            cast_from_float<T>(history1[lane]);
        final_target[final_base + 2 * final_tap_stride] =
            cast_from_float<T>(history2[lane]);
      }
    }
  }
}

template <typename T>
__global__ void causal_conv1d_k4_kernel(
    T *out, T *conv_state, T *final_conv_state, const T *qkv, const T *weight,
    const T *bias, const void *cu_seqlens, const void *initial_state_indices,
    const void *final_state_indices, bool has_bias, bool has_cu_seqlens,
    bool cu_seqlens_i64, bool initial_state_indices_i64,
    bool final_state_indices_i64, bool indexed_state_pool, size_t T_tokens,
    size_t C, size_t total_tokens, size_t pool_size, ptrdiff_t out_s0,
    ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t state_s0, ptrdiff_t state_s1,
    ptrdiff_t state_s2, ptrdiff_t final_s0, ptrdiff_t final_s1,
    ptrdiff_t final_s2, ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0,
    const T *history_snapshot) {
  const int request = blockIdx.x;
  const int channel = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
  if (channel >= C) {
    return;
  }

  int64_t token_begin = 0;
  int64_t token_end = static_cast<int64_t>(T_tokens);
  int token_batch = request;
  if (has_cu_seqlens) {
    token_begin = load_optional_index(cu_seqlens, cu_seqlens_i64, request, 0);
    token_end = load_optional_index(cu_seqlens, cu_seqlens_i64, request + 1, 0);
    token_batch = 0;
    if (token_begin < 0 || token_end < token_begin ||
        token_end > static_cast<int64_t>(total_tokens)) {
      return;
    }
  }
  const int64_t request_len = token_end - token_begin;

  int64_t read_slot =
      indexed_state_pool
          ? load_optional_index(initial_state_indices,
                                initial_state_indices_i64, request, request)
          : static_cast<int64_t>(request);
  int64_t write_slot =
      indexed_state_pool && final_state_indices != nullptr
          ? load_optional_index(final_state_indices, final_state_indices_i64,
                                request, request)
          : static_cast<int64_t>(request);
  if (read_slot < 0 || write_slot < 0 ||
      read_slot >= static_cast<int64_t>(pool_size) ||
      (final_state_indices != nullptr &&
       write_slot >= static_cast<int64_t>(pool_size))) {
    return;
  }

  const T *history_source = history_snapshot ? history_snapshot : conv_state;
  const ptrdiff_t history_s2 = history_snapshot ? 1 : state_s2;
  const ptrdiff_t state_base =
      history_snapshot ? (static_cast<ptrdiff_t>(request) * C + channel) * 3
                       : static_cast<ptrdiff_t>(read_slot) * state_s0 +
                             static_cast<ptrdiff_t>(channel) * state_s1;
  const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
  const float w0 = load_as_float(weight, weight_base);
  const float w1 = load_as_float(weight, weight_base + weight_s2);
  const float w2 = load_as_float(weight, weight_base + 2 * weight_s2);
  const float w3 = load_as_float(weight, weight_base + 3 * weight_s2);
  const float bias_value =
      has_bias ? load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0)
               : 0.0f;

  float history0 = load_as_float(history_source, state_base);
  float history1 = load_as_float(history_source, state_base + history_s2);
  float history2 = load_as_float(history_source, state_base + 2 * history_s2);

  ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0 +
                         static_cast<ptrdiff_t>(token_begin) * qkv_s1 +
                         static_cast<ptrdiff_t>(channel) * qkv_s2;
  ptrdiff_t out_offset = static_cast<ptrdiff_t>(token_batch) * out_s0 +
                         static_cast<ptrdiff_t>(token_begin) * out_s1 +
                         static_cast<ptrdiff_t>(channel) * out_s2;

  for (int64_t t = 0; t < request_len; ++t) {
    const float current = load_as_float(qkv, qkv_offset);
    float acc = conv4_with_bias<T>(w0, history0, w1, history1, w2, history2, w3,
                                   current, bias_value);
    out[out_offset] = cast_from_float<T>(acc);

    history0 = history1;
    history1 = history2;
    history2 = current;
    qkv_offset += qkv_s1;
    out_offset += out_s1;
  }

  T *final_target =
      final_state_indices != nullptr ? conv_state : final_conv_state;
  ptrdiff_t final_base;
  ptrdiff_t final_stride2;
  if (final_state_indices != nullptr) {
    final_base = static_cast<ptrdiff_t>(write_slot) * state_s0 +
                 static_cast<ptrdiff_t>(channel) * state_s1;
    final_stride2 = state_s2;
  } else {
    final_base = static_cast<ptrdiff_t>(request) * final_s0 +
                 static_cast<ptrdiff_t>(channel) * final_s1;
    final_stride2 = final_s2;
  }

  final_target[final_base] = cast_from_float<T>(history0);
  final_target[final_base + final_stride2] = cast_from_float<T>(history1);
  final_target[final_base + 2 * final_stride2] = cast_from_float<T>(history2);
}
}  // anonymous namespace
namespace {

// Snapshot before any request can write the shared pool. This also protects
// aliased explicit final-state views and arbitrary initial-slot permutations.
template <typename T>
__global__ void causal_conv1d_snapshot_history(T *snapshot, const T *state,
                                               const void *initial_indices,
                                               bool indices_i64, size_t count,
                                               size_t channels,
                                               size_t pool_size, ptrdiff_t s0,
                                               ptrdiff_t s1, ptrdiff_t s2) {
  for (size_t linear =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < count; linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t request = linear / (channels * 3);
    const int64_t slot =
        load_optional_index(initial_indices, indices_i64, request, request);
    if (slot < 0 || static_cast<size_t>(slot) >= pool_size) continue;
    snapshot[linear] =
        state[static_cast<ptrdiff_t>(slot) * s0 +
              static_cast<ptrdiff_t>((linear / 3) % channels) * s1 +
              static_cast<ptrdiff_t>(linear % 3) * s2];
  }
}

// One block owns a channel tile across all requests. The barrier protects
// every initial-slot read from every final-slot write, including permutations.
// This short recurrence needs no separate snapshot node or global workspace.
constexpr int kPoolChannelsPerBlock = 32;
constexpr int kMaxPoolRequestCapacity = 32;

template <typename T, int kRequestCapacity>
__global__
__launch_bounds__(kRequestCapacity *kPoolChannelsPerBlock) void causal_conv1d_k4_short_pool_kernel(
    T *out, T *state, const T *x, const T *weight, const T *bias,
    const void *initial_indices, const void *final_indices, bool initial_i64,
    bool final_i64, size_t requests, size_t tokens, size_t channels,
    size_t pool_size, ptrdiff_t x0, ptrdiff_t x1, ptrdiff_t x2, ptrdiff_t o0,
    ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t s0, ptrdiff_t s1, ptrdiff_t s2,
    ptrdiff_t w0, ptrdiff_t w2, ptrdiff_t b0) {
  const int request = threadIdx.x / kPoolChannelsPerBlock;
  const size_t channel =
      static_cast<size_t>(blockIdx.x) * kPoolChannelsPerBlock +
      threadIdx.x % kPoolChannelsPerBlock;
  bool active = request < requests && channel < channels;
  int64_t read_slot = 0, write_slot = 0;
  float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;
  if (active) {
    read_slot =
        load_optional_index(initial_indices, initial_i64, request, request);
    write_slot =
        load_optional_index(final_indices, final_i64, request, request);
    active = read_slot >= 0 && write_slot >= 0 &&
             static_cast<size_t>(read_slot) < pool_size &&
             static_cast<size_t>(write_slot) < pool_size;
    if (active) {
      const ptrdiff_t base = read_slot * s0 + channel * s1;
      h0 = load_as_float(state, base);
      h1 = load_as_float(state, base + s2);
      h2 = load_as_float(state, base + 2 * s2);
    }
  }
  // Padding and invalid-slot threads must participate before returning.
  __syncthreads();
  if (!active) return;
  const ptrdiff_t wb = channel * w0;
  const float a0 = load_as_float(weight, wb);
  const float a1 = load_as_float(weight, wb + w2);
  const float a2 = load_as_float(weight, wb + 2 * w2);
  const float a3 = load_as_float(weight, wb + 3 * w2);
  const float bias_value = bias ? load_as_float(bias, channel * b0) : 0.0f;
  ptrdiff_t input = static_cast<ptrdiff_t>(request) * x0 + channel * x2;
  ptrdiff_t output = static_cast<ptrdiff_t>(request) * o0 + channel * o2;
  for (size_t token = 0; token < tokens; ++token) {
    const float value = load_as_float(x, input);
    out[output] = cast_from_float<T>(
        conv4_with_bias<T>(a0, h0, a1, h1, a2, h2, a3, value, bias_value));
    h0 = h1;
    h1 = h2;
    h2 = value;
    input += x1;
    output += o1;
  }
  const ptrdiff_t target = write_slot * s0 + channel * s1;
  state[target] = cast_from_float<T>(h0);
  state[target + s2] = cast_from_float<T>(h1);
  state[target + 2 * s2] = cast_from_float<T>(h2);
}

// Read both channels for every request before the block-wide write barrier.
// Packed accesses retain the same state-pool permutation semantics.
template <typename T, int kCapacity, int kPairs = 16, bool kCompact = false,
          bool kWaveOwned = false>
__global__ __launch_bounds__(
    kCapacity *
        kPairs) void causal_conv1d_k4_short_pool_pair_kernel(T *out, T *state,
                                                             const T *x,
                                                             const T *weight,
                                                             const T *bias,
                                                             const int32_t
                                                                 *initial,
                                                             const int32_t
                                                                 *final,
                                                             std::conditional_t<
                                                                 kCompact ||
                                                                     kWaveOwned,
                                                                 uint32_t,
                                                                 size_t>
                                                                 requests,
                                                             std::conditional_t<
                                                                 kCompact,
                                                                 uint32_t,
                                                                 size_t>
                                                                 pool_size) {
  constexpr int kPairsPerBlock = kPairs;
  using Index = std::conditional_t<kCompact, uint32_t, ptrdiff_t>;
  constexpr int kChannels = 4096;
  static_assert(kCapacity <= 32 && 64 % kCapacity == 0);
  constexpr int kPairsPerWave =
      kPairs < 64 / kCapacity ? kPairs : 64 / kCapacity;
  static_assert(!kWaveOwned || kPairs % kPairsPerWave == 0);
  // Each channel pair owns a complete power-of-two request group in one
  // wave. Adjacent lanes within each request retain contiguous channels.
  const int request =
      kWaveOwned ? (threadIdx.x % (kCapacity * kPairsPerWave)) / kPairsPerWave
                 : threadIdx.x / kPairsPerBlock;
  const int pair =
      kWaveOwned ? (threadIdx.x / (kCapacity * kPairsPerWave)) * kPairsPerWave +
                       threadIdx.x % kPairsPerWave
                 : threadIdx.x % kPairsPerBlock;
  const int channel = (blockIdx.x * kPairsPerBlock + pair) * 2;
  bool active = request < requests;
  if constexpr (4096 % (kPairs * 2) != 0)
    active = active && channel < kChannels;
  int read_slot = 0, write_slot = 0;
  CausalHistoryPair history{};
  if (active) {
    read_slot = initial ? initial[request] : request;
    write_slot = final[request];
    if constexpr (kCompact) {
      active = static_cast<uint32_t>(read_slot) < pool_size &&
               static_cast<uint32_t>(write_slot) < pool_size;
    } else {
      active = read_slot >= 0 && write_slot >= 0 &&
               static_cast<size_t>(read_slot) < pool_size &&
               static_cast<size_t>(write_slot) < pool_size;
    }
    if (active)
      history = *reinterpret_cast<const CausalHistoryPair *>(
          causal_decode_indexed_offset<kCompact>(
              state,
              (static_cast<Index>(read_slot) * kChannels + channel) * 3));
  }
  if constexpr (kWaveOwned) {
    // Every request for a channel pair is in the same wave; all state
    // loads must complete before any lane overwrites another request's slot.
    __builtin_amdgcn_wave_barrier();
  } else {
    __syncthreads();
  }
  if (!active) return;
  const Index input = static_cast<Index>(request) * kChannels + channel;
  const uint4 weights = *reinterpret_cast<const uint4 *>(
      causal_decode_indexed_offset<kCompact>(weight, channel * 4));
  const unsigned int current = *reinterpret_cast<const unsigned int *>(
      causal_decode_indexed_offset<kCompact>(x, input));
  const unsigned int biases = *reinterpret_cast<const unsigned int *>(
      causal_decode_indexed_offset<kCompact>(bias, channel));
  if constexpr (std::is_same_v<T, half>) {
    *reinterpret_cast<unsigned int *>(
        causal_decode_indexed_offset<kCompact>(out, input)) =
        causal_conv1d_fp16_pair_output(history, weights, current, biases);
  } else {
    const unsigned int packed[4] = {weights.x, weights.y, weights.z, weights.w};
    const unsigned int raw[6] = {history.values[0], history.values[0] >> 16,
                                 history.values[1], history.values[1] >> 16,
                                 history.values[2], history.values[2] >> 16};
    unsigned short encoded[2];
#pragma unroll
    for (int lane = 0; lane < 2; ++lane) {
      const unsigned int first = packed[lane * 2],
                         second = packed[lane * 2 + 1];
      const float value = conv4_with_bias<T>(
          causal_decode_bits<T>(first), causal_decode_bits<T>(raw[lane * 3]),
          causal_decode_bits<T>(first >> 16),
          causal_decode_bits<T>(raw[lane * 3 + 1]),
          causal_decode_bits<T>(second),
          causal_decode_bits<T>(raw[lane * 3 + 2]),
          causal_decode_bits<T>(second >> 16),
          causal_decode_bits<T>(current >> (lane * 16)),
          causal_decode_bits<T>(biases >> (lane * 16)));
      encoded[lane] = causal_encode_bits<T>(value);
    }
    *reinterpret_cast<unsigned int *>(causal_decode_indexed_offset<kCompact>(
        out, input)) = static_cast<unsigned int>(encoded[0]) |
                       (static_cast<unsigned int>(encoded[1]) << 16);
  }
  CausalHistoryPair updated;
  updated.values[0] = (history.values[0] >> 16) | (history.values[1] << 16);
  updated.values[1] = (current & 0xffffu) | (history.values[2] << 16);
  updated.values[2] = (history.values[2] >> 16) | (current & 0xffff0000u);
  const Index target =
      (static_cast<Index>(write_slot) * kChannels + channel) * 3;
  *reinterpret_cast<CausalHistoryPair *>(
      causal_decode_indexed_offset<kCompact>(state, target)) = updated;
}

// ---------------------------------------------------------------------------
// Host launch policy. Metadata chooses a family; pointer/alias checks remain
// at invocation time. Device arithmetic and state synchronization are above.
// ---------------------------------------------------------------------------
inline int short_pool_capacity(const ConvInfo &info) {
  if (info.has_cu_seqlens || !info.final_indices_kind ||
      info.request_count <= 1 || info.request_count > kMaxPoolRequestCapacity ||
      info.T > 4)
    return 0;
  int capacity = 2;
  while (capacity < info.request_count) capacity *= 2;
  return capacity * kPoolChannelsPerBlock <= info.max_threads_per_block
             ? capacity
             : 0;
}

template <typename T, bool kVectorIO, int kTimeTile, int kThreads, int kVec,
          bool kIndexed>
void launch_k4_time_tiled_vec_impl(const ConvInfo &info, void *out,
                                   void *conv_state, void *final_conv_state,
                                   const void *qkv, const void *weight,
                                   const void *bias,
                                   const void *initial_state_indices,
                                   const void *final_state_indices,
                                   cudaStream_t stream, void *workspace) {
  constexpr unsigned int kChannelsPerBlock = kThreads * kVec;

  size_t time_chunks_size = (info.T + kTimeTile - 1) / kTimeTile;
  size_t channel_chunks_size =
      (info.C + kChannelsPerBlock - 1) / kChannelsPerBlock;
  if (time_chunks_size == 0) {
    time_chunks_size = 1;
  }
  if (channel_chunks_size == 0) {
    channel_chunks_size = 1;
  }

  const dim3 grid(static_cast<unsigned int>(info.request_count),
                  static_cast<unsigned int>(time_chunks_size),
                  static_cast<unsigned int>(channel_chunks_size));

  const ptrdiff_t final_s0 = info.final_conv_state_strides.empty()
                                 ? 0
                                 : info.final_conv_state_strides[0];
  const ptrdiff_t final_s1 = info.final_conv_state_strides.empty()
                                 ? 0
                                 : info.final_conv_state_strides[1];
  const ptrdiff_t final_s2 = info.final_conv_state_strides.empty()
                                 ? 0
                                 : info.final_conv_state_strides[2];

  causal_conv1d_k4_time_tiled_vec_kernel<T, kVec, kVectorIO, kTimeTile,
                                         kThreads, kIndexed>
      <<<grid, kThreads, 0, stream>>>(
          static_cast<T *>(out), static_cast<T *>(conv_state),
          static_cast<T *>(final_conv_state), static_cast<const T *>(qkv),
          static_cast<const T *>(weight), static_cast<const T *>(bias),
          initial_state_indices, final_state_indices, info.has_bias,
          info.initial_indices_kind == 2, info.final_indices_kind == 2, info.T,
          info.C, info.pool_size, info.out_strides[0], info.out_strides[1],
          info.out_strides[2], info.conv_state_strides[0],
          info.conv_state_strides[1], info.conv_state_strides[2], final_s0,
          final_s1, final_s2, info.qkv_strides[0], info.qkv_strides[1],
          info.qkv_strides[2], info.weight_strides[0], info.weight_strides[2],
          info.bias_strides.empty() ? 0 : info.bias_strides[0],
          static_cast<const T *>(info.history));
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv vec launch failed");
}

template <typename T>
bool can_use_k4_vector_io(const ConvInfo &info, const void *out,
                          const void *qkv) {
  constexpr size_t kAlignment = 16;
  const auto is_aligned = [](const void *ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & (kAlignment - 1)) == 0;
  };
  const auto stride_is_aligned = [](ptrdiff_t stride) {
    return (static_cast<size_t>(stride) * sizeof(T) & (kAlignment - 1)) == 0;
  };
  return is_aligned(out) && is_aligned(qkv) &&
         stride_is_aligned(info.out_strides[0]) &&
         stride_is_aligned(info.qkv_strides[0]) &&
         stride_is_aligned(info.out_strides[1]) &&
         stride_is_aligned(info.qkv_strides[1]);
}

template <typename T, int kTimeTile, int kThreads, int kVec, bool kIndexed>
void launch_k4_time_tiled_vec_for_tile(
    const ConvInfo &info, void *out, void *conv_state, void *final_conv_state,
    const void *qkv, const void *weight, const void *bias,
    const void *initial_state_indices, const void *final_state_indices,
    cudaStream_t stream, bool vector_io, void *workspace) {
  if (vector_io) {
    return launch_k4_time_tiled_vec_impl<T, true, kTimeTile, kThreads, kVec,
                                         kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, workspace);
  }
  return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, kThreads, kVec,
                                       kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream, workspace);
}

template <typename T, int kTimeTile, bool kIndexed>
void launch_k4_time_tiled_vec_for_channels(
    const ConvInfo &info, void *out, void *conv_state, void *final_conv_state,
    const void *qkv, const void *weight, const void *bias,
    const void *initial_state_indices, const void *final_state_indices,
    cudaStream_t stream, bool vector_io, void *workspace) {
  if constexpr (kIndexed) {
    return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, 64, 1, true>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, workspace);
  }
  if (info.C <= 512) {
    constexpr int kSmallVec = sizeof(T) == 4 ? 2 : 4;
    return launch_k4_time_tiled_vec_impl<T, false, kTimeTile, 64, kSmallVec,
                                         kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, workspace);
  }
  constexpr int kFullVec = sizeof(T) == 4 ? 4 : 8;
  if (info.C <= 4096) {
    return launch_k4_time_tiled_vec_for_tile<T, kTimeTile, 128, kFullVec,
                                             kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io,
        workspace);
  }
  return launch_k4_time_tiled_vec_for_tile<T, kTimeTile, 256, kFullVec,
                                           kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream, vector_io, workspace);
}

template <typename T, bool kIndexed>
void launch_k4_time_tiled_vec(const ConvInfo &info, void *out, void *conv_state,
                              void *final_conv_state, const void *qkv,
                              const void *weight, const void *bias,
                              const void *initial_state_indices,
                              const void *final_state_indices,
                              cudaStream_t stream, bool vector_io,
                              void *workspace) {
  if (info.T <= 16) {
    return launch_k4_time_tiled_vec_for_channels<T, 16, kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io,
        workspace);
  }
  if (info.T <= 64) {
    return launch_k4_time_tiled_vec_for_channels<T, 64, kIndexed>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io,
        workspace);
  }
  return launch_k4_time_tiled_vec_for_channels<T, 128, kIndexed>(
      info, out, conv_state, final_conv_state, qkv, weight, bias,
      initial_state_indices, final_state_indices, stream, vector_io, workspace);
}

template <typename T, int kTimeTile, int kChannelsPerThread>
void launch_k4_contiguous_tiled(const ConvInfo &info, void *out,
                                void *conv_state, const void *qkv,
                                const void *weight, const void *bias,
                                const void *initial_state_indices,
                                const void *final_state_indices,
                                cudaStream_t stream) {
  const size_t middle_chunks =
      info.T < 4 ? 0 : (info.T - 4 + kTimeTile - 1) / kTimeTile;
  const dim3 grid(static_cast<unsigned>(info.request_count),
                  static_cast<unsigned>(1 + middle_chunks),
                  4096 / (64 * kChannelsPerThread));
  const auto launch = [&](auto has_initial) {
    causal_conv1d_k4_contiguous_tiled_kernel<T, kTimeTile, kChannelsPerThread,
                                             decltype(has_initial)::value>
        <<<grid, 64, 0, stream>>>(
            static_cast<T *>(out), static_cast<T *>(conv_state),
            static_cast<const T *>(qkv), static_cast<const T *>(weight),
            static_cast<const T *>(bias),
            static_cast<const int32_t *>(initial_state_indices),
            static_cast<const int32_t *>(final_state_indices), info.T,
            info.pool_size, static_cast<const T *>(info.history));
  };
  if (initial_state_indices)
    launch(std::true_type{});
  else
    launch(std::false_type{});
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv contiguous launch failed");
}

// Resource-based policy shared by neighboring B/T values. BF16 conversion
// pressure needs more independent blocks before increasing channels/thread.
template <typename T, int kTimeTile>
void launch_k4_contiguous_family(const ConvInfo &info, void *out,
                                 void *conv_state, const void *qkv,
                                 const void *weight, const void *bias,
                                 const void *initial_state_indices,
                                 const void *final_state_indices,
                                 cudaStream_t stream) {
  const size_t time_blocks =
      1 + (info.T < 4 ? 0 : (info.T - 4 + kTimeTile - 1) / kTimeTile);
  constexpr size_t kBlocksPerCu = std::is_same_v<T, __nv_bfloat16> ? 8 : 2;
  const size_t paired_blocks = info.request_count * (4096 / 128) * time_blocks;
  if (paired_blocks >= static_cast<size_t>(info.compute_units) * kBlocksPerCu)
    return launch_k4_contiguous_tiled<T, kTimeTile, 2>(
        info, out, conv_state, qkv, weight, bias, initial_state_indices,
        final_state_indices, stream);
  return launch_k4_contiguous_tiled<T, kTimeTile, 1>(
      info, out, conv_state, qkv, weight, bias, initial_state_indices,
      final_state_indices, stream);
}

// Pointer alignment is checked at invocation time; descriptors may be reused
// with different storage carrying the same tensor metadata.
template <typename T>
bool try_launch_k4_decode(const ConvInfo &info, void *out, void *conv_state,
                          const void *qkv, const void *weight, const void *bias,
                          const void *initial_state_indices,
                          const void *final_state_indices,
                          cudaStream_t stream) {
  if constexpr (sizeof(T) == 2) {
    if (info.T == 1 && info.request_count <= 65535) {
      if ((reinterpret_cast<uintptr_t>(out) & 3) == 0 &&
          (reinterpret_cast<uintptr_t>(conv_state) & 3) == 0 &&
          (reinterpret_cast<uintptr_t>(qkv) & 3) == 0 &&
          (reinterpret_cast<uintptr_t>(bias) & 3) == 0 &&
          (reinterpret_cast<uintptr_t>(weight) & 15) == 0) {
        const auto launch_pair = [&](auto threads) {
          constexpr int kThreads = decltype(threads)::value;
          const auto launch_indexed = [&](auto compact) {
            constexpr bool kCompact = decltype(compact)::value;
            using PoolSize = std::conditional_t<kCompact, uint32_t, size_t>;
            causal_conv1d_k4_decode_pair_kernel<T, kThreads, kCompact>
                <<<dim3(4096 / (kThreads * 2), info.request_count), kThreads, 0,
                   stream>>>(
                    static_cast<T *>(out), static_cast<T *>(conv_state),
                    static_cast<const T *>(qkv), static_cast<const T *>(weight),
                    static_cast<const T *>(bias),
                    static_cast<const int32_t *>(initial_state_indices),
                    static_cast<const int32_t *>(final_state_indices),
                    static_cast<PoolSize>(info.pool_size),
                    static_cast<const T *>(info.history));
          };
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            constexpr size_t kMaxPool = 2147483647ULL / (4096 * 3 * sizeof(T));
            // The enclosing decode path bounds requests by 65535, so
            // its input and optional history snapshot also fit.
            if (info.pool_size <= kMaxPool) {
              launch_indexed(std::true_type{});
              return;
            }
          }
          launch_indexed(std::false_type{});
        };
        const size_t wide_blocks = info.request_count * (4096 / (128 * 2));
        if (std::is_same_v<T, half> &&
            wide_blocks < static_cast<size_t>(info.compute_units))
          launch_pair(std::integral_constant<int, 64>{});
        else
          launch_pair(std::integral_constant<int, 128>{});
      } else {
        constexpr int kThreads = 64;
        causal_conv1d_k4_decode_kernel<T, kThreads>
            <<<dim3(4096 / kThreads, info.request_count), kThreads, 0,
               stream>>>(static_cast<T *>(out), static_cast<T *>(conv_state),
                         static_cast<const T *>(qkv),
                         static_cast<const T *>(weight),
                         static_cast<const T *>(bias),
                         static_cast<const int32_t *>(initial_state_indices),
                         static_cast<const int32_t *>(final_state_indices),
                         info.pool_size, static_cast<const T *>(info.history));
      }
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon causal conv decode launch failed");
      return true;
    }
  }
  return false;
}

template <typename T>
void launch_k4_impl(const ConvInfo &info, void *out, void *conv_state,
                    void *final_conv_state, const void *qkv, const void *weight,
                    const void *bias, const void *cu_seqlens,
                    const void *initial_state_indices,
                    const void *final_state_indices, cudaStream_t stream,
                    void *workspace) {
  const size_t generic_tile = info.T <= 16 ? 16 : info.T <= 64 ? 64 : 128;
  const bool tiled_grid_fits =
      (info.T + generic_tile - 1) / generic_tile <= 65535;
  const size_t contiguous_tile = info.T <= 16 ? 4 : info.T <= 64 ? 8 : 16;
  const size_t contiguous_blocks =
      1 +
      (info.T < 4 ? 0 : (info.T - 4 + contiguous_tile - 1) / contiguous_tile);
  // The regular template updates history in its first block after consuming
  // it. Generic multi-tile indexed writes would race the first tile's reads;
  // keep that layout on the one-thread-per-channel recurrence instead.
  const bool history_tile_safe =
      info.history != nullptr || final_state_indices == nullptr ||
      info.T <= 128 ||
      (has_indexed_k4_layout(info) && contiguous_blocks <= 65535);
  if (!info.has_cu_seqlens && cu_seqlens == nullptr && out != qkv &&
      tiled_grid_fits && history_tile_safe &&
      (final_state_indices != nullptr ||
       !info.final_conv_state_strides.empty()) &&
      info.qkv_strides[2] == 1 && info.out_strides[2] == 1) {
    if (has_indexed_k4_layout(info) &&
        (final_conv_state == nullptr || final_conv_state == conv_state) &&
        final_state_indices != nullptr && bias != nullptr) {
      if (try_launch_k4_decode<T>(info, out, conv_state, qkv, weight, bias,
                                  initial_state_indices, final_state_indices,
                                  stream))
        return;
      const int tile = info.T <= 16 ? 4 : info.T <= 64 ? 8 : 16;
      const size_t blocks =
          1 + (info.T < 4 ? 0 : (info.T - 4 + tile - 1) / tile);
      if (blocks <= 65535) {
        if (tile == 4)
          return launch_k4_contiguous_family<T, 4>(
              info, out, conv_state, qkv, weight, bias, initial_state_indices,
              final_state_indices, stream);
        if (tile == 8)
          return launch_k4_contiguous_family<T, 8>(
              info, out, conv_state, qkv, weight, bias, initial_state_indices,
              final_state_indices, stream);
        return launch_k4_contiguous_family<T, 16>(
            info, out, conv_state, qkv, weight, bias, initial_state_indices,
            final_state_indices, stream);
      }
    }
    const bool vector_io = can_use_k4_vector_io<T>(info, out, qkv);
    if (info.indexed_state_pool)
      return launch_k4_time_tiled_vec<T, true>(
          info, out, conv_state, final_conv_state, qkv, weight, bias,
          initial_state_indices, final_state_indices, stream, vector_io,
          workspace);
    return launch_k4_time_tiled_vec<T, false>(
        info, out, conv_state, final_conv_state, qkv, weight, bias,
        initial_state_indices, final_state_indices, stream, vector_io,
        workspace);
  }
  constexpr unsigned int threads = 128;
  dim3 grid(static_cast<unsigned int>(info.request_count),
            (static_cast<unsigned int>(info.C) + threads - 1) / threads, 1);
  causal_conv1d_k4_kernel<T><<<grid, threads, 0, stream>>>(
      static_cast<T *>(out), static_cast<T *>(conv_state),
      static_cast<T *>(final_conv_state), static_cast<const T *>(qkv),
      static_cast<const T *>(weight), static_cast<const T *>(bias), cu_seqlens,
      initial_state_indices, final_state_indices, info.has_bias,
      info.has_cu_seqlens, info.cu_seqlens_kind == 2,
      info.initial_indices_kind == 2, info.final_indices_kind == 2,
      info.indexed_state_pool, info.T, info.C, info.total_tokens,
      info.pool_size, info.out_strides[0], info.out_strides[1],
      info.out_strides[2], info.conv_state_strides[0],
      info.conv_state_strides[1], info.conv_state_strides[2],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[0],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[1],
      info.final_conv_state_strides.empty() ? 0
                                            : info.final_conv_state_strides[2],
      info.qkv_strides[0], info.qkv_strides[1], info.qkv_strides[2],
      info.weight_strides[0], info.weight_strides[2],
      info.bias_strides.empty() ? 0 : info.bias_strides[0],
      static_cast<const T *>(info.history));
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv1d launch failed");
}

template <typename T, int kCapacity>
void launch_short_pool(const ConvInfo &p, void *out, void *state, const void *x,
                       const void *weight, const void *bias,
                       const void *initial, const void *final,
                       cudaStream_t stream) {
  if constexpr (sizeof(T) == 2) {
    if (p.T == 1 && has_indexed_k4_layout(p) &&
        (reinterpret_cast<uintptr_t>(out) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(state) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(bias) & 3) == 0 &&
        (reinterpret_cast<uintptr_t>(weight) & 15) == 0) {
      const auto launch_config = [&](auto pair_tag, auto compact_tag,
                                     auto wave_tag) {
        constexpr int kPairs = decltype(pair_tag)::value;
        constexpr bool kCompact = decltype(compact_tag)::value;
        constexpr bool kWaveOwned = decltype(wave_tag)::value;
        using RequestCount =
            std::conditional_t<kCompact || kWaveOwned, uint32_t, size_t>;
        using PoolSize = std::conditional_t<kCompact, uint32_t, size_t>;
        causal_conv1d_k4_short_pool_pair_kernel<T, kCapacity, kPairs, kCompact,
                                                kWaveOwned>
            <<<(4096 + kPairs * 2 - 1) / (kPairs * 2), kCapacity * kPairs, 0,
               stream>>>(static_cast<T *>(out), static_cast<T *>(state),
                         static_cast<const T *>(x),
                         static_cast<const T *>(weight),
                         static_cast<const T *>(bias),
                         static_cast<const int32_t *>(initial),
                         static_cast<const int32_t *>(final),
                         static_cast<RequestCount>(p.request_count),
                         static_cast<PoolSize>(p.pool_size));
      };
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        constexpr size_t kMaxPool = 2147483647ULL / (4096 * 3 * sizeof(T));
        if (p.pool_size <= kMaxPool)
          launch_config(std::integral_constant<int, 16>{}, std::true_type{},
                        std::true_type{});
        else
          launch_config(std::integral_constant<int, 16>{}, std::false_type{},
                        std::true_type{});
      } else {
        launch_config(std::integral_constant<int, 16>{}, std::false_type{},
                      std::false_type{});
      }
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon causal conv paired pool launch failed");
      return;
    }
  }
  causal_conv1d_k4_short_pool_kernel<T, kCapacity>
      <<<static_cast<unsigned>((p.C + kPoolChannelsPerBlock - 1) /
                               kPoolChannelsPerBlock),
         kCapacity * kPoolChannelsPerBlock, 0, stream>>>(
          static_cast<T *>(out), static_cast<T *>(state),
          static_cast<const T *>(x), static_cast<const T *>(weight),
          static_cast<const T *>(bias), initial, final,
          p.initial_indices_kind == 2, p.final_indices_kind == 2,
          p.request_count, p.T, p.C, p.pool_size, p.qkv_strides[0],
          p.qkv_strides[1], p.qkv_strides[2], p.out_strides[0],
          p.out_strides[1], p.out_strides[2], p.conv_state_strides[0],
          p.conv_state_strides[1], p.conv_state_strides[2], p.weight_strides[0],
          p.weight_strides[2], p.bias_strides.empty() ? 0 : p.bias_strides[0]);
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("Hygon causal conv short pool launch failed");
}

inline bool needs_history_snapshot(const ConvInfo &info, const void *state,
                                   const void *final_state,
                                   const void *initial_indices,
                                   const void *final_indices) {
  // Identical index views map each request back to its own initial row.
  // As in the public state-pool API, concurrent final slots must be unique.
  const bool same_indices =
      initial_indices && initial_indices == final_indices &&
      info.initial_indices_kind == info.final_indices_kind;
  return (final_indices && info.request_count > 1 && !same_indices &&
          !short_pool_capacity(info)) ||
         (final_state == state && final_state &&
          ((initial_indices && info.request_count > 1) ||
           (!info.has_cu_seqlens && info.T > 128)));
}

template <typename T>
void launch_k4(const ConvInfo &info, void *out, void *conv_state,
               void *final_conv_state, const void *qkv, const void *weight,
               const void *bias, const void *cu_seqlens,
               const void *initial_state_indices,
               const void *final_state_indices, cudaStream_t stream,
               void *workspace) {
  if (final_state_indices && !cu_seqlens) {
    switch (short_pool_capacity(info)) {
      case 2:
        return launch_short_pool<T, 2>(info, out, conv_state, qkv, weight, bias,
                                       initial_state_indices,
                                       final_state_indices, stream);
      case 4:
        return launch_short_pool<T, 4>(info, out, conv_state, qkv, weight, bias,
                                       initial_state_indices,
                                       final_state_indices, stream);
      case 8:
        return launch_short_pool<T, 8>(info, out, conv_state, qkv, weight, bias,
                                       initial_state_indices,
                                       final_state_indices, stream);
      case 16:
        return launch_short_pool<T, 16>(info, out, conv_state, qkv, weight,
                                        bias, initial_state_indices,
                                        final_state_indices, stream);
      case 32:
        return launch_short_pool<T, 32>(info, out, conv_state, qkv, weight,
                                        bias, initial_state_indices,
                                        final_state_indices, stream);
      default:
        break;
    }
  }
  if (!needs_history_snapshot(info, conv_state, final_conv_state,
                              initial_state_indices, final_state_indices))
    return launch_k4_impl<T>(info, out, conv_state, final_conv_state, qkv,
                             weight, bias, cu_seqlens, initial_state_indices,
                             final_state_indices, stream, workspace);
  const size_t count = info.request_count * info.C * 3;
  causal_conv1d_snapshot_history<T>
      <<<static_cast<unsigned>(std::min<size_t>(65535, (count + 255) / 256)),
         256, 0, stream>>>(
          static_cast<T *>(workspace), static_cast<const T *>(conv_state),
          initial_state_indices, info.initial_indices_kind == 2, count, info.C,
          info.pool_size, info.conv_state_strides[0],
          info.conv_state_strides[1], info.conv_state_strides[2]);
  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error(
        "Hygon causal conv history snapshot launch failed");
  ConvInfo staged_info = info;
  staged_info.history = workspace;
  launch_k4_impl<T>(staged_info, out, conv_state, final_conv_state, qkv, weight,
                    bias, cu_seqlens, initial_state_indices,
                    final_state_indices, stream, workspace);
}

}  // namespace
}  // namespace hygon_conv_detail

template <>
class Operator<CausalConv1d, Device::Type::kHygon> : public CausalConv1d {
 public:
  Operator(const Tensor qkv, const Tensor weight, Tensor conv_state,
           const std::optional<Tensor> bias,
           const std::optional<Tensor> cu_seqlens,
           const std::optional<Tensor> initial_state_indices,
           const std::optional<Tensor> final_state_indices, Tensor out,
           std::optional<Tensor> final_conv_state)
      : CausalConv1d(hygon_conv_detail::checked_input(
                         qkv, weight, conv_state, bias, cu_seqlens,
                         initial_state_indices, final_state_indices, out,
                         final_conv_state),
                     weight, conv_state, bias, cu_seqlens,
                     initial_state_indices, final_state_indices, out,
                     final_conv_state) {
    if (device_.type() != Device::Type::kHygon)
      throw std::invalid_argument("Hygon causal conv requires HYGON device");
    if (weight.shape()[2] != 4 || conv_state.shape()[2] != 3)
      throw std::invalid_argument(
          "Hygon causal conv kernel requires K4 and state_len3");
    check_device();
    if (requests_ >= static_cast<size_t>(INT_MAX) || channels_ > 65535u * 128u)
      throw std::invalid_argument(
          "Hygon causal conv request/channel launch extent unsupported");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device_.index()) != cudaSuccess)
      throw std::runtime_error(
          "Hygon causal conv device properties unavailable");
    info_.compute_units = std::max(1, properties.multiProcessorCount);
    info_.max_threads_per_block = properties.maxThreadsPerBlock;
    info_.request_count = requests_;
    info_.T = tokens_;
    info_.C = channels_;
    info_.pool_size = pool_;
    info_.total_tokens = qkv.numel() / (channels_ ? channels_ : 1);
    info_.has_bias = bias.has_value();
    info_.has_cu_seqlens = cu_seqlens.has_value();
    info_.indexed_state_pool =
        final_state_indices.has_value() || final_conv_state.has_value();
    info_.cu_seqlens_kind = index_kind_(cu_seqlens);
    info_.initial_indices_kind = index_kind_(initial_state_indices);
    info_.final_indices_kind = index_kind_(final_state_indices);
    auto fill3 = [&](ptrdiff_t dst[3], const std::optional<Tensor> &t) {
      if (!t) return;
      const auto &s = t->strides();
      for (size_t j = 0; j < s.size() && j < 3; ++j) dst[j] = s[j];
    };
    fill3(info_.out_strides, out);
    fill3(info_.conv_state_strides, conv_state);
    fill3(info_.qkv_strides, qkv);
    if (weight.strides().size() >= 2) {
      info_.weight_strides[0] = weight.strides()[0];
      info_.weight_strides[2] = weight.strides()[2];
    }
    if (bias)
      info_.bias_strides = {static_cast<ptrdiff_t>(
          bias->strides().empty() ? 1 : bias->strides()[0])};
    if (final_conv_state)
      for (auto s : final_conv_state->strides())
        info_.final_conv_state_strides.push_back(s);
    for (size_t i = 0; i < metadata_.size(); ++i)
      if (metadata_[i] && metadata_[i]->numel()) {
        size_t span = 1;
        for (size_t axis = 0; axis < metadata_[i]->ndim(); ++axis)
          span += (metadata_[i]->size(axis) - 1) * metadata_[i]->stride(axis);
        const auto type = metadata_[i]->dtype();
        spans_[i] =
            span * (type == DataType::kInt64 ? 8
                    : type == DataType::kFloat32 || type == DataType::kInt32
                        ? 4
                        : 2);
      }
    // Reserve the worst case for this metadata signature. Pointer aliases
    // and identical index buffers are checked again on every invocation.
    if ((final_state_indices && requests_ > 1 &&
         !hygon_conv_detail::short_pool_capacity(info_)) ||
        (final_conv_state && ((initial_state_indices && requests_ > 1) ||
                              (!cu_seqlens && tokens_ > 128)))) {
      const size_t width = dtype_ == DataType::kFloat32 ? 4 : 2;
      if (requests_ && channels_ > std::numeric_limits<size_t>::max() /
                                       requests_ / 3 / width)
        throw std::overflow_error("Hygon causal conv workspace size overflow");
      workspace_bytes_ = requests_ * channels_ * 3 * width;
    }
  }

  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_conv_state) const override {
    const Views views{qkv,
                      weight,
                      conv_state,
                      bias,
                      cu_seqlens,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_conv_state};
    for (size_t i = 0; i < views.size(); ++i) {
      const auto &t = views[i];
      if (t.has_value() != metadata_[i].has_value() ||
          (t && (!std::equal_to<Tensor>{}(*t, *metadata_[i]) ||
                 (t->numel() && !t->data()))))
        throw std::invalid_argument(
            "Hygon causal conv metadata changed or nonempty storage is null");
    }
    check_device();
    if (!requests_ || !channels_) return;
    check_aliases_(views);
    if (hygon_conv_detail::needs_history_snapshot(
            info_, conv_state.data(),
            final_conv_state ? final_conv_state->data() : nullptr,
            initial_state_indices ? initial_state_indices->data() : nullptr,
            final_state_indices ? final_state_indices->data() : nullptr) &&
        (workspace_ == nullptr || workspace_size_in_bytes_ < workspace_bytes_))
      throw std::invalid_argument(
          "Hygon causal conv workspace is missing or too small");
    auto ptr = [&](size_t i) -> void * {
      return views[i] ? const_cast<void *>(views[i]->data()) : nullptr;
    };
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
    switch (dtype_) {
      case DataType::kFloat32:
        hygon_conv_detail::launch_k4<float>(info_, ptr(7), ptr(2), ptr(8),
                                            ptr(0), ptr(1), ptr(3), ptr(4),
                                            ptr(5), ptr(6), stream, workspace_);
        break;
      case DataType::kFloat16:
        hygon_conv_detail::launch_k4<half>(info_, ptr(7), ptr(2), ptr(8),
                                           ptr(0), ptr(1), ptr(3), ptr(4),
                                           ptr(5), ptr(6), stream, workspace_);
        break;
      case DataType::kBFloat16:
        hygon_conv_detail::launch_k4<__nv_bfloat16>(
            info_, ptr(7), ptr(2), ptr(8), ptr(0), ptr(1), ptr(3), ptr(4),
            ptr(5), ptr(6), stream, workspace_);
        break;
      default:
        throw std::invalid_argument("Hygon causal conv dtype unsupported");
    }
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_bytes_;
  }

  // Every execution path submits GPU work only. Required workspace is
  // reported during preparation and owned by the caller/recorded Runtime.
  bool capture_safe() const override { return true; }

 private:
  hygon_conv_detail::ConvInfo info_{};
  std::size_t workspace_bytes_ = 0;
  std::array<size_t, 9> spans_{};
  void check_aliases_(const Views &views) const {
    const auto overlap = [&](size_t a, size_t b) {
      if (!spans_[a] || !spans_[b]) return false;
      const auto x = reinterpret_cast<uintptr_t>(views[a]->data());
      const auto y = reinterpret_cast<uintptr_t>(views[b]->data());
      return x <= y ? y - x < spans_[a] : x - y < spans_[b];
    };
    for (size_t dst : {size_t{7}, size_t{2}, size_t{8}}) {
      if (!views[dst] || (dst == 2 && !views[6])) continue;
      for (size_t src = 0; src < views.size(); ++src) {
        if (src == dst || !views[src] || !overlap(dst, src)) continue;
        const bool exact_layout =
            views[dst]->data() == views[src]->data() &&
            views[dst]->strides() == views[src]->strides();
        if (exact_layout && ((dst == 7 && src == 0) || (dst == 8 && src == 2)))
          continue;
        throw std::invalid_argument(
            "Hygon causal conv unsupported tensor overlap");
      }
    }
  }
  static int index_kind_(const std::optional<Tensor> &t) {
    if (!t) return 0;
    return t->dtype() == DataType::kInt64 ? 2 : 1;
  }
  void check_device() const {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
      throw std::runtime_error("Hygon causal conv device query failed");
    if (current != device_.index())
      throw std::invalid_argument("Hygon causal conv owner device mismatch");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CAUSAL_CONV1D_KERNEL_H_
