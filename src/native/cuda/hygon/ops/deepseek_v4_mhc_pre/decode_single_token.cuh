#pragma once
#include "decoder_projection.cuh"
#include "decoder_mix.cuh"

namespace infini::ops::v4_mhc_opt {
constexpr unsigned DecodeHidden = 4096;
constexpr unsigned DecodeCollapseBlocks = 8;

// A single token occupies only the first row of HCU's 16x16 tile. Read
// contiguous K values four at a time without changing MMAC4 accumulation order.
template<bool Squares>
static __device__ __forceinline__ void project_decode_tile(
    const __nv_bfloat16* x, const float* w, float* dots, float* squares) {
  constexpr unsigned width = 4 * DecodeHidden;
  constexpr unsigned split_width = width / Splits;
  const unsigned lane = threadIdx.x;
  const unsigned row = lane % 16;
  const unsigned group = lane / 16;
  const unsigned col = blockIdx.y * 16 + lane % 16;
  const unsigned start = blockIdx.z * split_width;
  Float4 dot = {};
  float square = 0.0f;
  for (unsigned tile = 0; tile < split_width; tile += 64) {
#pragma unroll
    for (unsigned chunk = 0; chunk < 4; ++chunk) {
      const unsigned k = start + tile + (chunk / 2 * 4 + group) * 8 + chunk % 2 * 4;
      const uint2 raw = row == 0 ? *reinterpret_cast<const uint2*>(x + k) : uint2{};
      const Float4 weights = col < Outputs ? *reinterpret_cast<const Float4*>(w + col * width + k)
                                          : Float4{};
      const float a0 = __uint_as_float(raw.x << 16);
      const float a1 = __uint_as_float(raw.x & 0xffff0000u);
      const float a2 = __uint_as_float(raw.y << 16);
      const float a3 = __uint_as_float(raw.y & 0xffff0000u);
      if constexpr (Squares) {
        square = fmaf(a0, a0, square);
        square = fmaf(a1, a1, square);
        square = fmaf(a2, a2, square);
        square = fmaf(a3, a3, square);
      }
      dot = __builtin_hcu_mmac_16x16x4_f32(a0, weights[0], dot);
      dot = __builtin_hcu_mmac_16x16x4_f32(a1, weights[1], dot);
      dot = __builtin_hcu_mmac_16x16x4_f32(a2, weights[2], dot);
      dot = __builtin_hcu_mmac_16x16x4_f32(a3, weights[3], dot);
    }
  }
  if constexpr (Squares) {
    const float pair = square + __shfl_xor(square, 32, 64);
    const float total = pair + __shfl_xor(pair, 16, 64);
    if (lane == 0) squares[blockIdx.z] = total;
  }
  if (row != 0) return;
#pragma unroll
  for (unsigned output = 0; output < 4; ++output) {
    const unsigned column = blockIdx.y * 16 + group + output * 4;
    if (column < Outputs) dots[blockIdx.z * Outputs + column] = dot[output];
  }
}

static __global__ void project_decode(const __nv_bfloat16* x, const float* w,
                                      float* dots, float* squares) {
  if (blockIdx.y == 0) project_decode_tile<true>(x, w, dots, squares);
  else project_decode_tile<false>(x, w, dots, squares);
}
// The reciprocal normalization input is a sum of nonnegative values.
// Swapping operands within a binary pair keeps the FP32 result and zero sign.
static __device__ __forceinline__ float mix_row_pairs(float value) {
  const float pair = value + __shfl_xor(value, 1, 64);
  return pair + __shfl_xor(pair, 2, 64);
}

// Matrix lanes are numbered row-major from 0. Even and odd rows use
// different first pairings to preserve the established column sum order.
static __device__ __forceinline__ float mix_column_pairs(float value, unsigned lane) {
  const unsigned partner = lane ^ ((lane & 4) ? 4 : 12);
  const float pair = value + wave_value(value, partner);
  return pair + __shfl_xor(pair, 8, 64);
}

static __global__ void mix_decode(
    const __nv_bfloat16* residual, const float* values, const float* scales, const float* bias,
    __nv_bfloat16* collapsed, float* post, float* combination, float eps, int64_t iterations,
    const float* squares, float norm_eps) {
  constexpr unsigned hidden = DecodeHidden;
  const unsigned lane = threadIdx.x;
  const float inverse = rsqrtf(ordered_splits(squares, 1) / float(4 * hidden) + norm_eps);
  // Matrix normalization and post gates depend on the projection only.
  // Run them in block zero while the other blocks collapse residual channels.
  if (blockIdx.x == 0) {
    if (lane < 16) {
      const float normalized = ordered_splits(values + 8 + lane, 24) * inverse;
      float matrix = normalized * scales[2] + bias[8 + lane];
      const unsigned row_start = lane & ~3u;
      float max_value = -INFINITY;
      max_value = fmaxf(max_value, wave_value(matrix, row_start));
      max_value = fmaxf(max_value, wave_value(matrix, row_start + 1));
      max_value = fmaxf(max_value, wave_value(matrix, row_start + 2));
      max_value = fmaxf(max_value, wave_value(matrix, row_start + 3));
      matrix = expf(matrix - max_value);
      matrix = matrix / mix_row_pairs(matrix) + eps;
      matrix /= mix_column_pairs(matrix, lane) + eps;
      for (int64_t iteration = 1; iteration < iterations; ++iteration) {
        matrix /= mix_row_pairs(matrix) + eps;
        matrix /= mix_column_pairs(matrix, lane) + eps;
      }
      combination[lane] = matrix;
    }
    if (lane >= 16 && lane < 20) {
      const unsigned index = lane - 12;
      const float normalized = ordered_splits(values + index, 24) * inverse;
      post[index - 4] = 2 * logistic<true>(normalized * scales[1] + bias[index]);
    }
    return;
  }

  __shared__ float pre[4];
  if (lane < 4) {
    const float normalized = ordered_splits(values + lane, 24) * inverse;
    pre[lane] = logistic<true>(normalized * scales[0] + bias[lane]) + eps;
  }
  __syncthreads();
  for (unsigned index = (blockIdx.x - 1) * blockDim.x + lane; index < hidden;
       index += DecodeCollapseBlocks * blockDim.x) {
    const float a = __bfloat162float(residual[index]);
    const float b = __bfloat162float(residual[hidden + index]);
    const float c = __bfloat162float(residual[2 * hidden + index]);
    const float d = __bfloat162float(residual[3 * hidden + index]);
    const float left = fmaf(a, pre[0], b * pre[1]);
    const float right = fmaf(c, pre[2], d * pre[3]);
    collapsed[index] = __float2bfloat16_rz(left + right);
  }
}
} // namespace infini::ops::v4_mhc_opt
