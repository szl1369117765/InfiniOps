#pragma once
#include "decoder_projection.cuh"

namespace infini::ops::v4_mhc_opt {
constexpr unsigned PrefillTileRows = 8;
constexpr unsigned PrefillTileColumns = 8;

// Each lane owns two adjacent positions of the original 128-lane reduction.
// The last addition joins the even and odd positions after both trees finish.
static __device__ __forceinline__ float reduce_prefill_pair(float even, float odd) {
#pragma unroll
  for (unsigned step = 32; step > 0; step /= 2) {
    even += __shfl_down(even, step, 64);
    odd += __shfl_down(odd, step, 64);
  }
  return even + odd;
}

// Reuse each weight pair across eight rows. Each accumulative dot or square
// follows the same K positions and FP32 FMA order as project_tiled.
static __global__ void project_prefill(const __nv_bfloat16* x, const float* w,
                                       float* values, unsigned rows, float eps) {
  constexpr unsigned width = 4 * 4096;
  const unsigned lane = threadIdx.x;
  const unsigned row_start = blockIdx.x * PrefillTileRows;
  const unsigned column_start = blockIdx.y * PrefillTileColumns;
  float dots[PrefillTileRows][PrefillTileColumns][2] = {};
  float squares[PrefillTileRows][2] = {};

#pragma unroll 2
  for (unsigned k = lane * 2; k < width; k += Threads) {
    float inputs[PrefillTileRows][2];
#pragma unroll
    for (unsigned row = 0; row < PrefillTileRows; ++row) {
      const unsigned index = row_start + row;
      const unsigned packed =
          index < rows ? *reinterpret_cast<const unsigned*>(x + index * width + k) : 0;
      const float even = __uint_as_float(packed << 16);
      const float odd = __uint_as_float(packed & 0xffff0000u);
      inputs[row][0] = even;
      inputs[row][1] = odd;
      squares[row][0] = fmaf(even, even, squares[row][0]);
      squares[row][1] = fmaf(odd, odd, squares[row][1]);
    }

#pragma unroll
    for (unsigned column = 0; column < PrefillTileColumns; ++column) {
      const auto weights =
          *reinterpret_cast<const Float2*>(w + (column_start + column) * width + k);
#pragma unroll
      for (unsigned row = 0; row < PrefillTileRows; ++row) {
        dots[row][column][0] = fmaf(inputs[row][0], weights[0], dots[row][column][0]);
        dots[row][column][1] = fmaf(inputs[row][1], weights[1], dots[row][column][1]);
      }
    }
  }

#pragma unroll
  for (unsigned row = 0; row < PrefillTileRows; ++row) {
    const float sum = reduce_prefill_pair(squares[row][0], squares[row][1]);
    const float inverse = rsqrtf(sum / float(width) + eps);
#pragma unroll
    for (unsigned column = 0; column < PrefillTileColumns; ++column) {
      const float dot = reduce_prefill_pair(dots[row][column][0], dots[row][column][1]);
      if (lane == 0 && row_start + row < rows) {
        values[(row_start + row) * Outputs + column_start + column] = dot * inverse;
      }
    }
  }
}
} // namespace infini::ops::v4_mhc_opt
