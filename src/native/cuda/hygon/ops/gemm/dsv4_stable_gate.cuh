#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops {

namespace hygon_stable_gate {
constexpr int kK = 4096;
constexpr int kN = 256;
constexpr int kTile = 16;
constexpr int kWave = 64;

using Bf16Pack = short __attribute__((ext_vector_type(4)));
using DotPack = float __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ Bf16Pack load_pack(const __nv_bfloat16* source) {
  Bf16Pack values;
  __builtin_memcpy(&values, source, sizeof(values));
  return values;
}

// One wave owns the tile for all 4096 K positions. Tail lanes supply zero
// inputs and join the matrix instructions before valid rows write their results.
static __global__ __launch_bounds__(kWave) void project(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, float* output,
    size_t rows, float alpha) {
  const unsigned lane_row = threadIdx.x % kTile;
  const unsigned lane_group = threadIdx.x / kTile;
  const size_t row = size_t(blockIdx.x) * kTile + lane_row;
  const unsigned col_start = blockIdx.y * kTile;
#if defined(__gfx936__)
  const bool valid = row < rows;
  const auto* input_row = valid ? input + row * kK : nullptr;
  const auto* weight_row = weight + (col_start + lane_row) * kK;
  DotPack dots{};
#pragma unroll 2
  for (int k = 0; k < kK; k += kTile) {
    const int offset = k + 4 * lane_group;
    const auto a = valid ? load_pack(input_row + offset) : Bf16Pack{};
    const auto b = load_pack(weight_row + offset);
    dots = __builtin_hcu_mmac_f32_16x16x16_bf16(a, b, dots);
  }
  if (!valid) return;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned col = col_start + lane_group + 4 * i;
    output[row * kN + col] = __fmul_rn(dots[i], alpha);
  }
#else
  if (row >= rows) return;
  const auto* input_row = input + row * kK;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned col = col_start + lane_group + 4 * i;
    const auto* weight_row = weight + col * kK;
    float dot = 0.0f;
    for (int k = 0; k < kK; ++k) {
      dot = fmaf(__bfloat162float(input_row[k]), __bfloat162float(weight_row[k]), dot);
    }
    output[row * kN + col] = __fmul_rn(dot, alpha);
  }
#endif
}
}  // namespace hygon_stable_gate

}  // namespace infini::ops
