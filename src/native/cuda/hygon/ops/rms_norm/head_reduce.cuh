#pragma once

#include "native/cuda/hygon/caster.cuh"

namespace infini::ops::hygon_rms_detail {

// One group of 32 lanes per 512-wide query head.
template <typename T>
static __device__ __forceinline__ float HeadInverseRms(const T* input, float epsilon) {
  const unsigned lane = threadIdx.x;
  float squared_sum = 0.0f;
#pragma unroll
  for (unsigned tile = 0; tile < 2; ++tile) {
#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
      const float x = Caster<Device::Type::kHygon>::template Cast<float>(
          input[tile * 256 + lane * 8 + element]);
      squared_sum = __fmaf_rn(x, x, squared_sum);
    }
  }
#pragma unroll
  for (unsigned offset = 16; offset; offset >>= 1)
    squared_sum += __shfl_xor_sync(0xffffffff, squared_sum, offset, 32);
  return rsqrtf(squared_sum / 512.0f + epsilon);
}

}  // namespace infini::ops::hygon_rms_detail
