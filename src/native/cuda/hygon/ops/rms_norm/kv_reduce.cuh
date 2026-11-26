#ifndef INFINI_OPS_HYGON_KV_RMS_REDUCE_CUH_
#define INFINI_OPS_HYGON_KV_RMS_REDUCE_CUH_

#include <cuda_runtime.h>

namespace infini::ops::hygon_rms_detail {

// The caller launches 256 threads, each owning two of the 512 KV channels.
static __device__ __forceinline__ float KvInverseRms(float x, float y, float epsilon,
                                                    float* partial) {
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread % 32;
  float sum = __fmaf_rn(y, y, __fmaf_rn(x, x, 0.0f));
#pragma unroll
  for (unsigned offset = 16; offset; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
  if (!lane) partial[thread / 32] = sum;
  __syncthreads();
  sum = partial[lane % 8];
#pragma unroll
  for (unsigned offset = 4; offset; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset, 8);
  return rsqrtf(sum / 512.0f + epsilon);
}
}  // namespace infini::ops::hygon_rms_detail

#endif
