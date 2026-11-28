#ifndef INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/hygon/ops/rms_norm/head_reduce.cuh"
#include "native/cuda/hygon/ops/rms_norm/kv_reduce.cuh"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {
namespace hygon_rms_detail {

// Warp-per-head reduction for the FP32 512-wide attention head path.
static __global__ void NormHeads(const float* input, const float* weight,
                                float* output, size_t heads, size_t rows,
                                int64_t input_row_stride, int64_t input_head_stride,
                                int64_t output_row_stride, int64_t output_head_stride,
                                float epsilon) {
  const unsigned lane = threadIdx.x;
  const size_t head = blockIdx.x;
  const size_t row = head / heads;
  if (row >= rows) return;
  input += row * input_row_stride + (head % heads) * input_head_stride;
  output += row * output_row_stride + (head % heads) * output_head_stride;
  const float inverse_rms = HeadInverseRms(input, epsilon);
  for (unsigned i = lane; i < 512; i += 32)
    output[i] = __fmul_rn(__fmul_rn(input[i], inverse_rms), weight[i]);
}
static __global__ void NormKv(const float* input, const float* weight,
                             float* output, int64_t input_stride,
                             int64_t output_stride, float epsilon) {
  __shared__ float partial[8];
  const unsigned thread = threadIdx.x;
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  const float inverse_rms = KvInverseRms(input[2*thread], input[2*thread+1], epsilon, partial);
#pragma unroll
  for (unsigned j=0;j<2;++j) {
    const unsigned i=2*thread+j;
    output[i]=__fmul_rn(__fmul_rn(input[i],inverse_rms),weight[i]);
  }
}
static __global__ __launch_bounds__(512) void NormHidden(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                 __nv_bfloat16* output, int64_t input_stride,
                                 int64_t output_stride, size_t width, float epsilon) {
  __shared__ float partial[8];
  const unsigned thread = threadIdx.x;
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  float squared_sum = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    const float value = __bfloat162float(input[thread * 8 + j]);
    squared_sum = __fmaf_rn(value, value, squared_sum);
  }
#pragma unroll
  for (unsigned offset = 32; offset; offset >>= 1)
    squared_sum += __shfl_xor_sync(0xffffffff, squared_sum, offset, 64);
  if (thread % 64 == 0) partial[thread / 64] = squared_sum;
  __syncthreads();
  if (thread < 64) {
    float total = thread < blockDim.x/64 ? partial[thread] : 0.0f;
#pragma unroll
    for (unsigned offset = 32; offset; offset >>= 1)
      total += __shfl_xor_sync(0xffffffff, total, offset, 64);
    if (thread == 0) partial[0] = rsqrtf(total / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float inverse_rms = partial[0];
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    const unsigned index = thread * 8 + j;
    const float normalized = __fmul_rn(__bfloat162float(input[index]), inverse_rms);
    output[index] = __float2bfloat16_rn(__fmul_rn(normalized, __bfloat162float(weight[index])));
  }
}
// The scalar path maps eight consecutive BF16 values to each logical thread.
static __device__ __forceinline__ void AccumulatePair(float& sum, unsigned pair) {
  const float lo = __uint_as_float(pair << 16);
  const float hi = __uint_as_float(pair & 0xffff0000u);
  sum = __fmaf_rn(lo, lo, sum);
  sum = __fmaf_rn(hi, hi, sum);
}

static __device__ __forceinline__ unsigned NormalizePair(unsigned input, unsigned weight,
                                                          float inverse_rms) {
  const float lo = __fmul_rn(__uint_as_float(input << 16), inverse_rms);
  const float hi = __fmul_rn(__uint_as_float(input & 0xffff0000u), inverse_rms);
  const auto low = __float2bfloat16_rn(__fmul_rn(lo, __uint_as_float(weight << 16)));
  const auto high = __float2bfloat16_rn(__fmul_rn(hi, __uint_as_float(weight & 0xffff0000u)));
  return static_cast<unsigned>(__bfloat16_as_ushort(low)) |
         (static_cast<unsigned>(__bfloat16_as_ushort(high)) << 16);
}

template <unsigned Groups>
static __global__ __launch_bounds__(512 / Groups) void NormAttention(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    int64_t input_stride, int64_t output_stride, float epsilon) {
  __shared__ float partial[8];
  constexpr unsigned threads = 512 / Groups;
  const unsigned thread = threadIdx.x;
  const auto* src = reinterpret_cast<const uint4*>(input + blockIdx.x * input_stride);
  const auto* weights = reinterpret_cast<const uint4*>(weight);
  auto* dst = reinterpret_cast<uint4*>(output + blockIdx.x * output_stride);
  uint4 values[Groups];
#pragma unroll
  for (unsigned group = 0; group < Groups; ++group) {
    const unsigned i = thread + group * threads;
    const uint4 x = src[i];
    values[group] = x;
    float sum = 0.0f;
    AccumulatePair(sum,x.x);
    AccumulatePair(sum,x.y);
    AccumulatePair(sum,x.z);
    AccumulatePair(sum,x.w);
#pragma unroll
    for (unsigned offset = 32; offset; offset >>= 1)
      sum += __shfl_xor_sync(0xffffffff, sum, offset, 64);
    if (thread % 64 == 0) partial[i / 64] = sum;
  }
  __syncthreads();
  if (thread < 64) {
    float total = thread < 8 ? partial[thread] : 0.0f;
#pragma unroll
    for (unsigned offset = 32; offset; offset >>= 1)
      total += __shfl_xor_sync(0xffffffff, total, offset, 64);
    if (thread == 0) partial[0] = rsqrtf(total / 4096.0f + epsilon);
  }
  __syncthreads();
  const float inverse_rms = partial[0];
#pragma unroll
  for (unsigned group = 0; group < Groups; ++group) {
    const unsigned i = thread + group * threads;
    const uint4 x = values[group];
    const uint4 w = weights[i];
    dst[i] = make_uint4(NormalizePair(x.x,w.x,inverse_rms), NormalizePair(x.y,w.y,inverse_rms),
                        NormalizePair(x.z,w.z,inverse_rms), NormalizePair(x.w,w.w,inverse_rms));
  }
}
}  // namespace hygon_rms_detail


template <>
class Operator<RmsNorm, Device::Type::kHygon>
    : public CudaRmsNorm<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kHygon>>::CudaRmsNorm;

  void operator()(const Tensor input, const Tensor weight, float epsilon,
                  Tensor out) const override {
    if (input.ndim() == 3 && input.size(-1) == 512 &&
        input.dtype() == DataType::kFloat32 && weight.dtype() == DataType::kFloat32 &&
        NativeSupports(input, weight, out)) {
      const size_t heads = input.size(1), rows = input.size(0);
      if (!heads || !rows) return;
      if (heads == 1) {
        hygon_rms_detail::NormKv<<<rows,256,0,static_cast<cudaStream_t>(stream_)>>>(
            static_cast<const float*>(input.data()),static_cast<const float*>(weight.data()),
            static_cast<float*>(out.data()),input.stride(0),out.stride(0),epsilon);
        if (cudaGetLastError()!=cudaSuccess) throw std::runtime_error("Hygon KV RMSNorm launch failed");
        return;
      }
      hygon_rms_detail::NormHeads<<<rows * heads, 32, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const float*>(input.data()), static_cast<const float*>(weight.data()),
          static_cast<float*>(out.data()), heads, rows, input.stride(0), input.stride(1),
          out.stride(0), out.stride(1), epsilon);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon attention head RMSNorm launch failed");
      return;
    }
    if (input.ndim() == 2 && (input.size(1) == 4096 || input.size(1) == 1024) &&
        input.dtype() == DataType::kBFloat16 && weight.dtype() == DataType::kBFloat16 &&
        NativeSupports(input, weight, out)) {
      if (!input.size(0)) return;
      if (input.size(1) == 4096 && input.stride(0) % 8 == 0 && out.stride(0) % 8 == 0 &&
          (reinterpret_cast<uintptr_t>(input.data()) | reinterpret_cast<uintptr_t>(weight.data()) |
           reinterpret_cast<uintptr_t>(out.data())) % alignof(uint4) == 0 &&
          dynamic_cast<const DeepseekV4AttentionRmsNormConfig*>(config_ptr_.get())) {
        const auto stream = static_cast<cudaStream_t>(stream_);
        const auto* x = static_cast<const __nv_bfloat16*>(input.data());
        const auto* w = static_cast<const __nv_bfloat16*>(weight.data());
        auto* y = static_cast<__nv_bfloat16*>(out.data());
        if (input.size(0) >= 128) {
          hygon_rms_detail::NormAttention<2><<<input.size(0),256,0,stream>>>(
              x,w,y,input.stride(0),out.stride(0),epsilon);
        } else {
          hygon_rms_detail::NormAttention<1><<<input.size(0),512,0,stream>>>(
              x,w,y,input.stride(0),out.stride(0),epsilon);
        }
        if (cudaGetLastError() != cudaSuccess)
          throw std::runtime_error("Hygon DeepSeek V4 attention RMSNorm launch failed");
        return;
      }
      hygon_rms_detail::NormHidden<<<input.size(0), input.size(1)/8, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const __nv_bfloat16*>(input.data()), static_cast<const __nv_bfloat16*>(weight.data()),
          static_cast<__nv_bfloat16*>(out.data()), input.stride(0), out.stride(0), input.size(1), epsilon);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon hidden RMSNorm launch failed");
      return;
    }
    CudaRmsNorm<Runtime<Device::Type::kHygon>>::operator()(input, weight, epsilon, out);
  }

  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
