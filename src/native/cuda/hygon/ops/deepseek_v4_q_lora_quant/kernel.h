#pragma once

#include "base/deepseek_v4_q_lora_quant.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"

#include <stdexcept>

namespace infini::ops {
namespace hygon_q_lora_quant {

constexpr unsigned kWave = 64;
constexpr unsigned kThreads = 128;
constexpr unsigned kValuesPerThread = 8;
constexpr unsigned kWidth = kThreads * kValuesPerThread;

static __global__ void NormalizeAndQuantize(const __nv_bfloat16* input,
                                           const __nv_bfloat16* weight,
                                           float epsilon, __nv_bfloat16* normalized,
                                           int8_t* packed, float* scale) {
  const unsigned thread = threadIdx.x;
  const unsigned start = thread * kValuesPerThread;
  const size_t row = blockIdx.x;
  input += row * kWidth;
  normalized += row * kWidth;
  packed += row * kWidth;
  __shared__ float partial[kThreads / kWave];
  float values[kValuesPerThread];
  float sum = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < kValuesPerThread; ++j) {
    values[j] = __bfloat162float(input[start + j]);
    sum = __fmaf_rn(values[j], values[j], sum);
  }
  // Match the 1024-wide Hygon RMSNorm reduction and BF16 writeback.
#pragma unroll
  for (unsigned offset = 32; offset; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset, kWave);
  if (thread % kWave == 0)
    partial[thread / kWave] = sum;
  __syncthreads();
  if (thread < kWave) {
    float total = thread < kThreads / kWave ? partial[thread] : 0.0f;
#pragma unroll
    for (unsigned offset = 32; offset; offset >>= 1)
      total += __shfl_xor_sync(0xffffffff, total, offset, kWave);
    if (thread == 0)
      partial[0] = rsqrtf(total / static_cast<float>(kWidth) + epsilon);
  }
  __syncthreads();
  const float inverse_rms = partial[0];
  float maximum = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < kValuesPerThread; ++j) {
    const unsigned i = start + j;
    const auto value = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(values[j], inverse_rms), __bfloat162float(weight[i])));
    normalized[i] = value;
    values[j] = __bfloat162float(value);
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  // The inputs to max reduction are non-negative. The same scale floor and
  // ties-to-even conversion as PerChannelQuantI8 apply to the rounded values.
#pragma unroll
  for (unsigned offset = 32; offset; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffffffffffull, maximum, offset, kWave));
  __shared__ float maxima[kThreads / kWave];
  if (thread % kWave == 0)
    maxima[thread / kWave] = maximum;
  __syncthreads();
  const float absmax = fmaxf(fmaxf(maxima[0], maxima[1]), 1.0e-10f);
  if (thread == 0)
    scale[row] = absmax / 127.0f;
  const float inverse = 127.0f / absmax;
#pragma unroll
  for (unsigned j = 0; j < kValuesPerThread; ++j) {
    const int q = __float2int_rn(values[j] * inverse);
    packed[start + j] = static_cast<int8_t>(min(max(q, -127), 127));
  }
}
}  // namespace hygon_q_lora_quant

template <>
class Operator<DeepseekV4QLoraQuant, Device::Type::kHygon>
    : public DeepseekV4QLoraQuant {
 public:
  using DeepseekV4QLoraQuant::DeepseekV4QLoraQuant;

  bool capture_safe() const override { return true; }

  static bool supports(Tensor input, Tensor weight, float epsilon,
                       Tensor normalized, Tensor packed, Tensor scale) {
    using namespace hygon_q_lora_quant;
    return input.device().type() == Device::Type::kHygon &&
           input.device() == weight.device() && input.device() == normalized.device() &&
           input.device() == packed.device() && input.device() == scale.device() &&
           input.ndim() == 2 && input.size(1) == kWidth &&
           input.dtype() == DataType::kBFloat16 && input.IsContiguous() &&
           weight.dtype() == input.dtype() && weight.shape() == Tensor::Shape{kWidth} &&
           weight.IsContiguous() && normalized.dtype() == input.dtype() &&
           normalized.shape() == input.shape() && normalized.IsContiguous() &&
           packed.dtype() == DataType::kInt8 && packed.shape() == input.shape() &&
           packed.IsContiguous() && scale.dtype() == DataType::kFloat32 &&
           scale.shape() == Tensor::Shape{input.size(0),1} && scale.IsContiguous() &&
           epsilon > 0;
  }

  void operator()(Tensor input, Tensor weight, float epsilon, Tensor normalized,
                  Tensor packed, Tensor scale) const override {
    using namespace hygon_q_lora_quant;
    if (!supports(input, weight, epsilon, normalized, packed, scale))
      throw std::invalid_argument("Unsupported Hygon DeepSeek V4 Q LoRA geometry");
    if (!input.size(0))
      return;
    NormalizeAndQuantize<<<input.size(0),kThreads,0,static_cast<cudaStream_t>(stream_)>>>(
        static_cast<const __nv_bfloat16*>(input.data()),
        static_cast<const __nv_bfloat16*>(weight.data()), epsilon,
        static_cast<__nv_bfloat16*>(normalized.data()), static_cast<int8_t*>(packed.data()),
        static_cast<float*>(scale.data()));
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon DeepSeek V4 Q LoRA launch failed");
  }
};
}  // namespace infini::ops
