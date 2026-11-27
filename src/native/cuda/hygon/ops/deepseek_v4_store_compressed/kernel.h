#pragma once

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "base/deepseek_v4_store_compressed.h"
#include "native/cuda/hygon/ops/deepseek_v4_store_mla/kernel.h"
#include "native/cuda/hygon/runtime_utils.h"
#include "native/cuda/ops/rms_norm/kernel.cuh"
#include "native/cuda/ops/rotary_embedding/kernel.cuh"

namespace infini::ops {
namespace v4_store_compressed_detail {

// The mirror assigns two adjacent FP32 inputs to each thread and reduces
// eight logical warp-32 partials. Only thread zero consumes the block sum.
static __device__ __forceinline__ float PairwiseSumSquared(const float* pooled) {
  const unsigned thread = threadIdx.x;
  const unsigned lane = thread % 32;
  const float a = pooled[thread * 2];
  const float b = pooled[thread * 2 + 1];
  float sum = __fmaf_rn(b, b, __fmul_rn(a, a));
#pragma unroll
  for (unsigned delta = 16; delta > 0; delta >>= 1) {
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, delta, 32));
  }
  __shared__ float warp_sums[8];
  if (lane == 0) warp_sums[thread / 32] = sum;
  __syncthreads();
  if (thread != 0) return 0.0f;

  const float even = __fadd_rn(__fadd_rn(warp_sums[0], warp_sums[4]),
                               __fadd_rn(warp_sums[2], warp_sums[6]));
  const float odd = __fadd_rn(__fadd_rn(warp_sums[1], warp_sums[5]),
                              __fadd_rn(warp_sums[3], warp_sums[7]));
  return __fadd_rn(even, odd);
}

template <unsigned BlockSize, typename Position, bool ReferenceRms>
static __global__ void Store(const float* input, const float* weight,
                             const Position* positions, const float* frequencies,
                             const int32_t* slots, float epsilon, size_t frequency_rows,
                             size_t page_tokens, size_t page_count, size_t page_stride,
                             uint8_t* pages) {
  constexpr auto kDev = Device::Type::kHygon;
  using Convert = Caster<kDev>;
  constexpr unsigned kNope = 448, kRope = 64, kWidth = 512, kValueBytes = 576;
  const auto row = blockIdx.x;
  const auto slot = slots[row];
  const auto position = positions[row];
  if (slot < 0 || size_t(slot) / page_tokens >= page_count ||
      position < 0 || size_t(position) >= frequency_rows) return;
  const auto* pooled = input + size_t(row) * kWidth;
  float sum;
  if constexpr (ReferenceRms) {
    static_assert(BlockSize == 256);
    sum = PairwiseSumSquared(pooled);
  } else {
    sum = SumSquared<BlockSize, kDev, float, float>(pooled, kWidth);
  }
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) inverse_rms = rsqrtf(sum / 512.0f + epsilon);
  __syncthreads();

  auto* page = pages + (size_t(slot) / page_tokens) * page_stride;
  const auto offset = size_t(slot) % page_tokens;
  auto* values = page + offset * kValueBytes;
  auto* exponents = page + page_tokens * kValueBytes + offset * 8;

  const auto normalize = [&](unsigned i) {
    return __fmul_rn(__fmul_rn(pooled[i], inverse_rms), weight[i]);
  };
  for (unsigned lane = threadIdx.x; lane < kWidth / 2; lane += BlockSize) {
    const unsigned channel = 2 * lane;
    float pair[2]{normalize(channel), normalize(channel + 1)};
    if (channel < kNope) {
      // The compressed path rounds NoPE before calculating the E4M3 scale.
      const float a = Convert::template Cast<float>(Convert::template Cast<__nv_bfloat16>(pair[0]));
      const float b = Convert::template Cast<float>(Convert::template Cast<__nv_bfloat16>(pair[1]));
      float maximum = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
      for (unsigned step = 16; step; step >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step, 32));
      const auto scale = v4_store_mla_detail::EncodeScale(maximum);
      const float inverse_scale = exp2f(127.0f - scale);
      values[channel] = __nv_cvt_float_to_fp8(a * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      values[channel + 1] = __nv_cvt_float_to_fp8(b * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      if (lane % 32 == 0) exponents[lane / 32] = scale;
    } else {
      const auto pair_index = (channel - kNope) / 2;
      ApplyRotaryPair<kDev, float, float, false>(
          pair, frequencies + size_t(position) * kRope + pair_index, 0, kRope / 2, false);
      auto* rope = reinterpret_cast<__nv_bfloat16*>(values + kNope);
      rope[2 * pair_index] = Convert::template Cast<__nv_bfloat16>(pair[0]);
      rope[2 * pair_index + 1] = Convert::template Cast<__nv_bfloat16>(pair[1]);
      if (pair_index == 0) exponents[7] = 0;
    }
  }
}

}  // namespace v4_store_compressed_detail

template <>
class Operator<DeepseekV4StoreCompressed, Device::Type::kHygon> : public DeepseekV4StoreCompressed {
 public:
  static bool supports(const Tensor pooled, const Tensor weight, const Tensor positions,
                       const Tensor frequencies, const Tensor slots, float epsilon,
                       int64_t page_tokens, const Tensor pages) {
    return (page_tokens == 64 || page_tokens == 2) &&
           Operator<DeepseekV4StoreMla, Device::Type::kHygon>::supports(pooled, slots, page_tokens, pages) &&
           pooled.dtype() == DataType::kFloat32 &&
           pooled.size(0) <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
           weight.device() == pooled.device() && weight.dtype() == DataType::kFloat32 &&
           weight.shape() == Tensor::Shape{512} && weight.IsContiguous() &&
           positions.device() == pooled.device() && positions.IsContiguous() &&
           (positions.dtype() == DataType::kInt32 || positions.dtype() == DataType::kInt64) &&
           positions.shape() == Tensor::Shape{pooled.size(0)} &&
           frequencies.device() == pooled.device() && frequencies.dtype() == DataType::kFloat32 &&
           frequencies.ndim() == 2 && frequencies.size(0) && frequencies.size(1) == 64 &&
           frequencies.IsContiguous() && std::isfinite(epsilon) && epsilon > 0;
  }

  Operator(const Tensor pooled, const Tensor weight, const Tensor positions,
           const Tensor frequencies, const Tensor slots, float epsilon,
           int64_t page_tokens, Tensor pages)
      : DeepseekV4StoreCompressed(pooled, weight, positions, frequencies, slots,
                                  epsilon, page_tokens, pages) {
    if (!supports(pooled, weight, positions, frequencies, slots, epsilon, page_tokens, pages))
      throw std::invalid_argument("Unsupported DeepSeek V4 fused compressed store inputs");
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor pooled, const Tensor weight, const Tensor positions,
                  const Tensor frequencies, const Tensor slots, float epsilon,
                  int64_t page_tokens, Tensor pages) const override {
    if (!supports(pooled, weight, positions, frequencies, slots, epsilon, page_tokens, pages))
      throw std::invalid_argument("DeepSeek V4 fused compressed store inputs changed");
    const auto stream = static_cast<cudaStream_t>(stream_);
    const auto blocks = static_cast<unsigned>(pooled.size(0));
    const auto block_size = RuntimeUtils<Device::Type::kHygon>::GetOptimalBlockSize();
    const char* mode = page_tokens == 64 && pooled.size(0) == 1
        ? std::getenv("INFINIOPS_DSV4_C4_STORE_RMS_REF") : nullptr;
    const bool reference_rms = mode && mode[0] == '1' && mode[1] == '\0';
    const auto launch = [&](auto position, auto block, auto reference) {
      using Position = decltype(position);
      constexpr unsigned kBlock = decltype(block)::value;
      constexpr bool kReferenceRms = decltype(reference)::value;
      v4_store_compressed_detail::Store<kBlock, Position, kReferenceRms><<<blocks, kBlock, 0, stream>>>(
          static_cast<const float*>(pooled.data()), static_cast<const float*>(weight.data()),
          static_cast<const Position*>(positions.data()),
          static_cast<const float*>(frequencies.data()), static_cast<const int32_t*>(slots.data()),
          epsilon, frequencies.size(0), page_tokens, pages.size(0), pages.size(1),
          static_cast<uint8_t*>(pages.data()));
    };
    const auto dispatch = [&](auto position) {
      if (reference_rms) {
        launch(position, std::integral_constant<unsigned, 256>{}, std::true_type{});
      } else if (block_size == 256) {
        launch(position, std::integral_constant<unsigned, 256>{}, std::false_type{});
      } else {
        launch(position, std::integral_constant<unsigned, 128>{}, std::false_type{});
      }
    };
    if (positions.dtype() == DataType::kInt32) dispatch(int32_t{});
    else dispatch(int64_t{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 fused compressed store launch failed");
  }
};

}  // namespace infini::ops
