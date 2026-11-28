#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "base/deepseek_v4_query_norm_rope.h"
#include "native/cuda/hygon/ops/rms_norm/head_reduce.cuh"
#include "native/cuda/hygon/ops/rms_norm/kv_reduce.cuh"
#include "native/cuda/hygon/runtime_utils.h"
#include "native/cuda/ops/rms_norm/kernel.cuh"
#include "native/cuda/ops/rotary_embedding/kernel.cuh"

namespace infini::ops {
namespace v4_query_norm_rope_detail {

template <typename Position, bool SingleHead>
static __global__ void Normalize(const __nv_bfloat16* input, const float* weight,
                                const Position* positions, const float* frequencies,
                                float epsilon, size_t heads, size_t frequency_rows,
                                __nv_bfloat16* output) {
  using Convert = Caster<Device::Type::kHygon>;
  const auto head = blockIdx.x;
  const auto* query = input + size_t(head) * 512;
  auto* result = output + size_t(head) * 512;
  const auto position = positions[head / heads];
  const auto* frequency = position >= 0 && size_t(position) < frequency_rows
      ? frequencies + size_t(position) * 64 : nullptr;
  const auto lane = threadIdx.x;
  const auto normalize = [&](unsigned i, float inverse) {
    const float x = Convert::template Cast<float>(query[i]);
    return Convert::template Cast<__nv_bfloat16>(
        __fmul_rn(__fmul_rn(x, inverse), weight[i]));
  };
  const auto write_pair = [&](unsigned pair, float inverse) {
    const unsigned i = 2 * pair;
    __nv_bfloat16 values[2]{normalize(i, inverse), normalize(i + 1, inverse)};
    if (frequency && i >= 448) {
      // Match the two contractions of the standalone Hygon BF16 rotation.
      const float x = Convert::template Cast<float>(values[0]);
      const float y = Convert::template Cast<float>(values[1]);
      const float cosine = frequency[pair - 224];
      const float sine = frequency[pair - 224 + 32];
      const float y_sine = __fmul_rn(y, sine);
      const float y_cosine = __fmul_rn(y, cosine);
      values[0] = Convert::template Cast<__nv_bfloat16>(__fmaf_rn(x, cosine, -y_sine));
      values[1] = Convert::template Cast<__nv_bfloat16>(__fmaf_rn(x, sine, y_cosine));
    }
    result[i] = values[0];
    result[i + 1] = values[1];
  };
  if constexpr (SingleHead) {
    __shared__ float partial[8];
    const float x = Convert::template Cast<float>(query[2 * lane]);
    const float y = Convert::template Cast<float>(query[2 * lane + 1]);
    write_pair(lane, hygon_rms_detail::KvInverseRms(x, y, epsilon, partial));
  } else {
    const float inverse = hygon_rms_detail::HeadInverseRms(query, epsilon);
#pragma unroll
    for (unsigned tile = 0; tile < 8; ++tile)
      write_pair(tile * 32 + lane, inverse);
  }
}

// The index key starts in FP32 and only rounds to BF16 after RoPE.
template <typename Position, unsigned BlockSize>
static __global__ void NormalizeIndexKey(const float* input, const float* weight,
                                        const Position* positions, const float* frequencies,
                                        float epsilon, size_t frequency_rows,
                                        __nv_bfloat16* output) {
  constexpr auto kDevice = Device::Type::kHygon;
  using Convert = Caster<kDevice>;
  constexpr size_t kWidth = 128;
  constexpr unsigned kRopePairs = 32;
  const auto* row = input + size_t(blockIdx.x) * kWidth;
  auto* result = output + size_t(blockIdx.x) * kWidth;
  // Keep the generic FP32 RmsNorm's block size, reduction and multiply order.
  const float squared_sum = SumSquared<BlockSize, kDevice, float, float>(row, kWidth);
  __shared__ float inverse;
  if (threadIdx.x == 0) {
    inverse = rsqrtf(squared_sum / static_cast<float>(kWidth) + epsilon);
  }
  __syncthreads();
  const unsigned pair = threadIdx.x;
  if (pair >= kWidth / 2) return;
  const unsigned lo = 2 * pair;
  const unsigned hi = lo + 1;
  float values[2]{
      __fmul_rn(__fmul_rn(row[lo], inverse), weight[lo]),
      __fmul_rn(__fmul_rn(row[hi], inverse), weight[hi])};
  const auto position = positions[blockIdx.x];
  if (pair >= kWidth / 2 - kRopePairs && position >= 0 && size_t(position) < frequency_rows) {
    const auto* frequency = frequencies + size_t(position) * (2 * kRopePairs)
        + pair - (kWidth / 2 - kRopePairs);
    ApplyRotaryPair<kDevice, float, float, false>(values, frequency, 0, kRopePairs, false);
  }
  result[lo] = Convert::template Cast<__nv_bfloat16>(values[0]);
  result[hi] = Convert::template Cast<__nv_bfloat16>(values[1]);
}

}  // namespace v4_query_norm_rope_detail

template <>
class Operator<DeepseekV4QueryNormRope, Device::Type::kHygon> : public DeepseekV4QueryNormRope {
 public:
  static bool supports(const Tensor query, const Tensor weight, const Tensor positions,
                       const Tensor frequencies, float epsilon, const Tensor output) {
    const auto max_grid = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    const bool query_heads = query.ndim() == 3 && query.size(0) && query.size(1) &&
        query.size(2) == 512 && query.numel() / 512 <= max_grid &&
        query.dtype() == DataType::kBFloat16;
    const bool index_key = query.ndim() == 2 && query.size(0) && query.size(0) <= max_grid &&
        query.size(1) == 128 && query.dtype() == DataType::kFloat32;
    return query.device().type() == Device::Type::kHygon &&
           (query_heads || index_key) && query.IsContiguous() &&
           weight.device() == query.device() && weight.dtype() == DataType::kFloat32 &&
           weight.shape() == Tensor::Shape{query.size(-1)} && weight.IsContiguous() &&
           positions.device() == query.device() && positions.IsContiguous() &&
           (positions.dtype() == DataType::kInt32 || positions.dtype() == DataType::kInt64) &&
           positions.shape() == Tensor::Shape{query.size(0)} &&
           frequencies.device() == query.device() && frequencies.dtype() == DataType::kFloat32 &&
           frequencies.ndim() == 2 && frequencies.size(0) && frequencies.size(1) == 64 &&
           frequencies.IsContiguous() && std::isfinite(epsilon) && epsilon > 0 &&
           output.device() == query.device() && output.dtype() == DataType::kBFloat16 &&
           output.shape() == query.shape() && output.IsContiguous();
  }

  Operator(const Tensor query, const Tensor weight, const Tensor positions,
           const Tensor frequencies, float epsilon, Tensor output)
      : DeepseekV4QueryNormRope(query, weight, positions, frequencies, epsilon, output) {
    if (!supports(query, weight, positions, frequencies, epsilon, output))
      throw std::invalid_argument("Unsupported DeepSeek V4 fused query inputs");
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor query, const Tensor weight, const Tensor positions,
                  const Tensor frequencies, float epsilon, Tensor output) const override {
    if (!supports(query, weight, positions, frequencies, epsilon, output))
      throw std::invalid_argument("DeepSeek V4 fused query inputs changed");
    const auto stream = static_cast<cudaStream_t>(stream_);
    if (query.ndim() == 2) {
      const auto blocks = static_cast<unsigned>(query.size(0));
      const auto block_size = RuntimeUtils<Device::Type::kHygon>::GetOptimalBlockSize();
      const auto launch_index_key = [&](auto position, auto block) {
        using Position = decltype(position);
        constexpr unsigned kBlockSize = decltype(block)::value;
        v4_query_norm_rope_detail::NormalizeIndexKey<Position, kBlockSize>
            <<<blocks, kBlockSize, 0, stream>>>(
                static_cast<const float*>(query.data()),
                static_cast<const float*>(weight.data()),
                static_cast<const Position*>(positions.data()),
                static_cast<const float*>(frequencies.data()), epsilon, frequencies.size(0),
                static_cast<__nv_bfloat16*>(output.data()));
      };
      const auto dispatch_index_key = [&](auto position) {
        if (block_size == 256) launch_index_key(position, std::integral_constant<unsigned, 256>{});
        else launch_index_key(position, std::integral_constant<unsigned, 128>{});
      };
      if (positions.dtype() == DataType::kInt32) dispatch_index_key(int32_t{});
      else dispatch_index_key(int64_t{});
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("DeepSeek V4 fused index key launch failed");
      return;
    }
    const auto blocks = static_cast<unsigned>(query.numel() / 512);
    const auto launch = [&](auto position, auto single) {
      constexpr bool kSingle = decltype(single)::value;
      using Position = decltype(position);
      v4_query_norm_rope_detail::Normalize<Position, kSingle>
          <<<blocks, kSingle ? 256 : 32, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(query.data()),
              static_cast<const float*>(weight.data()),
              static_cast<const Position*>(positions.data()),
              static_cast<const float*>(frequencies.data()), epsilon,
              query.size(1), frequencies.size(0), static_cast<__nv_bfloat16*>(output.data()));
    };
    const auto dispatch = [&](auto position) {
      if (query.size(1) == 1) launch(position, std::true_type{});
      else launch(position, std::false_type{});
    };
    if (positions.dtype() == DataType::kInt32) dispatch(int32_t{});
    else dispatch(int64_t{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 fused query launch failed");
  }
};

}  // namespace infini::ops
