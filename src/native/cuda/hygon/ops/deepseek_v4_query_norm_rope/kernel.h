#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "base/deepseek_v4_query_norm_rope.h"
#include "native/cuda/hygon/ops/rms_norm/head_reduce.cuh"
#include "native/cuda/hygon/ops/rms_norm/kv_reduce.cuh"

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

}  // namespace v4_query_norm_rope_detail

template <>
class Operator<DeepseekV4QueryNormRope, Device::Type::kHygon> : public DeepseekV4QueryNormRope {
 public:
  static bool supports(const Tensor query, const Tensor weight, const Tensor positions,
                       const Tensor frequencies, float epsilon, const Tensor output) {
    return query.device().type() == Device::Type::kHygon &&
           query.ndim() == 3 && query.size(0) && query.size(1) && query.size(2) == 512 &&
           query.numel() / 512 <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
           query.dtype() == DataType::kBFloat16 && query.IsContiguous() &&
           weight.device() == query.device() && weight.dtype() == DataType::kFloat32 &&
           weight.shape() == Tensor::Shape{512} && weight.IsContiguous() &&
           positions.device() == query.device() && positions.IsContiguous() &&
           (positions.dtype() == DataType::kInt32 || positions.dtype() == DataType::kInt64) &&
           positions.shape() == Tensor::Shape{query.size(0)} &&
           frequencies.device() == query.device() && frequencies.dtype() == DataType::kFloat32 &&
           frequencies.ndim() == 2 && frequencies.size(0) && frequencies.size(1) == 64 &&
           frequencies.IsContiguous() && std::isfinite(epsilon) && epsilon > 0 &&
           output.device() == query.device() && output.dtype() == query.dtype() &&
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
