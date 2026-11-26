#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_STORE_SWA_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_STORE_SWA_H_

#include <cmath>
#include <stdexcept>

#include "base/deepseek_v4_store_swa.h"
#include "native/cuda/hygon/ops/deepseek_v4_store_mla/kernel.h"
#include "native/cuda/hygon/ops/rms_norm/kv_reduce.cuh"
#include "native/cuda/ops/rotary_embedding/kernel.cuh"

namespace infini::ops {
namespace v4_store_swa_detail {

constexpr size_t kPageTokens = 256;
constexpr size_t kNope = 448;
constexpr size_t kRope = 64;
constexpr size_t kWidth = kNope + kRope;
constexpr size_t kValueBytes = kNope + 2 * kRope;

template <typename Position>
static __global__ void Store(const __nv_bfloat16* key, const float* weights,
                             const Position* positions, const float* frequencies,
                             const int32_t* slots, float epsilon, uint8_t* pages,
                             size_t rows, size_t num_positions, size_t page_count,
                             size_t page_stride) {
  using Convert = Caster<Device::Type::kHygon>;
  __shared__ float partial[8];
  const unsigned lane = threadIdx.x;
  const unsigned channel = 2 * lane;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const auto slot = slots[row];
    const auto position = positions[row];
    if (slot < 0 || size_t(slot) / kPageTokens >= page_count ||
        position < 0 || size_t(position) >= num_positions) continue;

    const auto* input = key + row * kWidth + channel;
    const float x = Convert::template Cast<float>(input[0]);
    const float y = Convert::template Cast<float>(input[1]);
    const float inverse = hygon_rms_detail::KvInverseRms(x, y, epsilon, partial);
    float a = __fmul_rn(__fmul_rn(x, inverse), weights[channel]);
    float b = __fmul_rn(__fmul_rn(y, inverse), weights[channel + 1]);
    const auto offset = size_t(slot) % kPageTokens;
    auto* page = pages + size_t(slot) / kPageTokens * page_stride;
    auto* values = page + offset * kValueBytes;
    auto* scales = page + kPageTokens * kValueBytes + offset * 8;

    if (channel < kNope) {
      float abs_max = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
      for (unsigned step = 16; step; step >>= 1)
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, step, 32));
      const auto scale = v4_store_mla_detail::EncodeScale(abs_max);
      const float inverse_scale = exp2f(127.0f - scale);
      values[channel] = __nv_cvt_float_to_fp8(a * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      values[channel + 1] = __nv_cvt_float_to_fp8(b * inverse_scale, __NV_SATFINITE, __NV_E4M3);
      if (lane % 32 == 0) scales[lane / 32] = scale;
    } else {
      const auto pair = (channel - kNope) / 2;
      float rotated[2]{a, b};
      ApplyRotaryPair<Device::Type::kHygon, float, float, false>(
          rotated, frequencies + size_t(position) * kRope + pair, 0, kRope / 2, false);
      auto* rope = reinterpret_cast<__nv_bfloat16*>(values + kNope);
      rope[2 * pair] = Convert::template Cast<__nv_bfloat16>(rotated[0]);
      rope[2 * pair + 1] = Convert::template Cast<__nv_bfloat16>(rotated[1]);
      if (pair == 0) scales[7] = 0;
    }
    __syncthreads();
  }
}
}  // namespace v4_store_swa_detail

template <>
class Operator<DeepseekV4StoreSwa, Device::Type::kHygon> : public DeepseekV4StoreSwa {
 public:
  static bool supports(const Tensor key, const Tensor weight, const Tensor positions,
                       const Tensor frequencies, const Tensor slots, float epsilon,
                       const Tensor pages) {
    using v4_store_swa_detail::kPageTokens;
    return Operator<DeepseekV4StoreMla, Device::Type::kHygon>::supports(key, slots, kPageTokens, pages)
        && key.dtype() == DataType::kBFloat16
        && weight.device() == key.device() && weight.dtype() == DataType::kFloat32
        && weight.shape() == Tensor::Shape{512} && weight.IsContiguous()
        && positions.device() == key.device() && positions.IsContiguous()
        && (positions.dtype() == DataType::kInt64 || positions.dtype() == DataType::kInt32)
        && positions.shape() == Tensor::Shape{key.size(0)}
        && frequencies.device() == key.device() && frequencies.dtype() == DataType::kFloat32
        && frequencies.ndim() == 2 && frequencies.size(0) > 0 && frequencies.size(1) == 64
        && frequencies.IsContiguous() && std::isfinite(epsilon) && epsilon > 0;
  }

  Operator(const Tensor key, const Tensor weight, const Tensor positions,
           const Tensor frequencies, const Tensor slots, float epsilon, Tensor pages)
      : DeepseekV4StoreSwa(key, weight, positions, frequencies, slots, epsilon, pages) {
    if (!supports(key, weight, positions, frequencies, slots, epsilon, pages))
      throw std::invalid_argument("Unsupported DeepSeek V4 fused SWA store inputs");
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor key, const Tensor weight, const Tensor positions,
                  const Tensor frequencies, const Tensor slots, float epsilon,
                  Tensor pages) const override {
    if (!supports(key, weight, positions, frequencies, slots, epsilon, pages))
      throw std::invalid_argument("DeepSeek V4 fused SWA store inputs changed");
    const auto rows = key.size(0);
    const auto blocks = static_cast<unsigned>(rows < 65535 ? rows : 65535);
    const auto launch = [&](auto tag) {
      using Position = decltype(tag);
      v4_store_swa_detail::Store<Position>
          <<<blocks, 256, 0, static_cast<cudaStream_t>(stream_)>>>(
              static_cast<const __nv_bfloat16*>(key.data()),
              static_cast<const float*>(weight.data()),
              static_cast<const Position*>(positions.data()),
              static_cast<const float*>(frequencies.data()),
              static_cast<const int32_t*>(slots.data()), epsilon,
              static_cast<uint8_t*>(pages.data()), rows, frequencies.size(0),
              pages.size(0), pages.size(1));
    };
    if (positions.dtype() == DataType::kInt32) launch(int32_t{});
    else launch(int64_t{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 fused SWA store launch failed");
  }
};
}  // namespace infini::ops

#endif
