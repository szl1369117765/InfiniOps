#pragma once
#include <optional>

#include "operator.h"
namespace infini::ops {
// WQ_A and WKV share one projection. The logical output buffer contains
// contiguous [M,1024] Q rows followed by contiguous [M,512] KV rows.
class DeepseekV4JoinedQkvConfig final : public Config {
 public:
  explicit DeepseekV4JoinedQkvConfig(bool scale_first, bool split_k = false)
      : scale_first(scale_first), split_k(split_k) {}
  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<DeepseekV4JoinedQkvConfig>(*this);
  }
  const bool scale_first;
  const bool split_k;
};

// Attention WO_B stores BF16-rounded values in a FP32 collective buffer.
class DeepseekV4WoBFloatConfig final : public Config {
 public:
  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<DeepseekV4WoBFloatConfig>(*this);
  }
};

class DeepseekV4WqBMatrixConfig final : public Config {
 public:
  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<DeepseekV4WqBMatrixConfig>(*this);
  }
};

class I8Gemm : public Operator<I8Gemm> {
 public:
  static constexpr bool kAutomaticDefault = true;
  I8Gemm(Tensor c, const Tensor a, const Tensor as, const Tensor b,
         const Tensor bs, const std::optional<Tensor> bias) {}
  virtual void operator()(Tensor c, const Tensor a, const Tensor as,
                          const Tensor b, const Tensor bs,
                          const std::optional<Tensor> bias) const = 0;
};
template <>
struct CacheKeyBuilder<I8Gemm> {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    if (const auto* joined = dynamic_cast<const DeepseekV4JoinedQkvConfig*>(&config)) {
      return detail::CacheKey::Build(config.implementation_index(), true, joined->scale_first, joined->split_k, args...);
    }
    if (dynamic_cast<const DeepseekV4WoBFloatConfig*>(&config)) {
      return detail::CacheKey::Build(config.implementation_index(), false, args...);
    }
    if (dynamic_cast<const DeepseekV4WqBMatrixConfig*>(&config)) {
      return detail::CacheKey::Build(config.implementation_index(), false, false, args...);
    }
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};
}  // namespace infini::ops
