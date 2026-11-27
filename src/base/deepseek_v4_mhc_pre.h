#pragma once
#include "operator.h"

namespace infini::ops {
class DeepseekV4MhcPreSingleTokenBatchConfig final : public Config {
 public:
  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<DeepseekV4MhcPreSingleTokenBatchConfig>(*this);
  }

  static bool Enabled(const Config* config) {
    return dynamic_cast<const DeepseekV4MhcPreSingleTokenBatchConfig*>(config) != nullptr;
  }
};

class DeepseekV4MhcPre : public Operator<DeepseekV4MhcPre> {
 public:
  DeepseekV4MhcPre(const Tensor residual,const Tensor projection,const Tensor scales,
                  const Tensor bias,double norm_eps,double mix_eps,int64_t iterations,
                  Tensor collapsed,Tensor post,Tensor combination) {}
  virtual void operator()(const Tensor residual,const Tensor projection,const Tensor scales,
                          const Tensor bias,double norm_eps,double mix_eps,int64_t iterations,
                          Tensor collapsed,Tensor post,Tensor combination) const = 0;
};

template <>
struct CacheKeyBuilder<DeepseekV4MhcPre> {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    if (DeepseekV4MhcPreSingleTokenBatchConfig::Enabled(&config)) {
      return detail::CacheKey::Build(config.implementation_index(), true, args...);
    }
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};
} // namespace infini::ops
