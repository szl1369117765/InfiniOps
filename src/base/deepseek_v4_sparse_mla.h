#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_H_

#include "operator.h"

namespace infini::ops {

class DeepseekV4PrefillPageWindowConfig final : public Config {
 public:
  DeepseekV4PrefillPageWindowConfig(size_t first, size_t count) : first(first), count(count) {}
  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<DeepseekV4PrefillPageWindowConfig>(*this);
  }
  const size_t first, count;
};

// Reference FlashMLA ABI: queries [rows, heads, 512]; KV pages are packed
// FP8 NoPE plus BF16 RoPE and separate per-token exponent bytes.
class DeepseekV4SparseMla : public Operator<DeepseekV4SparseMla> {
 public:
  DeepseekV4SparseMla(const Tensor query, const Tensor swa, const Tensor indices,
                      const Tensor lengths, const Tensor sink, const std::optional<Tensor> extra,
                      const std::optional<Tensor> extra_indices, const std::optional<Tensor> extra_lengths,
                      int64_t extra_page_size, double scale, Tensor out) {
    (void)query;
    (void)swa;
    (void)indices;
    (void)lengths;
    (void)sink;
    (void)extra;
    (void)extra_indices;
    (void)extra_lengths;
    (void)extra_page_size;
    (void)scale;
    (void)out;
  }

  virtual void operator()(const Tensor query, const Tensor swa, const Tensor indices,
                          const Tensor lengths, const Tensor sink,
                          const std::optional<Tensor> extra,
                          const std::optional<Tensor> extra_indices,
                          const std::optional<Tensor> extra_lengths,
                          int64_t extra_page_size, double scale, Tensor out) const = 0;
};

template <>
struct CacheKeyBuilder<DeepseekV4SparseMla> {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    if (const auto* window = dynamic_cast<const DeepseekV4PrefillPageWindowConfig*>(&config)) {
      return detail::CacheKey::Build(config.implementation_index(), true, window->first,
                                     window->count, args...);
    }
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};

}  // namespace infini::ops

#endif
