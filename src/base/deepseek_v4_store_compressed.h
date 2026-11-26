#pragma once

#include "operator.h"

namespace infini::ops {

// Normalize and rotate FP32 pooled keys, round to BF16, and write packed pages.
// Valid slots must be unique within a launch.
class DeepseekV4StoreCompressed : public Operator<DeepseekV4StoreCompressed> {
 public:
  DeepseekV4StoreCompressed(const Tensor pooled, const Tensor weight,
                            const Tensor positions, const Tensor frequencies,
                            const Tensor slots, float epsilon, int64_t page_tokens,
                            Tensor pages) {
    (void)pooled;
    (void)weight;
    (void)positions;
    (void)frequencies;
    (void)slots;
    (void)epsilon;
    (void)page_tokens;
    (void)pages;
  }

  virtual void operator()(const Tensor pooled, const Tensor weight,
                          const Tensor positions, const Tensor frequencies,
                          const Tensor slots, float epsilon, int64_t page_tokens,
                          Tensor pages) const = 0;
};

}  // namespace infini::ops
