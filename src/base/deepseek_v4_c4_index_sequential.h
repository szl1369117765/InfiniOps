#pragma once

#include "operator.h"

namespace infini::ops {

// Updates the C4 index-key cache and lists up to 512 visible keys in logical order.
class DeepseekV4C4IndexSequential : public Operator<DeepseekV4C4IndexSequential> {
 public:
  static constexpr size_t kMaxCandidates = 512;

  DeepseekV4C4IndexSequential(const Tensor key, const Tensor slots, const Tensor pages,
                              const Tensor lengths, Tensor cache, Tensor out) {
    (void)key;
    (void)slots;
    (void)pages;
    (void)lengths;
    (void)cache;
    (void)out;
  }

  virtual void operator()(const Tensor key, const Tensor slots, const Tensor pages,
                          const Tensor lengths, Tensor cache, Tensor out) const = 0;
};

}  // namespace infini::ops
