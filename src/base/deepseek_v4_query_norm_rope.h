#pragma once

#include "operator.h"

namespace infini::ops {

// RMS-normalize each BF16 query head and rotate its last 64 channels.
class DeepseekV4QueryNormRope : public Operator<DeepseekV4QueryNormRope> {
 public:
  DeepseekV4QueryNormRope(const Tensor query, const Tensor weight,
                         const Tensor positions, const Tensor frequencies,
                         float epsilon, Tensor output) {
    (void)query;
    (void)weight;
    (void)positions;
    (void)frequencies;
    (void)epsilon;
    (void)output;
  }

  virtual void operator()(const Tensor query, const Tensor weight,
                          const Tensor positions, const Tensor frequencies,
                          float epsilon, Tensor output) const = 0;
};

}  // namespace infini::ops
