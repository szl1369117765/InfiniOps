#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_STORE_SWA_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_STORE_SWA_H_

#include "operator.h"

namespace infini::ops {

// Normalize each projected SWA key, rotate its last 64 channels, and store it
// in a 256-token packed page. Valid slots must be unique within a launch.
class DeepseekV4StoreSwa : public Operator<DeepseekV4StoreSwa> {
 public:
  DeepseekV4StoreSwa(const Tensor key, const Tensor weight, const Tensor positions,
                     const Tensor frequencies, const Tensor slots, float epsilon,
                     Tensor pages) {
    (void)key;
    (void)weight;
    (void)positions;
    (void)frequencies;
    (void)slots;
    (void)epsilon;
    (void)pages;
  }

  virtual void operator()(const Tensor key, const Tensor weight, const Tensor positions,
                          const Tensor frequencies, const Tensor slots, float epsilon,
                          Tensor pages) const = 0;
};
}  // namespace infini::ops

#endif
