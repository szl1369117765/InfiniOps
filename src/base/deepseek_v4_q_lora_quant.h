#pragma once

#include "operator.h"

namespace infini::ops {

// Preserve the BF16 query needed by the C4 indexer and quantize its rounded values.
class DeepseekV4QLoraQuant : public Operator<DeepseekV4QLoraQuant> {
 public:
  DeepseekV4QLoraQuant(Tensor input, Tensor weight, float epsilon,
                      Tensor normalized, Tensor packed, Tensor scale) {}

  virtual void operator()(Tensor input, Tensor weight, float epsilon,
                          Tensor normalized, Tensor packed, Tensor scale) const = 0;
};

}  // namespace infini::ops
