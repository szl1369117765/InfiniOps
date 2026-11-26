#pragma once
#include "operator.h"

namespace infini::ops {
// Query and weights may omit a leading prefix of rows with at most 512 visible keys.
// Lengths, page tables and output cover every query row. Key and slots
// describe independent cache writes and may both have zero rows.
class DeepseekV4C4Index : public Operator<DeepseekV4C4Index> {
 public:
  DeepseekV4C4Index(const Tensor key, const Tensor slots, const Tensor query,
                    const Tensor weights, const Tensor pages, const Tensor lengths,
                    Tensor cache, Tensor out) {}
  virtual void operator()(const Tensor key, const Tensor slots, const Tensor query,
                          const Tensor weights, const Tensor pages, const Tensor lengths,
                          Tensor cache, Tensor out) const = 0;
};
}
