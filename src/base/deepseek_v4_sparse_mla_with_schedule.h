#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_WITH_SCHEDULE_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_WITH_SCHEDULE_H_

#include "operator.h"

namespace infini::ops {

// Consume a schedule prepared on the same stream for these exact length rows
// and index widths. Each invocation owns its attention scratch independently.
class DeepseekV4SparseMlaWithSchedule
    : public Operator<DeepseekV4SparseMlaWithSchedule> {
 public:
  DeepseekV4SparseMlaWithSchedule(const Tensor query, const Tensor swa,
                                  const Tensor indices, const Tensor lengths,
                                  const Tensor sink,
                                  const std::optional<Tensor> extra,
                                  const std::optional<Tensor> extra_indices,
                                  const std::optional<Tensor> extra_lengths,
                                  int64_t extra_page_size, double scale,
                                  const Tensor schedule, Tensor out) {
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
    (void)schedule;
    (void)out;
  }

  virtual void operator()(const Tensor query, const Tensor swa,
                          const Tensor indices, const Tensor lengths,
                          const Tensor sink, const std::optional<Tensor> extra,
                          const std::optional<Tensor> extra_indices,
                          const std::optional<Tensor> extra_lengths,
                          int64_t extra_page_size, double scale,
                          const Tensor schedule, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
