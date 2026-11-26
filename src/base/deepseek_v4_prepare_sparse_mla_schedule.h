#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_PREPARE_SPARSE_MLA_SCHEDULE_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_PREPARE_SPARSE_MLA_SCHEDULE_H_

#include "operator.h"

namespace infini::ops {

// Prepare an opaque Hygon FlashMLA schedule on the caller's stream.
class DeepseekV4PrepareSparseMlaSchedule : public Operator<DeepseekV4PrepareSparseMlaSchedule> {
 public:
  static constexpr size_t kScheduleBytes = 32832;

  DeepseekV4PrepareSparseMlaSchedule(const Tensor swa_lengths,
                                   const std::optional<Tensor> extra_lengths,
                                   int64_t swa_index_width, int64_t extra_index_width, Tensor schedule) {
    (void)swa_lengths;
    (void)extra_lengths;
    (void)swa_index_width;
    (void)extra_index_width;
    (void)schedule;
  }

  virtual void operator()(const Tensor swa_lengths, const std::optional<Tensor> extra_lengths,
                          int64_t swa_index_width, int64_t extra_index_width, Tensor schedule) const = 0;
};

}  // namespace infini::ops

#endif
