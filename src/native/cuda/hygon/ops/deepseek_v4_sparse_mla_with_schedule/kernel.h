#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_SPARSE_MLA_WITH_SCHEDULE_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_SPARSE_MLA_WITH_SCHEDULE_H_

#include "base/deepseek_v4_sparse_mla_with_schedule.h"
#include "native/cuda/hygon/ops/deepseek_v4_sparse_mla/kernel.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4SparseMlaWithSchedule, Device::Type::kHygon>
    : public DeepseekV4SparseMlaWithSchedule {
 public:
  static bool supports(const Tensor q, const Tensor swa, const Tensor indices, const Tensor lengths,
                       const Tensor sink, const std::optional<Tensor> extra,
                       const std::optional<Tensor> extra_indices, const std::optional<Tensor> extra_lengths,
                       int64_t extra_page_size, double scale, const Tensor schedule, const Tensor out) {
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    using Legacy = Operator<DeepseekV4SparseMla, Device::Type::kHygon>;
    return Legacy::supports(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size,
                            scale, out) &&
           q.dtype() == DataType::kBFloat16 && q.size(1) <= 64 && q.size(0) <= hygon_flash_mla::kMaxScheduleRows &&
           indices.size(1) <= std::numeric_limits<int>::max() &&
           (!extra_indices || extra_indices->size(1) <= std::numeric_limits<int>::max()) &&
           schedule.device() == q.device() && schedule.dtype() == DataType::kUInt8 &&
           schedule.shape() == Tensor::Shape{hygon_flash_mla::kScheduleStorageBytes} && schedule.IsContiguous();
#else
    return false;
#endif
  }

  Operator(const Tensor q, const Tensor swa, const Tensor indices, const Tensor lengths,
           const Tensor sink, const std::optional<Tensor> extra, const std::optional<Tensor> extra_indices,
           const std::optional<Tensor> extra_lengths, int64_t extra_page_size, double scale,
           const Tensor schedule, Tensor out)
      : DeepseekV4SparseMlaWithSchedule(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths,
                                       extra_page_size, scale, schedule, out) {
    if (!supports(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size,
                  scale, schedule, out))
      throw std::invalid_argument("DeepSeek V4 scheduled FlashMLA inputs are unsupported");
    partitions_ = v4_sparse_mla_detail::device_partitions(q);
    if (partitions_ < 1 || partitions_ > hygon_flash_mla::kMaxSchedulePartitions)
      throw std::invalid_argument("DeepSeek V4 scheduled FlashMLA device geometry is unsupported");
    workspace_bytes_ = hygon_flash_mla::scheduled_workspace_bytes(q.size(0), partitions_);
  }

  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override { return workspace_bytes_; }

  void operator()(const Tensor q, const Tensor swa, const Tensor indices, const Tensor lengths,
                  const Tensor sink, const std::optional<Tensor> extra, const std::optional<Tensor> extra_indices,
                  const std::optional<Tensor> extra_lengths, int64_t extra_page_size, double scale,
                  const Tensor schedule, Tensor out) const override {
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    using namespace v4_sparse_mla_detail;
    if (!supports(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size,
                  scale, schedule, out))
      throw std::invalid_argument("DeepSeek V4 scheduled FlashMLA inputs changed");
    const auto first = pages_for(swa, indices, lengths, 256);
    const auto second = extra ? pages_for(*extra, *extra_indices, *extra_lengths, extra_page_size) : Pages{};
    hygon_flash_mla::run_with_schedule(vendor_arguments(q, sink, out, first, second, workspace_,
        workspace_size_in_bytes_, stream_, partitions_, scale), schedule.data());
#else
    throw std::invalid_argument("DeepSeek V4 FlashMLA is not built");
#endif
  }

 private:
  int partitions_{};
  size_t workspace_bytes_{};
};

}  // namespace infini::ops

#endif
