#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_PREPARE_SPARSE_MLA_SCHEDULE_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_PREPARE_SPARSE_MLA_SCHEDULE_H_

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

#include "base/deepseek_v4_prepare_sparse_mla_schedule.h"
#include "native/cuda/hygon/flash_mla/bridge.hpp"

namespace infini::ops {

template <>
class Operator<DeepseekV4PrepareSparseMlaSchedule, Device::Type::kHygon>
    : public DeepseekV4PrepareSparseMlaSchedule {
 public:
  static bool supports(const Tensor swa_lengths,
                       const std::optional<Tensor> extra_lengths,
                       int64_t swa_index_width, int64_t extra_index_width,
                       const Tensor schedule) {
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    static_assert(kScheduleBytes == hygon_flash_mla::kScheduleStorageBytes);
    if (swa_lengths.device().type() != Device::Type::kHygon ||
        swa_lengths.dtype() != DataType::kInt32 || swa_lengths.ndim() != 1 ||
        swa_lengths.size(0) < 1 ||
        swa_lengths.size(0) > hygon_flash_mla::kMaxScheduleRows ||
        !swa_lengths.IsContiguous() || swa_index_width < 1 ||
        swa_index_width > std::numeric_limits<int>::max() ||
        static_cast<bool>(extra_lengths) != (extra_index_width != 0) ||
        extra_index_width < 0 ||
        extra_index_width > std::numeric_limits<int>::max() ||
        schedule.dtype() != DataType::kUInt8 ||
        schedule.shape() != Tensor::Shape{kScheduleBytes} ||
        schedule.device() != swa_lengths.device() || !schedule.IsContiguous()) {
      return false;
    }
    return !extra_lengths || (extra_lengths->device() == swa_lengths.device() &&
                              extra_lengths->dtype() == DataType::kInt32 &&
                              extra_lengths->IsContiguous() &&
                              extra_lengths->shape() == swa_lengths.shape());
#else
    return false;
#endif
  }

  Operator(const Tensor swa_lengths, const std::optional<Tensor> extra_lengths,
           int64_t swa_index_width, int64_t extra_index_width, Tensor schedule)
      : DeepseekV4PrepareSparseMlaSchedule(swa_lengths, extra_lengths,
                                           swa_index_width, extra_index_width,
                                           schedule) {
    if (!supports(swa_lengths, extra_lengths, swa_index_width,
                  extra_index_width, schedule) ||
        cudaDeviceGetAttribute(&partitions_, cudaDevAttrMultiProcessorCount,
                               swa_lengths.device().index()) != cudaSuccess ||
        partitions_ < 1 ||
        partitions_ > hygon_flash_mla::kMaxSchedulePartitions) {
      throw std::invalid_argument(
          "DeepSeek V4 FlashMLA shared schedule inputs are unsupported");
    }
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor swa_lengths,
                  const std::optional<Tensor> extra_lengths,
                  int64_t swa_index_width, int64_t extra_index_width,
                  Tensor schedule) const override {
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    if (!supports(swa_lengths, extra_lengths, swa_index_width,
                  extra_index_width, schedule))
      throw std::invalid_argument(
          "DeepSeek V4 FlashMLA shared schedule inputs changed");
    hygon_flash_mla::prepare_schedule(
        static_cast<int>(swa_lengths.size(0)), partitions_,
        static_cast<int>(swa_index_width), static_cast<int>(extra_index_width),
        static_cast<const int32_t*>(swa_lengths.data()),
        extra_lengths ? static_cast<const int32_t*>(extra_lengths->data())
                      : nullptr,
        schedule.data(), stream_);
#else
    throw std::invalid_argument("DeepSeek V4 FlashMLA is not built");
#endif
  }

 private:
  int partitions_{};
};

}  // namespace infini::ops

#endif
