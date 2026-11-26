#pragma once

#include "base/deepseek_v4_c4_index_sequential.h"
#include "native/cuda/hygon/ops/deepseek_v4_c4_index/kernel.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4C4IndexSequential, Device::Type::kHygon>
    : public DeepseekV4C4IndexSequential {
 public:
  static bool supports(Tensor key, Tensor slots, Tensor pages, Tensor lengths,
                       Tensor cache, Tensor out) {
    return v4_c4::supports_cache(key, slots, pages, lengths, cache, out);
  }

  Operator(Tensor key, Tensor slots, Tensor pages, Tensor lengths, Tensor cache, Tensor out)
      : DeepseekV4C4IndexSequential(key, slots, pages, lengths, cache, out) {
    if (!supports(key, slots, pages, lengths, cache, out))
      throw std::invalid_argument("Invalid DeepSeek V4 sequential C4 index geometry");
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor key, Tensor slots, Tensor pages, Tensor lengths,
                  Tensor cache, Tensor out) const override {
    if (!supports(key, slots, pages, lengths, cache, out))
      throw std::invalid_argument("Invalid DeepSeek V4 sequential C4 index execution");

    static_assert(kMaxCandidates == v4_c4::kTop);
    const auto stream = static_cast<cudaStream_t>(stream_);
    v4_c4::write_key_cache(key, slots, cache, stream);
    v4_c4::sequential_indices(pages, lengths, cache, out, stream);
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 sequential C4 index launch failed");
  }
};

}  // namespace infini::ops
