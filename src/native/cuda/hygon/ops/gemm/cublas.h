#ifndef INFINI_OPS_HYGON_GEMM_CUBLAS_H_
#define INFINI_OPS_HYGON_GEMM_CUBLAS_H_

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "dsv4_stable_gate.cuh"
#include "native/cuda/hygon/blas.h"
#include "native/cuda/ops/gemm/blas.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kHygon>
    : public BlasGemm<Blas<Device::Type::kHygon>> {
 public:
  using Base = BlasGemm<Blas<Device::Type::kHygon>>;
  Operator(Tensor a, Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Base(a,b,c,alpha,beta,trans_a,trans_b,y) {
    // Make runs before stream capture; initialize the rank's BLAS handle here.
    (void)this->GetHandle();
  }
  Operator(Tensor a, Tensor b, Tensor y)
      : Operator(a,b,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,y) {}
  bool capture_safe() const override { return true; }
  using Base::operator();

  void operator()(Tensor a, Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    const char* mode = std::getenv("INFINIOPS_DSV4_STABLE_GATE");
    if (mode && mode[0] == '1' && mode[1] == '\0' && !c &&
        !trans_a.value_or(trans_a_) && !trans_b.value_or(trans_b_) &&
        gate_layout(a, b, y)) {
      using namespace hygon_stable_gate;
      const auto rows = y.size(0);
      const dim3 tiles((rows + kTile - 1) / kTile, kN / kTile);
      project<<<tiles, kWave, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const __nv_bfloat16*>(a.data()),
          static_cast<const __nv_bfloat16*>(b.data()),
          static_cast<float*>(y.data()), rows, alpha.value_or(alpha_));
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("launching Hygon DeepSeek V4 stable MoE gate failed");
      return;
    }
    Base::operator()(a,b,c,alpha,beta,trans_a,trans_b,y);
  }

 private:
  static bool overlaps(const void* lhs, size_t lhs_bytes, const void* rhs,
                       size_t rhs_bytes) {
    const auto first = reinterpret_cast<uintptr_t>(lhs);
    const auto second = reinterpret_cast<uintptr_t>(rhs);
    return first <= second ? second - first < lhs_bytes
                           : first - second < rhs_bytes;
  }

  static bool gate_layout(const Tensor& a, const Tensor& b, const Tensor& y) {
    if (a.ndim() != 2 || b.ndim() != 2 || y.ndim() != 2 ||
        a.dtype() != DataType::kBFloat16 || b.dtype() != DataType::kBFloat16 ||
        y.dtype() != DataType::kFloat32 || a.size(1) != hygon_stable_gate::kK ||
        b.size(0) != hygon_stable_gate::kK || b.size(1) != hygon_stable_gate::kN ||
        y.size(0) < 1024 || y.size(0) > std::numeric_limits<int>::max() ||
        a.size(0) != y.size(0) || y.size(1) != hygon_stable_gate::kN ||
        a.stride(0) != hygon_stable_gate::kK || a.stride(1) != 1 ||
        b.stride(0) != 1 || b.stride(1) != hygon_stable_gate::kK ||
        y.stride(0) != hygon_stable_gate::kN || y.stride(1) != 1) {
      return false;
    }
    const auto output_bytes = y.numel() * sizeof(float);
    return !overlaps(y.data(), output_bytes, a.data(), a.numel() * sizeof(uint16_t)) &&
           !overlaps(y.data(), output_bytes, b.data(), b.numel() * sizeof(uint16_t));
  }

};

}  // namespace infini::ops

#endif
