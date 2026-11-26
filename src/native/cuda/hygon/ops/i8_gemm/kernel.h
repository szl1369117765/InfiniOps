#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <type_traits>

#include "base/i8_gemm.h"
#include "gemv_wave64.cuh"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace hygon_i8_gemm {

static __device__ __forceinline__ float ScaleDot(int32_t dot, float activation_scale,
                                                 float weight_scale) {
  return __fmul_rn(static_cast<float>(dot), __fmul_rn(activation_scale, weight_scale));
}

static __global__ void ScalePrefill(__nv_bfloat16* out, const int32_t* dot,
                                   const float* as, const float* bs, size_t rows, size_t cols) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * cols;
       i += size_t(gridDim.x) * blockDim.x) {
    const auto scaled = ScaleDot(dot[i], as[i / cols], bs[i % cols]);
    out[i] = Caster<Device::Type::kHygon>::template Cast<__nv_bfloat16>(scaled);
  }
}

class PrefillHandle {
 public:
  explicit PrefillHandle(int device) : device_(device) {
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Could not create Hygon INT8 GEMM handle");
  }
  PrefillHandle(const PrefillHandle&) = delete;
  PrefillHandle& operator=(const PrefillHandle&) = delete;
  ~PrefillHandle() {
    int previous{};
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    if (previous != device_ && cudaSetDevice(device_) != cudaSuccess) return;
    (void)cublasDestroy(handle_);
    if (previous != device_) (void)cudaSetDevice(previous);
  }

  static cublasHandle_t Get(int device) {
    thread_local std::map<int, PrefillHandle> handles;
    return handles.try_emplace(device, device).first->second.handle_;
  }

 private:
  int device_;
  cublasHandle_t handle_{};
};


template <class T>
__global__ void matmul(T* out, const int8_t* a, const float* as,
                       const int8_t* b, const float* bs, const T* bias,
                       size_t m, size_t n, size_t k, ptrdiff_t bias_row,
                       ptrdiff_t bias_col) {
  __shared__ int32_t partial[128];
  for (size_t idx = blockIdx.x; idx < m * n; idx += gridDim.x) {
    const size_t row = idx / n, col = idx % n;
    int32_t sum = 0;
    for (size_t j = threadIdx.x; j < k; j += blockDim.x)
      sum += int32_t(a[row * k + j]) * b[col * k + j];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = 64; step; step /= 2) {
      if (threadIdx.x < step)
        partial[threadIdx.x] += partial[threadIdx.x + step];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      using Convert = Caster<Device::Type::kHygon>;
      // The W8A8 epilogue rounds the joint scale before applying it to the dot.
      float value = ScaleDot(partial[0], as[row], bs[col]);
      if (bias)
        value += Convert::template Cast<float>(
            bias[row * bias_row + col * bias_col]);
      out[idx] = Convert::template Cast<T>(value);
    }
    __syncthreads();
  }
}

}  // namespace hygon_i8_gemm

template <>
class Operator<I8Gemm, Device::Type::kHygon> : public I8Gemm {
 public:
  Operator(Tensor c, const Tensor a, const Tensor as, const Tensor b, const Tensor bs,
           const std::optional<Tensor> bias) : I8Gemm(c, a, as, b, bs, bias) {
    if (UsePrefillBlas(c, a, b, bias)) {
      workspace_bytes_ = PrefillWorkspaceBytes(c);
      (void)hygon_i8_gemm::PrefillHandle::Get(c.device().index());
    }
  }
  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override { return workspace_bytes_; }
  static bool supports(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                       const std::optional<Tensor> bias) {
    if (a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 || a.size(1) == 0 ||
        a.size(1) > INT32_MAX / (128 * 128) || b.size(0) != a.size(1) ||
        a.dtype() != DataType::kInt8 || b.dtype() != DataType::kInt8 ||
        as.dtype() != DataType::kFloat32 || bs.dtype() != DataType::kFloat32 ||
        (c.dtype() != DataType::kFloat32 && c.dtype() != DataType::kBFloat16 &&
         c.dtype() != DataType::kFloat16) ||
        c.shape() != Tensor::Shape{a.size(0), b.size(1)} ||
        as.shape() != Tensor::Shape{a.size(0), 1} ||
        bs.shape() != Tensor::Shape{b.size(1), 1} || !a.IsContiguous() ||
        !c.IsContiguous() || !as.IsContiguous() || !bs.IsContiguous() ||
        b.stride(0) != 1 || b.stride(1) != static_cast<ptrdiff_t>(b.size(0)) ||
        c.device().type() != Device::Type::kHygon || a.device() != c.device() ||
        b.device() != c.device() || as.device() != c.device() ||
        bs.device() != c.device())
      return false;
    return !bias || (bias->device() == c.device() &&
                     bias->dtype() == c.dtype() && bias->shape() == c.shape() &&
                     bias->stride(0) >= 0 && bias->stride(1) >= 0);
  }
  void operator()(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                  const std::optional<Tensor> bias) const override {
    if (!supports(c, a, as, b, bs, bias))
      throw std::invalid_argument("Unsupported Hygon INT8 GEMM");
    if (!c.numel()) return;
    if (workspace_bytes_ && UsePrefillBlas(c, a, b, bias)) {
      const auto required_bytes = PrefillWorkspaceBytes(c);
      if (!workspace_ || workspace_size_in_bytes_ < required_bytes ||
          required_bytes > workspace_bytes_)
        throw std::invalid_argument("Invalid Hygon INT8 prefill workspace");

      const auto stream = static_cast<cudaStream_t>(stream_);
      const auto handle = hygon_i8_gemm::PrefillHandle::Get(c.device().index());
      if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Could not set Hygon INT8 GEMM stream");

      const int n = c.size(1), k = a.size(1);
      const auto* input = static_cast<const int8_t*>(a.data());
      const auto* weight = static_cast<const int8_t*>(b.data());
      const auto* input_scale = static_cast<const float*>(as.data());
      const auto* weight_scale = static_cast<const float*>(bs.data());
      auto* output = static_cast<__nv_bfloat16*>(c.data());
      auto* dot = static_cast<int32_t*>(workspace_);
      const int32_t alpha = 1, beta = 0;
      for (size_t first_row = 0; first_row < a.size(0);) {
        const int rows = std::min(kPrefillTileRows, a.size(0) - first_row);
        const size_t elements = static_cast<size_t>(rows) * n;
        const auto* tile = input + first_row * k;
        const auto* tile_scale = input_scale + first_row;
        auto* tile_output = output + first_row * n;
        auto status = CUBLAS_STATUS_NOT_SUPPORTED;
        if (rows >= kPrefillMinRows) {
          status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
              &alpha, weight, CUDA_R_8I, k, tile, CUDA_R_8I, k,
              &beta, dot, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
          hygon_i8_gemm::ScalePrefill<<<(elements + 255) / 256, 256, 0, stream>>>(
              tile_output, dot, tile_scale, weight_scale, rows, n);
        } else if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
          hygon_i8_gemm::matmul<__nv_bfloat16>
              <<<static_cast<unsigned>(std::min<size_t>(elements, 65535)), 128, 0, stream>>>(
                  tile_output, tile, tile_scale, weight, weight_scale, nullptr, rows, n, k, 0, 0);
        } else {
          throw std::runtime_error("Hygon INT8 GEMM failed: " + std::to_string(int(status)));
        }
        if (cudaGetLastError() != cudaSuccess)
          throw std::runtime_error("Hygon INT8 prefill tile launch failed");
        first_row += rows;
      }
      return;
    }
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      if (a.size(0) == 1 && (a.size(1) == 1024 || a.size(1) == 4096) &&
          c.size(1) <= INT32_MAX - 8 &&
          (reinterpret_cast<uintptr_t>(a.data()) & 3u) == 0 &&
          (reinterpret_cast<uintptr_t>(b.data()) & 3u) == 0) {
        auto run = [&](auto width, auto vectorized) {
          constexpr int k = decltype(width)::value;
          const int n = static_cast<int>(c.size(1));
          const int threads =
              ((k == 1024 && n == 4096) || vectorized) ? 512 : 256;
          const int blocks = (n + threads / 64 - 1) / (threads / 64);
          if constexpr (decltype(vectorized)::value) {
            hygon_i8_gemm::gemv_vector_hygon<k, T>
                <<<blocks, threads, 0, static_cast<cudaStream_t>(stream_)>>>(
                    static_cast<T*>(c.data()),
                    bias ? static_cast<const T*>(bias->data()) : nullptr,
                    static_cast<const int8_t*>(a.data()),
                    static_cast<const float*>(as.data()),
                    static_cast<const int8_t*>(b.data()),
                    static_cast<const float*>(bs.data()), n,
                    bias ? bias->stride(1) : 0);
          } else {
            hygon_i8_gemm::gemv_i8_hygon<k, T>
                <<<blocks, threads, 0, static_cast<cudaStream_t>(stream_)>>>(
                    static_cast<T*>(c.data()),
                    bias ? static_cast<const T*>(bias->data()) : nullptr,
                    static_cast<const int8_t*>(a.data()),
                    static_cast<const float*>(as.data()),
                    static_cast<const int8_t*>(b.data()),
                    static_cast<const float*>(bs.data()), n,
                    bias ? bias->stride(1) : 0);
          }
        };
        if (a.size(1) == 1024) {
          run(std::integral_constant<int, 1024>{}, std::false_type{});
        } else if (c.size(1) == 1280 &&
                   (reinterpret_cast<uintptr_t>(a.data()) & 15u) == 0 &&
                   (reinterpret_cast<uintptr_t>(b.data()) & 15u) == 0) {
          run(std::integral_constant<int, 4096>{}, std::true_type{});
        } else {
          run(std::integral_constant<int, 4096>{}, std::false_type{});
        }
        return;
      }
      hygon_i8_gemm::matmul<T>
          <<<static_cast<unsigned>(std::min<size_t>(c.numel(), 65535)), 128, 0,
             static_cast<cudaStream_t>(stream_)>>>(
              static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
              static_cast<const float*>(as.data()),
              static_cast<const int8_t*>(b.data()),
              static_cast<const float*>(bs.data()),
              bias ? static_cast<const T*>(bias->data()) : nullptr, c.size(0),
              c.size(1), a.size(1), bias ? bias->stride(0) : 0,
              bias ? bias->stride(1) : 0);
    };
    if (c.dtype() == DataType::kFloat32)
      launch(float{});
    else if (c.dtype() == DataType::kFloat16)
      launch(__half{});
    else
      launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon INT8 GEMM launch failed");
  }
 private:
  static constexpr int kPrefillMinRows = 64;
  static constexpr size_t kPrefillTileRows = 4096;

  static size_t PrefillWorkspaceBytes(const Tensor& output) {
    return std::min(kPrefillTileRows, output.size(0)) * output.size(1) * sizeof(int32_t);
  }

  static bool UsePrefillBlas(const Tensor& c, const Tensor& a, const Tensor& b,
                             const std::optional<Tensor>& bias) {
    if (a.ndim() != 2 || c.ndim() != 2 || bias || c.dtype() != DataType::kBFloat16 ||
        a.size(0) < kPrefillMinRows ||
        (reinterpret_cast<uintptr_t>(a.data()) & 3u) ||
        (reinterpret_cast<uintptr_t>(b.data()) & 3u)) return false;
    const auto k = a.size(1), n = c.size(1);
    return (k == 4096 && (n == 512 || n == 1024)) || (k == 1024 && (n == 4096 || n == 8192));
  }

  size_t workspace_bytes_{};
};

}  // namespace infini::ops
