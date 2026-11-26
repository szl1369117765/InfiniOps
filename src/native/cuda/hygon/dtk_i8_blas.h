#pragma once

#include <cstdint>
#include <map>

#include "native/cuda/hygon/blas.h"

namespace infini::ops::dtk_i8_blas {

using Backend = Blas<Device::Type::kHygon>;

// cuBLAS handles are mutable (stream and pointer mode) and belong to the
// device on which they are created. Do not share handles between rank threads.
class ThreadHandles final {
 public:
  ~ThreadHandles() {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    for (const auto& [device, handle] : handles_) {
      if (cudaSetDevice(device) == cudaSuccess)
        (void)Backend::BlasDestroy(handle);
    }
    (void)cudaSetDevice(previous);
  }

  cublasHandle_t Get(int device) {
    const auto existing = handles_.find(device);
    if (existing != handles_.end()) return existing->second;
    cublasHandle_t handle = nullptr;
    if (Backend::BlasCreate(&handle) != CUBLAS_STATUS_SUCCESS || !handle) {
      if (handle) (void)Backend::BlasDestroy(handle);
      return nullptr;
    }
    if (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) !=
        CUBLAS_STATUS_SUCCESS) {
      (void)Backend::BlasDestroy(handle);
      return nullptr;
    }
    try {
      handles_.emplace(device, handle);
    } catch (...) {
      (void)Backend::BlasDestroy(handle);
      return nullptr;
    }
    return handle;
  }

 private:
  std::map<int, cublasHandle_t> handles_;
};

inline cublasHandle_t Acquire(int device, cudaStream_t stream) {
  if (!stream) return nullptr;
  int active = -1;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaGetDevice(&active) != cudaSuccess || active != device ||
      cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone) {
    return nullptr;
  }
  thread_local ThreadHandles handles;
  const auto handle = handles.Get(device);
  if (!handle ||
      Backend::BlasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
    return nullptr;
  return handle;
}

inline cublasStatus_t Dot(cublasHandle_t handle, int32_t* output,
                          const int8_t* input, const int8_t* weights, int m,
                          int n, int k) {
  const int32_t alpha = 1, beta = 0;
  // The row-major [M,N] result is a column-major [N,M] view.
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                      weights, CUDA_R_8I, k, input, CUDA_R_8I, k, &beta, output,
                      CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
}

}  // namespace infini::ops::dtk_i8_blas
