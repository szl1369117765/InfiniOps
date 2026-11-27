#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "base/i8_gemm.h"
#include "gemv_wave64.cuh"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/dtk_i8_blas.h"

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
                       ptrdiff_t bias_col, bool activation_scale_first = false) {
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
      const float dot = static_cast<float>(partial[0]);
      float value = activation_scale_first ? __fmul_rn(__fmul_rn(dot, as[row]), bs[col])
                                           : ScaleDot(partial[0], as[row], bs[col]);
      if (bias)
        value += Convert::template Cast<float>(
            bias[row * bias_row + col * bias_col]);
      out[idx] = Convert::template Cast<T>(value);
    }
    __syncthreads();
  }
}

static __global__ void prefill_projection_epilogue(__nv_bfloat16* out, const int32_t* dots,
                                                    const float* as, const float* bs,
                                                    size_t elements, size_t columns) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) return;
  const float scale = __fmul_rn(as[idx / columns], bs[idx % columns]);
  const float value = __fmul_rn(static_cast<float>(dots[idx]), scale);
  out[idx] = Caster<Device::Type::kHygon>::Cast<__nv_bfloat16>(value);
}

static constexpr unsigned kProjectionRows = 4;
static constexpr unsigned kProjectionM12Tiles = 3;
static constexpr unsigned kProjectionM12Rows = kProjectionRows * kProjectionM12Tiles;
static constexpr unsigned kProjectionColumns = 4096;
static constexpr unsigned kProjectionK = 1024;
static constexpr unsigned kWave = 64;
static constexpr unsigned kWavesPerBlock = 4;

static __device__ __forceinline__ int32_t LoadProjectionInt8x4(const int8_t* input) {
  int32_t packed;
  __builtin_memcpy(&packed, __builtin_assume_aligned(input, alignof(int32_t)), sizeof(packed));
  return packed;
}

static __device__ __forceinline__ int32_t ProjectionDot4(int32_t lhs, int32_t rhs, int32_t sum) {
#if defined(__AMDGCN__) && __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(lhs, rhs, sum, false);
#else
  uint32_t a = static_cast<uint32_t>(lhs), b = static_cast<uint32_t>(rhs);
  for (unsigned i = 0; i < 4; ++i) {
    sum += (static_cast<int>((a & 0xff) ^ 0x80) - 128) *
           (static_cast<int>((b & 0xff) ^ 0x80) - 128);
    a >>= 8;
    b >>= 8;
  }
  return sum;
#endif
}

static __device__ __forceinline__ void ProjectionQ4Tile(
    __nv_bfloat16* out, const int8_t* a, const float* as, const int8_t* b, const float* bs) {
  const unsigned lane = threadIdx.x % kWave;
  const unsigned col = blockIdx.x * kWavesPerBlock + threadIdx.x / kWave;
  const int8_t* weight = b + col * kProjectionK;
  int32_t dot[kProjectionRows] = {};
  #pragma unroll
  for (unsigned k = lane * 4; k < kProjectionK; k += kWave * 4) {
    const int32_t packed_weight = LoadProjectionInt8x4(weight + k);
    #pragma unroll
    for (unsigned row = 0; row < kProjectionRows; ++row) {
      const int32_t packed_input = LoadProjectionInt8x4(a + row * kProjectionK + k);
      dot[row] = ProjectionDot4(packed_input, packed_weight, dot[row]);
    }
  }
  #pragma unroll
  for (unsigned offset = kWave / 2; offset; offset /= 2) {
    #pragma unroll
    for (unsigned row = 0; row < kProjectionRows; ++row) {
      dot[row] += __shfl_down(dot[row], offset, kWave);
    }
  }
  if (lane == 0) {
    using Convert = Caster<Device::Type::kHygon>;
    #pragma unroll
    for (unsigned row = 0; row < kProjectionRows; ++row) {
      const float scale = __fmul_rn(as[row], bs[col]);
      const float value = __fmul_rn(static_cast<float>(dot[row]), scale);
      out[row * kProjectionColumns + col] = Convert::template Cast<__nv_bfloat16>(value);
    }
  }
}

// Keep the same INT32 dot and BF16 epilogue as the scalar path while
// loading each INT8 weight fragment for up to four adjacent input rows.
template <unsigned K>
static __global__ void matmul_decode_batch(
    __nv_bfloat16* out, const int8_t* a, const float* as, const int8_t* b,
    const float* bs, unsigned rows, unsigned columns) {
  constexpr unsigned tile_rows = 4;
  const unsigned lane = threadIdx.x % kWave;
  const unsigned col = blockIdx.x * kWavesPerBlock + threadIdx.x / kWave;
  if (col >= columns) return;
  const unsigned first = blockIdx.y * tile_rows;
  const unsigned count = min(rows - first, tile_rows);
  const int8_t* weight = b + size_t(col) * K;
  int32_t dots[tile_rows] = {};
#pragma unroll
  for (unsigned k = lane * 4; k < K; k += kWave * 4) {
    const auto packed_weight = LoadProjectionInt8x4(weight + k);
#pragma unroll
    for (unsigned row = 0; row < tile_rows; ++row) {
      if (row < count) {
        const auto input = LoadProjectionInt8x4(a + size_t(first + row) * K + k);
        dots[row] = ProjectionDot4(input, packed_weight, dots[row]);
      }
    }
  }
#pragma unroll
  for (unsigned offset = kWave / 2; offset; offset /= 2) {
#pragma unroll
    for (unsigned row = 0; row < tile_rows; ++row)
      dots[row] += __shfl_down(dots[row], offset, kWave);
  }
  if (lane == 0) {
    using Convert = Caster<Device::Type::kHygon>;
#pragma unroll
    for (unsigned row = 0; row < tile_rows; ++row) {
      if (row < count) {
        const auto value = ScaleDot(dots[row], as[first + row], bs[col]);
        out[size_t(first + row) * columns + col] = Convert::template Cast<__nv_bfloat16>(value);
      }
    }
  }
}

static __global__ void matmul_q4_projection(__nv_bfloat16* out, const int8_t* a,
                                            const float* as, const int8_t* b, const float* bs) {
  ProjectionQ4Tile(out, a, as, b, bs);
}

static __global__ void matmul_m12_projection(__nv_bfloat16* out, const int8_t* a,
                                             const float* as, const int8_t* b, const float* bs) {
  const size_t row = size_t(blockIdx.y) * kProjectionRows;
  ProjectionQ4Tile(out + row * kProjectionColumns, a + row * kProjectionK, as + row, b, bs);
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
    if (prefill_shape(c, a, bias)) {
      prefill_workspace_bytes_ = c.numel() * sizeof(int32_t);
    }
  }
  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override {
    return prefill_enabled() ? std::max(workspace_bytes_, prefill_workspace_bytes_)
                             : workspace_bytes_;
  }
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
    if (c.dtype() == DataType::kBFloat16 && launch_prefill(c, a, as, b, bs, bias)) {
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon INT8 prefill launch failed");
      return;
    }
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
    bool used_m12_projection = false;
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const auto width = c.size(1);
        const char* exact = nullptr;
        if (width == 1024) exact = std::getenv("INFINIOPS_DSV4_WQA_SCALE_FIRST");
        if (width == 512) exact = std::getenv("INFINIOPS_DSV4_WKV_PREFILL_SCALE_FIRST");
        if (!bias && exact && exact[0] == '1' && exact[1] == '\0' &&
            a.size(0) > 4 && a.size(0) <= 16 && a.size(1) == 4096) {
          hygon_i8_gemm::matmul<T>
              <<<static_cast<unsigned>(c.numel()), 128, 0, static_cast<cudaStream_t>(stream_)>>>(
                  static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
                  static_cast<const float*>(as.data()), static_cast<const int8_t*>(b.data()),
                  static_cast<const float*>(bs.data()), nullptr, c.size(0), c.size(1),
                  a.size(1), 0, 0, true);
          return;
        }
      }

      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        using namespace hygon_i8_gemm;
        const bool m12 = c.size(0) == kProjectionM12Rows;
        if (!bias && (c.size(0) == kProjectionRows || m12) && c.size(1) == kProjectionColumns &&
            a.size(1) == kProjectionK &&
            reinterpret_cast<std::uintptr_t>(a.data()) % alignof(int32_t) == 0 &&
            reinterpret_cast<std::uintptr_t>(b.data()) % alignof(int32_t) == 0) {
          const char* value = std::getenv(m12 ? "INFINIOPS_V4_I8_PROJECTION_M12"
                                            : "INFINIOPS_V4_I8_PROJECTION_Q4");
          const bool enabled = value && value[0] == '1' && value[1] == '\0';
          if (enabled) {
            if (m12) {
              const dim3 grid(kProjectionColumns / kWavesPerBlock, kProjectionM12Tiles);
              matmul_m12_projection<<<grid, kWave * kWavesPerBlock, 0,
                                      static_cast<cudaStream_t>(stream_)>>>(
                  static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
                  static_cast<const float*>(as.data()), static_cast<const int8_t*>(b.data()),
                  static_cast<const float*>(bs.data()));
              used_m12_projection = true;
              return;
            }
            matmul_q4_projection<<<kProjectionColumns / kWavesPerBlock, kWave * kWavesPerBlock,
                                   0, static_cast<cudaStream_t>(stream_)>>>(
                static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
                static_cast<const float*>(as.data()), static_cast<const int8_t*>(b.data()),
                static_cast<const float*>(bs.data()));
            return;
          }
        }
      }
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        using namespace hygon_i8_gemm;
        const char* setting = std::getenv("INFINIOPS_DSV4_I8_DECODE_TILE");
        if (!bias && setting && setting[0] == '1' && setting[1] == '\0' &&
            a.size(0) >= 2 && a.size(0) <= 16 && c.size(1) <= 16384 &&
            (a.size(1) == 1024 || a.size(1) == 4096) &&
            (reinterpret_cast<std::uintptr_t>(a.data()) & 3u) == 0 &&
            (reinterpret_cast<std::uintptr_t>(b.data()) & 3u) == 0) {
          const unsigned rows = a.size(0), columns = c.size(1);
          const dim3 grid((columns + kWavesPerBlock - 1) / kWavesPerBlock, (rows + 3) / 4);
          const auto stream = static_cast<cudaStream_t>(stream_);
          const auto run = [&](auto width) {
            matmul_decode_batch<decltype(width)::value><<<grid, kWave * kWavesPerBlock, 0, stream>>>(
                static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
                static_cast<const float*>(as.data()), static_cast<const int8_t*>(b.data()),
                static_cast<const float*>(bs.data()), rows, columns);
          };
          if (a.size(1) == 1024) run(std::integral_constant<unsigned, 1024>{});
          else run(std::integral_constant<unsigned, 4096>{});
          return;
        }
      }
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
            const char* exact = nullptr;
            if (std::is_same_v<T, __nv_bfloat16> && !bias && k == 4096) {
              if (n == 1024) {
                exact = std::getenv("INFINIOPS_DSV4_WQA_DECODE_SCALE_FIRST");
              } else if (n == 512) {
                exact = std::getenv("INFINIOPS_DSV4_WKV_DECODE_SCALE_FIRST");
              }
            }
            const bool activation_scale_first = exact &&
                exact[0] == '1' && exact[1] == '\0';
            hygon_i8_gemm::gemv_i8_hygon<k, T>
                <<<blocks, threads, 0, static_cast<cudaStream_t>(stream_)>>>(
                    static_cast<T*>(c.data()),
                    bias ? static_cast<const T*>(bias->data()) : nullptr,
                    static_cast<const int8_t*>(a.data()),
                    static_cast<const float*>(as.data()),
                    static_cast<const int8_t*>(b.data()),
                    static_cast<const float*>(bs.data()), n,
                    bias ? bias->stride(1) : 0, activation_scale_first);
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
    if (used_m12_projection) {
      const char* trace = std::getenv("INFINIOPS_V4_I8_PROJECTION_M12_TRACE");
      if (trace && trace[0] == '1' && trace[1] == '\0')
        std::fprintf(stderr, "MTP_M12_I8_PROJECTION 12 4096 1024 device=%d\n",
                     static_cast<int>(c.device().index()));
    }
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

  static bool prefill_enabled() {
    const char* mode = std::getenv("INFINIOPS_V4_I8_PROJECTION_PREFILL_DTK");
    return mode && mode[0] == '1' && mode[1] == '\0';
  }

  static bool prefill_shape(Tensor c, Tensor a, const std::optional<Tensor>& bias) {
    if (bias || a.ndim() != 2 || c.ndim() != 2 || c.dtype() != DataType::kBFloat16) return false;
    return c.size(0) >= 64 && c.size(0) <= 256 && a.size(1) == 1024 &&
           (c.size(1) == 4096 || c.size(1) == 8192);
  }

  bool launch_prefill(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                      const std::optional<Tensor>& bias) const {
    if (!prefill_workspace_bytes_ || !workspace_ || !prefill_enabled() || !prefill_shape(c, a, bias) ||
        workspace_size_in_bytes_ < c.numel() * sizeof(int32_t)) return false;
    const auto addresses = reinterpret_cast<uintptr_t>(workspace_) |
                           reinterpret_cast<uintptr_t>(a.data()) |
                           reinterpret_cast<uintptr_t>(b.data());
    if (addresses % 16) return false;

    const auto stream = static_cast<cudaStream_t>(stream_);
    const auto handle = dtk_i8_blas::Acquire(c.device().index(), stream);
    if (!handle) return false;
    auto* dots = static_cast<int32_t*>(workspace_);
    const auto status = dtk_i8_blas::Dot(handle, dots, static_cast<const int8_t*>(a.data()),
        static_cast<const int8_t*>(b.data()), int(c.size(0)), int(c.size(1)), int(a.size(1)));
    if (status != CUBLAS_STATUS_SUCCESS) return false;
    constexpr unsigned threads = 256;
    hygon_i8_gemm::prefill_projection_epilogue<<<(c.numel() + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(c.data()), dots, static_cast<const float*>(as.data()),
        static_cast<const float*>(bs.data()), c.numel(), c.size(1));
    return true;
  }

  size_t prefill_workspace_bytes_{};
};

}  // namespace infini::ops
