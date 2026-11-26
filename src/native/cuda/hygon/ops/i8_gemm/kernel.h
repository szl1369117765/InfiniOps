#pragma once
#include <cuda_runtime.h>

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
      const float scale = __fmul_rn(as[row], bs[col]);
      float value = __fmul_rn(static_cast<float>(partial[0]), scale);
      if (bias)
        value += Convert::template Cast<float>(
            bias[row * bias_row + col * bias_col]);
      out[idx] = Convert::template Cast<T>(value);
    }
    __syncthreads();
  }
}

static __global__ void prefill_projection_epilogue(
    __nv_bfloat16* out, const int32_t* dots, const float* as, const float* bs,
    size_t elements, size_t columns) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elements) return;
  const float scale = __fmul_rn(as[idx / columns], bs[idx % columns]);
  const float value = __fmul_rn(static_cast<float>(dots[idx]), scale);
  out[idx] = Caster<Device::Type::kHygon>::Cast<__nv_bfloat16>(value);
}

static constexpr unsigned kProjectionRows = 4;
static constexpr unsigned kProjectionM12Tiles = 3;
static constexpr unsigned kProjectionM12Rows =
    kProjectionRows * kProjectionM12Tiles;
static constexpr unsigned kProjectionColumns = 4096;
static constexpr unsigned kProjectionK = 1024;
static constexpr unsigned kWave = 64;
static constexpr unsigned kWavesPerBlock = 4;

static __device__ __forceinline__ int32_t
LoadProjectionInt8x4(const int8_t* input) {
  int32_t packed;
  __builtin_memcpy(&packed, __builtin_assume_aligned(input, alignof(int32_t)),
                   sizeof(packed));
  return packed;
}

static __device__ __forceinline__ int32_t ProjectionDot4(int32_t lhs,
                                                         int32_t rhs,
                                                         int32_t sum) {
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

static __device__ __forceinline__ void ProjectionQ4Tile(__nv_bfloat16* out,
                                                        const int8_t* a,
                                                        const float* as,
                                                        const int8_t* b,
                                                        const float* bs) {
  const unsigned lane = threadIdx.x % kWave;
  const unsigned col = blockIdx.x * kWavesPerBlock + threadIdx.x / kWave;
  const int8_t* weight = b + col * kProjectionK;
  int32_t dot[kProjectionRows] = {};
#pragma unroll
  for (unsigned k = lane * 4; k < kProjectionK; k += kWave * 4) {
    const int32_t packed_weight = LoadProjectionInt8x4(weight + k);
#pragma unroll
    for (unsigned row = 0; row < kProjectionRows; ++row) {
      const int32_t packed_input =
          LoadProjectionInt8x4(a + row * kProjectionK + k);
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
      out[row * kProjectionColumns + col] =
          Convert::template Cast<__nv_bfloat16>(value);
    }
  }
}

static __global__ void matmul_q4_projection(__nv_bfloat16* out, const int8_t* a,
                                            const float* as, const int8_t* b,
                                            const float* bs) {
  ProjectionQ4Tile(out, a, as, b, bs);
}

static __global__ void matmul_m12_projection(__nv_bfloat16* out,
                                             const int8_t* a, const float* as,
                                             const int8_t* b, const float* bs) {
  const size_t row = size_t(blockIdx.y) * kProjectionRows;
  ProjectionQ4Tile(out + row * kProjectionColumns, a + row * kProjectionK,
                   as + row, b, bs);
}

}  // namespace hygon_i8_gemm

template <>
class Operator<I8Gemm, Device::Type::kHygon> : public I8Gemm {
 public:
  Operator(Tensor c, const Tensor a, const Tensor as, const Tensor b,
           const Tensor bs, const std::optional<Tensor> bias)
      : I8Gemm(c, a, as, b, bs, bias) {
    if (prefill_shape(c, a, bias)) {
      prefill_workspace_bytes_ = c.numel() * sizeof(int32_t);
    }
  }
  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override {
    return prefill_workspace_bytes_ && prefill_enabled()
               ? prefill_workspace_bytes_
               : 0;
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
    bool used_m12_projection = false;
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        using namespace hygon_i8_gemm;
        if (launch_prefill(c, a, as, b, bs, bias)) return;
        const bool m12 = c.size(0) == kProjectionM12Rows;
        if (!bias && (c.size(0) == kProjectionRows || m12) &&
            c.size(1) == kProjectionColumns && a.size(1) == kProjectionK &&
            reinterpret_cast<std::uintptr_t>(a.data()) % alignof(int32_t) ==
                0 &&
            reinterpret_cast<std::uintptr_t>(b.data()) % alignof(int32_t) ==
                0) {
          const char* value =
              std::getenv(m12 ? "INFINIOPS_V4_I8_PROJECTION_M12"
                              : "INFINIOPS_V4_I8_PROJECTION_Q4");
          const bool enabled = value && value[0] == '1' && value[1] == '\0';
          if (enabled) {
            if (m12) {
              const dim3 grid(kProjectionColumns / kWavesPerBlock,
                              kProjectionM12Tiles);
              matmul_m12_projection<<<grid, kWave * kWavesPerBlock, 0,
                                      static_cast<cudaStream_t>(stream_)>>>(
                  static_cast<T*>(c.data()),
                  static_cast<const int8_t*>(a.data()),
                  static_cast<const float*>(as.data()),
                  static_cast<const int8_t*>(b.data()),
                  static_cast<const float*>(bs.data()));
              used_m12_projection = true;
              return;
            }
            matmul_q4_projection<<<kProjectionColumns / kWavesPerBlock,
                                   kWave * kWavesPerBlock, 0,
                                   static_cast<cudaStream_t>(stream_)>>>(
                static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
                static_cast<const float*>(as.data()),
                static_cast<const int8_t*>(b.data()),
                static_cast<const float*>(bs.data()));
            return;
          }
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
    if (used_m12_projection) {
      const char* trace = std::getenv("INFINIOPS_V4_I8_PROJECTION_M12_TRACE");
      if (trace && trace[0] == '1' && trace[1] == '\0')
        std::fprintf(stderr, "MTP_M12_I8_PROJECTION 12 4096 1024 device=%d\n",
                     static_cast<int>(c.device().index()));
    }
  }

 private:
  static bool prefill_enabled() {
    const char* mode = std::getenv("INFINIOPS_V4_I8_PROJECTION_PREFILL_DTK");
    return mode && mode[0] == '1' && mode[1] == '\0';
  }

  static bool prefill_shape(Tensor c, Tensor a,
                            const std::optional<Tensor>& bias) {
    if (bias || a.ndim() != 2 || c.ndim() != 2 ||
        c.dtype() != DataType::kBFloat16)
      return false;
    return c.size(0) >= 64 && c.size(0) <= 256 && a.size(1) == 1024 &&
           (c.size(1) == 4096 || c.size(1) == 8192);
  }

  bool launch_prefill(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                      const std::optional<Tensor>& bias) const {
    if (!prefill_workspace_bytes_ || !workspace_ || !prefill_enabled() ||
        !prefill_shape(c, a, bias) ||
        workspace_size_in_bytes_ < c.numel() * sizeof(int32_t))
      return false;
    const auto addresses = reinterpret_cast<uintptr_t>(workspace_) |
                           reinterpret_cast<uintptr_t>(a.data()) |
                           reinterpret_cast<uintptr_t>(b.data());
    if (addresses % 16) return false;

    const auto stream = static_cast<cudaStream_t>(stream_);
    const auto handle = dtk_i8_blas::Acquire(c.device().index(), stream);
    if (!handle) return false;
    auto* dots = static_cast<int32_t*>(workspace_);
    const auto status =
        dtk_i8_blas::Dot(handle, dots, static_cast<const int8_t*>(a.data()),
                         static_cast<const int8_t*>(b.data()), int(c.size(0)),
                         int(c.size(1)), int(a.size(1)));
    if (status != CUBLAS_STATUS_SUCCESS) return false;
    constexpr unsigned threads = 256;
    hygon_i8_gemm::prefill_projection_epilogue<<<
        (c.numel() + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(c.data()), dots,
        static_cast<const float*>(as.data()),
        static_cast<const float*>(bs.data()), c.numel(), c.size(1));
    return true;
  }

  size_t prefill_workspace_bytes_{};
};

}  // namespace infini::ops
