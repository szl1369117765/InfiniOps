#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops::v4_mhc_post_opt {
// Reuse the four residual inputs for the four output streams. Do not reassociate
// the residual-first FMA chain or change the final round-toward-zero conversion.
static __global__ void mix_reuse(const __nv_bfloat16* activation,const __nv_bfloat16* residual,
    const float* post,const float* matrix,__nv_bfloat16* out,size_t rows,size_t hidden) {
  const size_t count=rows*hidden;
  for(size_t index=blockIdx.x*blockDim.x+threadIdx.x;index<count;index+=gridDim.x*blockDim.x) {
    const size_t row=index/hidden,col=index%hidden;
    const float r0=__bfloat162float(residual[row*4*hidden+col]);
    const float r1=__bfloat162float(residual[(row*4+1)*hidden+col]);
    const float r2=__bfloat162float(residual[(row*4+2)*hidden+col]);
    const float r3=__bfloat162float(residual[(row*4+3)*hidden+col]);
    const float x=__bfloat162float(activation[row*hidden+col]);
#pragma unroll
    for(unsigned stream=0;stream<4;++stream) {
      float value=matrix[row*16+stream]*r0;
      value=fmaf(post[row*4+stream],x,value);
      value=fmaf(matrix[row*16+4+stream],r1,value);
      value=fmaf(matrix[row*16+8+stream],r2,value);
      value=fmaf(matrix[row*16+12+stream],r3,value);
      out[(row*4+stream)*hidden+col]=__float2bfloat16_rz(value);
    }
  }
}

using FloatPair = float __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ FloatPair load_pair(const __nv_bfloat16* base) {
  const auto packed = *reinterpret_cast<const unsigned*>(base);
  return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000u)};
}

template<bool PackedFma>
static __device__ __forceinline__ FloatPair pair_fma(float scale, FloatPair source, FloatPair accum) {
#if defined(__gfx936__)
  if constexpr (PackedFma) {
    const FloatPair factors{scale,scale};
    FloatPair result;
    asm("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(result) : "v"(factors), "v"(source), "v"(accum));
    return result;
  }
#endif
  return {fmaf(scale, source[0], accum[0]), fmaf(scale, source[1], accum[1])};
}

// Each thread reads and writes two adjacent channels without changing
// either channel's residual-first FMA order or BF16 rounding.
template<bool PackedFma>
static __global__ void mix_pairs(const __nv_bfloat16* activation, const __nv_bfloat16* residual,
    const float* post, const float* matrix, __nv_bfloat16* out, size_t rows) {
  constexpr unsigned hidden = 4096, columns = 512, tiles = hidden / columns;
  for (size_t tile = blockIdx.x; tile < rows * tiles; tile += gridDim.x) {
    const size_t row = tile / tiles, col = tile % tiles * columns + threadIdx.x * 2;
    const auto* r = residual + row * 4 * hidden + col;
    const auto r0 = load_pair(r);
    const auto r1 = load_pair(r + hidden);
    const auto r2 = load_pair(r + 2 * hidden);
    const auto r3 = load_pair(r + 3 * hidden);
    const auto x = load_pair(activation + row * hidden + col);
#pragma unroll
    for (unsigned stream=0;stream<4;++stream) {
      const float initial = matrix[row*16+stream];
      FloatPair value{initial*r0[0],initial*r0[1]};
      value=pair_fma<PackedFma>(post[row*4+stream],x,value);
      value=pair_fma<PackedFma>(matrix[row*16+4+stream],r1,value);
      value=pair_fma<PackedFma>(matrix[row*16+8+stream],r2,value);
      value=pair_fma<PackedFma>(matrix[row*16+12+stream],r3,value);
      const auto left=__float2bfloat16_rz(value[0]), right=__float2bfloat16_rz(value[1]);
      const unsigned bits=unsigned(__bfloat16_as_ushort(left))
          | (unsigned(__bfloat16_as_ushort(right)) << 16);
      auto* dst=out+(row*4+stream)*hidden+col;
      *reinterpret_cast<unsigned*>(dst)=bits;
    }
  }
}

} // namespace infini::ops::v4_mhc_post_opt
