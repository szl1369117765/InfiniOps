#include "gfx93/decode/sparse_fp8/splitkv_mla_hip.cuh"

namespace gfx93::decode::sparse_fp8 {
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 16>(
    const SparseAttnDecodeParams &params);
}
