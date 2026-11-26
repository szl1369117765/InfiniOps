#include "bridge.hpp"

#include "params_hip.h"
#include "gfx9/decode/combine/combine_hip.h"
#include "gfx9/decode/get_decoding_sched_meta/get_decoding_sched_meta_hip.h"
#include "gfx93/decode/sparse_fp8/splitkv_mla_hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infini::ops::hygon_flash_mla {
namespace {
constexpr int kHeads = 64;
constexpr int kDimension = 512;

void check(hipError_t status) {
  if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

__global__ void pad_input(const uint16_t* query, const float* sink, uint16_t* padded,
                          float* padded_sink, int rows, int heads) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(rows) * kHeads * kDimension;
  for (size_t i = index; i < count; i += gridDim.x * blockDim.x) {
    const size_t head = (i / kDimension) % kHeads;
    const size_t row = i / (kHeads * kDimension);
    padded[i] = head < heads ? query[(row * heads + head) * kDimension + i % kDimension] : 0;
  }
  if (index < kHeads) padded_sink[index] = index < heads ? sink[index] : 0.0f;
}

__global__ void unpad_output(const uint16_t* padded, uint16_t* output, int rows, int heads) {
  const size_t count = static_cast<size_t>(rows) * heads * kDimension;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    const size_t row = i / (heads * kDimension);
    output[i] = padded[row * kHeads * kDimension + i % (heads * kDimension)];
  }
}

class Workspace {
 public:
  explicit Workspace(void* data) : next_(reinterpret_cast<uintptr_t>(data)) {}

  template<class T>
  T* take(size_t count) {
    next_ = (next_ + 255) & ~uintptr_t(255);
    auto* result = reinterpret_cast<T*>(next_);
    next_ += count * sizeof(T);
    return result;
  }

 private:
  uintptr_t next_;
};

struct Schedule {
  DecodingSchedMeta* metadata;
  int* splits;
};

Schedule schedule_view(const void* storage, int rows, int partitions) {
  static_assert(sizeof(DecodingSchedMeta) == 32 &&
                alignof(DecodingSchedMeta) == 32);
  static_assert(31 + sizeof(DecodingSchedMeta) * kMaxSchedulePartitions +
                    sizeof(int) * (kMaxScheduleRows + 1) <=
                kScheduleStorageBytes);
  if (!storage || rows <= 0 || rows > kMaxScheduleRows || partitions <= 0 ||
      partitions > kMaxSchedulePartitions)
    throw std::invalid_argument(
        "Unsupported FlashMLA shared schedule geometry");
  const auto address =
      (reinterpret_cast<uintptr_t>(storage) + 31) & ~uintptr_t(31);
  auto* metadata = reinterpret_cast<DecodingSchedMeta*>(address);
  auto* splits =
      reinterpret_cast<int*>(address + sizeof(DecodingSchedMeta) * partitions);
  return {metadata, splits};
}

void launch_schedule(int rows, int partitions, int swa_width, int extra_width,
                     const int32_t* swa_lengths, const int32_t* extra_lengths,
                     Schedule schedule, hipStream_t stream) {
  GetDecodeSchedMetaParams scheduler{rows,
                                     1,
                                     64,
                                     5,
                                     swa_width,
                                     extra_width,
                                     const_cast<int32_t*>(swa_lengths),
                                     const_cast<int32_t*>(extra_lengths),
                                     nullptr,
                                     schedule.metadata,
                                     schedule.splits,
                                     partitions,
                                     stream};
  gfx9::decode::run_get_decoding_sched_meta_kernel(scheduler);
}

void run_impl(const Arguments& a, const void* prepared_schedule) {
  const auto required = prepared_schedule
                            ? scheduled_workspace_bytes(a.rows, a.partitions)
                            : workspace_bytes(a.rows, a.partitions);
  if (!a.workspace || a.workspace_bytes < required)
    throw std::invalid_argument("FlashMLA workspace is too small");
  Workspace work(a.workspace);
  const size_t query_count = static_cast<size_t>(a.rows) * kHeads;
  const size_t split_count = static_cast<size_t>(a.rows + a.partitions) * kHeads;
  auto* query = work.take<cutlass::bfloat16_t>(query_count * kDimension);
  auto* output = work.take<cutlass::bfloat16_t>(query_count * kDimension);
  auto* sink = work.take<float>(kHeads);
  auto* lse = work.take<float>(query_count);
  auto* lse_accum = work.take<float>(split_count);
  auto* out_accum = work.take<float>(split_count * kDimension);
  const Schedule schedule =
      prepared_schedule ? schedule_view(prepared_schedule, a.rows, a.partitions)
                        : Schedule{work.take<DecodingSchedMeta>(a.partitions),
                                   work.take<int>(a.rows + 1)};
  const auto stream = static_cast<hipStream_t>(a.stream);
  const unsigned grid = std::min<size_t>((query_count * kDimension + 255) / 256, 65535);
  pad_input<<<grid,256,0,stream>>>(static_cast<const uint16_t*>(a.query),a.sink,
      reinterpret_cast<uint16_t*>(query),sink,a.rows,a.heads);
  check(hipGetLastError());

  SparseAttnDecodeParams p{};
  p.b=a.rows;
  p.s_q=1;
  p.h_q=kHeads;
  p.h_kv=1;
  p.d_qk=kDimension;
  p.d_v=kDimension;
  p.sm_scale=a.scale;
  p.sm_scale_div_log2=a.scale*1.4426950408889634f;
  p.num_blocks=a.swa.blocks;
  p.page_block_size=a.swa.page_tokens;
  p.topk=a.swa.index_width;
  p.model_type=ModelType::MODEL1;
  p.q=query;
  p.kv=reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(a.swa.data));
  p.indices=const_cast<int32_t*>(a.swa.indices);
  p.topk_length=const_cast<int32_t*>(a.swa.lengths);
  p.attn_sink=sink;
  p.lse=lse;
  p.out=output;
  p.stride_q_b=kHeads*kDimension;
  p.stride_q_s_q=kHeads*kDimension;
  p.stride_q_h_q=kDimension;
  p.stride_kv_block=a.swa.page_stride;
  p.stride_kv_row=584;
  p.stride_indices_b=a.swa.index_width;
  p.stride_indices_s_q=a.swa.index_width;
  p.stride_lse_b=kHeads;
  p.stride_lse_s_q=kHeads;
  p.stride_o_b=kHeads*kDimension;
  p.stride_o_s_q=kHeads*kDimension;
  p.stride_o_h_q=kDimension;
  p.extra_num_blocks=a.extra.blocks;
  p.extra_page_block_size=a.extra.page_tokens;
  p.extra_topk=a.extra.index_width;
  p.extra_kv=reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(a.extra.data));
  p.extra_indices=const_cast<int32_t*>(a.extra.indices);
  p.extra_topk_length=const_cast<int32_t*>(a.extra.lengths);
  p.stride_extra_kv_block=a.extra.page_stride;
  p.stride_extra_kv_row=584;
  p.stride_extra_indices_b=a.extra.index_width;
  p.stride_extra_indices_s_q=a.extra.index_width;
  p.stream=stream;
  p.num_sm_parts=a.partitions;
  p.tile_scheduler_metadata_ptr = schedule.metadata;
  p.num_splits_ptr = schedule.splits;
  p.lse_accum=lse_accum;
  p.o_accum=out_accum;
  p.stride_lse_accum_split=kHeads;
  p.stride_lse_accum_s_q=kHeads;
  p.stride_o_accum_split=kHeads*kDimension;
  p.stride_o_accum_s_q=kHeads*kDimension;
  p.stride_o_accum_h_q=kDimension;
  if (!prepared_schedule) {
    launch_schedule(a.rows, a.partitions, a.swa.index_width,
                    a.extra.index_width, a.swa.lengths, a.extra.lengths,
                    schedule, stream);
  }
  gfx93::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1,kHeads>(p);

  CombineParams combine{};
  combine.b=a.rows;
  combine.s_q=1;
  combine.h_q=kHeads;
  combine.d_v=kDimension;
  combine.lse=lse;
  combine.out=output;
  combine.stride_lse_b=kHeads;
  combine.stride_lse_s_q=kHeads;
  combine.stride_o_b=kHeads*kDimension;
  combine.stride_o_s_q=kHeads*kDimension;
  combine.stride_o_h_q=kDimension;
  combine.lse_accum=lse_accum;
  combine.o_accum=out_accum;
  combine.stride_lse_accum_split=kHeads;
  combine.stride_lse_accum_s_q=kHeads;
  combine.stride_o_accum_split=kHeads*kDimension;
  combine.stride_o_accum_s_q=kHeads*kDimension;
  combine.stride_o_accum_h_q=kDimension;
  combine.tile_scheduler_metadata_ptr = schedule.metadata;
  combine.num_splits_ptr = schedule.splits;
  combine.num_sm_parts=a.partitions;
  combine.attn_sink=sink;
  combine.stream=stream;
  gfx9::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine);
  unpad_output<<<grid,256,0,stream>>>(reinterpret_cast<const uint16_t*>(output),
      static_cast<uint16_t*>(a.output),a.rows,a.heads);
  check(hipGetLastError());
}
}  // namespace

void run(const Arguments& a) { run_impl(a, nullptr); }

void prepare_schedule(int rows, int partitions, int swa_width, int extra_width,
                      const int32_t* swa_lengths, const int32_t* extra_lengths,
                      void* schedule, void* stream) {
  launch_schedule(rows, partitions, swa_width, extra_width, swa_lengths,
                  extra_lengths, schedule_view(schedule, rows, partitions),
                  static_cast<hipStream_t>(stream));
}

void run_with_schedule(const Arguments& a, const void* schedule) {
  if (!schedule)
    throw std::invalid_argument("FlashMLA shared schedule is missing");
  run_impl(a, schedule);
}

}  // namespace infini::ops::hygon_flash_mla
