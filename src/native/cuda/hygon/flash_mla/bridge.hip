#include "bridge.hpp"

#include "params_hip.h"
#include "gfx9/decode/combine/combine_hip.h"
#include "gfx9/decode/get_decoding_sched_meta/get_decoding_sched_meta_hip.h"
#include "gfx93/decode/sparse_fp8/splitkv_mla_hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace infini::ops::hygon_flash_mla {
namespace {
constexpr int kDimension = 512;

void check(hipError_t status) {
  if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

template<int kHeads>
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

template<int kHeads>
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
  if (!storage || rows <= 0 || rows > kMaxScheduleRows || partitions <= 0 ||
      partitions > kMaxSchedulePartitions)
    throw std::invalid_argument(
        "Unsupported FlashMLA shared schedule geometry");
  const auto address =
      (reinterpret_cast<uintptr_t>(storage) + 31) & ~uintptr_t(31);
  const auto alignment = address - reinterpret_cast<uintptr_t>(storage);
  if (alignment + sizeof(DecodingSchedMeta) * partitions + sizeof(int) * (rows+1) >
      kScheduleStorageBytes)
    throw std::invalid_argument("FlashMLA shared schedule storage is too small");
  auto* metadata = reinterpret_cast<DecodingSchedMeta*>(address);
  auto* splits =
      reinterpret_cast<int*>(address + sizeof(DecodingSchedMeta) * partitions);
  return {metadata, splits};
}

__global__ void schedule_independent(const int* swa, const int* extra, int rows, int partitions,
    int swa_width, int extra_width, int max_blocks, DecodingSchedMeta* metadata, int* prefix) {
    __shared__ int counts[32];
    if (threadIdx.x < rows) {
        const int short_length = swa ? swa[threadIdx.x] : swa_width;
        const int extra_length = extra ? extra[threadIdx.x] : extra_width;
        const int short_blocks = (max(short_length, 1)+63)/64;
        counts[threadIdx.x] = short_blocks + (extra_length+63)/64;
    }
    __syncthreads();
    if (threadIdx.x <= rows) {
        int count = 0;
        for (int item=0; item<threadIdx.x; ++item) count += counts[item];
        prefix[threadIdx.x] = count;
    }
    for (int part = threadIdx.x; part < partitions; part += blockDim.x) {
        const int row = part/max_blocks, block=part%max_blocks;
        DecodingSchedMeta entry{};
        if (row < rows && block < counts[row]) {
            const int split = counts[row]>1;
            entry = {row,row,block,block+1,block,split,split,{0}};
        } else {
            entry.begin_req_idx = rows;
            entry.end_req_idx = rows-1;
        }
        metadata[part] = entry;
    }
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
  // The split scheduler uses five LDS arrays sized by the query count.
  // Wide prefill uses the reference's no-split partitioning instead.
  if (rows > 3072) {
    gfx9::decode::run_get_decoding_sched_meta_no_split_kernel(scheduler);
  } else {
    gfx9::decode::run_get_decoding_sched_meta_kernel(scheduler);
  }
}

bool enabled(const char* name) {
  const auto* value=std::getenv(name);
  return value && value[0]=='1' && value[1]=='\0';
}

int independent_blocks(int rows, int partitions, int swa_width, int extra_width) {
  const bool use_all = enabled("INFINIOPS_DSV4_MLA_ALL_SPLITS");
  const bool use_c4 = enabled("INFINIOPS_DSV4_MLA_INDEPENDENT_SPLITS") &&
      rows>1 && extra_width==512;
  if ((!use_all && !use_c4) || rows<=0 || rows>kMaxScheduleRows || swa_width<=0)
    return 0;
  const int max_blocks=(swa_width+63)/64+(extra_width+63)/64;
  return rows*max_blocks<=partitions && max_blocks+5<=partitions ? max_blocks : 0;
}

void launch_selected_schedule(int rows, int partitions, int swa_width, int extra_width,
    const int32_t* swa_lengths, const int32_t* extra_lengths, Schedule schedule, hipStream_t stream) {
  const auto blocks=independent_blocks(rows,partitions,swa_width,extra_width);
  if (blocks) {
    // Preserve the one-block split of the full-partition single-query schedule.
    hipLaunchKernelGGL(schedule_independent,dim3(1),dim3(128),0,stream,
        swa_lengths,extra_lengths,rows,rows*blocks,swa_width,extra_width,blocks,
        schedule.metadata,schedule.splits);
    check(hipGetLastError());
  } else {
    launch_schedule(rows,partitions,swa_width,extra_width,swa_lengths,extra_lengths,schedule,stream);
  }
}

template<int kHeads>
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
  pad_input<kHeads><<<grid,256,0,stream>>>(static_cast<const uint16_t*>(a.query),a.sink,
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
  const auto independent = independent_blocks(
      a.rows, a.partitions, a.swa.index_width, a.extra.index_width);
  p.num_sm_parts=independent ? a.rows * independent : a.partitions;
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
    launch_selected_schedule(a.rows, a.partitions, a.swa.index_width,
        a.extra.index_width, a.swa.lengths, a.extra.lengths, schedule, stream);
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
  unpad_output<kHeads><<<grid,256,0,stream>>>(reinterpret_cast<const uint16_t*>(output),
      static_cast<uint16_t*>(a.output),a.rows,a.heads);
  check(hipGetLastError());
}
}  // namespace

bool use_compact_heads(const Arguments& a) {
  const auto* value=std::getenv("INFINIOPS_DSV4_FLASHMLA_H16");
  return a.heads <= 16 && a.rows <= 16 && value && value[0] == '1' && value[1] == '\0';
}

void run(const Arguments& a) {
  if (use_compact_heads(a)) run_impl<16>(a, nullptr);
  else run_impl<64>(a, nullptr);
}

void prepare_schedule(int rows, int partitions, int swa_width, int extra_width,
                      const int32_t* swa_lengths, const int32_t* extra_lengths,
                      void* schedule, void* stream) {
  launch_selected_schedule(rows, partitions, swa_width, extra_width, swa_lengths,
      extra_lengths, schedule_view(schedule, rows, partitions), static_cast<hipStream_t>(stream));
}

void run_with_schedule(const Arguments& a, const void* schedule) {
  if (!schedule)
    throw std::invalid_argument("FlashMLA shared schedule is missing");
  if (use_compact_heads(a)) run_impl<16>(a, schedule);
  else run_impl<64>(a, schedule);
}
} // namespace infini::ops::hygon_flash_mla
