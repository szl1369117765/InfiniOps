#pragma once

#include <cstddef>
#include <cstdint>

namespace infini::ops::hygon_flash_mla {

struct Pages {
  const void* data;
  const int32_t* indices;
  const int32_t* lengths;
  int blocks;
  int page_tokens;
  int page_stride;
  int index_width;
};

struct Arguments {
  const void* query;
  const float* sink;
  void* output;
  void* workspace;
  size_t workspace_bytes;
  void* stream;
  int rows;
  int heads;
  int partitions;
  float scale;
  Pages swa;
  Pages extra;
};

inline constexpr int kMaxScheduleRows = 4;
inline constexpr int kMaxSchedulePartitions = 1024;
inline constexpr size_t kScheduleStorageBytes = 32 * kMaxSchedulePartitions + 64;

inline size_t scheduled_workspace_bytes(size_t rows, size_t partitions) {
  constexpr size_t heads = 64, dim = 512;
  const size_t queries = rows * heads;
  const size_t splits = (rows + partitions) * heads;
  // Each allocation in the bridge starts at a 256-byte boundary.
  return 6 * 256 + queries * dim * 4 + queries * 4 +
      splits * (dim + 1) * 4 + heads * 4;
}

inline size_t workspace_bytes(size_t rows, size_t partitions) {
  return scheduled_workspace_bytes(rows, partitions) + 2 * 256 +
      partitions * 32 + (rows + 1) * 4;
}

void run(const Arguments& args);
void prepare_schedule(int rows, int partitions, int swa_width, int extra_width,
                      const int32_t* swa_lengths, const int32_t* extra_lengths,
                      void* schedule, void* stream);
void run_with_schedule(const Arguments& args, const void* schedule);

}  // namespace infini::ops::hygon_flash_mla
