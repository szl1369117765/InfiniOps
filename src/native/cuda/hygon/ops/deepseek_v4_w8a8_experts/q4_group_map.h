#pragma once

#include <cstddef>
#include <cstdint>

namespace infini::ops::deepseek_v4_w8a8_detail {

constexpr unsigned kGroupTokens = 4;
constexpr unsigned kGroupTopK = 6;

struct Q4ExpertGroup {
  int expert = -1;
  int slots[kGroupTokens] = {-1, -1, -1, -1};
};

// Duplicate IDs in the same token remain distinct slots. Group the n-th
// occurrence in each token and assign ownership to its first slot.
template <unsigned TopK>
__host__ __device__ inline Q4ExpertGroup FindQ4ExpertGroup(const int64_t* ids,
                                                           unsigned slot,
                                                           size_t experts) {
  const auto expert = ids[slot];
  Q4ExpertGroup group;
  if (expert < 0 || static_cast<size_t>(expert) >= experts) return group;

  unsigned occurrence = 0;
  const unsigned token = slot / TopK;
  for (unsigned prior = token * TopK; prior < slot; ++prior) {
    if (ids[prior] == expert) ++occurrence;
  }
  group.expert = static_cast<int>(expert);
  group.slots[token] = static_cast<int>(slot);
  for (unsigned row = 0; row < kGroupTokens; ++row) {
    if (row == token) continue;
    unsigned seen = 0;
    for (unsigned k = 0; k < TopK; ++k) {
      const unsigned other = row * TopK + k;
      if (ids[other] == expert && seen++ == occurrence) {
        if (other < slot) return Q4ExpertGroup{};
        group.slots[row] = static_cast<int>(other);
        break;
      }
    }
  }
  return group;
}

}  // namespace infini::ops::deepseek_v4_w8a8_detail
