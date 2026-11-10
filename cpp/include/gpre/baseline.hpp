// C++20 CPU baseline for gpre (FR-25b). M1 exposes the shared Philox generator; the
// multithreaded GBM pricer over path blocks (std::jthread pool) arrives in M6.
#ifndef GPRE_BASELINE_HPP_
#define GPRE_BASELINE_HPP_

#include <array>
#include <cstdint>

#include "philox4x32.cuh"

namespace gpre {

// Convenience wrapper: Philox4x32-10 of a counter under a 64-bit seed.
inline std::array<std::uint32_t, 4> philox_words(std::uint64_t seed,
                                                 const std::array<std::uint32_t, 4>& counter) {
  Philox4x32Counter ctr{{counter[0], counter[1], counter[2], counter[3]}};
  const Philox4x32Counter out = philox4x32(ctr, philox_key_from_seed(seed));
  return {out.v[0], out.v[1], out.v[2], out.v[3]};
}

}  // namespace gpre

#endif  // GPRE_BASELINE_HPP_
