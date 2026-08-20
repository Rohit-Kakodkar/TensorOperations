#pragma once

namespace sfpp_min {

inline constexpr int NGLL = 5;
inline constexpr int DEG  = NGLL - 1;

inline constexpr int kStorageChunk = 32;
inline constexpr int kExecChunk    = 4;

static_assert(kStorageChunk != kExecChunk,
              "storage tiling (32) and the team's element count (4) are "
              "deliberately different in SPECFEM++; unifying them deletes the "
              "mismatch this library exists to study");

static_assert(kStorageChunk % kExecChunk == 0,
              "a team's elements must not straddle a storage tile");

using real_t = float;

}  // namespace sfpp_min
