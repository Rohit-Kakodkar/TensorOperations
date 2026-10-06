#pragma once

namespace sfpp_min {

// GLL points per axis. The calibrated ground truth is NGLL = 5; other orders
// are built as separate targets with -DSFPP_MIN_NGLL=<n>.
#ifndef SFPP_MIN_NGLL
#define SFPP_MIN_NGLL 5
#endif
inline constexpr int NGLL = SFPP_MIN_NGLL;
static_assert(NGLL >= 2, "a GLL rule needs at least the two endpoints");
inline constexpr int DEG = NGLL - 1;

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
