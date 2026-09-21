#pragma once

#include <config.hpp>

#include <Kokkos_Core.hpp>

namespace sfpp_min {

inline constexpr int kPointsPerElement = NGLL * NGLL * NGLL;
inline constexpr int kWorkItemsPerTeam = kExecChunk * kPointsPerElement;
inline constexpr int kTeamSize         = 256;
inline constexpr int kWarpSize         = 32;

static_assert(kWorkItemsPerTeam == 500,
              "the ground truth was measured at 4 elements x 125 points");

struct WorkItem {
  int ielement;
  int iz;
  int iy;
  int ix;
};

KOKKOS_INLINE_FUNCTION WorkItem decompose(int i) {
  WorkItem w;
  w.ielement    = i % kExecChunk;
  const int zyx = i / kExecChunk;
  w.iz          = zyx / (NGLL * NGLL);
  w.iy          = (zyx / NGLL) % NGLL;
  w.ix          = zyx % NGLL;
  return w;
}

KOKKOS_INLINE_FUNCTION int num_teams(int nspec) {
  return (nspec + kExecChunk - 1) / kExecChunk;
}

}  // namespace sfpp_min
