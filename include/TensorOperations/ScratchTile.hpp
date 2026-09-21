#pragma once
// Team-scratch tile allocation, and the tile arithmetic that sizes it.
//
// Extracted from Evaluator/Team.hpp when the TeamPolicyTag v1 path was removed.
// Nothing here is tied to an execution mode: these are functions of a tile spec
// and a team handle, used by the slot store (SlotStore.hpp), the level graph's
// grid sizing (LevelGraph.hpp) and the evaluators (Evaluator/Team2.hpp).
#include <TensorOperations/TiledLayout.hpp>
#include <TensorOperations/Tiling.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {
namespace Impl {

// Ceil-division. The template form binds the divisor at compile time so the
// compiler lowers it to a multiply-shift instead of a runtime integer division
// (expensive on GPU); the runtime form is used only when the divisor genuinely
// isn't known (dynamic tiles).
template <int Divisor>
KOKKOS_FORCEINLINE_FUNCTION int ceil_div(int n) noexcept {
  static_assert(Divisor > 0, "ceil_div divisor must be positive");
  return (n + Divisor - 1) / Divisor;
}
KOKKOS_FORCEINLINE_FUNCTION int ceil_div(int n, int divisor) noexcept {
  return (n + divisor - 1) / divisor;
}

// Number of tiles that cover extent `ext` along dim `d` of `tile`. For a static
// tile the divisor is compile-time, so we pick the matching dim's constexpr
// extent (keeping the cheap multiply-shift); dynamic tiles use a runtime
// divide.
template <typename Tile>
KOKKOS_FUNCTION int tile_count_along(const Tile& tile, int d,
                                     int ext) noexcept {
  if constexpr (Tile::is_static) {
    int n = 0;
    [&]<int... Ds>(std::integer_sequence<int, Ds...>) {
      ((d == Ds ? (n = ceil_div<Tile::extent(Ds)>(ext)) : 0), ...);
    }(std::make_integer_sequence<int, Tile::rank>{});
    return n;
  } else {
    return ceil_div(ext, tile.extent(d));
  }
}

// --- team-scratch tile allocation -------------------------------------------
//
// A staged tile is a LayoutRight scratch view sized by a tile spec.
// alloc_scratch_tile carves one out of team scratch (each call advances the
// team's scratch cursor); alloc_scratch_tile_at places one on a pointer the
// caller already owns.
template <typename ValueType, typename ES>
using scratch_backing_t =
    Kokkos::View<ValueType*, typename ES::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename ValueType, typename ES, typename Team, typename Tile>
KOKKOS_FORCEINLINE_FUNCTION auto alloc_scratch_tile(const Team& team,
                                                    const Tile& tile) {
  const auto layout   = make_tile_layout(tile, LayoutRight{});
  using tile_layout_t = std::decay_t<decltype(layout)>;
  scratch_backing_t<ValueType, ES> backing(
      team.team_scratch(0), static_cast<std::size_t>(layout.size()));
  return ScratchView<ValueType, ES, tile_layout_t>{backing, layout};
}

// The same tile, over a buffer the CALLER chose instead of a fresh bump of the
// team cursor. Returns the identical type to alloc_scratch_tile, which is what
// lets a pooled slot store (SlotStore.hpp) place two slots with disjoint live
// ranges on one pointer without any node, evaluator or store type changing.
//
// The pointer must have room for make_tile_layout(tile).size() elements and the
// alignment of a team_scratch allocation -- both guaranteed when it comes from
// a pool carved by alloc_scratch_tile's own backing type.
template <typename ValueType, typename ES, typename Tile>
KOKKOS_FORCEINLINE_FUNCTION auto alloc_scratch_tile_at(ValueType*  ptr,
                                                       const Tile& tile) {
  const auto layout   = make_tile_layout(tile, LayoutRight{});
  using tile_layout_t = std::decay_t<decltype(layout)>;
  scratch_backing_t<ValueType, ES> backing(
      ptr, static_cast<std::size_t>(layout.size()));
  return ScratchView<ValueType, ES, tile_layout_t>{backing, layout};
}

// The matching per-tile contribution to a launch's scratch_size_per_team.
//
// KOKKOS_FUNCTION because a device-side arena may advance by exactly this.
// Without it nvcc emits only warning #20011 and the kernel traps at runtime as
// "unspecified launch failure" while the CPU build stays green.
template <typename ValueType, typename ES, typename Tile>
KOKKOS_FUNCTION std::size_t scratch_tile_bytes(const Tile& tile) {
  return scratch_backing_t<ValueType, ES>::shmem_size(
      static_cast<std::size_t>(make_tile_layout(tile, LayoutRight{}).size()));
}

}  // namespace Impl
}  // namespace TensorOperations
