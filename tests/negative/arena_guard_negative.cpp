// ===========================================================================
// arena_guard_negative.cpp — slot stores that MUST NOT COMPILE.
//
// The arena lays every slot at a COMPILE-TIME offset, so it can only size a
// tile whose extents are known at compile time. A DynamicTile has none, and
// without the guard the failure surfaces deep inside make_tile_layout as a
// missing `num_elements` on a layout the caller never named -- true, but
// unreadable, and pointing at the wrong file.
//
// Asserting `!DynamicTile<2>::is_static` in an ordinary test would prove
// nothing: a guard that is defined, correct, and never read looks identical to
// a working one from inside a static_assert. So this instantiates the real
// carve on a tile it forbids, and ctest asserts the build fails WITH THE
// GUARD'S OWN DIAGNOSTIC -- a bare failure would let an unrelated breakage in
// this file masquerade as the guard working.
//
// Selected by -DARENA_NEG_CASE=<n>; exactly one case per target.
//   1  a DynamicTile handed to the arena carve
//   0  the control: the same carve on static tiles, which must COMPILE.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/SlotStore.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

#ifndef ARENA_NEG_CASE
#error "define ARENA_NEG_CASE"
#endif

namespace {

constexpr int kI = 4, kJ = 8;

using TileS = StaticTile<kI, kJ>;
using TileD = DynamicTile<2>;

// The sizing path alone fires the guard -- it is where the offsets are built,
// and it is callable on the host, so no kernel is needed to reach it.
template <typename... Tiles>
std::size_t size_store(const Tiles&... tiles) {
  return arena_slot_store_bytes<float, ES>(tiles...);
}

}  // namespace

int main() {
#if ARENA_NEG_CASE == 0
  // CONTROL: two static tiles. Every extent is known, so the offsets are
  // constants and this MUST build.
  return static_cast<int>(size_store(TileS{}, TileS{}) > 0);

#elif ARENA_NEG_CASE == 1
  // A dynamic tile carries its extents as runtime members, so slot 1's offset
  // could not be a compile-time constant.
  return static_cast<int>(
      size_store(TileS{}, TileD{Kokkos::Array<int, 2>{kI, kJ}}) > 0);

#else
#error "unknown ARENA_NEG_CASE"
#endif
}
