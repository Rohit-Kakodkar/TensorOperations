// ===========================================================================
// slot_guard_negative.cpp — configurations that MUST NOT COMPILE.
//
// Asserting `!Impl::operand_stageable_v<...>` in an ordinary test proves the
// predicate is false. It does NOT prove the evaluator consults it. A guard that
// is defined, correct, and never read looks identical to a working one from
// inside a static_assert -- and that is precisely the failure this file exists
// to catch.
//
// So each case below builds the real evaluator on a configuration the guard
// forbids. CMake compiles each as its own target and ctest asserts the build
// FAILS (WILL_FAIL). A case that starts compiling means a guard came unwired.
//
// Selected by -DSLOT_NEG_CASE=<n>; exactly one case per target.
//   1  permuted slot in a contraction's A slot
//   2  permuted slot in a contraction's B slot
//   0  the control: the same shape WITHOUT the violation, which must COMPILE.
//      Without it, a typo that breaks the file for unrelated reasons would
//      register as every guard working perfectly.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

#ifndef SLOT_NEG_CASE
#error "define SLOT_NEG_CASE"
#endif

namespace {

constexpr int kI = 16, kK = 8, kL = 32;

using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

using TileIK = StaticTile<kI, kK>;
using TileKI = StaticTile<kK, kI>;
using TileKL = StaticTile<kK, kL>;
using TileIL = StaticTile<kI, kL>;
using TileLK = StaticTile<kL, kK>;

template <typename Tile>
using SlotView = decltype(Impl::alloc_scratch_tile<float, ES>(
    std::declval<team_t>(), std::declval<Tile>()));

}  // namespace

// Instantiating the Evaluator is what fires the guard: the static_asserts live
// in its class body, where no `requires` can see them.
template <typename Node, typename Tile>
void instantiate() {
  using Eval = Evaluator<TeamPolicyTag<ES>, Node, Tile>;
  (void)sizeof(Eval);
}

int main() {
#if SLOT_NEG_CASE == 0
  // CONTROL: slot labelled {i,k} in the A slot of C{i,l} = A{i,k} B{k,l}.
  // permA is the identity, so the guard permits it and this MUST build.
  auto slot = make_slot_node<0, 'i', 'k'>(SlotView<TileIK>{},
                                          Kokkos::Array<int, 2>{kI, kK});
  auto node = make_contraction_node<'i', 'l'>(
      slot, make_input_node(make_handle<'k', 'l'>(View2{})));
  instantiate<decltype(node), Tile<TileIK, TileKL, TileIL>>();

#elif SLOT_NEG_CASE == 1
  // A-slot violation. The slot is labelled {k,i} but the contraction's
  // canonical A order is freeA ++ contracted = {i,k}, so permA = (1,0). Staging
  // that would reorder the shared buffer in place.
  auto slot = make_slot_node<0, 'k', 'i'>(SlotView<TileKI>{},
                                          Kokkos::Array<int, 2>{kK, kI});
  auto node = make_contraction_node<'i', 'l'>(
      slot, make_input_node(make_handle<'k', 'l'>(View2{})));
  instantiate<decltype(node), Tile<TileKI, TileKL, TileIL>>();

#elif SLOT_NEG_CASE == 2
  // B-slot violation. Canonical B order is contracted ++ freeB = {k,l}; the
  // slot is labelled {l,k}, so permB = (1,0).
  auto slot = make_slot_node<0, 'l', 'k'>(SlotView<TileLK>{},
                                          Kokkos::Array<int, 2>{kL, kK});
  auto node = make_contraction_node<'i', 'l'>(
      make_input_node(make_handle<'i', 'k'>(View2{})), slot);
  instantiate<decltype(node), Tile<TileIK, TileLK, TileIL>>();

#else
#error "unknown SLOT_NEG_CASE"
#endif
  return 0;
}
