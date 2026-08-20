// ===========================================================================
// level_guard_negative.cpp — level configurations that MUST NOT COMPILE.
//
// test_level_graph.cpp asserts the predicates return false on illegal levels.
// That proves the predicates are correct. It does NOT prove LevelPlan consults
// them -- a guard that is defined, right, and never read is indistinguishable
// from a working one when you only ever look at it from inside a static_assert.
//
// So each case here instantiates the real LevelPlan on a level it forbids.
// CMake builds each as its own target and ctest asserts the build fails WITH
// THAT GUARD'S OWN DIAGNOSTIC: a bare failure would let an unrelated breakage
// in this file masquerade as every guard working.
//
// Selected by -DLEVEL_NEG_CASE=<n>; exactly one case per target.
//   1  a level mixing a contraction member and a combine member
//   2  contraction members whose SA/SB disagree
//   3  combine members whose output tile layouts disagree
//   4  a member reading a slot its OWN level produces
//   5  STAGED members whose iteration spaces differ -- the guard that makes
//      fusing a stage level's copies into one range sound
//   0  the control: a legal two-member level, which must COMPILE.
// ===========================================================================
#include <TensorOperations/LabelTiles.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

#ifndef LEVEL_NEG_CASE
#error "define LEVEL_NEG_CASE"
#endif

namespace {

constexpr int         kN = 5, kTE = 16, kOdd = 7;
constexpr std::size_t kStages = 2;

template <int32_t... M>
using Modes = std::integer_sequence<int32_t, M...>;

template <std::size_t I, typename ModesSeq, typename Tile>
using Slot = decltype(make_slot_node_seq<I, ModesSeq>(
    std::declval<SlotView<float, ES, Tile>>(),
    std::declval<Kokkos::Array<int, ModesSeq::size()>>()));

using NodeH = Slot<0, Modes<'q', 'a'>, StaticTile<kN, kN>>;
using NodeU = Slot<1, Modes<'e', 'a', 'b', 'c'>, StaticTile<kTE, kN, kN, kN>>;

// Same shape as NodeH but a differing free extent, so a contraction against it
// produces a different SA.
using NodeR = Slot<0, Modes<'r', 'a'>, StaticTile<kOdd, kN>>;

// Slot 2 is level 0's FIRST OUTPUT -- naming it from inside level 0 is the
// sibling read.
using NodeSib = Slot<kStages, Modes<'q', 'a'>, StaticTile<kN, kN>>;

// Two staged members over tensors of different rank, so their output tiles --
// and therefore their iteration spaces -- cannot agree. A stage level runs ONE
// TeamVectorRange over the shared space, so this must not compile.
using V2s = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using V4s = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using StageMap =
    LabelTiles<LabelTile<'e', kTE>, LabelWhole<'q', kN>, LabelWhole<'a', kN>,
               LabelWhole<'b', kN>, LabelWhole<'c', kN>>;
template <typename Raw>
using Resolved = typename Impl::lg_resolve_member<StageMap, Raw>::type;
using StageH   = Resolved<decltype(make_stage_node(
    make_input_node(make_handle<'q', 'a'>(std::declval<V2s>()))))>;
using StageU   = Resolved<decltype(make_stage_node(
    make_input_node(make_handle<'e', 'a', 'b', 'c'>(std::declval<V4s>()))))>;

using NodeP4 = Slot<0, Modes<'q', 'e', 'b', 'c'>, StaticTile<kN, kTE, kN, kN>>;
using NodeP2 = Slot<1, Modes<'q', 'e'>, StaticTile<kN, kTE>>;

using Ga   = decltype(make_contraction_node<'q', 'e', 'b', 'c'>(
    std::declval<NodeH>(), std::declval<NodeU>()));
using Gb   = decltype(make_contraction_node<'q', 'e', 'a', 'c'>(
    std::declval<decltype(std::declval<NodeH>().as<'q', 'b'>())>(),
    std::declval<NodeU>()));
using Gr   = decltype(make_contraction_node<'r', 'e', 'b', 'c'>(
    std::declval<NodeR>(), std::declval<NodeU>()));
using Gsib = decltype(make_contraction_node<'q', 'e', 'b', 'c'>(
    std::declval<NodeSib>(), std::declval<NodeU>()));

struct Id4 {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float v) const {
    return v;
  }
};
struct Id2 {
  KOKKOS_FUNCTION float operator()(int, int, float v) const { return v; }
};

using Cmb4 = decltype(make_combine_node<'q', 'e', 'b', 'c'>(
    std::declval<NodeP4>(), std::declval<Id4>()));
using Cmb2 = decltype(make_combine_node<'q', 'e'>(std::declval<NodeP2>(),
                                                  std::declval<Id2>()));

#if LEVEL_NEG_CASE == 0
using Levels = DeviceTuple<DeviceTuple<Ga, Gb>>;
#elif LEVEL_NEG_CASE == 1
using Levels = DeviceTuple<DeviceTuple<Ga, Cmb4>>;
#elif LEVEL_NEG_CASE == 2
using Levels = DeviceTuple<DeviceTuple<Ga, Gr>>;
#elif LEVEL_NEG_CASE == 3
using Levels = DeviceTuple<DeviceTuple<Cmb4, Cmb2>>;
#elif LEVEL_NEG_CASE == 4
using Levels = DeviceTuple<DeviceTuple<Ga, Gsib>>;
#elif LEVEL_NEG_CASE == 5
using Levels = DeviceTuple<DeviceTuple<StageH, StageU>>;
#else
#error "LEVEL_NEG_CASE must be 0..5"
#endif

using Plan = LevelPlan<Levels, kStages>;
static_assert(Plan::num_levels == 1);

}  // namespace

int main() { return 0; }
