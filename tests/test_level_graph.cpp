// ===========================================================================
// test_level_graph.cpp — LevelPlan: everything the level graph must KNOW
// before anything runs.
//
// Three separable claims, all compile-time:
//
//   1. SLOT ALGEBRA. A flat slot space -- stages [0, NumStages), then each
//      level's members in order -- with a two-level inverse. Checked over a
//      synthetic level list with MIXED ARITIES, because a multi-output combine
//      is the only thing that makes member index and slot index diverge; a
//      list of arity-1 members would pass every one of these with the two
//      conflated.
//
//   2. TILE DERIVATION. Every member's output tile is a type-level function of
//      its node. This is what lets add() take no tile and the graph carry no
//      runtime tile values. A wrong derivation here does NOT fail to compile --
//      it silently yields a WRONG TILE, and every downstream extent assert then
//      compares the wrong things.
//
//   3. LEVEL PREDICATES. Whether a proposed level is legal at all. The
//      corresponding ILLEGAL levels live in tests/negative/, because a
//      predicate that is correct and never consulted looks identical to a
//      working one from inside a static_assert.
//
// LABELS AND EXTENTS. Where a permutation is under test the extents are
// PAIRWISE DISTINCT (7/16/3/4), so no transposition can hide behind two equal
// axes, and the relabel is a 3-CYCLE rather than a swap. A swap is an
// involution: it equals its own inverse, so it cannot distinguish a gather
// permutation from a scatter one, and a direction bug survives it untouched.
// ===========================================================================
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

template <int32_t... M>
using Modes = std::integer_sequence<int32_t, M...>;

template <std::size_t I, typename ModesSeq, typename Tile>
using Slot = decltype(make_slot_node_seq<I, ModesSeq>(
    std::declval<SlotView<float, ES, Tile>>(),
    std::declval<Kokkos::Array<int, ModesSeq::size()>>()));

// --- 1. slot algebra -------------------------------------------------------
//
// Synthetic members: only output_arity is consulted, so a bare tag carrying
// NumOut is the whole of what this section needs. Level 1 mixes a 3-output
// member with a 1-output one, which is what forces member != slot.

template <int N>
struct MultiOut {
  static constexpr int NumOut = N;
};
struct SingleOut {};

using LA    = DeviceTuple<SingleOut, SingleOut, SingleOut>;
using LB    = DeviceTuple<MultiOut<3>, SingleOut>;
using LC    = DeviceTuple<MultiOut<2>>;
using Synth = DeviceTuple<LA, LB, LC>;
constexpr std::size_t kSynthStages = 5;

static_assert(Impl::lg_total_members_v<Synth> == 6);
static_assert(Impl::lg_num_slots_v<Synth, kSynthStages> == 14,
              "5 stages + 3 + 4 + 2");
static_assert(Impl::lg_level_slots_v<Synth, 0> == 3);
static_assert(Impl::lg_level_slots_v<Synth, 1> == 4, "the 3-output member");
static_assert(Impl::lg_level_slots_v<Synth, 2> == 2);

static_assert(Impl::lg_level_base_v<Synth, kSynthStages, 0> == 5,
              "level 0 starts after the stages, not at 0");
static_assert(Impl::lg_level_base_v<Synth, kSynthStages, 1> == 8);
static_assert(Impl::lg_level_base_v<Synth, kSynthStages, 2> == 12);

static_assert(Impl::lg_member_base_v<Synth, kSynthStages, 0, 0> == 5);
static_assert(Impl::lg_member_base_v<Synth, kSynthStages, 0, 2> == 7);
static_assert(Impl::lg_member_base_v<Synth, kSynthStages, 1, 0> == 8);
static_assert(Impl::lg_member_base_v<Synth, kSynthStages, 1, 1> == 11,
              "member 1 of level 1 starts 3 slots in, not 1");
static_assert(Impl::lg_member_base_v<Synth, kSynthStages, 2, 0> == 12);

static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 5> == 0);
static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 7> == 0);
static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 8> == 1);
static_assert(
    Impl::lg_slot_level_v<Synth, kSynthStages, 10> == 1,
    "a middle slot of a 3-output member is still that member's level");
static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 11> == 1);
static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 13> == 2);

static_assert(Impl::lg_slot_member_v<Synth, kSynthStages, 8> == 0);
static_assert(Impl::lg_slot_member_v<Synth, kSynthStages, 10> == 0);
static_assert(Impl::lg_slot_member_v<Synth, kSynthStages, 11> == 1);
static_assert(Impl::lg_slot_member_v<Synth, kSynthStages, 13> == 0);

static_assert(Impl::lg_is_stage_slot_v<kSynthStages, 4>);
static_assert(!Impl::lg_is_stage_slot_v<kSynthStages, 5>);
static_assert(Impl::lg_slot_level_v<Synth, kSynthStages, 0> == 3,
              "a stage slot belongs to no level, and says so");

template <std::size_t L, std::size_t M>
inline constexpr bool kRoundTrip =
    Impl::lg_slot_level_v<Synth, kSynthStages,
                          Impl::lg_member_base_v<Synth, kSynthStages, L, M>> ==
        L &&
    Impl::lg_slot_member_v<Synth, kSynthStages,
                           Impl::lg_member_base_v<Synth, kSynthStages, L, M>> ==
        M;
static_assert(kRoundTrip<0, 0> && kRoundTrip<0, 1> && kRoundTrip<0, 2>);
static_assert(kRoundTrip<1, 0> && kRoundTrip<1, 1>);
static_assert(kRoundTrip<2, 0>);

using NoLevels = DeviceTuple<>;
static_assert(Impl::lg_num_slots_v<NoLevels, 3> == 3);
static_assert(Impl::lg_total_members_v<NoLevels> == 0);

// --- 2. tile derivation ----------------------------------------------------

constexpr int kQ = 7, kA = 5, kE = 16, kB = 3, kC = 4;

using TileH = StaticTile<kQ, kA>;
using TileU = StaticTile<kE, kA, kB, kC>;
using TileP = StaticTile<kQ, kE, kB, kC>;

using NodeH = Slot<0, Modes<'q', 'a'>, TileH>;
using NodeU = Slot<1, Modes<'e', 'a', 'b', 'c'>, TileU>;
using NodeP = Slot<2, Modes<'q', 'e', 'b', 'c'>, TileP>;

using Cnx = decltype(make_contraction_node<'q', 'e', 'b', 'c'>(
    std::declval<NodeH>(), std::declval<NodeU>()));
static_assert(std::is_same_v<member_out_tile_t<Cnx>, TileP>,
              "freeA in A's order, then freeB in B's order");

struct Id4 {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float v) const {
    return v;
  }
};

using CmbId = decltype(make_combine_node<'q', 'e', 'b', 'c'>(
    std::declval<NodeP>(), std::declval<Id4>()));
static_assert(std::is_same_v<member_out_tile_t<CmbId>, TileP>,
              "an identity-labelled combine keeps its operand's tile");

using Cmb3 = decltype(make_combine_node<'e', 'b', 'q', 'c'>(
    std::declval<NodeP>(), std::declval<Id4>()));
static_assert(
    std::is_same_v<member_out_tile_t<Cmb3>, StaticTile<kE, kB, kQ, kC>>,
    "a 3-cycle relabel pins the DIRECTION of the label permutation; "
    "an involution cannot, because it equals its own inverse");

// --- 3. level predicates, positive side ------------------------------------
//
// The shape the SEM port actually needs: one staged u read by three
// contractions that differ only in which axis they contract, the differing
// axis supplied by relabelling the SAME staged H through as<>(). All three
// outputs are StaticTile<N,TE,N,N>, so SA and SB agree across the level while
// SK need not.

constexpr int kN = 5, kTE = 16;

using SemH = Slot<0, Modes<'q', 'a'>, StaticTile<kN, kN>>;
using SemU = Slot<1, Modes<'e', 'a', 'b', 'c'>, StaticTile<kTE, kN, kN, kN>>;
using SemQ = StaticTile<kN, kTE, kN, kN>;
constexpr std::size_t kSemStages = 2;

using Ga = decltype(make_contraction_node<'q', 'e', 'b', 'c'>(
    std::declval<SemH>(), std::declval<SemU>()));
using Gb = decltype(make_contraction_node<'q', 'e', 'a', 'c'>(
    std::declval<decltype(std::declval<SemH>().as<'q', 'b'>())>(),
    std::declval<SemU>()));
using Gc = decltype(make_contraction_node<'q', 'e', 'a', 'b'>(
    std::declval<decltype(std::declval<SemH>().as<'q', 'c'>())>(),
    std::declval<SemU>()));

static_assert(std::is_same_v<member_out_tile_t<Ga>, SemQ>);
static_assert(std::is_same_v<member_out_tile_t<Gb>, SemQ>);
static_assert(std::is_same_v<member_out_tile_t<Gc>, SemQ>);

static_assert(Impl::lg_member_sa_v<Ga> == kN);
static_assert(Impl::lg_member_sb_v<Ga> == kTE * kN * kN);
static_assert(Impl::lg_member_sa_v<Gb> == kN);
static_assert(Impl::lg_member_sb_v<Gb> == kTE * kN * kN);
static_assert(Impl::lg_member_sa_v<Gc> == kN);
static_assert(Impl::lg_member_sb_v<Gc> == kTE * kN * kN);

using GradLevel = DeviceTuple<Ga, Gb, Gc>;
using SemLevels = DeviceTuple<GradLevel>;

static_assert(Impl::lg_level_homogeneous_v<GradLevel>);
static_assert(Impl::lg_level_space_agrees_v<GradLevel>);
static_assert(Impl::lg_levels_reads_v<SemLevels, kSemStages>,
              "all three read stage slots only");

using SemPlan = LevelPlan<SemLevels, kSemStages>;
static_assert(SemPlan::num_levels == 1);
static_assert(SemPlan::num_members == 3);
static_assert(SemPlan::num_slots == 5, "2 stages + 3 gradients");

// A member's operand slots are all strictly below its own level's base -- the
// property the negative file's sibling case violates.
static_assert(Impl::lg_max_operand_slot_v<Ga> == 1);
static_assert(
    Impl::lg_max_operand_slot_v<Ga> <
    static_cast<int>(Impl::lg_level_base_v<SemLevels, kSemStages, 0>));

// --- 4. the staged-node read, in the DAG liveness fold ---------------------
//
// dag_note_node_reads is a CLOSED set of node kinds, and it had no StagedTag
// branch -- so it silently stopped being closed the moment a staged node
// appeared, and a slot a staged node reads looked dead at its definition.
//
// This lives here rather than in test_slot_liveness.cpp on purpose: that file
// is the CONTROL on the same three-line edit, and a control that changed
// alongside the code it controls is not one. Neither it nor test_dag_graph.cpp
// exercises a staged node, which is exactly why the hole survived.
//
//   node 0  a contraction              -> slot 0
//   node 1  a stage node reading slot 0 -> slot 1
//   node 2  a contraction              -> slot 2, the root
//
// With the branch, slot 0 is live to node 1 and slot 1 cannot share its pool.
// Without it, slot 0 dies at its own definition and all three slots collapse
// into one pool -- which is the silent miscompilation, not a lost optimization.

using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

template <int32_t A, int32_t B>
using In2 = decltype(make_input_node(make_handle<A, B>(std::declval<View2>())));

using Cnx0 = decltype(make_contraction_node<'i', 'l'>(
    std::declval<In2<'i', 'k'>>(), std::declval<In2<'k', 'l'>>()));
using Cnx2 = decltype(make_contraction_node<'x', 'z'>(
    std::declval<In2<'x', 'y'>>(), std::declval<In2<'y', 'z'>>()));

using Reads0 = Slot<0, Modes<'i', 'l'>, StaticTile<4, 6>>;
using Staged = decltype(make_stage_node(std::declval<Reads0>()));

using StagedNodes = DeviceTuple<Cnx0, Staged, Cnx2>;

static_assert(Impl::dag_num_slots<StagedNodes>() == 3);
static_assert(Impl::dag_operand_slot<Reads0>() == 0,
              "the staged node's operand names slot 0");

constexpr auto kStagedPlan = Impl::dag_pool_of_slot<StagedNodes, 2>();
static_assert(kStagedPlan[0] == 0);
static_assert(kStagedPlan[1] == 1,
              "slot 1 may not take slot 0's buffer: the staged node READS slot "
              "0 while writing slot 1");
static_assert(kStagedPlan[2] == 0, "slot 2 is free to reclaim slot 0");
static_assert(Impl::dag_pool_count<StagedNodes, 2>() == 2,
              "one pool here would mean the staged read went unrecorded");

}  // namespace

// The file's content is its static_asserts; these pin the same arithmetic at
// runtime so a green ctest run reports on it rather than on an empty binary.
TEST(LevelPlanTest, SlotSpaceStartsAfterTheStages) {
  EXPECT_EQ((Impl::lg_num_slots_v<Synth, kSynthStages>), 14u);
  EXPECT_EQ((Impl::lg_level_base_v<Synth, kSynthStages, 0>), 5u);
  EXPECT_EQ((Impl::lg_level_base_v<Synth, kSynthStages, 2>), 12u);
}

TEST(LevelPlanTest, AMultiOutputMemberAdvancesTheSlotCursorByItsArity) {
  EXPECT_EQ((Impl::lg_member_base_v<Synth, kSynthStages, 1, 0>), 8u);
  EXPECT_EQ((Impl::lg_member_base_v<Synth, kSynthStages, 1, 1>), 11u);
  EXPECT_EQ((Impl::lg_slot_member_v<Synth, kSynthStages, 10>), 0u);
  EXPECT_EQ((Impl::lg_slot_member_v<Synth, kSynthStages, 11>), 1u);
}

TEST(LevelPlanTest, TheSemGradientLevelIsOneLegalLevel) {
  EXPECT_EQ(SemPlan::num_slots, 5u);
  EXPECT_EQ((Impl::lg_member_sa_v<Ga>), kN);
  EXPECT_EQ((Impl::lg_member_sb_v<Ga>), kTE * kN * kN);
}
