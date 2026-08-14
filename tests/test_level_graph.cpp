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
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

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

// ===========================================================================
// The runtime half: a graph GENERATES the kernel LevelPlan only describes, and
// its output equals a hand-driven reference. Grid is the last stage, tiled over
// the batch mode 'e'; the stages materialize both operands once per team and
// the contraction level reads them.
//
//   stage H[q,a]          (batch-independent, one tile every team re-reads)
//   stage U[e,a,b,c]      (the grid: E/TE teams, one 'e'-tile each)
//   level: Gd[q,e,b,c] = sum_a Hd[q,d] U[e,a,b,c]   for d in {a,b,c}, the
//          differing axis relabelled onto the SAME staged H.
// ===========================================================================
namespace rt {

constexpr int rN = 5, rTE = 2, rE = 8;

using ViewH  = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewU  = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using ViewC  = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using TileHr = StaticTile<rN, rN>;
using TileUr = StaticTile<rTE, rN, rN, rN>;

float hval(int q, int a) { return 0.1f + 0.3f * q - 0.2f * a + 0.05f * q * a; }
float uval(int e, int a, int b, int c) {
  return 0.2f + 0.11f * e - 0.07f * a + 0.13f * b - 0.03f * c +
         0.01f * (e + a) * (b + 1);
}

TEST(LevelGraphRuntime, SingleContractionMemberEqualsReference) {
  ViewH Hd("H", rN, rN);
  ViewU Ud("U", rE, rN, rN, rN);
  ViewC Cd("C", rN, rE, rN, rN);

  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < rN; ++q)
    for (int a = 0; a < rN; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < rE; ++e)
    for (int a = 0; a < rN; ++a)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileHr{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileUr{});
  auto ga       = make_contraction_node<'q', 'e', 'b', 'c'>(h, u);
  auto [g3, ca] = g2.add(ga);

  g3.outputs(ca).execute(TeamPolicyTag2<ES>{}, Cd);
  Kokkos::fence();

  auto Ch = Kokkos::create_mirror_view(Cd);
  Kokkos::deep_copy(Ch, Cd);

  double max_err = 0.0;
  for (int q = 0; q < rN; ++q)
    for (int e = 0; e < rE; ++e)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) {
          double ref = 0.0;
          for (int a = 0; a < rN; ++a) ref += hval(q, a) * uval(e, a, b, c);
          max_err = std::max(max_err, std::abs(ref - Ch(q, e, b, c)));
        }
  EXPECT_LT(max_err, 1e-4) << "graph contraction != reference";
}

TEST(LevelGraphRuntime, ThreeMemberFusedGradientLevelEqualsReference) {
  ViewH Hd("H", rN, rN);
  ViewU Ud("U", rE, rN, rN, rN);
  ViewC Ax("Ax", rN, rE, rN, rN);
  ViewC Bx("Bx", rN, rE, rN, rN);
  ViewC Cx("Cx", rN, rE, rN, rN);

  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < rN; ++q)
    for (int a = 0; a < rN; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < rE; ++e)
    for (int a = 0; a < rN; ++a)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileHr{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileUr{});

  auto gax = make_contraction_node<'q', 'e', 'b', 'c'>(h, u);
  auto gbx =
      make_contraction_node<'q', 'e', 'a', 'c'>(h.template as<'q', 'b'>(), u);
  auto gcx =
      make_contraction_node<'q', 'e', 'a', 'b'>(h.template as<'q', 'c'>(), u);
  auto [g3, ha, hb, hc] = g2.add(gax, gbx, gcx);

  g3.outputs(ha, hb, hc).execute(TeamPolicyTag2<ES>{}, Ax, Bx, Cx);
  Kokkos::fence();

  auto Ah  = Kokkos::create_mirror_view(Ax);
  auto Bh  = Kokkos::create_mirror_view(Bx);
  auto Chh = Kokkos::create_mirror_view(Cx);
  Kokkos::deep_copy(Ah, Ax);
  Kokkos::deep_copy(Bh, Bx);
  Kokkos::deep_copy(Chh, Cx);

  // A modes (q,e,b,c): sum_a H[q,a] U[e,a,b,c]
  // B modes (q,e,a,c): sum_b H[q,b] U[e,a,b,c]
  // C modes (q,e,a,b): sum_c H[q,c] U[e,a,b,c]
  double ea = 0.0, eb = 0.0, ec = 0.0;
  for (int q = 0; q < rN; ++q)
    for (int e = 0; e < rE; ++e)
      for (int m = 0; m < rN; ++m)
        for (int n = 0; n < rN; ++n) {
          double ra = 0.0, rb = 0.0, rc = 0.0;
          for (int k = 0; k < rN; ++k) {
            ra += hval(q, k) * uval(e, k, m, n);
            rb += hval(q, k) * uval(e, m, k, n);
            rc += hval(q, k) * uval(e, m, n, k);
          }
          ea = std::max(ea, std::abs(ra - Ah(q, e, m, n)));
          eb = std::max(eb, std::abs(rb - Bh(q, e, m, n)));
          ec = std::max(ec, std::abs(rc - Chh(q, e, m, n)));
        }
  EXPECT_LT(ea, 1e-4) << "member a";
  EXPECT_LT(eb, 1e-4) << "member b";
  EXPECT_LT(ec, 1e-4) << "member c";
}

struct ScaleG {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float g) const {
    return 2.0f * g + 0.5f;
  }
};

struct DupG {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int, int, int, int,
                                                     float g) const {
    return {2.0f * g, 3.0f * g};
  }
};

// A combine LEVEL reading the slot a prior contraction level wrote: the first
// multi-level graph and the first exercise of the combine driver. The pointwise
// op is coordinate-free, so the tile-local coord the fused range passes is
// irrelevant and the reference is exactly 2*grad + 0.5.
TEST(LevelGraphRuntime, CombineLevelReadingContractionEqualsReference) {
  ViewH Hd("H", rN, rN);
  ViewU Ud("U", rE, rN, rN, rN);
  ViewC Px("P", rN, rE, rN, rN);

  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < rN; ++q)
    for (int a = 0; a < rN; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < rE; ++e)
    for (int a = 0; a < rN; ++a)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileHr{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileUr{});

  auto ga       = make_contraction_node<'q', 'e', 'b', 'c'>(h, u);
  auto [g3, ca] = g2.add(ga);
  auto cmb      = make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleG{});
  auto [g4, pa] = g3.add(cmb);

  g4.outputs(pa).execute(TeamPolicyTag2<ES>{}, Px);
  Kokkos::fence();

  auto Ph = Kokkos::create_mirror_view(Px);
  Kokkos::deep_copy(Ph, Px);

  double max_err = 0.0;
  for (int q = 0; q < rN; ++q)
    for (int e = 0; e < rE; ++e)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) {
          double g = 0.0;
          for (int a = 0; a < rN; ++a) g += hval(q, a) * uval(e, a, b, c);
          double ref = 2.0 * g + 0.5;
          max_err    = std::max(max_err, std::abs(ref - Ph(q, e, b, c)));
        }
  EXPECT_LT(max_err, 1e-4) << "combine-over-contraction != reference";
}

// One combine member emitting TWO outputs in a single pass. Pins the slot
// fan-out (member != slot: o0 and o1 are two consecutive level slots off one
// member) and the slot-indexed carve that gives each its own buffer.
TEST(LevelGraphRuntime, MultiOutputCombineEqualsReference) {
  ViewH Hd("H", rN, rN);
  ViewU Ud("U", rE, rN, rN, rN);
  ViewC O0("O0", rN, rE, rN, rN);
  ViewC O1("O1", rN, rE, rN, rN);

  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < rN; ++q)
    for (int a = 0; a < rN; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < rE; ++e)
    for (int a = 0; a < rN; ++a)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileHr{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileUr{});

  auto ga           = make_contraction_node<'q', 'e', 'b', 'c'>(h, u);
  auto [g3, ca]     = g2.add(ga);
  auto cmb          = make_combine_node<'q', 'e', 'b', 'c'>(ca, DupG{});
  auto [g4, o0, o1] = g3.add(cmb);

  g4.outputs(o0, o1).execute(TeamPolicyTag2<ES>{}, O0, O1);
  Kokkos::fence();

  auto H0 = Kokkos::create_mirror_view(O0);
  auto H1 = Kokkos::create_mirror_view(O1);
  Kokkos::deep_copy(H0, O0);
  Kokkos::deep_copy(H1, O1);

  double e0 = 0.0, e1 = 0.0;
  for (int q = 0; q < rN; ++q)
    for (int e = 0; e < rE; ++e)
      for (int b = 0; b < rN; ++b)
        for (int c = 0; c < rN; ++c) {
          double g = 0.0;
          for (int a = 0; a < rN; ++a) g += hval(q, a) * uval(e, a, b, c);
          e0 = std::max(e0, std::abs(2.0 * g - H0(q, e, b, c)));
          e1 = std::max(e1, std::abs(3.0 * g - H1(q, e, b, c)));
        }
  EXPECT_LT(e0, 1e-4) << "multi-output combine o0";
  EXPECT_LT(e1, 1e-4) << "multi-output combine o1";
}

}  // namespace rt

// ===========================================================================
// Declared-order contraction output. A contraction STORES canonically
// (freeA ++ freeB) but is DECLARED, and therefore read, in whatever order its
// labels say -- the permutation rides in the slot's view type, never in the
// stored bytes.
//
// Every extent here is pairwise distinct and every permutation is a 3-CYCLE,
// both deliberately. An involution cannot test a permutation's direction (it
// passes against its own inverse), and equal extents hide a direction error
// outright: the extents-only wall in Team2.hpp cannot see axis order, so with
// rt's (5,2,5,5) shapes a fully inverted perm still type-checks.
// ===========================================================================
namespace declared {

constexpr int dQ = 7, dA = 3, dB = 4, dC = 5, dTE = 2, dE = 8;

using ViewH      = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewU      = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using ViewO      = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using ViewW      = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using TileH      = StaticTile<dQ, dA>;
using TileU      = StaticTile<dTE, dA, dB, dC>;
constexpr int dP = 6;
using TileW      = StaticTile<dP, dQ>;

float hval(int q, int a) {
  return 0.3f + 0.17f * q - 0.23f * a + 0.04f * q * a;
}
float wval(int p, int q) { return 0.6f - 0.11f * p + 0.19f * q; }
float uval(int e, int a, int b, int c) {
  return 0.7f - 0.13f * e + 0.09f * a - 0.05f * b + 0.21f * c +
         0.02f * (e + b) * (c + 1);
}

struct ScaleD {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float g) const {
    return 2.0f * g + 0.5f;
  }
};

void fill(ViewH Hd, ViewU Ud) {
  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < dQ; ++q)
    for (int a = 0; a < dA; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < dE; ++e)
    for (int a = 0; a < dA; ++a)
      for (int b = 0; b < dB; ++b)
        for (int c = 0; c < dC; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);
}

double gref(int q, int e, int b, int c) {
  double g = 0.0;
  for (int a = 0; a < dA; ++a) g += hval(q, a) * uval(e, a, b, c);
  return g;
}

// The contraction's canonical order is <q,e,b,c>; it is DECLARED <e,b,q,c>,
// a 3-cycle on the first three axes, and rooted directly. Pins that the store
// path reconciles a non-identity permC against canonical scratch.
TEST(LevelGraphDeclaredOrder, NonCanonicalContractionRootEqualsReference) {
  ViewH Hd("H", dQ, dA);
  ViewU Ud("U", dE, dA, dB, dC);
  ViewO Od("O", dE, dB, dQ, dC);
  fill(Hd, Ud);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileH{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileU{});

  auto ga       = make_contraction_node<'e', 'b', 'q', 'c'>(h, u);
  auto [g3, ca] = g2.add(ga);

  g3.outputs(ca).execute(TeamPolicyTag2<ES>{}, Od);
  Kokkos::fence();

  auto Oh = Kokkos::create_mirror_view(Od);
  Kokkos::deep_copy(Oh, Od);

  double max_err = 0.0;
  for (int e = 0; e < dE; ++e)
    for (int b = 0; b < dB; ++b)
      for (int q = 0; q < dQ; ++q)
        for (int c = 0; c < dC; ++c)
          max_err =
              std::max(max_err, std::abs(gref(q, e, b, c) - Oh(e, b, q, c)));
  EXPECT_LT(max_err, 1e-4) << "declared-order contraction != reference";
}

// The consumer's order differs from BOTH the storage order and the producer's
// declared order, so the read permutation is the composition of the two. This
// is the SEM shape: one gradient slot read by combines in different frames.
TEST(LevelGraphDeclaredOrder, CombineReadsDeclaredSlotInItsOwnOrder) {
  ViewH Hd("H", dQ, dA);
  ViewU Ud("U", dE, dA, dB, dC);
  ViewO Od("O", dQ, dC, dE, dB);
  fill(Hd, Ud);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileH{});
  auto [g2, u] =
      g1.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileU{});

  auto ga       = make_contraction_node<'e', 'b', 'q', 'c'>(h, u);
  auto [g3, ca] = g2.add(ga);
  auto cmb      = make_combine_node<'q', 'c', 'e', 'b'>(ca, ScaleD{});
  auto [g4, pa] = g3.add(cmb);

  g4.outputs(pa).execute(TeamPolicyTag2<ES>{}, Od);
  Kokkos::fence();

  auto Oh = Kokkos::create_mirror_view(Od);
  Kokkos::deep_copy(Oh, Od);

  double max_err = 0.0;
  for (int q = 0; q < dQ; ++q)
    for (int c = 0; c < dC; ++c)
      for (int e = 0; e < dE; ++e)
        for (int b = 0; b < dB; ++b) {
          const double ref = 2.0 * gref(q, e, b, c) + 0.5;
          max_err          = std::max(max_err, std::abs(ref - Oh(q, c, e, b)));
        }
  EXPECT_LT(max_err, 1e-4) << "combine over declared-order slot != reference";
}

// A declared-order slot consumed by ANOTHER CONTRACTION, which is the only
// consumer that reads a slot handle's shape() as well as its labels. Level 2
// reads level 1's slot through a non-identity presentation permutation, and
// declares its own output non-canonically in turn.
TEST(LevelGraphDeclaredOrder, ContractionReadsDeclaredSlotOperand) {
  ViewH Hd("H", dQ, dA);
  ViewW Wd("W", dP, dQ);
  ViewU Ud("U", dE, dA, dB, dC);
  ViewO Od("O", dE, dB, dP, dC);
  fill(Hd, Ud);

  auto Wh = Kokkos::create_mirror_view(Wd);
  for (int p = 0; p < dP; ++p)
    for (int q = 0; q < dQ; ++q) Wh(p, q) = wval(p, q);
  Kokkos::deep_copy(Wd, Wh);

  auto g0      = make_level_graph<float, ES>();
  auto [g1, h] = g0.stage(make_input_node(make_handle<'q', 'a'>(Hd)), TileH{});
  auto [g2, w] = g1.stage(make_input_node(make_handle<'p', 'q'>(Wd)), TileW{});
  auto [g3, u] =
      g2.stage(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud)), TileU{});

  auto ga       = make_contraction_node<'e', 'b', 'q', 'c'>(h, u);
  auto [g4, ca] = g3.add(ga);
  auto gb       = make_contraction_node<'e', 'b', 'p', 'c'>(w, ca);
  auto [g5, cb] = g4.add(gb);

  g5.outputs(cb).execute(TeamPolicyTag2<ES>{}, Od);
  Kokkos::fence();

  auto Oh = Kokkos::create_mirror_view(Od);
  Kokkos::deep_copy(Oh, Od);

  double max_err = 0.0;
  for (int e = 0; e < dE; ++e)
    for (int b = 0; b < dB; ++b)
      for (int p = 0; p < dP; ++p)
        for (int c = 0; c < dC; ++c) {
          double ref = 0.0;
          for (int q = 0; q < dQ; ++q) ref += wval(p, q) * gref(q, e, b, c);
          max_err = std::max(max_err, std::abs(ref - Oh(e, b, p, c)));
        }
  EXPECT_LT(max_err, 1e-4)
      << "contraction over declared-order slot != reference";
}

}  // namespace declared

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
