// ===========================================================================
// test_level_liveness.cpp — pooling the LEVEL graph's slot store by live range.
//
// A LevelGraph used to give every slot its own buffer for the whole kernel.
// Most die long before that, so liveness puts the ones that cannot be alive at
// once on a single buffer.
//
// THE FAILURE MODE THIS FILE EXISTS FOR IS SILENT. An over-eager plan does not
// crash and does not fault -- it points a live slot at a buffer some other
// level is about to overwrite, and the kernel returns plausible wrong numbers.
// So the plan is checked in two independent ways:
//
//   THE PLAN ITSELF, at compile time. lg_pool_of_slot is constexpr, so the
//   expected assignment for a hand-traceable graph is a static_assert -- no
//   kernel involved, and a regression fails the BUILD rather than a tolerance.
//
//   THE NUMBERS, at run time, against a host reference computed independently.
//   The chain below is built so reuse actually happens; if the ranges were
//   computed wrongly this is exactly the graph that would corrupt.
//
// Both matter. The static_asserts would pass a plan that is right about
// liveness and wrong about where it points, and the numeric test alone would
// pass a plan that simply never reuses anything.
//
// WHAT THIS FILE COVERS THAT test_slot_liveness.cpp CANNOT: the timeline is
// LEVELS, not members. A DagGraph node reads its operands and writes its output
// in one evaluation; a level does not, because barriers exist only at level
// ends and every member of a level runs interleaved inside one TeamVectorRange.
// So every slot a level touches is live for the whole level, and the two
// SameLevel tests below pin exactly that.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

// i is multi-tiled (32 over a 16 tile) for the same reason as test_dag_graph:
// a single work item would let a wrong index still produce the right answer.
constexpr int kI = 32, kK = 8, kL = 32, kM = 32;
constexpr int kTI = 16;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

using TA = StaticTile<kTI, kK>;
using TB = StaticTile<kK, kL>;
using TE = StaticTile<kL, kM>;
using TC = StaticTile<kTI, kL>;  // == StaticTile<kTI, kM>

// One extent per label. i is multi-tiled (32 over a 16 tile); every other axis
// is whole, so only i is gridded.
using Map = LabelTiles<LabelTile<'i', kTI>, LabelWhole<'k', kK>,
                       LabelWhole<'l', kL>, LabelWhole<'m', kM>>;

float fill_val(int r, int c, int salt) {
  return 0.5f + 0.25f * r - 0.125f * c +
         0.0625f * ((r * 7 + c * 3 + salt) % 11);
}

View2 make_view(const char* name, int r, int c, int salt) {
  ViewH h(name, r, c);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) h(i, j) = fill_val(i, j, salt);
  View2 d(name, r, c);
  Kokkos::deep_copy(d, h);
  return d;
}

// --- the chain: C2 = ((A B) E) F, one member per level ---------------------
//
// A IS STAGED LAST ON PURPOSE. lg_execute takes the LAST STAGE as the grid
// node, and every mode the outputs tile over must come from it -- i is
// multi-tiled here, so a graph that staged something without an i (F, say)
// would leave i ungridded and every team would compute the first tile.
//
// Four stages then three levels, so the slot order is
//   0 E, 1 F, 2 B, 3 A, 4 C0, 5 C1, 6 C2.
auto level_chain(View2 a, View2 b, View2 e, View2 f) {
  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, E] = g0.stage(make_input_node(make_handle<'l', 'm'>(e)));
  auto [g2, F] = g1.stage(make_input_node(make_handle<'l', 'm'>(f)));
  auto [g3, B] = g2.stage(make_input_node(make_handle<'k', 'l'>(b)));
  auto [g4, A] = g3.stage(make_input_node(make_handle<'i', 'k'>(a)));

  auto [g5, c0] = g4.add(make_contraction_node<'i', 'l'>(A, B));
  auto [g6, c1] =
      g5.add(make_contraction_node<'i', 'm'>(c0.template as<'i', 'l'>(), E));
  auto [g7, c2] =
      g6.add(make_contraction_node<'i', 'm'>(c1.template as<'i', 'l'>(), F));
  return std::make_tuple(g7, c2);
}

using ChainGraph  = std::decay_t<decltype(std::get<0>(
    level_chain(std::declval<View2>(), std::declval<View2>(),
                std::declval<View2>(), std::declval<View2>())))>;
using ChainLevels = ChainGraph::levels_type;

// Hand-traced. Time 0 is the stages, level L is time L+1, and the root store
// runs at time 4. Ranges are CLOSED at both ends.
//
//   slot 0  E   staged, last read by level 1  -> [0,2]
//   slot 1  F   staged, last read by level 2  -> [0,3]
//   slot 2  B   staged, last read by level 0  -> [0,1]
//   slot 3  A   staged, last read by level 0  -> [0,1]
//   slot 4  C0  written level 0, read level 1 -> [1,2]
//   slot 5  C1  written level 1, read level 2 -> [2,3]
//   slot 6  C2  written level 2, root         -> [3,4]
//
// A and B die before C1 is written, so C1 reclaims B's buffer and C2 reclaims
// E's. F is live across every death and can never be touched.
constexpr auto kChainPlan = Impl::lg_pool_of_slot<ChainLevels, 4, /*root=*/6>();
static_assert(kChainPlan.size() == 7, "4 stages + 3 level outputs");
static_assert(kChainPlan[0] == 0);
static_assert(kChainPlan[1] == 1);
static_assert(kChainPlan[2] == 2);
static_assert(kChainPlan[3] == 3);
static_assert(kChainPlan[4] == 4, "C0 is live alongside every stage");
static_assert(kChainPlan[5] == 2, "C1 reclaims B's buffer");
static_assert(kChainPlan[6] == 0, "C2 reclaims E's buffer");
static_assert(Impl::lg_pool_count<ChainLevels, 4, 6>() == 5,
              "seven slots must fit in five pools");

// --- the same graph, with a SECOND member in the last level ----------------
//
// Used only for the plan, never launched: two members of one level run
// interleaved with no barrier between them, so their outputs are simultaneously
// live and must never share. This is the property a per-MEMBER port of the DAG
// analysis would get wrong, and it would get it wrong silently.
auto level_wide(View2 a, View2 b, View2 e, View2 f) {
  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, E] = g0.stage(make_input_node(make_handle<'l', 'm'>(e)));
  auto [g2, F] = g1.stage(make_input_node(make_handle<'l', 'm'>(f)));
  auto [g3, B] = g2.stage(make_input_node(make_handle<'k', 'l'>(b)));
  auto [g4, A] = g3.stage(make_input_node(make_handle<'i', 'k'>(a)));

  auto [g5, c0] = g4.add(make_contraction_node<'i', 'l'>(A, B));
  auto [g6, x, y] =
      g5.add(make_contraction_node<'i', 'm'>(c0.template as<'i', 'l'>(), E),
             make_contraction_node<'i', 'm'>(c0.template as<'i', 'l'>(), F));
  return std::make_tuple(g6, x, y);
}

using WideGraph  = std::decay_t<decltype(std::get<0>(
    level_wide(std::declval<View2>(), std::declval<View2>(),
               std::declval<View2>(), std::declval<View2>())))>;
using WideLevels = WideGraph::levels_type;

constexpr auto kWidePlan = Impl::lg_pool_of_slot<WideLevels, 4, 5, 6>();
static_assert(kWidePlan[5] != kWidePlan[6],
              "two members of ONE level are simultaneously live: there is no "
              "barrier between them, so their outputs may never share a pool");
static_assert(kWidePlan[4] != kWidePlan[5] && kWidePlan[4] != kWidePlan[6],
              "a slot DEFINED at level L and one LAST READ at level L overlap "
              "at L -- closed ranges are what make that safe");

}  // namespace

// The plan is only worth anything if it actually reuses. A plan that assigned
// every slot its own pool would satisfy every disjointness property in this
// file and buy nothing.
TEST(LevelLivenessTest, PoolingActuallyReclaimsBuffers) {
  static_assert(Impl::lg_pool_count<ChainLevels, 4, 6>() <
                    Impl::lg_num_slots_v<ChainLevels, 4>,
                "pooling must use fewer pools than there are slots");
  // Bound to locals first: the commas in the template arguments would be read
  // as extra macro arguments.
  constexpr std::size_t pools = Impl::lg_pool_count<ChainLevels, 4, 6>();
  constexpr std::size_t slots = Impl::lg_num_slots_v<ChainLevels, 4>;
  EXPECT_EQ(pools, 5u);
  EXPECT_EQ(slots, 7u);
}

// The numbers, against a host reference. This is the graph whose plan reuses,
// so a wrong live range corrupts it rather than merely mis-sizing it.
TEST(LevelLivenessTest, PooledChainEqualsReference) {
  View2 a = make_view("A", kI, kK, 0);
  View2 b = make_view("B", kK, kL, 1);
  View2 e = make_view("E", kL, kM, 2);
  View2 f = make_view("F", kL, kM, 3);
  View2 out("C2", kI, kM);

  auto [g, c2] = level_chain(a, b, e, f);
  g.outputs(c2).execute(TeamPolicyTag2<ES>{}, out);
  Kokkos::fence();

  ViewH oh("oh", kI, kM);
  Kokkos::deep_copy(oh, out);

  // ((A B) E) F, computed independently on the host.
  std::vector<double> c0(static_cast<std::size_t>(kI) * kL, 0.0);
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      double s = 0.0;
      for (int k = 0; k < kK; ++k) s += fill_val(i, k, 0) * fill_val(k, l, 1);
      c0[static_cast<std::size_t>(i) * kL + l] = s;
    }
  std::vector<double> c1(static_cast<std::size_t>(kI) * kM, 0.0);
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) {
      double s = 0.0;
      for (int l = 0; l < kL; ++l)
        s += c0[static_cast<std::size_t>(i) * kL + l] * fill_val(l, m, 2);
      c1[static_cast<std::size_t>(i) * kM + m] = s;
    }

  // RELATIVE error: three chained contractions over 8x32x32 take the result to
  // ~5e6, where float epsilon alone is worth ~0.6 in absolute terms. An
  // absolute tolerance here would be measuring float, not the plan.
  double max_rel = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) {
      double ref = 0.0;
      for (int l = 0; l < kL; ++l)
        ref += c1[static_cast<std::size_t>(i) * kM + l] * fill_val(l, m, 3);
      const double got = static_cast<double>(oh(i, m));
      const double den = std::max(1.0, std::abs(ref));
      max_rel          = std::max(max_rel, std::abs(ref - got) / den);
    }
  // An aliasing bug does not land near the answer -- the un-gridded version of
  // this same graph missed by a relative 1e+00, so this bound has enormous
  // margin against the failure it is guarding.
  EXPECT_LT(max_rel, 1e-5) << "pooled level graph != reference";
}

// Pooling must SHRINK the request, and the launch must ask for the pooled
// figure rather than the un-pooled one -- asking for less than the carve uses
// overruns the team allocation on GPU only, since Serial's 32 KB absorbs it.
TEST(LevelLivenessTest, PooledBytesAreSmallerAndAreWhatIsRequested) {
  View2 a = make_view("A", kI, kK, 0);
  View2 b = make_view("B", kK, kL, 1);
  View2 e = make_view("E", kL, kM, 2);
  View2 f = make_view("F", kL, kM, 3);

  auto [g, c2] = level_chain(a, b, e, f);
  auto outs    = g.outputs(c2);

  EXPECT_LT(outs.scratch_bytes(), outs.slot_bytes())
      << "pooling did not reduce the scratch request";
  EXPECT_EQ(decltype(outs)::num_pools, 5u);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
