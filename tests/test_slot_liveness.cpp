// ===========================================================================
// test_slot_liveness.cpp — pooling the slot store by live range.
//
// A DagGraph used to give every node output its own buffer for the whole
// kernel. Most of them are dead long before that: in a chain, node K's result
// is read by node K+1 and never again. Liveness puts the slots that cannot be
// alive at once on ONE buffer.
//
// THE FAILURE MODE THIS FILE EXISTS FOR IS SILENT. An over-eager plan does not
// crash and does not fault -- it points a live slot at a buffer some other node
// is about to overwrite, and the kernel returns plausible wrong numbers. So the
// plan is checked in two independent ways:
//
//   THE PLAN ITSELF, at compile time. dag_pool_of_slot is constexpr, so the
//   expected assignment for a hand-traceable graph is a static_assert -- no
//   kernel involved, and a regression fails the BUILD rather than a tolerance.
//
//   THE NUMBERS, at run time, against a host reference computed independently.
//   The chain below is built so that reuse actually happens (slot 2 lands on
//   slot 0's buffer); if the ranges were computed wrongly this is exactly the
//   graph that would corrupt.
//
// Both matter. The static_asserts would pass a plan that is right about
// liveness and wrong about where it points, and the numeric test alone would
// pass a plan that simply never reuses anything.
// ===========================================================================
#include <TensorOperations/DagGraph.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
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
using TC = StaticTile<kTI, kL>;  // == StaticTile<kTI, kM>: every slot's tile
using TE = StaticTile<kL, kM>;
using Bundle0 = Tile<TA, TB, TC>;  // C0{i,l} = A{i,k} B{k,l}
using Bundle1 = Tile<TC, TE, TC>;  // C{i,m}  = C_prev{i,l} E{l,m}

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

ViewH to_host(const View2& d) {
  ViewH h("h", d.extent(0), d.extent(1));
  Kokkos::deep_copy(h, d);
  return h;
}

// --- the chain: C2 = ((A B) E) F -------------------------------------------
//
// Three slots, and the middle one is what forces the interesting plan. Slot 0
// dies at node 1, so slot 2 -- defined at node 2 -- can have its buffer.
auto dag_chain(View2 a, View2 b, View2 e, View2 f) {
  auto [g0, c0] =
      make_dag<float, ES>().add(make_contraction_node<'i', 'l'>(
                                    make_input_node(make_handle<'i', 'k'>(a)),
                                    make_input_node(make_handle<'k', 'l'>(b))),
                                Bundle0{});
  auto [g1, c1] = g0.add(make_contraction_node<'i', 'm'>(
                             c0.template as<'i', 'l'>(),
                             make_input_node(make_handle<'l', 'm'>(e))),
                         Bundle1{});
  auto [g2, c2] = g1.add(make_contraction_node<'i', 'm'>(
                             c1.template as<'i', 'l'>(),
                             make_input_node(make_handle<'l', 'm'>(f))),
                         Bundle1{});
  return std::make_tuple(g2, c2);
}

using ChainGraph = std::decay_t<decltype(std::get<0>(
    dag_chain(std::declval<View2>(), std::declval<View2>(),
              std::declval<View2>(), std::declval<View2>())))>;
using ChainNodes = ChainGraph::nodes_type;

// Slot 2 is the only root, so it is live past the last node; slots 0 and 1 die
// at their consumers. Hand-traced:
//
//   slot 0  written node 0, last read node 1  -> [0,1]
//   slot 1  written node 1, last read node 2  -> [1,2]
//   slot 2  written node 2, read by store_roots after node 2 -> [2,3]
//
// so 0 and 2 are disjoint and share, and 1 overlaps both.
constexpr auto kChainPlan = Impl::dag_pool_of_slot<ChainNodes, 2>();
static_assert(kChainPlan.size() == 3, "three node outputs, three slots");
static_assert(kChainPlan[0] == 0);
static_assert(kChainPlan[1] == 1, "slot 1 is live across slot 0's death");
static_assert(kChainPlan[2] == 0, "slot 2 reclaims slot 0's buffer");
static_assert(Impl::dag_pool_count<ChainNodes, 2>() == 2,
              "three slots must fit in two pools");

// A named functor, not a lambda: nvcc rejects an extended __host__ __device__
// lambda inside a function with a deduced return type, which every graph
// builder here has.
struct Add {
  KOKKOS_FUNCTION float operator()(int, int, float x, float y) const {
    return x + y;
  }
};

// --- the diamond: one node, two consumers, joined by a combine -------------
auto dag_diamond(View2 a, View2 b, View2 e, View2 f) {
  auto [g0, c0] =
      make_dag<float, ES>().add(make_contraction_node<'i', 'l'>(
                                    make_input_node(make_handle<'i', 'k'>(a)),
                                    make_input_node(make_handle<'k', 'l'>(b))),
                                Bundle0{});
  auto [g1, p] = g0.add(make_contraction_node<'i', 'm'>(
                            c0.template as<'i', 'l'>(),
                            make_input_node(make_handle<'l', 'm'>(e))),
                        Bundle1{});
  auto [g2, q] = g1.add(make_contraction_node<'i', 'm'>(
                            c0.template as<'i', 'l'>(),  // the SAME slot again
                            make_input_node(make_handle<'l', 'm'>(f))),
                        Bundle1{});
  auto [g3, r] =
      g2.add(make_combine_node<'i', 'm'>(p.template as<'i', 'm'>(),
                                         q.template as<'i', 'm'>(), Add{}),
             CombineTile<TC, TC, TC>{});
  return std::make_tuple(g3, r);
}

using DiamondGraph = std::decay_t<decltype(std::get<0>(
    dag_diamond(std::declval<View2>(), std::declval<View2>(),
                std::declval<View2>(), std::declval<View2>())))>;
using DiamondNodes = DiamondGraph::nodes_type;

// Fan-out extends a live range to its LAST consumer, which is the part a naive
// "free it at its first read" rule gets wrong:
//
//   slot 0  [0,2]   read by BOTH node 1 and node 2
//   slot 1  [1,3]
//   slot 2  [2,3]
//   slot 3  [3,4]   the root
//
// 0 and 3 are the only disjoint pair, so four slots need three pools.
constexpr auto kDiamondPlan = Impl::dag_pool_of_slot<DiamondNodes, 3>();
static_assert(kDiamondPlan[0] == 0);
static_assert(kDiamondPlan[1] == 1);
static_assert(kDiamondPlan[2] == 2,
              "slot 2 overlaps both 0 (still fanning out) and 1");
static_assert(kDiamondPlan[3] == 0, "the root reclaims the shared node's slot");
static_assert(Impl::dag_pool_count<DiamondNodes, 3>() == 3);

// Host reference: plain triple loops, independent of every tiling decision.
ViewH host_matmul(const ViewH& x, const ViewH& y) {
  const int r = static_cast<int>(x.extent(0)),
            k = static_cast<int>(x.extent(1)),
            c = static_cast<int>(y.extent(1));
  ViewH     out("ref", r, c);
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < c; ++j) {
      float s = 0.0f;
      for (int p = 0; p < k; ++p) s += x(i, p) * y(p, j);
      out(i, j) = s;
    }
  return out;
}

}  // namespace

// A slot that dies must be reclaimed, and the kernel must still be right. The
// two claims are one test on purpose: either alone is satisfiable by a broken
// implementation.
TEST(SlotLivenessTest, ChainReusesADeadSlotAndStaysCorrect) {
  View2 a = make_view("a", kI, kK, 1), b = make_view("b", kK, kL, 2);
  View2 e = make_view("e", kL, kM, 3), f = make_view("f", kL, kM, 4);
  View2 out("out", kI, kM);

  auto [g, root] = dag_chain(a, b, e, f);
  auto outs      = g.outputs(root);
  outs.execute(TeamPolicyTag<ES>{}, out);

  const ViewH ref = host_matmul(
      host_matmul(host_matmul(to_host(a), to_host(b)), to_host(e)), to_host(f));
  const ViewH got = to_host(out);
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m)
      EXPECT_NEAR(got(i, m), ref(i, m), 2e-2f * std::abs(ref(i, m)) + 1e-2f)
          << "at (" << i << "," << m << ")";

  // Three slots in two pools: exactly one tile's worth must have come back.
  EXPECT_EQ(decltype(outs)::num_pools, 2u);
  EXPECT_LT(outs.slot_bytes(), g.slot_bytes());
  EXPECT_EQ(g.slot_bytes() - outs.slot_bytes(),
            g.slot_bytes() / 3);  // one of three equal tiles
  std::printf(
      "[   INFO   ] chain slots: %zu B unpooled -> %zu B in %zu pools\n",
      g.slot_bytes(), outs.slot_bytes(),
      static_cast<std::size_t>(decltype(outs)::num_pools));
}

// The fan-out case, where the shared node stays live past its first reader.
TEST(SlotLivenessTest, FanOutKeepsTheSharedSlotLiveToItsLastReader) {
  View2 a = make_view("a", kI, kK, 5), b = make_view("b", kK, kL, 6);
  View2 e = make_view("e", kL, kM, 7), f = make_view("f", kL, kM, 8);
  View2 out("out", kI, kM);

  auto [g, root] = dag_diamond(a, b, e, f);
  auto outs      = g.outputs(root);
  outs.execute(TeamPolicyTag<ES>{}, out);

  const ViewH ab  = host_matmul(to_host(a), to_host(b));
  const ViewH lhs = host_matmul(ab, to_host(e));
  const ViewH rhs = host_matmul(ab, to_host(f));
  const ViewH got = to_host(out);
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) {
      const float ref = lhs(i, m) + rhs(i, m);
      EXPECT_NEAR(got(i, m), ref, 2e-2f * std::abs(ref) + 1e-2f)
          << "at (" << i << "," << m << ")";
    }

  EXPECT_EQ(decltype(outs)::num_pools, 3u);
  EXPECT_LT(outs.slot_bytes(), g.slot_bytes());
  std::printf(
      "[   INFO   ] diamond slots: %zu B unpooled -> %zu B in %zu pools\n",
      g.slot_bytes(), outs.slot_bytes(),
      static_cast<std::size_t>(decltype(outs)::num_pools));
}

// A pool is only ever as big as its largest occupant, so pooling can never
// request MORE than one-buffer-per-slot however the ranges fall. Worth its own
// assertion: it is the property that makes a bad plan cost savings rather than
// correctness.
TEST(SlotLivenessTest, PoolingNeverCostsMoreThanNoPooling) {
  View2 a = make_view("a", kI, kK, 9), b = make_view("b", kK, kL, 10);
  View2 e = make_view("e", kL, kM, 11), f = make_view("f", kL, kM, 12);

  auto [gc, rc] = dag_chain(a, b, e, f);
  auto [gd, rd] = dag_diamond(a, b, e, f);
  EXPECT_LE(gc.outputs(rc).slot_bytes(), gc.slot_bytes());
  EXPECT_LE(gd.outputs(rd).slot_bytes(), gd.slot_bytes());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
