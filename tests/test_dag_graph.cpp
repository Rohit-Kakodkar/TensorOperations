// ===========================================================================
// test_dag_graph.cpp — the flat, topologically-ordered driver.
//
// Stage 5 of the DAG evaluator. Two things have to be true, and they are
// different claims:
//
//   EQUIVALENCE. A tree re-expressed as a DAG *with no sharing* must produce
//   bit-identical results AND request identical scratch. Equal scratch with no
//   sharing is what proves the flat sizing agrees with the recursive sizing --
//   checked before sharing muddies the comparison, so a discrepancy can only
//   mean the driver, not the deduplication.
//
//   SHARING. A diamond -- one node feeding two consumers -- must give the same
//   numbers as the tree that spells that node twice, at strictly less scratch.
//   That is the whole point of the exercise, and the scratch inequality is the
//   part a correctness test alone would miss.
//
// Data is non-symmetric with mutually distinct fills, for the reason spelled
// out in test_slot_node.cpp: an all-ones fill cannot tell a correct contraction
// from one that transposed an index.
// ===========================================================================
#include <TensorOperations/DagGraph.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

namespace {

// i is deliberately multi-tiled (32 over a 16 tile => 2 teams) so the driver's
// label-gathered index scheme is actually exercised; a single work item would
// let a wrong index still produce the right answer.
constexpr int kI = 32, kK = 8, kL = 32, kM = 32;
constexpr int kTI = 16;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

// Node 0: C0{i,l} = A{i,k} B{k,l}
using TA      = StaticTile<kTI, kK>;
using TB      = StaticTile<kK, kL>;
using TC0     = StaticTile<kTI, kL>;
using Bundle0 = Tile<TA, TB, TC0>;

// Node 1/2: {i,m} = C0{i,l} E{l,m}
using TE      = StaticTile<kL, kM>;
using TC1     = StaticTile<kTI, kM>;
using Bundle1 = Tile<TC0, TE, TC1>;

// Root combine over the two {i,m} results.
using CombRoot = CombineTile<TC1, TC1, TC1>;

struct Blend {
  KOKKOS_FUNCTION float operator()(int, int, float p, float q) const {
    return 2.0f * p - 3.0f * q;
  }
};

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

// --- the two-level chain, both spellings -----------------------------------

auto tree_chain_node(View2 a, View2 b, View2 e) {
  return make_contraction_node<'i', 'm'>(
      make_contraction_node<'i', 'l'>(
          make_input_node(make_handle<'i', 'k'>(a)),
          make_input_node(make_handle<'k', 'l'>(b))),
      make_input_node(make_handle<'l', 'm'>(e)));
}

void run_tree_chain(View2 a, View2 b, View2 e, View2 out) {
  auto g       = make_graph();
  auto [g1, t] = g.ops(tree_chain_node(a, b, e));
  g1.execute(TeamPolicyTag<ES>{}, Tile<Bundle0, TE, TC1>{}, out);
}

std::size_t tree_chain_bytes() {
  using Node = decltype(tree_chain_node(
      std::declval<View2>(), std::declval<View2>(), std::declval<View2>()));
  using Eval = Evaluator<TeamPolicyTag<ES>, Node, Tile<Bundle0, TE, TC1>>;
  return Eval::scratch_size_per_team(Tile<Bundle0, TE, TC1>{});
}

// The same computation as a DAG: node 0 is the inner contraction, node 1 names
// it. No sharing -- one consumer -- so this must match the tree exactly.
auto dag_chain(View2 a, View2 b, View2 e) {
  auto [g0, c0] =
      make_dag<float, ES>().add(make_contraction_node<'i', 'l'>(
                                    make_input_node(make_handle<'i', 'k'>(a)),
                                    make_input_node(make_handle<'k', 'l'>(b))),
                                Bundle0{});
  auto [g1, c1] = g0.add(make_contraction_node<'i', 'm'>(
                             c0.template as<'i', 'l'>(),
                             make_input_node(make_handle<'l', 'm'>(e))),
                         Bundle1{});
  return std::make_tuple(g1, c1);
}

void run_dag_chain(View2 a, View2 b, View2 e, View2 out) {
  auto [g, root] = dag_chain(a, b, e);
  g.outputs(root).execute(TeamPolicyTag<ES>{}, out);
}

std::size_t dag_chain_bytes(View2 a, View2 b, View2 e) {
  auto [g, root] = dag_chain(a, b, e);
  return g.scratch_bytes();
}

// --- the diamond: one node, two consumers ----------------------------------

auto tree_diamond_node(View2 a, View2 b, View2 e, View2 f) {
  return make_combine_node<'i', 'm'>(
      make_contraction_node<'i', 'm'>(
          make_contraction_node<'i', 'l'>(
              make_input_node(make_handle<'i', 'k'>(a)),
              make_input_node(make_handle<'k', 'l'>(b))),
          make_input_node(make_handle<'l', 'm'>(e))),
      make_contraction_node<'i', 'm'>(
          make_contraction_node<'i', 'l'>(
              make_input_node(make_handle<'i', 'k'>(a)),
              make_input_node(make_handle<'k', 'l'>(b))),
          make_input_node(make_handle<'l', 'm'>(f))),
      Blend{});
}

using TreeDiamondTile =
    CombineTile<TC1, Tile<Bundle0, TE, TC1>, Tile<Bundle0, TE, TC1>>;

void run_tree_diamond(View2 a, View2 b, View2 e, View2 f, View2 out) {
  auto g       = make_graph();
  auto [g1, t] = g.ops(tree_diamond_node(a, b, e, f));
  g1.execute(TeamPolicyTag<ES>{}, TreeDiamondTile{}, out);
}

std::size_t tree_diamond_bytes() {
  using Node =
      decltype(tree_diamond_node(std::declval<View2>(), std::declval<View2>(),
                                 std::declval<View2>(), std::declval<View2>()));
  using Eval = Evaluator<TeamPolicyTag<ES>, Node, TreeDiamondTile>;
  return Eval::scratch_size_per_team(TreeDiamondTile{});
}

// The DAG spelling: A*B appears ONCE and is named by both consumers.
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
  auto [g2, q] = g1.add(
      make_contraction_node<'i', 'm'>(
          c0.template as<'i', 'l'>(),  // the SAME slot, named a second time
          make_input_node(make_handle<'l', 'm'>(f))),
      Bundle1{});
  auto [g3, r] =
      g2.add(make_combine_node<'i', 'm'>(p.template as<'i', 'm'>(),
                                         q.template as<'i', 'm'>(), Blend{}),
             CombRoot{});
  return std::make_tuple(g3, r);
}

void run_dag_diamond(View2 a, View2 b, View2 e, View2 f, View2 out) {
  auto [g, root] = dag_diamond(a, b, e, f);
  g.outputs(root).execute(TeamPolicyTag<ES>{}, out);
}

std::size_t dag_diamond_bytes(View2 a, View2 b, View2 e, View2 f) {
  auto [g, root] = dag_diamond(a, b, e, f);
  return g.scratch_bytes();
}

// --- a multi-output node: M outputs, M slots, ONE evaluation ---------------
//
// SplitPair emits two tensors from one pass over its operands. SplitP/SplitQ
// are the same arithmetic as two separate single-output functors, defined
// THROUGH SplitPair so the split spelling cannot drift from the merged one --
// which is the whole basis of comparing them.

using CombPair = CombineTile<TC0, TC0, TC0>;

struct SplitPair {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int, int, float g,
                                                     float d) const {
    return {2.0f * g - 0.5f * d, -3.0f * g + 1.25f * d};
  }
};
struct SplitP {
  KOKKOS_FUNCTION float operator()(int a, int b, float g, float d) const {
    return SplitPair{}(a, b, g, d)[0];
  }
};
struct SplitQ {
  KOKKOS_FUNCTION float operator()(int a, int b, float g, float d) const {
    return SplitPair{}(a, b, g, d)[1];
  }
};

// Node 0 alone, so both spellings below start from the same gradient-like node.
template <typename Dag>
auto add_split_source(const Dag& d0, View2 a, View2 b) {
  return d0.add(make_contraction_node<'i', 'l'>(
                    make_input_node(make_handle<'i', 'k'>(a)),
                    make_input_node(make_handle<'k', 'l'>(b))),
                Bundle0{});
}

// MERGED: one node, two outputs, two slots. The pair is computed once.
auto dag_pair_merged(View2 a, View2 b, View2 d) {
  auto [g0, c0] = add_split_source(make_dag<float, ES>(), a, b);
  auto [g1, p, q] =
      g0.add(make_combine_node<'i', 'l'>(
                 c0.template as<'i', 'l'>(),
                 make_input_node(make_handle<'i', 'l'>(d)), SplitPair{}),
             CombPair{});
  return std::make_tuple(g1, p, q);
}

// SPLIT: two single-output nodes over the same shared source. Same numbers,
// but the operand staging and the fn call happen twice.
auto dag_pair_split(View2 a, View2 b, View2 d) {
  auto [g0, c0] = add_split_source(make_dag<float, ES>(), a, b);
  auto [g1, p] =
      g0.add(make_combine_node<'i', 'l'>(
                 c0.template as<'i', 'l'>(),
                 make_input_node(make_handle<'i', 'l'>(d)), SplitP{}),
             CombPair{});
  auto [g2, q] =
      g1.add(make_combine_node<'i', 'l'>(
                 c0.template as<'i', 'l'>(),
                 make_input_node(make_handle<'i', 'l'>(d)), SplitQ{}),
             CombPair{});
  return std::make_tuple(g2, p, q);
}

// Both outputs consumed by later nodes, then merged: the shape the SEM
// stiffness graph actually uses. `p` feeds a contraction against E, `q` one
// against F, and a root combine blends them.
template <typename Dag, typename HP, typename HQ>
auto dag_pair_tail(const Dag& g, HP p, HQ q, View2 e, View2 f) {
  auto [g1, t0] = g.add(
      make_contraction_node<'i', 'm'>(
          p.template as<'i', 'l'>(), make_input_node(make_handle<'l', 'm'>(e))),
      Bundle1{});
  auto [g2, t1] = g1.add(
      make_contraction_node<'i', 'm'>(
          q.template as<'i', 'l'>(), make_input_node(make_handle<'l', 'm'>(f))),
      Bundle1{});
  auto [g3, r] =
      g2.add(make_combine_node<'i', 'm'>(t0.template as<'i', 'm'>(),
                                         t1.template as<'i', 'm'>(), Blend{}),
             CombRoot{});
  return std::make_tuple(g3, r);
}

}  // namespace

// EQUIVALENCE. No sharing anywhere, so the DAG must be indistinguishable from
// the tree -- in results and in what it asks the team for.
TEST(DagGraphTest, NoSharingMatchesTheTreeExactly) {
  const auto a = make_view("a", kI, kK, 0);
  const auto b = make_view("b", kK, kL, 1);
  const auto e = make_view("e", kL, kM, 2);

  View2 out_tree("out_tree", kI, kM), out_dag("out_dag", kI, kM);
  run_tree_chain(a, b, e, out_tree);
  run_dag_chain(a, b, e, out_dag);
  Kokkos::fence();

  const auto ht = to_host(out_tree);
  const auto hd = to_host(out_dag);

  double mag = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) mag += std::abs(static_cast<double>(ht(i, m)));
  EXPECT_GT(mag, 1.0) << "the tree produced nothing to compare against";

  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m)
      EXPECT_EQ(ht(i, m), hd(i, m)) << "at (" << i << "," << m << ")";

  // The decisive half: identical scratch with no sharing proves the flat sum
  // agrees with the recursive one.
  EXPECT_EQ(dag_chain_bytes(a, b, e), tree_chain_bytes());
}

// SHARING. Same numbers as the tree that spells the shared node twice, at
// strictly less scratch.
TEST(DagGraphTest, SharedNodeMatchesTheDuplicatedTreeAndCostsLess) {
  const auto a = make_view("a", kI, kK, 0);
  const auto b = make_view("b", kK, kL, 1);
  const auto e = make_view("e", kL, kM, 2);
  const auto f = make_view("f", kL, kM, 3);

  View2 out_tree("out_tree", kI, kM), out_dag("out_dag", kI, kM);
  run_tree_diamond(a, b, e, f, out_tree);
  run_dag_diamond(a, b, e, f, out_dag);
  Kokkos::fence();

  const auto ht = to_host(out_tree);
  const auto hd = to_host(out_dag);

  double mag = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) mag += std::abs(static_cast<double>(ht(i, m)));
  EXPECT_GT(mag, 1.0);

  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m)
      EXPECT_EQ(ht(i, m), hd(i, m)) << "at (" << i << "," << m << ")";

  // The point of the whole exercise: the shared subtree is stored and evaluated
  // once, so the DAG asks the team for strictly less than the tree does.
  const std::size_t dag_b  = dag_diamond_bytes(a, b, e, f);
  const std::size_t tree_b = tree_diamond_bytes();
  EXPECT_LT(dag_b, tree_b) << "dag " << dag_b << " vs tree " << tree_b;
  // Printed, not merely asserted: the magnitude is the result, and a test that
  // only says "less" would hide a win shrinking to nothing.
  std::printf("[   INFO   ] scratch/team: tree %zu B, dag %zu B (%.2fx less)\n",
              tree_b, dag_b,
              static_cast<double>(tree_b) / static_cast<double>(dag_b));
}

// The label-gathered index scheme's precondition, asked host-side.
TEST(DagGraphTest, IndexConsistencyIsCheckable) {
  const auto a = make_view("a", kI, kK, 0);
  const auto b = make_view("b", kK, kL, 1);
  const auto e = make_view("e", kL, kM, 2);
  const auto f = make_view("f", kL, kM, 3);

  auto [gc, rc] = dag_chain(a, b, e);
  EXPECT_TRUE(gc.index_consistent());

  auto [gd, rd] = dag_diamond(a, b, e, f);
  EXPECT_TRUE(gd.index_consistent());

  // Sizing splits the way the driver assumes: store + what evaluators still
  // carve once their outputs are adopted.
  EXPECT_EQ(gd.scratch_bytes(), gd.slot_bytes() + gd.operand_bytes());
  EXPECT_GT(gd.slot_bytes(), 0u);
}

// MULTI-OUTPUT, WRITTEN STRAIGHT OUT. Both slots of one node designated as
// graph outputs -- which is also the case that proves a root is addressed by
// SLOT and not by node index, since output 1 lives at slot 2 of a two-node
// graph.
TEST(DagGraphTest, MultiOutputSlotsAreBothGraphOutputs) {
  const auto a = make_view("a", kI, kK, 0);
  const auto b = make_view("b", kK, kL, 1);
  const auto d = make_view("d", kI, kL, 4);

  View2 mp("mp", kI, kL), mq("mq", kI, kL);
  View2 sp("sp", kI, kL), sq("sq", kI, kL);
  {
    auto [g, p, q] = dag_pair_merged(a, b, d);
    g.outputs(p, q).execute(TeamPolicyTag<ES>{}, mp, mq);
  }
  {
    auto [g, p, q] = dag_pair_split(a, b, d);
    g.outputs(p, q).execute(TeamPolicyTag<ES>{}, sp, sq);
  }
  Kokkos::fence();

  const auto hmp = to_host(mp), hmq = to_host(mq);
  const auto hsp = to_host(sp), hsq = to_host(sq);

  // The two outputs must not be the same tensor, or this test would pass on an
  // evaluator that wrote output 0 into both slots.
  double spread = 0.0, mag = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      spread += std::abs(static_cast<double>(hmp(i, l) - hmq(i, l)));
      mag += std::abs(static_cast<double>(hmp(i, l)));
    }
  EXPECT_GT(mag, 1.0) << "the merged spelling produced nothing";
  EXPECT_GT(spread, 1.0) << "the two outputs are indistinguishable";

  for (int i = 0; i < kI; ++i)
    for (int l = 0; l < kL; ++l) {
      EXPECT_EQ(hmp(i, l), hsp(i, l))
          << "output 0 at (" << i << "," << l << ")";
      EXPECT_EQ(hmq(i, l), hsq(i, l))
          << "output 1 at (" << i << "," << l << ")";
    }
}

// MULTI-OUTPUT AS OPERANDS. Each output feeds a different consumer -- the SEM
// stiffness shape. Same numbers as the split spelling, at no more scratch: the
// slots are the same buffers either way, but the merged node stages its
// operands and calls fn ONCE where the split one does it twice.
TEST(DagGraphTest, MultiOutputFeedsTwoConsumersAndCostsLess) {
  const auto a = make_view("a", kI, kK, 0);
  const auto b = make_view("b", kK, kL, 1);
  const auto d = make_view("d", kI, kL, 4);
  const auto e = make_view("e", kL, kM, 2);
  const auto f = make_view("f", kL, kM, 3);

  View2       out_merged("out_merged", kI, kM), out_split("out_split", kI, kM);
  std::size_t merged_b = 0, split_b = 0, merged_slots = 0, split_slots = 0;
  {
    auto [g0, p, q] = dag_pair_merged(a, b, d);
    auto [g, r]     = dag_pair_tail(g0, p, q, e, f);
    EXPECT_TRUE(g.index_consistent());
    merged_b     = g.scratch_bytes();
    merged_slots = g.slot_bytes();
    g.outputs(r).execute(TeamPolicyTag<ES>{}, out_merged);
  }
  {
    auto [g0, p, q] = dag_pair_split(a, b, d);
    auto [g, r]     = dag_pair_tail(g0, p, q, e, f);
    split_b         = g.scratch_bytes();
    split_slots     = g.slot_bytes();
    g.outputs(r).execute(TeamPolicyTag<ES>{}, out_split);
  }
  Kokkos::fence();

  const auto hm = to_host(out_merged);
  const auto hs = to_host(out_split);

  double mag = 0.0;
  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m) mag += std::abs(static_cast<double>(hs(i, m)));
  EXPECT_GT(mag, 1.0);

  for (int i = 0; i < kI; ++i)
    for (int m = 0; m < kM; ++m)
      EXPECT_EQ(hs(i, m), hm(i, m)) << "at (" << i << "," << m << ")";

  // The slot store is the same size either way -- two outputs is two buffers
  // however they are spelled -- so any saving is in what the second combine
  // evaluator no longer carves. Stated as two assertions because a merged
  // spelling that grew the STORE would be a bug this file should catch.
  EXPECT_EQ(merged_slots, split_slots);
  EXPECT_LE(merged_b, split_b);
  std::printf(
      "[   INFO   ] scratch/team: split %zu B, merged %zu B (slots %zu B "
      "both)\n",
      split_b, merged_b, merged_slots);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
