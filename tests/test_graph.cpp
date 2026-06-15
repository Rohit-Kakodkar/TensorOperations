#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Tests operate on Kokkos::View tensors so value_type propagates correctly.
// ---------------------------------------------------------------------------

TEST(GraphTest, SingleContractionNode) {
  // A_{i,j,k}: (3, 4, 5)   B_{j,k,l}: (4, 5, 6)  →  C_{i,l}: (3, 6)
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View3 A("A", 3, 4, 5);
  View3 B("B", 4, 5, 6);

  auto hA =
      make_input_node(make_handle(A, std::array<int32_t, 3>{'i', 'j', 'k'}));
  auto hB =
      make_input_node(make_handle(B, std::array<int32_t, 3>{'j', 'k', 'l'}));

  auto g = make_graph();
  auto [g1, out1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'l'}));
  auto [T1] = out1;

  // Compile-time index counts: C_{i,l} survives 2 free modes, contracts j,k.
  static_assert(decltype(g1)::num_free_indices() == 2);
  static_assert(decltype(g1)::num_contraction_indices() == 2);

  static_assert(decltype(T1)::Rank == 2);
  static_assert(decltype(T1)::NumContracted == 2);
  static_assert(std::is_same_v<decltype(T1)::value_type, float>);

  auto s = T1.shape();
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 6);

  auto m = T1.modes();
  EXPECT_EQ(m[0], 'i');
  EXPECT_EQ(m[1], 'l');

  // Work items over participating modes [i=3, l=6, j=4, k=5] (ceil-div).
  EXPECT_EQ((g1.execute<RangePolicyTag>(StaticTile<3, 6, 4, 5>{})), 1u);
  EXPECT_EQ((g1.execute<RangePolicyTag>(StaticTile<1, 1, 1, 1>{})), 360u);
  EXPECT_EQ((g1.execute<RangePolicyTag>(StaticTile<3, 3, 4, 5>{})), 2u);
}

TEST(GraphTest, MultiLevelChain) {
  // A_{i,k,m}: (3,4,5)  B_{k,m,j}: (4,5,6)  →  AB_{i,j}: (3,6)
  // C_{p,q,r}: (2,3,4)  D_{q,r,s}: (3,4,5)  →  CD_{p,s}: (2,5)
  // AB_{i,j} ⊗ CD_{p,s} →  T3_{i,j,p,s}: (3,6,2,5)
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View3 A("A", 3, 4, 5);
  View3 B("B", 4, 5, 6);
  View3 C("C", 2, 3, 4);
  View3 D("D", 3, 4, 5);

  auto hA =
      make_input_node(make_handle(A, std::array<int32_t, 3>{'i', 'k', 'm'}));
  auto hB =
      make_input_node(make_handle(B, std::array<int32_t, 3>{'k', 'm', 'j'}));
  auto hC =
      make_input_node(make_handle(C, std::array<int32_t, 3>{'p', 'q', 'r'}));
  auto hD =
      make_input_node(make_handle(D, std::array<int32_t, 3>{'q', 'r', 's'}));

  auto g = make_graph();

  auto [g1, lvl1] =
      g.ops(make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'j'}),
            make_contraction_node(hC, hD, std::array<int32_t, 2>{'p', 's'}));
  auto [T1, T2] = lvl1;

  static_assert(decltype(T1)::Rank == 2);
  static_assert(decltype(T2)::Rank == 2);

  auto [g2, lvl2] = g1.ops(make_contraction_node(
      T1, T2, std::array<int32_t, 4>{'i', 'j', 'p', 's'}));
  auto [T3]       = lvl2;

  static_assert(decltype(T3)::Rank == 4);
  static_assert(decltype(T3)::NumContracted == 0);  // outer product

  auto s = T3.shape();
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 6);
  EXPECT_EQ(s[2], 2);
  EXPECT_EQ(s[3], 5);

  // Compile-time index counts over the whole graph: T3 keeps 4 free modes;
  // the two leaf contractions each contract 2 modes (AB: k,m; CD: q,r) and the
  // top-level outer product contracts 0 → 2 + 2 + 0 = 4.
  static_assert(decltype(g2)::num_free_indices() == 4);
  static_assert(decltype(g2)::num_contraction_indices() == 4);

  // T3 is an outer product (NumContracted 0): participating modes == output
  // modes [3, 6, 2, 5].
  EXPECT_EQ((g2.execute<TeamPolicyTag>(StaticTile<1, 1, 1, 1>{})), 180u);
  EXPECT_EQ((g2.execute<RangePolicyTag>(StaticTile<3, 6, 2, 5>{})), 1u);
}

TEST(GraphTest, ScalarInferredFromNode) {
  // Inferred scalar must flow from input through contraction
  using View2f = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using View2d = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2f Af("Af", 3, 4);
  View2d Ad("Ad", 4, 5);

  auto hAf = make_input_node(make_handle(Af, std::array<int32_t, 2>{'i', 'j'}));
  auto hAd = make_input_node(make_handle(Ad, std::array<int32_t, 2>{'j', 'k'}));

  auto nf = make_contraction_node(hAf, hAd, std::array<int32_t, 2>{'i', 'k'});
  static_assert(std::is_same_v<decltype(nf)::value_type, float>);

  // Explicit override still works
  auto nd =
      make_contraction_node<double>(hAf, hAd, std::array<int32_t, 2>{'i', 'k'});
  static_assert(std::is_same_v<decltype(nd)::value_type, double>);
}

TEST(GraphTest, HookPreservesType) {
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2 A("A", 3, 4);
  View2 B("B", 4, 5);

  auto hA = make_input_node(make_handle(A, std::array<int32_t, 2>{'i', 'j'}));
  auto hB = make_input_node(make_handle(B, std::array<int32_t, 2>{'j', 'k'}));

  [[maybe_unused]] int call_count = 0;
  auto                 relu       = [&]() { ++call_count; };

  auto nc =
      make_contraction_node(hA, hB, std::array<int32_t, 2>{'i', 'k'}, relu);

  // Hook type is the lambda — no type erasure
  static_assert(!std::is_same_v<decltype(nc)::hook_type, NoHook>);
  static_assert(!std::is_same_v<decltype(nc)::hook_type, void>);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
