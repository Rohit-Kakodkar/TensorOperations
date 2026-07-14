#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time label-plan checks. For a combine node the output labels ARE the
// canonical order; each operand's label_perm gathers its axes into that order.
// P{i,j} = fn(A{i,j}, B{j,i}): A is already canonical (identity), B is
// reversed.
// ---------------------------------------------------------------------------
namespace {
using OutM = std::integer_sequence<int32_t, 'i', 'j'>;
using AM   = std::integer_sequence<int32_t, 'i', 'j'>;
using BM   = std::integer_sequence<int32_t, 'j', 'i'>;
// A already in output order -> identity gather.
static_assert(Impl::is_identity_seq(Impl::label_perm_seq_t<OutM, AM>{}));
// B is [j,i]; output axis i (0) draws from B axis 1, output axis j (1) from B
// axis 0 -> perm = {1,0}.
static_assert(std::is_same_v<Impl::label_perm_seq_t<OutM, BM>,
                             std::integer_sequence<int, 1, 0>>);
static_assert(Impl::same_label_set<AM, OutM>());
static_assert(Impl::same_label_set<BM, OutM>());
static_assert(
    !Impl::same_label_set<std::integer_sequence<int32_t, 'i', 'k'>, OutM>());
}  // namespace

// Combine functors are named structs (not device lambdas): GoogleTest's
// TestBody is private, so a KOKKOS_LAMBDA there cannot be used as a device
// functor. fn receives the global output coordinate then both operand values.
struct MulPlusCoord {
  KOKKOS_FUNCTION float operator()(int i, int j, float a, float b) const {
    return a * b + static_cast<float>(i) * 100.0f + static_cast<float>(j);
  }
};

struct AddOp {
  KOKKOS_FUNCTION float operator()(int, int, float a, float b) const {
    return a + b;
  }
};

struct FmaCoord3 {
  KOKKOS_FUNCTION float operator()(int i, int j, float a, float b,
                                   float c) const {
    return a * b + c + static_cast<float>(i) * 100.0f + static_cast<float>(j);
  }
};

// Multi-output combine: returns Kokkos::Array<float,2> -> two output tensors.
struct SumDiff {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int j, float a,
                                                     float b) const {
    return {a * b, a - b + static_cast<float>(i - j)};  // {p0, p1}
  }
};

// ---------------------------------------------------------------------------

TEST(CombineNodeTest, ShapeAndModes) {
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2 A("A", 12, 8);
  View2 B("B", 12, 8);

  auto na = make_input_node(make_handle<'i', 'j'>(A));
  auto nb = make_input_node(make_handle<'i', 'j'>(B));

  auto np = make_combine_node<'i', 'j'>(na, nb, AddOp{});

  static_assert(decltype(np)::Rank == 2);
  static_assert(std::is_same_v<decltype(np)::value_type, float>);
  static_assert(std::is_same_v<decltype(np)::modes_seq,
                               std::integer_sequence<int32_t, 'i', 'j'>>);

  auto s = np.shape();
  EXPECT_EQ(s[0], 12);
  EXPECT_EQ(s[1], 8);
}

TEST(CombineNodeTest, SameOrderMultiTileTeam) {
  // P{i,j} = A{i,j}*B{i,j} + 100*i + j, non-symmetric data, tiled so i and j
  // each split into multiple tiles. The coordinate term catches any global-vs-
  // tile-local index bug (like InputHookGlobalIndexAcrossTiles for contraction)
  // and the multiply term guards the plain combine.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 12, J = 8;
  View2         A("A", I, J);
  View2         B("B", I, J);
  View2         P("P", I, J);

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      Ah(i, j) = static_cast<float>((i + 2 * j) % 5 + 1) * 0.5f;
      Bh(i, j) = static_cast<float>((3 * i + j) % 4 + 1) * 0.25f;
    }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(P, 0.0f);

  auto na       = make_input_node(make_handle<'i', 'j'>(A));
  auto nb       = make_input_node(make_handle<'i', 'j'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'j'>(na, nb, MulPlusCoord{}));

  // Single output tile <4,4>: i -> 3 tiles, j -> 2 tiles (6 teams).
  const int wk = g1.execute(TeamPolicyTag<>{}, StaticTile<4, 4>{}, P);
  EXPECT_EQ(wk, 6);

  auto Ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      EXPECT_FLOAT_EQ(Ph(i, j), Ah(i, j) * Bh(i, j) +
                                    static_cast<float>(i) * 100.0f +
                                    static_cast<float>(j))
          << "i=" << i << " j=" << j;
}

TEST(CombineNodeTest, PermutedOperandTeam) {
  // P{i,j} = A{i,j} + B{j,i}: B is stored transposed. Non-square I != J so a
  // missed transpose is not masked. Exercises label_perm_seq +
  // canonicalize_input (the permuted-operand staging path) on the second
  // operand.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 12, J = 8;
  View2         A("A", I, J);  // axes i,j
  View2         B("B", J, I);  // axes j,i
  View2         P("P", I, J);  // axes i,j

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      Ah(i, j) = static_cast<float>((i + 2 * j) % 7 + 1);
  for (int j = 0; j < J; ++j)
    for (int i = 0; i < I; ++i)
      Bh(j, i) = static_cast<float>((3 * j + i) % 5 + 1);
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(P, 0.0f);

  auto na       = make_input_node(make_handle<'i', 'j'>(A));
  auto nb       = make_input_node(make_handle<'j', 'i'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'j'>(na, nb, AddOp{}));

  g1.execute(TeamPolicyTag<>{}, StaticTile<4, 4>{}, P);

  auto Ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      EXPECT_FLOAT_EQ(Ph(i, j), Ah(i, j) + Bh(j, i)) << "i=" << i << " j=" << j;
}

TEST(CombineNodeTest, TernaryPermutedTeam) {
  // Three operands: P{i,j} = A{i,j}*B{i,j} + C{j,i} + 100*i + j, with C stored
  // transposed. Exercises the N-ary (DeviceTuple) path with a mix of identity
  // and permuted operands, non-symmetric non-square data, multiple tiles.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 12, J = 8;
  View2         A("A", I, J);  // axes i,j
  View2         B("B", I, J);  // axes i,j
  View2         C("C", J, I);  // axes j,i (permuted)
  View2         P("P", I, J);  // axes i,j

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  auto Ch = Kokkos::create_mirror_view(C);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      Ah(i, j) = static_cast<float>((i + 2 * j) % 5 + 1) * 0.5f;
      Bh(i, j) = static_cast<float>((3 * i + j) % 4 + 1) * 0.25f;
    }
  for (int j = 0; j < J; ++j)
    for (int i = 0; i < I; ++i)
      Ch(j, i) = static_cast<float>((2 * j + 3 * i) % 6 + 1);
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(C, Ch);
  Kokkos::deep_copy(P, 0.0f);

  auto na       = make_input_node(make_handle<'i', 'j'>(A));
  auto nb       = make_input_node(make_handle<'i', 'j'>(B));
  auto nc       = make_input_node(make_handle<'j', 'i'>(C));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'j'>(na, nb, nc, FmaCoord3{}));

  static_assert(decltype(o1)::NumOps == 3);

  g1.execute(TeamPolicyTag<>{}, StaticTile<4, 4>{}, P);

  auto Ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      EXPECT_FLOAT_EQ(Ph(i, j), Ah(i, j) * Bh(i, j) + Ch(j, i) +
                                    static_cast<float>(i) * 100.0f +
                                    static_cast<float>(j))
          << "i=" << i << " j=" << j;
}

TEST(CombineNodeTest, MultiOutputTeam) {
  // fn returns Kokkos::Array<float,2>, so the node emits TWO tensors in one
  // pass:
  //   p0 = A*B ,  p1 = A - B + (i - j).
  // ops() expands the node into two handles; execute takes two views. Data is
  // non-symmetric / non-square, checked against a host reference.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 12, J = 8;
  View2         A("A", I, J);
  View2         B("B", I, J);
  View2         P0("P0", I, J);
  View2         P1("P1", I, J);

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      Ah(i, j) = static_cast<float>((i + 2 * j) % 5 + 1) * 0.5f;
      Bh(i, j) = static_cast<float>((3 * i + j) % 4 + 1) * 0.25f;
    }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(P0, 0.0f);
  Kokkos::deep_copy(P1, 0.0f);

  auto na = make_input_node(make_handle<'i', 'j'>(A));
  auto nb = make_input_node(make_handle<'i', 'j'>(B));
  auto g  = make_graph();

  auto [g1, p0, p1] = g.ops(make_combine_node<'i', 'j'>(na, nb, SumDiff{}));
  static_assert(decltype(p0)::OutputIndex == 0);
  static_assert(decltype(p1)::OutputIndex == 1);

  g1.execute(TeamPolicyTag<>{}, StaticTile<4, 4>{}, P0, P1);

  auto P0h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P0);
  auto P1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P1);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      EXPECT_FLOAT_EQ(P0h(i, j), Ah(i, j) * Bh(i, j))
          << "i=" << i << " j=" << j;
      EXPECT_FLOAT_EQ(P1h(i, j),
                      Ah(i, j) - Bh(i, j) + static_cast<float>(i - j))
          << "i=" << i << " j=" << j;
    }
}

TEST(CombineNodeTest, MultiOutputPermutedOperandTeam) {
  // Multi-output x permuted operand composed: fn returns Kokkos::Array<float,2>
  // while B is stored transposed (axes j,i). The single-feature tests cover the
  // NumOut > 1 output path and the label_perm_seq + canonicalize_input staging
  // path only separately; this exercises them together. Non-square I != J so a
  // missed transpose is not masked.
  //   p0 = A{i,j} * B{j,i} ,  p1 = A{i,j} - B{j,i} + (i - j).
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 12, J = 8;
  View2         A("A", I, J);  // axes i,j
  View2         B("B", J, I);  // axes j,i (permuted)
  View2         P0("P0", I, J);
  View2         P1("P1", I, J);

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      Ah(i, j) = static_cast<float>((i + 2 * j) % 5 + 1) * 0.5f;
  for (int j = 0; j < J; ++j)
    for (int i = 0; i < I; ++i)
      Bh(j, i) = static_cast<float>((3 * j + i) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(P0, 0.0f);
  Kokkos::deep_copy(P1, 0.0f);

  auto na = make_input_node(make_handle<'i', 'j'>(A));
  auto nb = make_input_node(make_handle<'j', 'i'>(B));
  auto g  = make_graph();

  auto [g1, p0, p1] = g.ops(make_combine_node<'i', 'j'>(na, nb, SumDiff{}));

  g1.execute(TeamPolicyTag<>{}, StaticTile<4, 4>{}, P0, P1);

  auto P0h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P0);
  auto P1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P1);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      EXPECT_FLOAT_EQ(P0h(i, j), Ah(i, j) * Bh(j, i))
          << "i=" << i << " j=" << j;
      EXPECT_FLOAT_EQ(P1h(i, j),
                      Ah(i, j) - Bh(j, i) + static_cast<float>(i - j))
          << "i=" << i << " j=" << j;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
