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

// ---------------------------------------------------------------------------
// Shared setup for the contraction-operand tests:
//   c{i,l} = sum_{j,k} A{i,j,k} * B{j,k,l}   (contraction operand)
//   D{i,l}                                   (plain input operand)
// Exercises the CombineTile bundle path (per-operand tile specs): operand 0
// gets a Tile<A,B,C> for the inner contraction, operand 1 gets the combine
// output tile for the plain input D. The contraction's canonical output tile
// must equal the combine output tile so all operand scratch views collapse to
// the same scratch_view_t.
//
// Static-tile register-kernel constraints on CPU: the contraction operand's
// SA (output row extent) must be a multiple of MT=8, flattened SK a multiple
// of NT=8, and SB (output col extent) a multiple of NR=2*simd_width (32 on
// AVX-512). Team scratch on the serial backend is capped near 32 KB, so we
// keep the K stage small: J=2, K=4 gives SK=8 (minimum) and shrinks the
// per-team footprint below the cap. Tile choices: SA=16 (%MT=8), flattened
// SK=8*32=256 (%NT=8), SB=32 (%NR up to 32).
// ---------------------------------------------------------------------------
namespace {
struct ContractionCombineData {
  using View2            = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3            = Kokkos::View<float***, Kokkos::LayoutRight>;
  static constexpr int I = 16, J = 2, K = 4, L = 32;
  using OutTile = StaticTile<16, 32>;
  using CBundle =
      Tile<StaticTile<16, 2, 4>, StaticTile<2, 4, 32>, StaticTile<16, 32>>;

  View3                   A{"A", I, J, K};
  View3                   B{"B", J, K, L};
  View2                   D{"D", I, L};
  View3::host_mirror_type Ah{Kokkos::create_mirror_view(A)};
  View3::host_mirror_type Bh{Kokkos::create_mirror_view(B)};
  View2::host_mirror_type Dh{Kokkos::create_mirror_view(D)};

  ContractionCombineData() {
    for (int i = 0; i < I; ++i)
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          Ah(i, j, k) =
              static_cast<float>(((i * 7 + j * 3 + k) % 5) + 1) * 0.25f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k)
        for (int l = 0; l < L; ++l)
          Bh(j, k, l) =
              static_cast<float>(((j * 5 + k * 2 + l) % 4) + 1) * 0.5f;
    for (int i = 0; i < I; ++i)
      for (int l = 0; l < L; ++l)
        Dh(i, l) = static_cast<float>((i + 3 * l) % 7 + 1) * 0.125f;
    Kokkos::deep_copy(A, Ah);
    Kokkos::deep_copy(B, Bh);
    Kokkos::deep_copy(D, Dh);
  }

  // Canonical output order: freeA(i) ++ freeB(l) == user output <i,l>, so the
  // contraction's permC is identity (required by the combine evaluator).
  auto contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i',
                                 'l'>(
        make_input_node(make_handle<'i', 'j', 'k'>(A)),
        make_input_node(make_handle<'j', 'k', 'l'>(B)));
  }
  auto d_node() const { return make_input_node(make_handle<'i', 'l'>(D)); }
  static auto combine_tile() {
    return make_combine_tile(OutTile{}, CBundle{}, OutTile{});
  }

  // Host reference for the contraction operand c{i,l}.
  float host_c(int i, int l) const {
    float c = 0.0f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k) c += Ah(i, j, k) * Bh(j, k, l);
    return c;
  }
};
}  // namespace

// ---------------------------------------------------------------------------
// Combine with a contraction operand:
//   P{i,l} = fn(c{i,l}, D{i,l}) = c{i,l} * D{i,l} + 100*i + l
// ---------------------------------------------------------------------------
TEST(CombineNodeTest, ContractionOperandTeam) {
  const ContractionCombineData  d;
  ContractionCombineData::View2 P("P", d.I, d.L);
  Kokkos::deep_copy(P, 0.0f);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'l'>(
      d.contraction_node(), d.d_node(), MulPlusCoord{}));
  g1.execute(TeamPolicyTag<>{}, d.combine_tile(), P);

  auto Ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  for (int i = 0; i < d.I; ++i)
    for (int l = 0; l < d.L; ++l) {
      const float expected = d.host_c(i, l) * d.Dh(i, l) +
                             static_cast<float>(i) * 100.0f +
                             static_cast<float>(l);
      EXPECT_FLOAT_EQ(Ph(i, l), expected) << "i=" << i << " l=" << l;
    }
}

// ---------------------------------------------------------------------------
// Multi-output combine with a contraction operand: fn returns
// Kokkos::Array<float,2>, and operand 0 is a contraction. Verifies that the
// NumOut > 1 output path composes with the contraction-operand staging path.
//   p0{i,l} = c{i,l} * D{i,l} + 100*i + l    (uses coord + both operands)
//   p1{i,l} = c{i,l} - D{i,l} + (i - l)      (independent formula)
// Multi-output only adds one extra output tile to the scratch total.
// ---------------------------------------------------------------------------
struct MulPlusCoordAndSubDiff {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int i, int l, float c,
                                                     float d) const {
    return {c * d + static_cast<float>(i) * 100.0f + static_cast<float>(l),
            c - d + static_cast<float>(i - l)};
  }
};

TEST(CombineNodeTest, MultiOutputContractionOperandTeam) {
  const ContractionCombineData  d;
  ContractionCombineData::View2 P0("P0", d.I, d.L);
  ContractionCombineData::View2 P1("P1", d.I, d.L);
  Kokkos::deep_copy(P0, 0.0f);
  Kokkos::deep_copy(P1, 0.0f);

  auto g            = make_graph();
  auto [g1, p0, p1] = g.ops(make_combine_node<'i', 'l'>(
      d.contraction_node(), d.d_node(), MulPlusCoordAndSubDiff{}));
  static_assert(decltype(p0)::OutputIndex == 0);
  static_assert(decltype(p1)::OutputIndex == 1);

  g1.execute(TeamPolicyTag<>{}, d.combine_tile(), P0, P1);

  auto P0h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P0);
  auto P1h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P1);
  for (int i = 0; i < d.I; ++i)
    for (int l = 0; l < d.L; ++l) {
      const float c  = d.host_c(i, l);
      const float dv = d.Dh(i, l);
      const float expected0 =
          c * dv + static_cast<float>(i) * 100.0f + static_cast<float>(l);
      const float expected1 = c - dv + static_cast<float>(i - l);
      EXPECT_FLOAT_EQ(P0h(i, l), expected0) << "p0 i=" << i << " l=" << l;
      EXPECT_FLOAT_EQ(P1h(i, l), expected1) << "p1 i=" << i << " l=" << l;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
