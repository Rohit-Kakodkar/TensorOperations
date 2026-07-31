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

// A do-nothing store hook. Used only to give a node a non-NoHook hook_type in
// the compile-time staging-rule assertions below.
struct NoopHook {
  KOKKOS_FUNCTION void operator()(int, int, float&) const {}
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
  // contraction's permC is identity.
  auto contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i',
                                 'l'>(
        make_input_node(make_handle<'i', 'j', 'k'>(A)),
        make_input_node(make_handle<'j', 'k', 'l'>(B)));
  }

  // The SAME contraction declared in a non-canonical position: identical
  // operands, so freeA ++ freeB is still [i,l], but the user output order is
  // <l,i>, giving permC == [1,0]. The combine matches operands on their
  // CANONICAL modes, so this operand's label gather stays identity and it is
  // consumed by the plain zero-copy passthrough -- what differs is purely the
  // frame its tile bundle is spelled in (below).
  auto noncanonical_contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'l',
                                 'i'>(
        make_input_node(make_handle<'i', 'j', 'k'>(A)),
        make_input_node(make_handle<'j', 'k', 'l'>(B)));
  }

  auto d_node() const { return make_input_node(make_handle<'i', 'l'>(D)); }
  static auto combine_tile() {
    return make_combine_tile(OutTile{}, CBundle{}, OutTile{});
  }

  // Bundle for the non-canonical node. A tile bundle's `.c` slot is always in
  // its own node's USER order, so here it is L x I -- the transpose of the
  // 16x32 scratch that contraction actually writes. The evaluator recovers the
  // canonical shape by reading `.c` through permC (Impl::operand_leaf_tile_t).
  using NCBundle =
      Tile<StaticTile<16, 2, 4>, StaticTile<2, 4, 32>, StaticTile<32, 16>>;
  static auto noncanonical_combine_tile() {
    return make_combine_tile(OutTile{}, NCBundle{}, OutTile{});
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

// ---------------------------------------------------------------------------
// Combine with a NON-CANONICAL contraction operand (permC != identity).
//
// Same numbers as ContractionOperandTeam -- only the operand's declared output
// order changes, from <i,l> to <l,i>. That leaves its canonical (freeA++freeB)
// order, its scratch, and its label gather into the combine untouched; the one
// thing that moves is the frame its tile bundle's `.c` slot is written in.
// Isolates requirement (2): the evaluator must read that slot through permC
// rather than assuming it is already canonical. Rejected before this change by
// the combine evaluator's identity-permC static_assert.
//
// The combine analogue of GraphTest.FusedOperandNonIdentityPermCTeam.
// ---------------------------------------------------------------------------
TEST(CombineNodeTest, NonCanonicalContractionOperandTeam) {
  const ContractionCombineData  d;
  ContractionCombineData::View2 P("P", d.I, d.L);
  Kokkos::deep_copy(P, 0.0f);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'l'>(
      d.noncanonical_contraction_node(), d.d_node(), MulPlusCoord{}));
  g1.execute(TeamPolicyTag<>{}, d.noncanonical_combine_tile(), P);

  // Identical expected values to ContractionOperandTeam: re-declaring the
  // operand's output order must not change what it computes.
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
// Combine with a PERMUTED contraction operand, at UNEQUAL extents.
//
//   c{l,i} = sum_{j,k} A{l,j,k} * B{j,k,i}      (canonical order [l,i])
//   P{i,l} = c{l,i} * D{i,l} + 100*i + l        (combine output order [i,l])
//
// The operand's canonical order is the reverse of the combine's output order,
// so its label gather is [1,0] and its 64x32 C scratch must be read as 32x64.
// Isolates requirement (1), and does so at extents that can ONLY be served by
// the zero-copy relabel: an in-place axis swap is a self-map of a fixed buffer
// only at equal extents, so the path the contraction evaluator uses for its own
// permuted fused operands could never stage this (asserted below). That is the
// capability this test exists to pin, over and above the refactor.
//
// Register-kernel constraints apply to the operand in ITS canonical order, so
// with A and B swapped relative to ContractionCombineData it is I -- not L --
// that carries the output-column (NR) constraint: SA = L = 64 (%MT=8),
// flattened SK = J*K = 8 (%NT=8), SB = I = 32 (%NR up to 32). Team scratch is
// 2048+1024+8192 (operand a+b+c) + 8192 (D staging) + 8192 (output) = 27 KB,
// inside the serial backend's ~32 KB cap.
// ---------------------------------------------------------------------------
namespace {
struct PermutedContractionCombineData {
  using View2            = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3            = Kokkos::View<float***, Kokkos::LayoutRight>;
  static constexpr int I = 32, J = 2, K = 4, L = 64;
  using OutTile = StaticTile<32, 64>;  // combine output <i,l>
  // `.c` is 64x32: the operand's own canonical (and user) order [l,i].
  using CBundle =
      Tile<StaticTile<64, 2, 4>, StaticTile<2, 4, 32>, StaticTile<64, 32>>;

  View3                   A{"A", L, J, K};
  View3                   B{"B", J, K, I};
  View2                   D{"D", I, L};
  View3::host_mirror_type Ah{Kokkos::create_mirror_view(A)};
  View3::host_mirror_type Bh{Kokkos::create_mirror_view(B)};
  View2::host_mirror_type Dh{Kokkos::create_mirror_view(D)};

  PermutedContractionCombineData() {
    for (int l = 0; l < L; ++l)
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          Ah(l, j, k) =
              static_cast<float>(((l * 7 + j * 3 + k) % 5) + 1) * 0.25f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k)
        for (int i = 0; i < I; ++i)
          Bh(j, k, i) =
              static_cast<float>(((j * 5 + k * 2 + i) % 4) + 1) * 0.5f;
    for (int i = 0; i < I; ++i)
      for (int l = 0; l < L; ++l)
        Dh(i, l) = static_cast<float>((i + 3 * l) % 7 + 1) * 0.125f;
    Kokkos::deep_copy(A, Ah);
    Kokkos::deep_copy(B, Bh);
    Kokkos::deep_copy(D, Dh);
  }

  // freeA(l) ++ freeB(i) == user output <l,i>, so permC is identity here and
  // the only thing out of step with the combine is the label gather.
  auto contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'l',
                                 'i'>(
        make_input_node(make_handle<'l', 'j', 'k'>(A)),
        make_input_node(make_handle<'j', 'k', 'i'>(B)));
  }
  // The same node carrying a hook. Not executed -- it exists so the static_
  // asserts below can pin that hooking an operand takes it off the relabel
  // path, which is otherwise only observable by reading the evaluator.
  auto hooked_contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'l',
                                 'i'>(
        make_input_node(make_handle<'l', 'j', 'k'>(A)),
        make_input_node(make_handle<'j', 'k', 'i'>(B)), NoopHook{});
  }

  auto d_node() const { return make_input_node(make_handle<'i', 'l'>(D)); }
  static auto combine_tile() {
    return make_combine_tile(OutTile{}, CBundle{}, OutTile{});
  }

  // Host reference for the contraction operand, in ITS order: c{l,i}.
  float host_c(int l, int i) const {
    float c = 0.0f;
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k) c += Ah(l, j, k) * Bh(j, k, i);
    return c;
  }
};

// The divergence between the two staging rules, asserted in both directions
// over the exact node/tile/perm the test below runs.
//
// Impl::operand_stageable_v is what the CONTRACTION evaluator gates its own
// operands on, and it rejects this one: a 64x32 tile has no room to become
// 32x16 inside its own storage. Impl::operand_relabelable_v is the rule the
// COMBINE evaluator uses instead, and it accepts it, because a relabel moves no
// data and so has nothing to say about extents. The test below proves that
// acceptance is correct.
//
// Asserting both is what keeps the two from being "unified" later: relaxing
// operand_stageable_v would let the contraction evaluator through to an
// in-place reorder it cannot perform.
using PermutedFusedOp =
    decltype(std::declval<PermutedContractionCombineData>().contraction_node());
using PermutedFusedTile = PermutedContractionCombineData::CBundle;
using PermutedFusedPerm =
    Impl::label_perm_seq_t<std::integer_sequence<int32_t, 'i', 'l'>,
                           PermutedFusedOp::modes_seq>;
static_assert(
    std::is_same_v<PermutedFusedPerm, std::integer_sequence<int, 1, 0>>,
    "operand canonical order is [l,i]; gathering it into the "
    "combine's [i,l] must be the transposition");
static_assert(!Impl::operand_stageable_v<PermutedFusedOp, PermutedFusedTile,
                                         PermutedFusedPerm>,
              "a 64x32 fused operand tile must NOT be stageable under a "
              "transposing perm -- it cannot be reordered in place");
static_assert(Impl::operand_relabelable_v<PermutedFusedOp, PermutedFusedPerm>,
              "the same operand MUST be relabelable -- that is the path the "
              "combine evaluator reaches it by, and the reason the test below "
              "compiles at all");

// The three ways out of the relabel path, each pinned. Together with the
// positive above these are the whole of operand_relabelable_v.
static_assert(!Impl::operand_relabelable_v<PermutedFusedOp,
                                           std::integer_sequence<int, 0, 1>>,
              "an unpermuted operand stays on the zero-copy passthrough");
static_assert(
    !Impl::operand_relabelable_v<
        decltype(std::declval<PermutedContractionCombineData>().d_node()),
        PermutedFusedPerm>,
    "an INPUT operand must never be relabeled -- its storage is the user's "
    "global tensor, not private scratch");
static_assert(
    !Impl::operand_relabelable_v<
        decltype(std::declval<PermutedContractionCombineData>()
                     .hooked_contraction_node()),
        PermutedFusedPerm>,
    "a HOOKED operand stays on the staging path, where the hook is already "
    "written back into its own scratch");
}  // namespace

TEST(CombineNodeTest, PermutedContractionOperandTeam) {
  const PermutedContractionCombineData  d;
  PermutedContractionCombineData::View2 P("P", d.I, d.L);
  Kokkos::deep_copy(P, 0.0f);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'l'>(
      d.contraction_node(), d.d_node(), MulPlusCoord{}));
  g1.execute(TeamPolicyTag<>{}, d.combine_tile(), P);

  auto Ph = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, P);
  for (int i = 0; i < d.I; ++i)
    for (int l = 0; l < d.L; ++l) {
      // host_c is indexed in the OPERAND's order (l,i) while the output is
      // (i,l): a missed relabel shows up here, and cannot hide behind a square
      // tile because I != L.
      const float expected = d.host_c(l, i) * d.Dh(i, l) +
                             static_cast<float>(i) * 100.0f +
                             static_cast<float>(l);
      EXPECT_FLOAT_EQ(Ph(i, l), expected) << "i=" << i << " l=" << l;
    }
}

// ---------------------------------------------------------------------------
// Combine node AS AN OPERAND (phase 1: single-output combines).
//
//   P{i,k} = A{i,k} * D{i,k} + 100i + k        (combine, the fused operand)
//   c{i,l} = sum_k P{i,k} * B{k,l}             (contraction consuming it)
//
// K = 16 against a k-tile of 8 deliberately gives the parent TWO contracted
// tiles, so the combine evaluator is re-invoked per k-tile -- the path that
// distinguishes a fused operand from a one-shot one, and the reason its
// per-operand allocators are built once in the constructor.
//
// Register-kernel divisibility, both backends (CPU MT=8/NT=8/NR=2*W=32; GPU
// MT=4/NT=2/NR=2): SA=16, SK=8, SB=32.
//
// Every array is index-dependent and non-symmetric, and I != L, so a dropped
// relabel or a transposed index cannot pass.
// ---------------------------------------------------------------------------
namespace {
struct CombineOperandData {
  using View2            = Kokkos::View<float**, Kokkos::LayoutRight>;
  static constexpr int I = 16, K = 16, L = 32;

  // Tiles: the combine's own bundle (both its operands are inputs, so their
  // tiles equal its output tile), nested into the contraction's A slot.
  using OpTile   = StaticTile<16, 8>;  // combine output tile <i,k>
  using CombTile = CombineTile<OpTile, OpTile, OpTile>;
  using CBundle  = Tile<CombTile, StaticTile<8, 32>, StaticTile<16, 32>>;

  View2                   A{"A", I, K};
  View2                   D{"D", I, K};
  View2                   B{"B", K, L};
  View2::host_mirror_type Ah{Kokkos::create_mirror_view(A)};
  View2::host_mirror_type Dh{Kokkos::create_mirror_view(D)};
  View2::host_mirror_type Bh{Kokkos::create_mirror_view(B)};

  CombineOperandData() {
    for (int i = 0; i < I; ++i)
      for (int k = 0; k < K; ++k) {
        Ah(i, k) = static_cast<float>(((i * 7 + k * 3) % 5) + 1) * 0.25f;
        Dh(i, k) = static_cast<float>(((i * 2 + k * 5) % 7) + 1) * 0.125f;
      }
    for (int k = 0; k < K; ++k)
      for (int l = 0; l < L; ++l)
        Bh(k, l) = static_cast<float>(((k * 5 + l * 2) % 4) + 1) * 0.5f;
    Kokkos::deep_copy(A, Ah);
    Kokkos::deep_copy(D, Dh);
    Kokkos::deep_copy(B, Bh);
  }

  auto combine_node() const {
    return make_combine_node<'i', 'k'>(
        make_input_node(make_handle<'i', 'k'>(A)),
        make_input_node(make_handle<'i', 'k'>(D)), MulPlusCoord{});
  }
  auto b_node() const { return make_input_node(make_handle<'k', 'l'>(B)); }

  // c{i,l} = sum_k P{i,k} B{k,l}; canonical (freeA ++ freeB) == user <i,l>.
  auto contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i',
                                 'l'>(combine_node(), b_node());
  }

  float host_p(int i, int k) const {
    return Ah(i, k) * Dh(i, k) + static_cast<float>(i) * 100.0f +
           static_cast<float>(k);
  }
  float host_c(int i, int l) const {
    float c = 0.0f;
    for (int k = 0; k < K; ++k) c += host_p(i, k) * Bh(k, l);
    return c;
  }
};

// The staging rules over the exact node/tile/perm a fused combine operand
// reaches them by -- the regression guard for the predicate fix this feature
// turned on.
//
// operand_stageable_v used to answer "is it a contraction?" and so returned
// TRUE for any combine operand, on the reasoning that everything else is
// copied into a fresh staging buffer and therefore shape-unconstrained. That
// is wrong: a combine produces its OWN scratch, so a consumer reorders it in
// place, and a 16x8 tile cannot become 8x16 inside its own storage. Asserting
// the false here pins the fix -- a hard error from inside an evaluator's class
// body is otherwise unobservable to a test.
using FusedCombineOp =
    decltype(std::declval<CombineOperandData>().combine_node());
using FusedCombineTile = CombineOperandData::CombTile;
using FusedCombineSwap = std::integer_sequence<int, 1, 0>;
static_assert(Impl::produces_own_scratch_v<FusedCombineOp>,
              "a combine node hands back its own scratch, exactly as a fused "
              "contraction does");
static_assert(Impl::fusable_operand_v<FusedCombineOp>,
              "and is therefore a legal operand");
static_assert(!Impl::operand_stageable_v<FusedCombineOp, FusedCombineTile,
                                         FusedCombineSwap>,
              "a 16x8 fused combine operand tile must NOT be stageable under a "
              "transposing perm -- it would be reordered in place");
static_assert(Impl::operand_stageable_v<FusedCombineOp, FusedCombineTile,
                                        std::integer_sequence<int, 0, 1>>,
              "the identity perm is a true zero-copy passthrough and stays "
              "stageable");
static_assert(Impl::operand_relabelable_v<FusedCombineOp, FusedCombineSwap>,
              "a permuted combine operand IS relabelable -- it carries no hook "
              "at all, so the unhooked half of the rule is vacuous");
static_assert(!Impl::operand_relabelable_v<FusedCombineOp,
                                           std::integer_sequence<int, 0, 1>>,
              "an unpermuted one stays on the zero-copy passthrough");

// Phase-1 gate. A multi-output combine (SumDiff -> Kokkos::Array<float,2>) is
// still a legal graph output but NOT a legal operand: instantiating an
// evaluator over one fails in Impl::single_result with
//   "a multi-output combine node cannot yet be consumed as an operand"
// Left as prose rather than a test -- the repo has no negative-compilation
// harness, and a live case would simply not build.
}  // namespace

// A combine node in the contraction's A slot.
TEST(CombineNodeTest, ContractionOverCombineOperandTeam) {
  const CombineOperandData  d;
  CombineOperandData::View2 C("C", d.I, d.L);
  Kokkos::deep_copy(C, 0.0f);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(d.contraction_node());
  const int wk =
      g1.execute(TeamPolicyTag<>{}, CombineOperandData::CBundle{}, C);
  EXPECT_EQ(wk, 1);  // one 16x32 output tile

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < d.I; ++i)
    for (int l = 0; l < d.L; ++l)
      EXPECT_FLOAT_EQ(Ch(i, l), d.host_c(i, l)) << "i=" << i << " l=" << l;
}

// ---------------------------------------------------------------------------
// The same combine in the contraction's B slot, since a fused operand may sit
// in either position:
//   Q{k,l} = E{k,l} * F{k,l} + 100k + l
//   c{i,l} = sum_k A{i,k} * Q{k,l}
// ---------------------------------------------------------------------------
namespace {
struct CombineOperandBData {
  using View2            = Kokkos::View<float**, Kokkos::LayoutRight>;
  static constexpr int I = 16, K = 16, L = 32;

  using OpTile   = StaticTile<8, 32>;  // combine output tile <k,l>
  using CombTile = CombineTile<OpTile, OpTile, OpTile>;
  using CBundle  = Tile<StaticTile<16, 8>, CombTile, StaticTile<16, 32>>;

  View2                   A{"A", I, K};
  View2                   E{"E", K, L};
  View2                   F{"F", K, L};
  View2::host_mirror_type Ah{Kokkos::create_mirror_view(A)};
  View2::host_mirror_type Eh{Kokkos::create_mirror_view(E)};
  View2::host_mirror_type Fh{Kokkos::create_mirror_view(F)};

  CombineOperandBData() {
    for (int i = 0; i < I; ++i)
      for (int k = 0; k < K; ++k)
        Ah(i, k) = static_cast<float>(((i * 3 + k * 7) % 6) + 1) * 0.25f;
    for (int k = 0; k < K; ++k)
      for (int l = 0; l < L; ++l) {
        Eh(k, l) = static_cast<float>(((k * 5 + l * 3) % 4) + 1) * 0.5f;
        Fh(k, l) = static_cast<float>(((k + l * 2) % 5) + 1) * 0.125f;
      }
    Kokkos::deep_copy(A, Ah);
    Kokkos::deep_copy(E, Eh);
    Kokkos::deep_copy(F, Fh);
  }

  auto contraction_node() const {
    return make_contraction_node<float, Kokkos::DefaultExecutionSpace, 'i',
                                 'l'>(
        make_input_node(make_handle<'i', 'k'>(A)),
        make_combine_node<'k', 'l'>(make_input_node(make_handle<'k', 'l'>(E)),
                                    make_input_node(make_handle<'k', 'l'>(F)),
                                    MulPlusCoord{}));
  }

  float host_q(int k, int l) const {
    return Eh(k, l) * Fh(k, l) + static_cast<float>(k) * 100.0f +
           static_cast<float>(l);
  }
  float host_c(int i, int l) const {
    float c = 0.0f;
    for (int k = 0; k < K; ++k) c += Ah(i, k) * host_q(k, l);
    return c;
  }
};
}  // namespace

TEST(CombineNodeTest, ContractionOverCombineOperandBTeam) {
  const CombineOperandBData  d;
  CombineOperandBData::View2 C("C", d.I, d.L);
  Kokkos::deep_copy(C, 0.0f);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(d.contraction_node());
  g1.execute(TeamPolicyTag<>{}, CombineOperandBData::CBundle{}, C);

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < d.I; ++i)
    for (int l = 0; l < d.L; ++l)
      EXPECT_FLOAT_EQ(Ch(i, l), d.host_c(i, l)) << "i=" << i << " l=" << l;
}

// ---------------------------------------------------------------------------
// A combine consuming a combine (zero-copy passthrough -- identical modes, so
// the label gather is the identity):
//   P{i,j} = A{i,j} * B{i,j} + 100i + j
//   Q{i,j} = P{i,j} + C{i,j}
// ---------------------------------------------------------------------------
TEST(CombineNodeTest, CombineOverCombineOperandTeam) {
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 16, J = 32;
  View2         A("A", I, J), B("B", I, J), C("C", I, J), Q("Q", I, J);
  auto          Ah = Kokkos::create_mirror_view(A);
  auto          Bh = Kokkos::create_mirror_view(B);
  auto          Ch = Kokkos::create_mirror_view(C);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      Ah(i, j) = static_cast<float>(((i * 7 + j * 3) % 5) + 1) * 0.25f;
      Bh(i, j) = static_cast<float>(((i * 2 + j * 5) % 7) + 1) * 0.125f;
      Ch(i, j) = static_cast<float>(((i + j * 3) % 4) + 1) * 0.5f;
    }
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);
  Kokkos::deep_copy(C, Ch);
  Kokkos::deep_copy(Q, 0.0f);

  using OutTile   = StaticTile<8, 16>;
  using InnerTile = CombineTile<OutTile, OutTile, OutTile>;

  auto inner = make_combine_node<'i', 'j'>(
      make_input_node(make_handle<'i', 'j'>(A)),
      make_input_node(make_handle<'i', 'j'>(B)), MulPlusCoord{});

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'j'>(
      inner, make_input_node(make_handle<'i', 'j'>(C)), AddOp{}));
  g1.execute(TeamPolicyTag<>{},
             make_combine_tile(OutTile{}, InnerTile{}, OutTile{}), Q);

  auto Qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Q);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      const float p = Ah(i, j) * Bh(i, j) + static_cast<float>(i) * 100.0f +
                      static_cast<float>(j);
      EXPECT_FLOAT_EQ(Qh(i, j), p + Ch(i, j)) << "i=" << i << " j=" << j;
    }
}

// ---------------------------------------------------------------------------
// A PERMUTED combine operand, consumed by the zero-copy relabel path:
//   P{l,i} = E{l,i} * F{l,i} + 100l + i      (declared in <l,i> order)
//   Q{i,l} = P{l,i} * D{i,l} + 100i + l      (consumed in <i,l> order)
//
// A combine node's modes ARE its canonical order, so there is no permC here --
// the whole permutation is the outer combine's label gather. I != L, so the
// tile is not square and the relabel (rather than an in-place reorder) is the
// only path that can work; operand_relabelable_v is what selects it.
// ---------------------------------------------------------------------------
TEST(CombineNodeTest, PermutedCombineOperandTeam) {
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  constexpr int I = 16, L = 32;
  View2         E("E", L, I), F("F", L, I), D("D", I, L), Q("Q", I, L);
  auto          Eh = Kokkos::create_mirror_view(E);
  auto          Fh = Kokkos::create_mirror_view(F);
  auto          Dh = Kokkos::create_mirror_view(D);
  for (int l = 0; l < L; ++l)
    for (int i = 0; i < I; ++i) {
      Eh(l, i) = static_cast<float>(((l * 3 + i * 7) % 5) + 1) * 0.25f;
      Fh(l, i) = static_cast<float>(((l * 5 + i * 2) % 7) + 1) * 0.125f;
    }
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l)
      Dh(i, l) = static_cast<float>(((i + 3 * l) % 7) + 1) * 0.5f;
  Kokkos::deep_copy(E, Eh);
  Kokkos::deep_copy(F, Fh);
  Kokkos::deep_copy(D, Dh);
  Kokkos::deep_copy(Q, 0.0f);

  using OutTile     = StaticTile<16, 32>;  // <i,l>
  using InnerTile   = StaticTile<32, 16>;  // <l,i> -- the transposition
  using InnerBundle = CombineTile<InnerTile, InnerTile, InnerTile>;

  auto inner = make_combine_node<'l', 'i'>(
      make_input_node(make_handle<'l', 'i'>(E)),
      make_input_node(make_handle<'l', 'i'>(F)), MulPlusCoord{});

  // The operand really is reached by the relabel, not by a copy.
  static_assert(Impl::operand_relabelable_v<decltype(inner),
                                            std::integer_sequence<int, 1, 0>>);

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_combine_node<'i', 'l'>(
      inner, make_input_node(make_handle<'i', 'l'>(D)), MulPlusCoord{}));
  g1.execute(TeamPolicyTag<>{},
             make_combine_tile(OutTile{}, InnerBundle{}, OutTile{}), Q);

  auto Qh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Q);
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l) {
      // host_p is indexed in the OPERAND's order (l,i) while the output is
      // (i,l): a missed relabel shows up here.
      const float p = Eh(l, i) * Fh(l, i) + static_cast<float>(l) * 100.0f +
                      static_cast<float>(i);
      const float expected =
          p * Dh(i, l) + static_cast<float>(i) * 100.0f + static_cast<float>(l);
      EXPECT_FLOAT_EQ(Qh(i, l), expected) << "i=" << i << " l=" << l;
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
