#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/Graph.hpp>
#include <TensorOperations/Tiling.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time einsum plan checks (gather convention: perm[i] is the user axis
// that becomes canonical axis i). For A[j,i,k] * B[k,j,l] -> C[l,i]:
// contracted = {j,k} in A's order (Korder), freeA = {i}, freeB = {l}, so
// canonical A = [i,j,k], canonical B = [j,k,l], canonical C = [i,l].
// ---------------------------------------------------------------------------
namespace {
using AM = std::integer_sequence<int32_t, 'j', 'i', 'k'>;
using BM = std::integer_sequence<int32_t, 'k', 'j', 'l'>;
using OM = std::integer_sequence<int32_t, 'l', 'i'>;
static_assert(std::is_same_v<Impl::permA_seq_t<AM, BM>,
                             std::integer_sequence<int, 1, 0, 2>>);
static_assert(std::is_same_v<Impl::permB_seq_t<AM, BM>,
                             std::integer_sequence<int, 1, 0, 2>>);
static_assert(std::is_same_v<Impl::canonC_modes_seq_t<2, AM, BM>,
                             std::integer_sequence<int32_t, 'i', 'l'>>);
// canonical C mode i sits at user position 1, mode l at user position 0.
static_assert(std::is_same_v<Impl::permC_seq_t<2, AM, BM, OM>,
                             std::integer_sequence<int, 1, 0>>);
static_assert(std::is_same_v<
              Impl::inverse_perm_seq_t<std::integer_sequence<int, 1, 2, 0>>,
              std::integer_sequence<int, 2, 0, 1>>);
// Identity plans collapse to identity sequences (fast-path guarantee).
using CM = std::integer_sequence<int32_t, 'i', 'j', 'k'>;
using DM = std::integer_sequence<int32_t, 'j', 'k', 'l'>;
static_assert(Impl::is_identity_seq(Impl::permA_seq_t<CM, DM>{}));
static_assert(Impl::is_identity_seq(Impl::permB_seq_t<CM, DM>{}));
}  // namespace

// ---------------------------------------------------------------------------
// Tests operate on Kokkos::View tensors so value_type propagates correctly.
// ---------------------------------------------------------------------------

TEST(GraphTest, SingleContractionNode) {
  // A_{i,j,k}: (3, 4, 5)   B_{j,k,l}: (4, 5, 6)  →  C_{i,l}: (3, 6)
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View3 A("A", 3, 4, 5);
  View3 B("B", 4, 5, 6);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g          = make_graph();
  auto [g1, out1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  auto [T1]       = out1;

  static_assert(decltype(T1)::Rank == 2);
  static_assert(decltype(T1)::NumContracted == 2);
  static_assert(std::is_same_v<decltype(T1)::value_type, float>);
  static_assert(std::is_same_v<decltype(T1)::modes_seq,
                               std::integer_sequence<int32_t, 'i', 'l'>>);

  auto s = T1.shape();
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 6);

  (void)g1;
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

  auto hA = make_input_node(make_handle<'i', 'k', 'm'>(A));
  auto hB = make_input_node(make_handle<'k', 'm', 'j'>(B));
  auto hC = make_input_node(make_handle<'p', 'q', 'r'>(C));
  auto hD = make_input_node(make_handle<'q', 'r', 's'>(D));

  auto g = make_graph();

  auto [g1, lvl1] = g.ops(make_contraction_node<'i', 'j'>(hA, hB),
                          make_contraction_node<'p', 's'>(hC, hD));
  auto [T1, T2]   = lvl1;

  static_assert(decltype(T1)::Rank == 2);
  static_assert(decltype(T2)::Rank == 2);

  auto [g2, lvl2] = g1.ops(make_contraction_node<'i', 'j', 'p', 's'>(T1, T2));
  auto [T3]       = lvl2;

  static_assert(decltype(T3)::Rank == 4);
  static_assert(decltype(T3)::NumContracted == 0);  // outer product

  auto s = T3.shape();
  EXPECT_EQ(s[0], 3);
  EXPECT_EQ(s[1], 6);
  EXPECT_EQ(s[2], 2);
  EXPECT_EQ(s[3], 5);

  (void)g2;
}

TEST(GraphTest, ScalarInferredFromNode) {
  // Inferred scalar must flow from input through contraction
  using View2f = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  using View2d = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2f Af("Af", 3, 4);
  View2d Ad("Ad", 4, 5);

  auto hAf = make_input_node(make_handle<'i', 'j'>(Af));
  auto hAd = make_input_node(make_handle<'j', 'k'>(Ad));

  auto nf = make_contraction_node<'i', 'k'>(hAf, hAd);
  static_assert(std::is_same_v<decltype(nf)::value_type, float>);

  // Explicit override still works
  auto nd =
      make_contraction_node<double, Kokkos::DefaultExecutionSpace, 'i', 'k'>(
          hAf, hAd);
  static_assert(std::is_same_v<decltype(nd)::value_type, double>);
}

TEST(GraphTest, HookPreservesType) {
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2 A("A", 3, 4);
  View2 B("B", 4, 5);

  auto hA = make_input_node(make_handle<'i', 'j'>(A));
  auto hB = make_input_node(make_handle<'j', 'k'>(B));

  [[maybe_unused]] int call_count = 0;
  auto                 relu       = [&](int, int, float&) { ++call_count; };

  auto nc = make_contraction_node<'i', 'k'>(hA, hB, relu);

  // Hook type is the lambda — no type erasure
  static_assert(!std::is_same_v<decltype(nc)::hook_type, NoHook>);
  static_assert(!std::is_same_v<decltype(nc)::hook_type, void>);
}

TEST(GraphTest, SingleContractionExecutionTeam) {
  // Same contraction as SingleContractionExecution, run on the team/scratch
  // tier. C_{i,l} = sum_{j,k} A_{i,j,k} * B_{j,k,l}; A=B=1  →  C[i,l] = 4*4
  // = 16.
  // Tile extents are chosen so the staged 2-D GEMM dims divide the register-
  // block factors on both backends (SA=16, SK=4*4=16, SB=32): the kernel
  // assumes SA%MT==0, SK%NT==0, SB%NR==0 with no remainder path, and the CPU
  // column block NR=2*W is 32 on AVX-512, so SB must be at least 32.
  // Views use DefaultMemorySpace (CudaSpace when CUDA is enabled) so the
  // kernel can access them from the GPU.
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
  View3 A("A", 16, 4, 4);
  View3 B("B", 4, 4, 32);
  View2 C("C", 16, 32);

  Kokkos::deep_copy(A, 1.0f);
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  auto [T1]     = o1;

  // One team, one tile: A[i,j,k], B[j,k,l], C[i,l] each cover the full extent.
  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<16, 4, 4>, StaticTile<4, 4, 32>, StaticTile<16, 32>>{},
      C);

  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < 16; ++i)
    for (int l = 0; l < 32; ++l) EXPECT_FLOAT_EQ(C_host(i, l), 16.0f);
}

TEST(GraphTest, MultiTileExecutionTeam) {
  // Same contraction on the team tier, but tiled: the i mode splits into 3
  // output tiles (3 teams) and the contracted j mode splits into 2 accumulation
  // blocks, exercising multiple teams plus the k-tile reduction loop. Result is
  // C[i,l] = sum_{j,k} 1*1 = 4*4 = 16.
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
  View3 A("A", 48, 4, 4);
  View3 B("B", 4, 4, 32);
  View2 C("C", 48, 32);

  Kokkos::deep_copy(A, 1.0f);
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  auto [T1]     = o1;

  // A[i=48,j=4,k=4], B[j=4,k=4,l=32], C[i=48,l=32]: i → 3 tiles, j → 2 tiles
  // (all divide their extents). Per output tile SA=16, SB=32; per accumulation
  // block SK=2*4=8 — all divisible by the register-block factors on both
  // backends (SB=32 meets the CPU column block NR=2*W=32 on AVX-512). 3 output
  // tiles, 2 contracted-tile blocks per output.
  const std::size_t wk = g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<16, 2, 4>, StaticTile<2, 4, 32>, StaticTile<16, 32>>{},
      C);
  EXPECT_EQ(wk, 3u);  // one team per output tile of C[i,l] = [3,1]

  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < 48; ++i)
    for (int l = 0; l < 32; ++l) EXPECT_FLOAT_EQ(C_host(i, l), 16.0f);
}

TEST(GraphTest, NonSquareNonSymmetricTeam) {
  // Guards the class of bug that uniform (all-ones) inputs cannot detect:
  // a transposed or free-mode-swapped output is invisible when every element
  // is identical. Uses a *non-symmetric* deterministic fill AND a *non-square*
  // output tile (TI=64, TL=32) over a non-square problem (I=128, L=32), checked
  // against an independent host reference. TL=32 meets the CPU column block
  // NR=2*W=32 on AVX-512; I=128 splits into 2 output tiles along i.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3     = Kokkos::View<float***, Kokkos::LayoutRight>;
  constexpr int I = 128, J = 4, K = 4, L = 32;
  View3         A("A", I, J, K);
  View3         B("B", J, K, L);
  View2         C("C", I, L);

  // Fill on host (deterministic, non-symmetric) then copy to device — avoids a
  // device lambda inside GoogleTest's private TestBody.
  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k)
        Ah(i, j, k) = static_cast<float>((i + 2 * j + 3 * k) % 5 + 1) * 0.5f;
  for (int j = 0; j < J; ++j)
    for (int k = 0; k < K; ++k)
      for (int l = 0; l < L; ++l)
        Bh(j, k, l) = static_cast<float>((3 * j + k + 2 * l) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));
  auto [T1]     = o1;

  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<64, 4, 4>, StaticTile<4, 4, 32>, StaticTile<64, 32>>{},
      C);

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l) {
      double acc = 0.0;
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          acc += static_cast<double>(Ah(i, j, k)) * Bh(j, k, l);
      EXPECT_NEAR(Ch(i, l), static_cast<float>(acc), 1e-3f)
          << "i=" << i << " l=" << l;
    }
}

TEST(GraphTest, PermutedInputOutputTeam) {
  // A[j,i,k] * B[j,k,l] -> C[l,i]: the free mode 'i' sits in the middle of A
  // and the output is reversed. Exercises permA (canonicalize A to [i,j,k]) and
  // permC (write the canonical [i,l] result into the user's [l,i] view).
  // Non-symmetric data + a host reference guard against transposition bugs.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3     = Kokkos::View<float***, Kokkos::LayoutRight>;
  constexpr int I = 128, J = 4, K = 4, L = 32;
  View3         A("A", J, I, K);  // axes j,i,k
  View3         B("B", J, K, L);  // axes j,k,l
  View2         C("C", L, I);     // axes l,i

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int j = 0; j < J; ++j)
    for (int i = 0; i < I; ++i)
      for (int k = 0; k < K; ++k)
        Ah(j, i, k) = static_cast<float>((i + 2 * j + 3 * k) % 5 + 1) * 0.5f;
  for (int j = 0; j < J; ++j)
    for (int k = 0; k < K; ++k)
      for (int l = 0; l < L; ++l)
        Bh(j, k, l) = static_cast<float>((3 * j + k + 2 * l) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto hA       = make_input_node(make_handle<'j', 'i', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'l', 'i'>(hA, hB));

  // Tiles in user (data-axis) order: A[j,i,k]=<TJ,TI,TK>, B[j,k,l]=<TJ,TK,TL>,
  // C[l,i]=<TL,TI>.
  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<4, 64, 4>, StaticTile<4, 4, 32>, StaticTile<32, 64>>{},
      C);

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int l = 0; l < L; ++l)
    for (int i = 0; i < I; ++i) {
      double acc = 0.0;
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          acc += static_cast<double>(Ah(j, i, k)) * Bh(j, k, l);
      EXPECT_NEAR(Ch(l, i), static_cast<float>(acc), 1e-3f)
          << "l=" << l << " i=" << i;
    }
}

TEST(GraphTest, PermutedContractedOrderTeam) {
  // A[i,j,k] * B[k,j,l] -> C[i,l]: the contracted modes appear in opposite
  // order in A (j,k) vs B (k,j). Exercises permB reordering B's contracted axes
  // to match A's Korder. J != K so a mis-ordering is not masked.
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3     = Kokkos::View<float***, Kokkos::LayoutRight>;
  constexpr int I = 128, J = 8, K = 4, L = 32;
  View3         A("A", I, J, K);  // axes i,j,k
  View3         B("B", K, J, L);  // axes k,j,l
  View2         C("C", I, L);     // axes i,l

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k)
        Ah(i, j, k) = static_cast<float>((i + 2 * j + 3 * k) % 5 + 1) * 0.5f;
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < J; ++j)
      for (int l = 0; l < L; ++l)
        Bh(k, j, l) = static_cast<float>((3 * j + k + 2 * l) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'k', 'j', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));

  // Tiles in user order: A[i,j,k]=<TI,TJ,TK>, B[k,j,l]=<TK,TJ,TL>,
  // C[i,l]=<TI,TL>.
  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<64, 8, 4>, StaticTile<4, 8, 32>, StaticTile<64, 32>>{},
      C);

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < I; ++i)
    for (int l = 0; l < L; ++l) {
      double acc = 0.0;
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          acc += static_cast<double>(Ah(i, j, k)) * Bh(k, j, l);
      EXPECT_NEAR(Ch(i, l), static_cast<float>(acc), 1e-3f)
          << "i=" << i << " l=" << l;
    }
}

TEST(GraphTest, PermutedOutputOnlyTeam) {
  // A[i,j,k] * B[j,k,l] -> C[l,i]: inputs canonical, only the output is
  // reversed. Isolates permC (identity permA/permB).
  using View2     = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3     = Kokkos::View<float***, Kokkos::LayoutRight>;
  constexpr int I = 128, J = 4, K = 4, L = 32;
  View3         A("A", I, J, K);
  View3         B("B", J, K, L);
  View2         C("C", L, I);  // axes l,i

  auto Ah = Kokkos::create_mirror_view(A);
  auto Bh = Kokkos::create_mirror_view(B);
  for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < K; ++k)
        Ah(i, j, k) = static_cast<float>((i + 2 * j + 3 * k) % 5 + 1) * 0.5f;
  for (int j = 0; j < J; ++j)
    for (int k = 0; k < K; ++k)
      for (int l = 0; l < L; ++l)
        Bh(j, k, l) = static_cast<float>((3 * j + k + 2 * l) % 4 + 1) * 0.25f;
  Kokkos::deep_copy(A, Ah);
  Kokkos::deep_copy(B, Bh);

  auto hA       = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB       = make_input_node(make_handle<'j', 'k', 'l'>(B));
  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'l', 'i'>(hA, hB));

  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<64, 4, 4>, StaticTile<4, 4, 32>, StaticTile<32, 64>>{},
      C);

  auto Ch = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int l = 0; l < L; ++l)
    for (int i = 0; i < I; ++i) {
      double acc = 0.0;
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < K; ++k)
          acc += static_cast<double>(Ah(i, j, k)) * Bh(j, k, l);
      EXPECT_NEAR(Ch(l, i), static_cast<float>(acc), 1e-3f)
          << "l=" << l << " i=" << i;
    }
}

// Input hooks are applied at load time, when the operand tile is staged into
// scratch (contraction hooks are applied at store; see HookPreservesType).
struct Doubler {
  KOKKOS_FUNCTION void operator()(int, int, int, float& v) const { v *= 2.0f; }
};

TEST(GraphTest, InputHookAppliedTeam) {
  // Same contraction as SingleContractionExecutionTeam (A=B=1 → C=16), but A
  // carries a doubling hook, so every staged A element becomes 2 → C=32.
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
  View3 A("A", 16, 4, 4);
  View3 B("B", 4, 4, 32);
  View2 C("C", 16, 32);

  Kokkos::deep_copy(A, 1.0f);
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A), Doubler{});
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));

  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<16, 4, 4>, StaticTile<4, 4, 32>, StaticTile<16, 32>>{},
      C);

  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < 16; ++i)
    for (int l = 0; l < 32; ++l) EXPECT_FLOAT_EQ(C_host(i, l), 32.0f);
}

struct WriteRowIndex {
  KOKKOS_FUNCTION void operator()(int i, int, int, float& v) const {
    v = static_cast<float>(i);
  }
};

TEST(GraphTest, InputHookGlobalIndexAcrossTiles) {
  // A's i mode splits into 3 load tiles (16 each; same tiling as
  // MultiTileExecutionTeam). The hook ignores A's data and writes its own
  // row index instead, so C[i,l] = sum_{j,k} i = 16*i holds only if the hook
  // observed each element's *global* row index (i in [0,48)) rather than a
  // tile-local one (which would reset to [0,16) on every tile and corrupt
  // tiles 1 and 2).
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
  View3 A("A", 48, 4, 4);
  View3 B("B", 4, 4, 32);
  View2 C("C", 48, 32);

  Kokkos::deep_copy(A, 1.0f);
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A), WriteRowIndex{});
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g        = make_graph();
  auto [g1, o1] = g.ops(make_contraction_node<'i', 'l'>(hA, hB));

  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<16, 2, 4>, StaticTile<2, 4, 32>, StaticTile<16, 32>>{},
      C);

  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < 48; ++i)
    for (int l = 0; l < 32; ++l)
      EXPECT_FLOAT_EQ(C_host(i, l), 16.0f * static_cast<float>(i))
          << "i=" << i << " l=" << l;
}

struct WriteOutputIndex {
  KOKKOS_FUNCTION void operator()(int i, int l, float& v) const {
    v = static_cast<float>(i) * 100.0f + static_cast<float>(l);
  }
};

TEST(GraphTest, ContractionHookGlobalIndexAcrossTiles) {
  // C's i mode splits into 3 store tiles (16 each). The hook ignores the
  // computed contraction value and writes an index-encoded value instead, so
  // C[i,l] == 100*i+l holds only if the store hook saw each element's
  // *global* output coordinate rather than a tile-local one.
  using View2 = Kokkos::View<float**, Kokkos::LayoutRight>;
  using View3 = Kokkos::View<float***, Kokkos::LayoutRight>;
  View3 A("A", 48, 4, 4);
  View3 B("B", 4, 4, 32);
  View2 C("C", 48, 32);

  Kokkos::deep_copy(A, 1.0f);
  Kokkos::deep_copy(B, 1.0f);
  Kokkos::deep_copy(C, 0.0f);

  auto hA = make_input_node(make_handle<'i', 'j', 'k'>(A));
  auto hB = make_input_node(make_handle<'j', 'k', 'l'>(B));

  auto g = make_graph();
  auto [g1, o1] =
      g.ops(make_contraction_node<'i', 'l'>(hA, hB, WriteOutputIndex{}));

  g1.execute(
      TeamPolicyTag<>{},
      Tile<StaticTile<16, 2, 4>, StaticTile<2, 4, 32>, StaticTile<16, 32>>{},
      C);

  auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, C);
  for (int i = 0; i < 48; ++i)
    for (int l = 0; l < 32; ++l)
      EXPECT_FLOAT_EQ(C_host(i, l),
                      static_cast<float>(i) * 100.0f + static_cast<float>(l))
          << "i=" << i << " l=" << l;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
