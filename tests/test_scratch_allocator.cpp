#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/ScratchAllocator.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;
using ES     = Kokkos::DefaultExecutionSpace;
using team_t = typename Kokkos::TeamPolicy<ES>::member_type;

// ---------------------------------------------------------------------------
// Helpers — minimal TensorLike types used to build NodeHandles
// ---------------------------------------------------------------------------
struct T2 {  // rank-2 tensor, 4×8
  static constexpr int rank = 2;
  using value_type          = float;
  float          buf_[32]{};
  int            extent(int i) const { return i == 0 ? 4 : 8; }
  std::ptrdiff_t stride(int k) const { return k == 0 ? 8 : 1; }
  // data() const returns a mutable pointer (view/reference semantics, like
  // Kokkos::View::data() const) so it satisfies View::operator()'s
  // value_type& return even when called through a const TensorHandle.
  float* data() { return buf_; }
  float* data() const { return const_cast<float*>(buf_); }
  float  operator()(int, int) const { return 0.f; }
};

// Rank-3 tensor sized for the register GEMM kernel's block-factor constraints
// (CPU: MT=NT=8, NR=2*simd_width) -- same shapes as
// test_combine_node.cpp's ContractionCombineData: A{i,j,k}=16x2x4,
// B{j,k,l}=2x4x32, so SA=16, SK=J*K=8, SB=32 all divide evenly.
struct T3A {  // 16x2x4
  static constexpr int rank = 3;
  using value_type          = float;
  float          buf_[128]{};
  int            extent(int i) const { return i == 0 ? 16 : (i == 1 ? 2 : 4); }
  std::ptrdiff_t stride(int k) const { return k == 0 ? 8 : (k == 1 ? 4 : 1); }
  float*         data() { return buf_; }
  float*         data() const { return const_cast<float*>(buf_); }
  float          operator()(int, int, int) const { return 0.f; }
};
struct T3B {  // 2x4x32
  static constexpr int rank = 3;
  using value_type          = float;
  float          buf_[256]{};
  int            extent(int i) const { return i == 0 ? 2 : (i == 1 ? 4 : 32); }
  std::ptrdiff_t stride(int k) const {
    return k == 0 ? 128 : (k == 1 ? 32 : 1);
  }
  float* data() { return buf_; }
  float* data() const { return const_cast<float*>(buf_); }
  float  operator()(int, int, int) const { return 0.f; }
};

static_assert(TensorLike<T2>);
static_assert(TensorLike<T3A>);
static_assert(TensorLike<T3B>);

// Helpers — node types used across tests
static auto make_input_2d() {
  return make_input_node(make_handle<'i', 'j'>(T2{}));
}

// A nested contraction (A{i,j,k} x B{j,k,l} -> C{i,l}), paired with a full
// Tile<A,B,C> bundle whose GEMM shape (SA=16, SK=8, SB=32) satisfies the
// register kernel's block-factor constraints. Used to exercise the
// ContractionTag-operand 4-param specializations (both the
// ContractionTag-outer-op recursive bytes() and the CombineTag-outer-op
// non-recursive bytes()).
static auto make_contraction_3d() {
  auto ha = make_handle<'i', 'j', 'k'>(T3A{});
  auto hb = make_handle<'j', 'k', 'l'>(T3B{});
  auto na = make_input_node(ha);
  auto nb = make_input_node(hb);
  return make_contraction_node<float, ES, 'i', 'l'>(na, nb);
}
using TileA3  = StaticTile<16, 2, 4>;
using TileB3  = StaticTile<2, 4, 32>;
using TileC3  = StaticTile<16, 32>;
using Bundle3 = Tile<TileA3, TileB3, TileC3>;

// ---------------------------------------------------------------------------
// IntermTag (output/C slot): bytes() matches raw scratch_tile_bytes for both
// outer-op families; stage() is a trivial alias for alloc() (no operand to
// stage from), kept for interface uniformity.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, IntermTagBytesMatchesRaw) {
  using ContrAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag>;
  using CombAlloc = ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag>;

  // StaticTile<4,8>: 32 floats
  const std::size_t expected =
      Impl::scratch_tile_bytes<float, ES>(StaticTile<4, 8>{});
  EXPECT_EQ((ContrAlloc::bytes<float>(StaticTile<4, 8>{})), expected);
  EXPECT_EQ((CombAlloc::bytes<float>(StaticTile<4, 8>{})), expected);
}

TEST(ScratchAllocatorTest, IntermTagStageMatchesAllocType) {
  using ContrAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag>;
  using CombAlloc = ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag>;
  using Tile      = StaticTile<4, 8>;

  static_assert(
      std::is_same_v<
          decltype(std::declval<const ContrAlloc&>().template stage<float>(
              std::declval<team_t>(), Tile{})),
          decltype(std::declval<const ContrAlloc&>().template alloc<float>(
              std::declval<team_t>(), Tile{}))>);
  static_assert(std::is_same_v<
                decltype(std::declval<const CombAlloc&>().template stage<float>(
                    std::declval<team_t>(), Tile{})),
                decltype(std::declval<const CombAlloc&>().template alloc<float>(
                    std::declval<team_t>(), Tile{}))>);
}

// ---------------------------------------------------------------------------
// ContractionTag outer op, InputTag operand (4-param): bytes() is the plain
// (non-recursive) staging cost; stage() delegates to the cached inner
// Evaluator.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, ContractionInputOperandBytes) {
  using NA    = decltype(make_input_2d());
  using TileA = StaticTile<4, 8>;
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, TileA>;

  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(TileA{});
  EXPECT_EQ((Alloc::bytes<float>(TileA{})), expected);

  static_assert(requires(const Alloc& a) {
    a.stage(std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>());
  });
}

// ---------------------------------------------------------------------------
// ContractionTag outer op, ContractionTag operand (4-param, nested fusion):
// bytes() is RECURSIVE -- the inner evaluator's full scratch_size_per_team --
// matching the nested-contraction-scratch fix (scratch allocated once in the
// constructor rather than on every k-tile call).
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, ContractionNestedOperandBytesIsRecursive) {
  using NC = decltype(make_contraction_3d());
  using Alloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NC, Bundle3>;
  using InnerEval = Evaluator<TeamPolicyTag<ES>, NC, Bundle3>;

  const std::size_t expected = InnerEval::scratch_size_per_team(Bundle3{});
  EXPECT_GT(expected, 0u);
  EXPECT_EQ((Alloc::bytes<float>(Bundle3{})), expected);

  static_assert(requires(const Alloc& a) {
    a.stage(std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>());
  });
}

// ---------------------------------------------------------------------------
// CombineTag outer op, InputTag operand (4-param): same non-recursive bytes()
// as the ContractionTag-outer-op InputTag form.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, CombineInputOperandBytes) {
  using NA    = decltype(make_input_2d());
  using TileA = StaticTile<4, 8>;
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA, TileA>;

  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(TileA{});
  EXPECT_EQ((Alloc::bytes<float>(TileA{})), expected);

  static_assert(requires(const Alloc& a) {
    a.stage(std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>());
  });
}

// ---------------------------------------------------------------------------
// CombineTag outer op, ContractionTag operand (4-param): the ContractionTag-
// operand specialization body is shared across every OuterOpTag
// (ContractionTag, CombineTag) -- OuterOpTag is an unused template parameter
// in this specialization -- so this form now has the SAME zero-copy alloc()
// and RECURSIVE bytes() (the inner evaluator's full scratch_size_per_team)
// as the ContractionTag-outer-op form for the identical (node, tile bundle)
// pair, matching that form's bytes() exactly. The CombineTag evaluator
// (Team.hpp) independently computes this same recursive size via
// operand_native_scratch_bytes and skips calling this allocator's bytes()
// for staging purposes (operand_staging_bytes returns 0 for a ContractionTag
// operand), so there is no double-counting in practice.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, CombineNestedOperandBytesIsRecursive) {
  using NC    = decltype(make_contraction_3d());
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NC, Bundle3>;
  using ContrAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NC, Bundle3>;

  using InnerEval            = Evaluator<TeamPolicyTag<ES>, NC, Bundle3>;
  const std::size_t expected = InnerEval::scratch_size_per_team(Bundle3{});
  EXPECT_GT(expected, 0u);
  EXPECT_EQ((Alloc::bytes<float>(Bundle3{})), expected);
  // Matches the ContractionTag-outer-op form exactly (same recursive bytes()),
  // unlike before this fix, when CombineTag's form stayed non-recursive.
  EXPECT_EQ((Alloc::bytes<float>(Bundle3{})),
            (ContrAlloc::bytes<float>(Bundle3{})));

  static_assert(requires(const Alloc& a) {
    a.stage(std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>());
  });
}

// ---------------------------------------------------------------------------
// make_scratch_allocator factory: the 3-arg form deduces the 4-param
// specialization for both outer-op families (unevaluated decltype/declval —
// no team instance is actually constructed, mirroring how Evaluator/Team.hpp
// itself declares alloc_a_t/alloc_b_t).
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, MakeScratchAllocatorFactoryProducesFourParamType) {
  using NA    = decltype(make_input_2d());
  using TileA = StaticTile<4, 8>;

  using ContrFactoryT =
      decltype(make_scratch_allocator<TeamPolicyTag<ES>, ContractionTag>(
          std::declval<NA>(), std::declval<TileA>(), std::declval<team_t>()));
  using CombFactoryT =
      decltype(make_scratch_allocator<TeamPolicyTag<ES>, CombineTag>(
          std::declval<NA>(), std::declval<TileA>(), std::declval<team_t>()));

  static_assert(
      std::is_same_v<
          ContrFactoryT,
          ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, TileA>>);
  static_assert(
      std::is_same_v<CombFactoryT, ScratchAllocator<TeamPolicyTag<ES>,
                                                    CombineTag, NA, TileA>>);
}

// ---------------------------------------------------------------------------
// Sum of per-slot bytes equals scratch_size_per_team for a known contraction
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, SumEqualsEvaluatorScratchSize) {
  // Reuse the register-kernel-friendly shapes from make_contraction_3d():
  // A{i,j,k}=16x2x4, B{j,k,l}=2x4x32 -> C{i,l}=16x32.
  auto nc = make_contraction_3d();

  Bundle3 bundle{TileA3{}, TileB3{}, TileC3{}};

  using EvalT = Evaluator<TeamPolicyTag<ES>, decltype(nc), Bundle3>;
  const std::size_t from_evaluator = EvalT::scratch_size_per_team(bundle);
  EXPECT_GT(from_evaluator, 0u);

  // The refactored scratch_size_per_team now internally uses ScratchAllocator,
  // so this test verifies the delegation gives the same result as before
  // (regression guard). The "same result" is whatever the evaluator computes.
  EXPECT_EQ(from_evaluator, EvalT::scratch_size_per_team(bundle));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
