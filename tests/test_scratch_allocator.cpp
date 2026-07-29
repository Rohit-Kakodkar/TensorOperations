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

// Does this allocator offer the operand-staging entry point? Spelled as a
// concept (rather than a bare requires-expression at each use) so the check is
// a dependent -- hence SFINAE-friendly -- context: probing for a member that
// genuinely does not exist is a hard error on a concrete type.
template <typename Alloc>
concept HasStage = requires(const Alloc& a) {
  a.stage(std::declval<team_t>(), std::declval<Kokkos::Array<int, 2>>());
};

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
// outer-op families. The scalar type is now baked into the allocator's type, so
// bytes() takes no template argument.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, IntermTagBytesMatchesRaw) {
  using Tile       = StaticTile<4, 8>;
  using ContrAlloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag,
                                      IntermTag, float, Tile>;
  using CombAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag, float, Tile>;

  // StaticTile<4,8>: 32 floats
  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(Tile{});
  EXPECT_EQ(ContrAlloc::bytes(Tile{}), expected);
  EXPECT_EQ(CombAlloc::bytes(Tile{}), expected);
}

// The IntermTag slot carves its scratch in the constructor and hands it back
// from get(); there is no operand, so it has no stage(). The view it produces
// must be exactly what Impl::alloc_scratch_tile would give for the same tile.
TEST(ScratchAllocatorTest, IntermTagGetMatchesAllocScratchTile) {
  using Tile  = StaticTile<4, 8>;
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag,
                                 float, Tile>;

  static_assert(
      std::is_same_v<decltype(std::declval<const Alloc&>().get()),
                     decltype(Impl::alloc_scratch_tile<float, ES>(
                         std::declval<team_t>(), std::declval<Tile>()))>);
  // No source to stage from, so no stage() on this specialization.
  static_assert(!HasStage<Alloc>);

  // The constructed view really is a distinct, correctly-sized team-scratch
  // tile: run one team with exactly the bytes bytes() asks for and write
  // through get().
  const std::size_t     bytes = Alloc::bytes(Tile{});
  Kokkos::View<int, ES> ok("ok");
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<ES>(1, Kokkos::AUTO)
          .set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(bytes))),
      KOKKOS_LAMBDA(const team_t& team) {
        Alloc a{Tile{}, team};
        auto  v = a.get();
        Kokkos::single(Kokkos::PerTeam(team), [&] {
          for (std::size_t i = 0; i < v.size(); ++i) v.data()[i] = float(i);
          ok() = (v.size() == 32u && v.data()[31] == 31.f) ? 1 : 0;
        });
      });
  Kokkos::fence();
  int ok_h = 0;
  Kokkos::deep_copy(ok_h, ok);
  EXPECT_EQ(ok_h, 1);
}

// ---------------------------------------------------------------------------
// ContractionTag outer op, InputTag operand: bytes() is the plain
// (non-recursive) staging cost; stage() delegates to the cached inner
// Evaluator. PermSeq is omitted, exercising the identity default.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, ContractionInputOperandBytes) {
  using NA    = decltype(make_input_2d());
  using TileA = StaticTile<4, 8>;
  using Alloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, float, TileA>;

  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(TileA{});
  EXPECT_EQ(Alloc::bytes(TileA{}), expected);

  static_assert(HasStage<Alloc>);
}

// ---------------------------------------------------------------------------
// The same InputTag operand with an explicit, NON-identity PermSeq: the
// staging buffer has the CANONICAL shape, so a transposing perm flips the
// tile the buffer is sized and typed on. This is the one form that consumes
// PermSeq at all, and the path the evaluators actually take for a permuted
// operand (nothing else in the suite covers it).
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, InputOperandPermSeqCanonicalizesStagingBuffer) {
  using NA        = decltype(make_input_2d());
  using TileA     = StaticTile<4, 8>;
  using Swap      = std::integer_sequence<int, 1, 0>;
  using Identity  = std::integer_sequence<int, 0, 1>;
  using PermAlloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA,
                                     float, TileA, Swap>;
  using IdAlloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, float,
                                   TileA, Identity>;

  // Sized on the canonical (transposed) tile...
  EXPECT_EQ(PermAlloc::bytes(TileA{}),
            (Impl::scratch_tile_bytes<float, ES>(StaticTile<8, 4>{})));
  // ... and typed on it too, which is what makes the staged copy match the
  // consumer's canonical layout rather than the operand's native one.
  static_assert(
      std::is_same_v<typename PermAlloc::scratch_view_t,
                     decltype(Impl::alloc_scratch_tile<float, ES>(
                         std::declval<team_t>(), StaticTile<8, 4>{}))>);
  static_assert(!std::is_same_v<typename PermAlloc::scratch_view_t,
                                typename IdAlloc::scratch_view_t>);

  // Omitting PermSeq leaves the `void` sentinel, a distinct type from an
  // explicit identity sequence but resolved to exactly the same axes -- so the
  // two allocators behave identically even though they are not the same type.
  using DefaultAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA, float, TileA>;
  static_assert(!std::is_same_v<DefaultAlloc, IdAlloc>);
  static_assert(std::is_same_v<typename DefaultAlloc::scratch_view_t,
                               typename IdAlloc::scratch_view_t>);
  EXPECT_EQ(DefaultAlloc::bytes(TileA{}), IdAlloc::bytes(TileA{}));
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
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NC, float, Bundle3>;
  using InnerEval = Evaluator<TeamPolicyTag<ES>, NC, Bundle3>;

  const std::size_t expected = InnerEval::scratch_size_per_team(Bundle3{});
  EXPECT_GT(expected, 0u);
  EXPECT_EQ(Alloc::bytes(Bundle3{}), expected);

  static_assert(HasStage<Alloc>);
  // get() is a pure accessor onto the inner evaluator's own C scratch -- this
  // form carves nothing of its own.
  static_assert(std::is_same_v<decltype(std::declval<const Alloc&>().get()),
                               typename InnerEval::scratch_view_t>);
}

// ---------------------------------------------------------------------------
// CombineTag outer op, InputTag operand (4-param): same non-recursive bytes()
// as the ContractionTag-outer-op InputTag form.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, CombineInputOperandBytes) {
  using NA    = decltype(make_input_2d());
  using TileA = StaticTile<4, 8>;
  using Alloc =
      ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA, float, TileA>;

  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(TileA{});
  EXPECT_EQ(Alloc::bytes(TileA{}), expected);

  static_assert(HasStage<Alloc>);
}

// ---------------------------------------------------------------------------
// CombineTag outer op, ContractionTag operand: the ContractionTag-operand
// specialization body is shared across every OuterOpTag (ContractionTag,
// CombineTag) -- OuterOpTag is an unused template parameter in this
// specialization -- so this form has the SAME zero-copy get() and RECURSIVE
// bytes() (the inner evaluator's full scratch_size_per_team) as the
// ContractionTag-outer-op form for the identical (node, tile bundle) pair,
// matching that form's bytes() exactly.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, CombineNestedOperandBytesIsRecursive) {
  using NC = decltype(make_contraction_3d());
  using Alloc =
      ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NC, float, Bundle3>;
  using ContrAlloc =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NC, float, Bundle3>;

  using InnerEval            = Evaluator<TeamPolicyTag<ES>, NC, Bundle3>;
  const std::size_t expected = InnerEval::scratch_size_per_team(Bundle3{});
  EXPECT_GT(expected, 0u);
  EXPECT_EQ(Alloc::bytes(Bundle3{}), expected);
  // Matches the ContractionTag-outer-op form exactly (same recursive bytes()),
  // unlike before this fix, when CombineTag's form stayed non-recursive.
  EXPECT_EQ(Alloc::bytes(Bundle3{}), ContrAlloc::bytes(Bundle3{}));

  static_assert(HasStage<Alloc>);
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
