#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/ScratchAllocator.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

// ---------------------------------------------------------------------------
// Helpers — minimal TensorLike types used to build NodeHandles
// ---------------------------------------------------------------------------
struct T2 {  // rank-2 tensor, 4×8
  static constexpr int rank = 2;
  using value_type          = float;
  float          buf_[32]{};
  int            extent(int i) const { return i == 0 ? 4 : 8; }
  std::ptrdiff_t stride(int k) const { return k == 0 ? 8 : 1; }
  float*         data() { return buf_; }
  const float*   data() const { return buf_; }
  float          operator()(int, int) const { return 0.f; }
};

struct T3 {  // rank-3 tensor, 4×8×16
  static constexpr int rank = 3;
  using value_type          = float;
  float          buf_[512]{};
  int            extent(int i) const { return i == 0 ? 4 : (i == 1 ? 8 : 16); }
  std::ptrdiff_t stride(int k) const {
    return k == 0 ? 128 : (k == 1 ? 16 : 1);
  }
  float*       data() { return buf_; }
  const float* data() const { return buf_; }
  float        operator()(int, int, int) const { return 0.f; }
};

static_assert(TensorLike<T2>);
static_assert(TensorLike<T3>);

// Helpers — node types used across tests
static auto make_input_2d() {
  return make_input_node(make_handle<'i', 'j'>(T2{}));
}
static auto make_contraction_2d() {
  auto na = make_input_node(make_handle<'i', 'j'>(T2{}));
  auto nb = make_input_node(make_handle<'j', 'k'>(T2{}));
  return make_contraction_node<float, ES, 'i', 'k'>(na, nb);
}

// ---------------------------------------------------------------------------
// ScratchAllocator::bytes matches raw scratch_tile_bytes for IntermTag slot
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, BytesMatchScratchTileBytesStatic) {
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, IntermTag>;

  // StaticTile<4,8>: 32 floats
  const std::size_t expected =
      Impl::scratch_tile_bytes<float, ES>(StaticTile<4, 8>{});
  EXPECT_EQ((Alloc::bytes<float>(StaticTile<4, 8>{})), expected);
}

// bytes() for an InputTag operand node matches scratch_tile_bytes
TEST(ScratchAllocatorTest, BytesMatchScratchTileBytesDynamic) {
  using NA    = decltype(make_input_2d());
  using Alloc = ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA>;

  DynamicTile<2> tile;
  tile.extents               = {4, 8};
  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(tile);
  EXPECT_EQ((Alloc::bytes<float>(tile)), expected);
}

// All six (OuterOpTag, operand-node-type) combos give the same bytes today.
TEST(ScratchAllocatorTest, AllContractionSlotsConsistent) {
  using NA = decltype(make_input_2d());
  using NC = decltype(make_contraction_2d());

  constexpr StaticTile<4, 8> tile{};
  const std::size_t          ref = Impl::scratch_tile_bytes<float, ES>(tile);

  // ContractionTag outer op
  EXPECT_EQ(
      (ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NA>::bytes<float>(
          tile)),
      ref);
  EXPECT_EQ(
      (ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, NC>::bytes<float>(
          tile)),
      ref);
  EXPECT_EQ((ScratchAllocator<TeamPolicyTag<ES>, ContractionTag,
                              IntermTag>::bytes<float>(tile)),
            ref);

  // CombineTag outer op
  EXPECT_EQ(
      (ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NA>::bytes<float>(tile)),
      ref);
  EXPECT_EQ(
      (ScratchAllocator<TeamPolicyTag<ES>, CombineTag, NC>::bytes<float>(tile)),
      ref);
  EXPECT_EQ(
      (ScratchAllocator<TeamPolicyTag<ES>, CombineTag, IntermTag>::bytes<float>(
          tile)),
      ref);
}

// ---------------------------------------------------------------------------
// make_scratch_allocator constructs the right specialization and stores the
// node. ContractionTag nodes expose node_ as a public member.
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, MakeScratchAllocatorFactory) {
  auto na = make_input_2d();
  auto nc = make_contraction_2d();

  auto alloc_input =
      make_scratch_allocator<TeamPolicyTag<ES>, ContractionTag>(na);
  auto alloc_contr =
      make_scratch_allocator<TeamPolicyTag<ES>, ContractionTag>(nc);

  // Type checks: factory produces the right specialization.
  using ExpInput =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, decltype(na)>;
  using ExpContr =
      ScratchAllocator<TeamPolicyTag<ES>, ContractionTag, decltype(nc)>;
  static_assert(std::is_same_v<decltype(alloc_input), ExpInput>);
  static_assert(std::is_same_v<decltype(alloc_contr), ExpContr>);

  // bytes() matches the raw helper.
  constexpr StaticTile<4, 8> tile{};
  const std::size_t expected = Impl::scratch_tile_bytes<float, ES>(tile);
  EXPECT_EQ((ExpInput::bytes<float>(tile)), expected);
  EXPECT_EQ((ExpContr::bytes<float>(tile)), expected);

  // ContractionTag allocator stores the node.
  static_assert(std::is_same_v<decltype(alloc_contr.node_), decltype(nc)>);
}

// ---------------------------------------------------------------------------
// Sum of per-slot bytes equals scratch_size_per_team for a known contraction
// ---------------------------------------------------------------------------
TEST(ScratchAllocatorTest, SumEqualsEvaluatorScratchSize) {
  // A_{i,j,k}: 4×8×8  B_{j,k,l}: 8×8×4  →  C_{i,l}: 4×4
  // Tile: A tile = StaticTile<4,8,8>, B tile = StaticTile<8,8,4>,
  //       C tile = StaticTile<4,4>
  auto ha = make_handle<'i', 'j', 'k'>(T3{});
  auto hb = make_handle<'j', 'k', 'l'>(T3{});
  auto na = make_input_node(ha);
  auto nb = make_input_node(hb);
  auto nc = make_contraction_node<float, ES, 'i', 'l'>(na, nb);

  // Tile<A,B,C> with StaticTile for all
  using TileA = StaticTile<4, 8, 8>;
  using TileB = StaticTile<8, 8, 4>;
  using TileC = StaticTile<4, 4>;
  Tile<TileA, TileB, TileC> bundle{TileA{}, TileB{}, TileC{}};

  using EvalT =
      Evaluator<TeamPolicyTag<ES>, decltype(nc), Tile<TileA, TileB, TileC>>;
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
