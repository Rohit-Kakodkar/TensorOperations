#include <Kokkos_Core.hpp>
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <TensorOperations/Tiling.hpp>
#include <array>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// StaticTile — compile-time extents.
// ---------------------------------------------------------------------------
TEST(TilingTest, StaticTile) {
  using Tile = StaticTile<2, 3, 4>;
  static_assert(Tile::rank == 3);
  static_assert(Tile::is_static);
  static_assert(Tile::extent(0) == 2);
  static_assert(Tile::extent(1) == 3);
  static_assert(Tile::extent(2) == 4);

  // Degenerate unit tile is the un-staged baseline.
  static_assert(StaticTile<1, 1>::rank == 2);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// DynamicTile — runtime extents, compile-time rank.
// ---------------------------------------------------------------------------
TEST(TilingTest, DynamicTile) {
  using Tile = DynamicTile<3>;
  static_assert(Tile::rank == 3);
  static_assert(!Tile::is_static);

  Tile t{{2, 4, 6}};
  EXPECT_EQ(t.extent(0), 2);
  EXPECT_EQ(t.extent(1), 4);
  EXPECT_EQ(t.extent(2), 6);
}

using View2D = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>;

// ---------------------------------------------------------------------------
// Evaluator specialization selection — purely type-level; no kernels run.
// C_{ik} = sum_j A_{ij} B_{jk}: output rank 2, NumContracted 1.
// ---------------------------------------------------------------------------
TEST(TilingTest, EvaluatorSelection) {
  using NC = decltype(make_contraction_node<float>(
      make_input_node(make_handle(std::declval<View2D>(),
                                  std::array<int32_t, 2>{'i', 'j'})),
      make_input_node(make_handle(std::declval<View2D>(),
                                  std::array<int32_t, 2>{'j', 'k'})),
      std::array<int32_t, 2>{'i', 'k'}));

  static_assert(NC::Rank == 2);
  static_assert(NC::NumContracted == 1);

  // -- Team + Contraction + Tile<TileA,TileB,TileC> -> scratch tier ----------
  // Each operand carries its own tile: A:(i,j), B:(j,k), C:(i,k), all rank 2.
  using TTile  = Tile<DynamicTile<2>, DynamicTile<2>, DynamicTile<2>>;
  using TEvalC = Evaluator<TeamPolicyTag<>, NC, TTile>;
  static_assert(std::is_same_v<TEvalC::tiling_type, TTile>);
  static_assert(TEvalC::Rank == 2);  // output rank, not tile rank
  static_assert(TEvalC::scratch_view_t::rank == 2);

  // -- Team also accepts StaticTile operands on the scratch tier -------------
  using TTileS  = Tile<StaticTile<2, 2>, StaticTile<2, 2>, StaticTile<2, 2>>;
  using TEvalCS = Evaluator<TeamPolicyTag<>, NC, TTileS>;
  static_assert(std::is_same_v<TEvalCS::tiling_type, TTileS>);
  static_assert(TEvalCS::scratch_view_t::rank == 2);

  SUCCEED();
}

// ---------------------------------------------------------------------------
// reshape(view, tile) — reinterpret a View under a new shape from a Tile.
// ---------------------------------------------------------------------------

using Buf1D = Kokkos::View<float*, Kokkos::LayoutRight, Kokkos::HostSpace>;

TEST(TilingTest, ReshapeViewStaticRight) {
  Buf1D buf("buf", 32);
  for (int k = 0; k < 32; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, StaticTileLayoutRight<32>> src{buf, {}};
  auto                                   r = reshape(src, StaticTile<4, 8>{});
  static_assert(
      std::is_same_v<decltype(r), View<Buf1D, StaticTileLayoutRight<4, 8>>>);

  EXPECT_EQ(r.size(), 32);
  EXPECT_EQ(r.extent(0), 4);
  EXPECT_EQ(r.extent(1), 8);
  EXPECT_EQ(r.stride(0), 8);
  EXPECT_EQ(r.stride(1), 1);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(r(i, j), static_cast<float>(i * 8 + j));
}

TEST(TilingTest, ReshapeViewStaticLeft) {
  Buf1D buf("buf", 32);
  for (int k = 0; k < 32; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, StaticTileLayoutLeft<32>> src{buf, {}};
  auto                                  r = reshape(src, StaticTile<4, 8>{});
  static_assert(
      std::is_same_v<decltype(r), View<Buf1D, StaticTileLayoutLeft<4, 8>>>);

  EXPECT_EQ(r.size(), 32);
  EXPECT_EQ(r.stride(0), 1);
  EXPECT_EQ(r.stride(1), 4);

  // column-major: element (i,j) is at flat offset i*1 + j*4
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(r(i, j), static_cast<float>(i + j * 4));
}

TEST(TilingTest, ReshapeViewDynamicRight) {
  Buf1D buf("buf", 32);
  for (int k = 0; k < 32; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, DynamicTileLayoutRight<1>> src{
      buf, DynamicTileLayoutRight<1>{Kokkos::Array<int, 1>{32}}};
  auto r = reshape(src, DynamicTile<2>{{4, 8}});
  static_assert(
      std::is_same_v<decltype(r), View<Buf1D, DynamicTileLayoutRight<2>>>);

  EXPECT_EQ(r.size(), 32);
  EXPECT_EQ(r.extent(0), 4);
  EXPECT_EQ(r.extent(1), 8);
  EXPECT_EQ(r.stride(0), 8);
  EXPECT_EQ(r.stride(1), 1);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(r(i, j), static_cast<float>(i * 8 + j));
}

TEST(TilingTest, ReshapeViewDynamicLeft) {
  Buf1D buf("buf", 32);
  for (int k = 0; k < 32; ++k) buf(k) = static_cast<float>(k);

  View<Buf1D, DynamicTileLayoutLeft<1>> src{
      buf, DynamicTileLayoutLeft<1>{Kokkos::Array<int, 1>{32}}};
  auto r = reshape(src, DynamicTile<2>{{4, 8}});
  static_assert(
      std::is_same_v<decltype(r), View<Buf1D, DynamicTileLayoutLeft<2>>>);

  EXPECT_EQ(r.size(), 32);
  EXPECT_EQ(r.stride(0), 1);
  EXPECT_EQ(r.stride(1), 4);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j)
      EXPECT_FLOAT_EQ(r(i, j), static_cast<float>(i + j * 4));
}

// ---------------------------------------------------------------------------
// reshape(layout, tile) — reinterpret TileLayout under a new shape from a Tile.
// ---------------------------------------------------------------------------
TEST(TilingTest, ReshapeStaticRight) {
  using Old = StaticTileLayoutRight<32>;
  using New = StaticTileLayoutRight<4, 8>;
  using R   = decltype(reshape(Old{}, StaticTile<4, 8>{}));
  static_assert(std::is_same_v<R, New>);

  constexpr R layout{};
  static_assert(layout.size() == 32);
  static_assert(layout.extent(0) == 4);
  static_assert(layout.extent(1) == 8);
  static_assert(layout.stride(0) == 8);  // row-major: stride[0] = extent[1]
  static_assert(layout.stride(1) == 1);

  // round-trip: flat(i,j) → operator[] → {i,j}
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
      auto idx = layout[static_cast<int>(layout.flat(i, j))];
      EXPECT_EQ(idx[0], i);
      EXPECT_EQ(idx[1], j);
    }
}

TEST(TilingTest, ReshapeStaticLeft) {
  using Old = StaticTileLayoutLeft<32>;
  using New = StaticTileLayoutLeft<4, 8>;
  using R   = decltype(reshape(Old{}, StaticTile<4, 8>{}));
  static_assert(std::is_same_v<R, New>);

  constexpr R layout{};
  static_assert(layout.size() == 32);
  static_assert(layout.extent(0) == 4);
  static_assert(layout.extent(1) == 8);
  static_assert(layout.stride(0) == 1);  // column-major: stride[0] = 1
  static_assert(layout.stride(1) == 4);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
      auto idx = layout[static_cast<int>(layout.flat(i, j))];
      EXPECT_EQ(idx[0], i);
      EXPECT_EQ(idx[1], j);
    }
}

TEST(TilingTest, ReshapeDynamicRight) {
  auto src      = DynamicTileLayoutRight<1>{Kokkos::Array<int, 1>{32}};
  auto reshaped = reshape(src, DynamicTile<2>{{4, 8}});
  static_assert(std::is_same_v<decltype(reshaped), DynamicTileLayoutRight<2>>);

  EXPECT_EQ(reshaped.size(), 32);
  EXPECT_EQ(reshaped.extent(0), 4);
  EXPECT_EQ(reshaped.extent(1), 8);
  EXPECT_EQ(reshaped.stride(0), 8);
  EXPECT_EQ(reshaped.stride(1), 1);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
      auto idx = reshaped[static_cast<int>(
          reshaped.flat(Kokkos::Array<int, 2>{i, j}))];
      EXPECT_EQ(idx[0], i);
      EXPECT_EQ(idx[1], j);
    }
}

TEST(TilingTest, ReshapeDynamicLeft) {
  auto src      = DynamicTileLayoutLeft<1>{Kokkos::Array<int, 1>{32}};
  auto reshaped = reshape(src, DynamicTile<2>{{4, 8}});
  static_assert(std::is_same_v<decltype(reshaped), DynamicTileLayoutLeft<2>>);

  EXPECT_EQ(reshaped.size(), 32);
  EXPECT_EQ(reshaped.extent(0), 4);
  EXPECT_EQ(reshaped.extent(1), 8);
  EXPECT_EQ(reshaped.stride(0), 1);
  EXPECT_EQ(reshaped.stride(1), 4);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 8; ++j) {
      auto idx = reshaped[static_cast<int>(
          reshaped.flat(Kokkos::Array<int, 2>{i, j}))];
      EXPECT_EQ(idx[0], i);
      EXPECT_EQ(idx[1], j);
    }
}

// ---------------------------------------------------------------------------
// tile_layout — factory: TileLayout × Tile → 2N-dimensional TileLayout.
//
// First N extents: outer tile counts (src.extent(d) / tile.extent(d)).
// Last  N extents: inner tile sizes.
// Memory order (Right/Left) is preserved from the source layout.
// ---------------------------------------------------------------------------

TEST(TilingTest, TileLayoutStaticRight) {
  // Static src + static tile → fully compile-time
  // StaticTileLayoutRight<4,3,2,4>
  using Src  = StaticTileLayoutRight<8, 12>;
  using Tile = StaticTile<2, 4>;
  using R    = decltype(tile_layout(Src{}, Tile{}));
  static_assert(std::is_same_v<R, StaticTileLayoutRight<4, 3, 2, 4>>);

  constexpr R layout{};
  static_assert(layout.rank == 4);
  // outer counts
  static_assert(layout.extent(0) == 4);
  static_assert(layout.extent(1) == 3);
  // inner tile sizes
  static_assert(layout.extent(2) == 2);
  static_assert(layout.extent(3) == 4);
  // Right strides: fastest dim last
  static_assert(layout.stride(3) == 1);
  static_assert(layout.stride(2) == 4);
  static_assert(layout.stride(1) == 8);
  static_assert(layout.stride(0) == 24);

  // round-trip: flat(i,j,k,l) → operator[] → {i,j,k,l}
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 4; ++l) {
          auto idx = layout[static_cast<int>(layout.flat(i, j, k, l))];
          EXPECT_EQ(idx[0], i);
          EXPECT_EQ(idx[1], j);
          EXPECT_EQ(idx[2], k);
          EXPECT_EQ(idx[3], l);
        }
}

TEST(TilingTest, TileLayoutStaticLeft) {
  // Static src + static tile → fully compile-time StaticTileLayoutLeft<4,3,2,4>
  using Src  = StaticTileLayoutLeft<8, 12>;
  using Tile = StaticTile<2, 4>;
  using R    = decltype(tile_layout(Src{}, Tile{}));
  static_assert(std::is_same_v<R, StaticTileLayoutLeft<4, 3, 2, 4>>);

  constexpr R layout{};
  static_assert(layout.rank == 4);
  static_assert(layout.extent(0) == 4);
  static_assert(layout.extent(1) == 3);
  static_assert(layout.extent(2) == 2);
  static_assert(layout.extent(3) == 4);
  // Left strides: fastest dim first
  static_assert(layout.stride(0) == 1);
  static_assert(layout.stride(1) == 4);
  static_assert(layout.stride(2) == 12);
  static_assert(layout.stride(3) == 24);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 4; ++l) {
          auto idx = layout[static_cast<int>(layout.flat(i, j, k, l))];
          EXPECT_EQ(idx[0], i);
          EXPECT_EQ(idx[1], j);
          EXPECT_EQ(idx[2], k);
          EXPECT_EQ(idx[3], l);
        }
}

TEST(TilingTest, TileLayoutDynamicSrcStaticTileRight) {
  // Dynamic src + static tile → DynamicTileLayoutRight<4>
  auto src = DynamicTileLayoutRight<2>{Kokkos::Array<int, 2>{8, 12}};
  auto tl  = tile_layout(src, StaticTile<2, 4>{});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutRight<4>>);

  EXPECT_EQ(tl.rank, 4);
  EXPECT_EQ(tl.extent(0), 4);  // outer: 8/2
  EXPECT_EQ(tl.extent(1), 3);  // outer: 12/4
  EXPECT_EQ(tl.extent(2), 2);  // inner tile size
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(3), 1);
  EXPECT_EQ(tl.stride(2), 4);
  EXPECT_EQ(tl.stride(1), 8);
  EXPECT_EQ(tl.stride(0), 24);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 4; ++l) {
          auto idx =
              tl[static_cast<int>(tl.flat(Kokkos::Array<int, 4>{i, j, k, l}))];
          EXPECT_EQ(idx[0], i);
          EXPECT_EQ(idx[1], j);
          EXPECT_EQ(idx[2], k);
          EXPECT_EQ(idx[3], l);
        }
}

TEST(TilingTest, TileLayoutDynamicSrcStaticTileLeft) {
  // Dynamic src + static tile → DynamicTileLayoutLeft<4>
  auto src = DynamicTileLayoutLeft<2>{Kokkos::Array<int, 2>{8, 12}};
  auto tl  = tile_layout(src, StaticTile<2, 4>{});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutLeft<4>>);

  EXPECT_EQ(tl.extent(0), 4);
  EXPECT_EQ(tl.extent(1), 3);
  EXPECT_EQ(tl.extent(2), 2);
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(0), 1);
  EXPECT_EQ(tl.stride(1), 4);
  EXPECT_EQ(tl.stride(2), 12);
  EXPECT_EQ(tl.stride(3), 24);

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 4; ++l) {
          auto idx =
              tl[static_cast<int>(tl.flat(Kokkos::Array<int, 4>{i, j, k, l}))];
          EXPECT_EQ(idx[0], i);
          EXPECT_EQ(idx[1], j);
          EXPECT_EQ(idx[2], k);
          EXPECT_EQ(idx[3], l);
        }
}

TEST(TilingTest, TileLayoutStaticSrcDynamicTileRight) {
  // Static src + dynamic tile → DynamicTileLayoutRight<4>
  auto tl = tile_layout(StaticTileLayoutRight<8, 12>{}, DynamicTile<2>{{2, 4}});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutRight<4>>);

  EXPECT_EQ(tl.extent(0), 4);
  EXPECT_EQ(tl.extent(1), 3);
  EXPECT_EQ(tl.extent(2), 2);
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(3), 1);
  EXPECT_EQ(tl.stride(0), 24);
}

TEST(TilingTest, TileLayoutStaticSrcDynamicTileLeft) {
  // Static src + dynamic tile → DynamicTileLayoutLeft<4>
  auto tl = tile_layout(StaticTileLayoutLeft<8, 12>{}, DynamicTile<2>{{2, 4}});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutLeft<4>>);

  EXPECT_EQ(tl.extent(0), 4);
  EXPECT_EQ(tl.extent(1), 3);
  EXPECT_EQ(tl.extent(2), 2);
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(0), 1);
  EXPECT_EQ(tl.stride(3), 24);
}

TEST(TilingTest, TileLayoutDynamicRight) {
  // Dynamic src + dynamic tile → DynamicTileLayoutRight<4>
  auto src = DynamicTileLayoutRight<2>{Kokkos::Array<int, 2>{8, 12}};
  auto tl  = tile_layout(src, DynamicTile<2>{{2, 4}});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutRight<4>>);

  EXPECT_EQ(tl.extent(0), 4);
  EXPECT_EQ(tl.extent(1), 3);
  EXPECT_EQ(tl.extent(2), 2);
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(3), 1);
  EXPECT_EQ(tl.stride(0), 24);
}

TEST(TilingTest, TileLayoutDynamicLeft) {
  // Dynamic src + dynamic tile → DynamicTileLayoutLeft<4>
  auto src = DynamicTileLayoutLeft<2>{Kokkos::Array<int, 2>{8, 12}};
  auto tl  = tile_layout(src, DynamicTile<2>{{2, 4}});
  static_assert(std::is_same_v<decltype(tl), DynamicTileLayoutLeft<4>>);

  EXPECT_EQ(tl.extent(0), 4);
  EXPECT_EQ(tl.extent(1), 3);
  EXPECT_EQ(tl.extent(2), 2);
  EXPECT_EQ(tl.extent(3), 4);
  EXPECT_EQ(tl.stride(0), 1);
  EXPECT_EQ(tl.stride(3), 24);
}

// ---------------------------------------------------------------------------
// prefix_product — collapse a tile to a 2D [before, after] shape at a split.
// ---------------------------------------------------------------------------
TEST(TilingTest, PrefixProductStaticMidSplit) {
  using R = decltype(prefix_product(StaticTile<2, 4, 8>{}, rank_c<1>));
  static_assert(std::is_same_v<R, StaticTile<2, 32>>);  // [2] x [4*8]
  static_assert(R::extent(0) == 2);
  static_assert(R::extent(1) == 32);
  SUCCEED();
}

TEST(TilingTest, PrefixProductStaticSplitZero) {
  using R = decltype(prefix_product(StaticTile<2, 4, 8>{}, rank_c<0>));
  static_assert(std::is_same_v<R, StaticTile<1, 64>>);  // [] x [2*4*8]
  SUCCEED();
}

TEST(TilingTest, PrefixProductStaticSplitFull) {
  using R = decltype(prefix_product(StaticTile<2, 4, 8>{}, rank_c<3>));
  static_assert(std::is_same_v<R, StaticTile<64, 1>>);  // [2*4*8] x []
  SUCCEED();
}

TEST(TilingTest, PrefixProductDynamic) {
  auto tile   = DynamicTile<3>{{2, 4, 8}};
  auto result = prefix_product(tile, rank_c<1>);
  static_assert(decltype(result)::rank == 2);  // always 2D: [before, after]
  EXPECT_EQ(result.extent(0), 2);              // before split: 2
  EXPECT_EQ(result.extent(1), 32);             // from split:   4*8
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
