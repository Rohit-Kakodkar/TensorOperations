#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <mesh.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace sfpp_min;

namespace {

constexpr MeshDims kSmall{4, 3, 2};
constexpr MeshDims kPerf{60, 48, 9};

struct Built {
  ElementSet set;
  IglobMap   g;
  int        nglob = 0;
};

Built build(const MeshDims& d, const ElementSet& set, bool access_order) {
  Built b;
  b.set   = set;
  b.g     = IglobMap(set.nspec());
  b.nglob = access_order ? renumber_access_order(d, set, b.g)
                         : renumber_grid_order(d, set, b.g);
  return b;
}

}  // namespace

TEST(SfppMinMesh, WorkItemDecompositionIsElementFastest) {
  for (int i = 0; i < kWorkItemsPerTeam; ++i) {
    const WorkItem w = decompose(i);
    ASSERT_GE(w.ielement, 0);
    ASSERT_LT(w.ielement, kExecChunk);
    ASSERT_LT(w.iz, NGLL);
    ASSERT_LT(w.iy, NGLL);
    ASSERT_LT(w.ix, NGLL);
  }
  for (int i = 0; i + 1 < kExecChunk; ++i) {
    const WorkItem a = decompose(i);
    const WorkItem b = decompose(i + 1);
    ASSERT_EQ(b.ielement - a.ielement, 1);
    ASSERT_EQ(a.iz, b.iz);
    ASSERT_EQ(a.iy, b.iy);
    ASSERT_EQ(a.ix, b.ix);
  }
  const WorkItem w4 = decompose(kExecChunk);
  EXPECT_EQ(w4.ielement, 0);
  EXPECT_EQ(w4.ix, 1);
}

TEST(SfppMinMesh, WorkItemDecompositionIsABijection) {
  std::vector<char> seen(kWorkItemsPerTeam, 0);
  for (int i = 0; i < kWorkItemsPerTeam; ++i) {
    const WorkItem w = decompose(i);
    const int      k =
        ((w.iz * NGLL + w.iy) * NGLL + w.ix) * kExecChunk + w.ielement;
    ASSERT_EQ(seen[k], 0) << "i=" << i;
    seen[k] = 1;
  }
}

TEST(SfppMinMesh, InteriorSetMatchesTheMeasuredElementCount) {
  const auto interior = interior_elements(kPerf);
  EXPECT_EQ(kPerf.nspec_grid(), 25920);
  EXPECT_EQ(interior.nspec(), 21344);
  EXPECT_EQ(interior.nspec(), 58 * 46 * 8);
  EXPECT_EQ(num_teams(interior.nspec()), 5336);
  EXPECT_EQ(kPerf.nspec_grid() - interior.nspec(), 4576);
  EXPECT_EQ(num_teams(kPerf.nspec_grid() - interior.nspec()), 1144);
}

TEST(SfppMinMesh, NodeCountMatchesTheClosedFormOnTheFullGrid) {
  const auto b = build(kSmall, all_elements(kSmall), true);
  EXPECT_EQ(static_cast<std::size_t>(b.nglob), kSmall.nnode_grid());
  EXPECT_EQ(kSmall.nnode_grid(), 17u * 13u * 9u);
  EXPECT_EQ(kPerf.nnode_grid(), 241u * 193u * 37u);
  EXPECT_EQ(kPerf.nnode_grid(), 1720981u);
}

TEST(SfppMinMesh, AdjacentElementsShareAFace) {
  const auto  b = build(kSmall, all_elements(kSmall), true);
  const auto& h = b.g.h_map;
  for (int ez = 0; ez < kSmall.nez; ++ez)
    for (int ey = 0; ey < kSmall.ney; ++ey)
      for (int ex = 0; ex + 1 < kSmall.nex; ++ex) {
        const int a = grid_element_index(kSmall, ex, ey, ez);
        const int c = grid_element_index(kSmall, ex + 1, ey, ez);
        for (int iz = 0; iz < NGLL; ++iz)
          for (int iy = 0; iy < NGLL; ++iy)
            ASSERT_EQ(h(a, iz, iy, DEG), h(c, iz, iy, 0));
      }
  for (int ez = 0; ez + 1 < kSmall.nez; ++ez)
    for (int ey = 0; ey < kSmall.ney; ++ey)
      for (int ex = 0; ex < kSmall.nex; ++ex) {
        const int a = grid_element_index(kSmall, ex, ey, ez);
        const int c = grid_element_index(kSmall, ex, ey, ez + 1);
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix)
            ASSERT_EQ(h(a, DEG, iy, ix), h(c, 0, iy, ix));
      }
}

TEST(SfppMinMesh, NumberingIsAPermutationOfZeroToNglob) {
  for (bool access : {true, false}) {
    const auto       b = build(kSmall, all_elements(kSmall), access);
    std::vector<int> hits(b.nglob, 0);
    const auto&      h = b.g.h_map;
    for (int ispec = 0; ispec < b.set.nspec(); ++ispec)
      for (int iz = 0; iz < NGLL; ++iz)
        for (int iy = 0; iy < NGLL; ++iy)
          for (int ix = 0; ix < NGLL; ++ix) {
            const int v = h(ispec, iz, iy, ix);
            ASSERT_GE(v, 0);
            ASSERT_LT(v, b.nglob);
            ++hits[v];
          }
    EXPECT_EQ(std::count(hits.begin(), hits.end(), 0), 0);
  }
}

TEST(SfppMinMesh, InteriorPointsBelongToExactlyOneElement) {
  const auto       b = build(kSmall, all_elements(kSmall), true);
  const auto&      h = b.g.h_map;
  std::vector<int> hits(b.nglob, 0);
  for (int ispec = 0; ispec < b.set.nspec(); ++ispec)
    for (int iz = 1; iz < DEG; ++iz)
      for (int iy = 1; iy < DEG; ++iy)
        for (int ix = 1; ix < DEG; ++ix) ++hits[h(ispec, iz, iy, ix)];
  for (int v = 0; v < b.nglob; ++v) ASSERT_LE(hits[v], 1);
  const int interior = (DEG - 1) * (DEG - 1) * (DEG - 1) * b.set.nspec();
  EXPECT_EQ(std::count(hits.begin(), hits.end(), 1), interior);
}

TEST(SfppMinMesh, PartialStorageChunkIsNotSkipped) {
  const MeshDims d{5, 5, 3};
  ASSERT_NE(d.nspec_grid() % kStorageChunk, 0);
  const auto b = build(d, all_elements(d), true);
  EXPECT_EQ(static_cast<std::size_t>(b.nglob), d.nnode_grid());
  const auto& h = b.g.h_map;
  for (int ispec = 0; ispec < b.set.nspec(); ++ispec)
    ASSERT_GE(h(ispec, 0, 0, 0), 0) << "ispec=" << ispec << " never assigned";
}

TEST(SfppMinMesh, PredictedAtomicSectorsMatchGroundTruth) {
  const auto   b = build(kPerf, interior_elements(kPerf), true);
  const double s = predicted_atomic_sectors_per_point(b.set, b.g, b.nglob);
  std::printf(
      "[ INFO     ] interior + access-order : %.4f sectors/point "
      "(ground truth 1.076)\n",
      s);
  EXPECT_NEAR(s, 1.076, 0.15 * 1.076);
}

TEST(SfppMinMesh, ElementSetAndNumberingBothMoveTheSectorCount) {
  const auto   ia = build(kPerf, interior_elements(kPerf), true);
  const auto   ig = build(kPerf, interior_elements(kPerf), false);
  const auto   aa = build(kPerf, all_elements(kPerf), true);
  const double s_ia =
      predicted_atomic_sectors_per_point(ia.set, ia.g, ia.nglob);
  const double s_ig =
      predicted_atomic_sectors_per_point(ig.set, ig.g, ig.nglob);
  const double s_aa =
      predicted_atomic_sectors_per_point(aa.set, aa.g, aa.nglob);
  std::printf(
      "[ INFO     ] interior/access = %.4f  interior/grid = %.4f  "
      "wholecube/access = %.4f   (truth 1.076)\n",
      s_ia, s_ig, s_aa);
  EXPECT_GT(s_ig, s_ia);
  EXPECT_LT(s_aa, s_ia);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
