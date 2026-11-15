#include "kernel_case.hpp"

using namespace sfpp_min;
using namespace sfpp_min::test;

namespace {

template <typename ScratchLayout = Kokkos::LayoutLeft, typename Off>
ErrorReport run_and_compare(Case<Off>& k) {
  dummy_stiffness<ScratchLayout>(make_args(k, make_gll_views()));
  return compare_to_oracle(k);
}

template <typename Off>
void expect_layout_invariant(const char* name) {
  auto k = make_linear_case<Off>();
  expect_matches_oracle(run_and_compare(k), name);
}

template <typename Off>
void expect_aos_invariant(const char* name) {
  auto               k = make_linear_case<Off>();
  MetricsAoS<Off>    m(k.set.nspec());
  PropertiesAoS<Off> p(k.set.nspec());
  fill_aos(k, m, p);
  k.g.to_device();
  dummy_stiffness(make_args(k, make_gll_views(), m, p, k.g.map));
  expect_matches_oracle(compare_to_oracle(k), name);
}

}  // namespace

TEST(SfppMinKernel, MatchesSerialOracleStaticExtents) {
  expect_layout_invariant<ChunkTiledOffset>("static extents");
}

TEST(SfppMinKernel, MatchesSerialOracleDynamicExtents) {
  expect_layout_invariant<ChunkTiledDynamicOffset>("dynamic extents");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutRight) {
  expect_layout_invariant<PlainOffset>("LayoutRight (PlainOffset)");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutRightDynamic) {
  expect_layout_invariant<LayoutRightDynamicOffset>("LayoutRight dynamic");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutLeft) {
  expect_layout_invariant<LayoutLeftOffset>("LayoutLeft");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutLeftDynamic) {
  expect_layout_invariant<LayoutLeftDynamicOffset>("LayoutLeft dynamic");
}

TEST(SfppMinKernel, MatchesSerialOracleAoSStaticExtents) {
  expect_aos_invariant<ChunkTiledOffset>("AoS static extents");
}

TEST(SfppMinKernel, MatchesSerialOracleAoSDynamicExtents) {
  expect_aos_invariant<ChunkTiledDynamicOffset>("AoS dynamic extents");
}

// Scratch LayoutRight is a reshuffle of team-scratch storage; the numeric
// result must be scratch-layout-invariant.
TEST(SfppMinKernel, MatchesSerialOracleScratchLayoutRight) {
  auto k = make_linear_case<ChunkTiledDynamicOffset>();
  expect_matches_oracle(run_and_compare<Kokkos::LayoutRight>(k),
                        "scratch LayoutRight");
}

TEST(SfppMinKernel, RigidBodyTranslationIsAtTheFloatRoundoffFloor) {
  EXPECT_LT(rigid_body_ratio([](const auto& a) { dummy_stiffness(a); }), 1e-5);
}

TEST(SfppMinKernel, ResolvedTeamSizeAndScratchMatchTheGroundTruth) {
  std::printf("[ INFO     ] dummy_shmem_size() = %zu B\n", dummy_shmem_size());
  EXPECT_EQ(dummy_shmem_size(), 24124u)
      << "scratch request must be 6008 + 18008 + 108; the launch adds "
         "8*(team_size+2) on top to reach the measured 26188 B";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
