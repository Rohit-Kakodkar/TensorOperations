#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

namespace {

using ES = Kokkos::Cuda;

constexpr int N = 5;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using View4 = Kokkos::View<float****, Kokkos::LayoutRight, ES>;

template <int TE>
using Map =
    LabelTiles<LabelWhole<'q', N>, LabelWhole<'a', N>, LabelTile<'e', TE>,
               LabelWhole<'b', N>, LabelWhole<'c', N>>;

template <int TE>
struct Contraction {
  ViewH H;
  View4 U;
  View4 C;

  explicit Contraction(int E)
      : H("H", N, N), U("U", E, N, N, N), C("C", N, E, N, N) {
    Kokkos::deep_copy(H, 0.5f);
    Kokkos::deep_copy(U, 0.25f);
    Kokkos::deep_copy(C, -999.0f);
  }

  auto outputs() const {
    auto g0 = make_level_graph<float, ES>(Map<TE>{});
    auto [g1, h] =
        g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(H))));
    auto [g2, u] = g1.add(
        make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(U))));
    auto [g3, c] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(h, u));
    return g3.outputs(c);
  }
};

bool synced() {
  return cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
}

int changed_from_sentinel(const View4& v) {
  auto h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  int  bad = 0;
  for (std::size_t n = 0; n < h.size(); ++n)
    if (h.data()[n] != -999.0f) ++bad;
  return bad;
}

}  // namespace

TEST(CuteLevelGraph, LeagueMatchesTeamBackend) {
  constexpr int   E = 8, TE = 2;
  Contraction<TE> p(E);
  const auto      out = p.outputs();

  const int cute_league = out.execute(CutePolicyTag<>{}, p.C);
  ASSERT_TRUE(synced());
  const int team_league = out.execute(TeamPolicyTag<ES>{}, p.C);
  ASSERT_TRUE(synced());

  EXPECT_EQ(cute_league, E / TE);
  EXPECT_EQ(cute_league, team_league);
}

TEST(CuteLevelGraph, EmptyKernelWritesNothing) {
  Contraction<2> p(8);
  p.outputs().execute(CutePolicyTag<>{}, p.C);
  ASSERT_TRUE(synced());
  EXPECT_EQ(changed_from_sentinel(p.C), 0);
}

TEST(CuteLevelGraph, ExplicitTeamSize) {
  Contraction<2> p(8);
  EXPECT_EQ(p.outputs().team_size(64).execute(CutePolicyTag<>{}, p.C), 4);
  ASSERT_TRUE(synced());
}

TEST(CuteLevelGraph, ScratchAbove48KiB) {
  constexpr int   E = 100, TE = 50;
  Contraction<TE> p(E);
  const auto      out = p.outputs();
  ASSERT_GT(out.scratch_bytes(), std::size_t{48 * 1024});

  EXPECT_EQ(out.execute(CutePolicyTag<>{}, p.C), E / TE);
  ASSERT_TRUE(synced());
  EXPECT_EQ(changed_from_sentinel(p.C), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
