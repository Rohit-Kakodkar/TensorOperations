#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

using namespace TensorOperations;

namespace {

using ES = Kokkos::Cuda;

static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<cute::Shape<cute::_5, cute::_5, cute::_5>,
                                      std::integer_sequence<int, 0, 1, 2>, 128>,
              cute::Layout<cute::Shape<cute::_5, cute::_5, cute::_5>,
                           cute::Stride<cute::_1, cute::_5, cute::Int<25>>>>);
static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<
                  cute::Shape<cute::_2, cute::_5, cute::_5, cute::_5>,
                  std::integer_sequence<int, 3, 2, 1, 0>, 128>,
              cute::Layout<cute::Shape<cute::_1, cute::_5, cute::_5, cute::_5>,
                           cute::Stride<cute::Int<125>, cute::Int<25>, cute::_5,
                                        cute::_1>>>);
static_assert(std::is_same_v<
              Impl::cute_thr_layout_t<cute::Shape<cute::_3, cute::_16>,
                                      std::integer_sequence<int, 0, 1>, 32>,
              cute::Layout<cute::Shape<cute::_3, cute::_8>,
                           cute::Stride<cute::_1, cute::_3>>>);

constexpr int N = 5;

template <typename Layout>
using View4 = Kokkos::View<float****, Layout, ES>;
using ViewR = View4<Kokkos::LayoutRight>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

template <int TE>
using Map4 = LabelTiles<LabelTile<'e', TE>, LabelWhole<'a', N>,
                        LabelWhole<'b', N>, LabelWhole<'c', N>>;

template <typename V>
void fill(const V& v, float seed) {
  auto h = Kokkos::create_mirror_view(v);
  for (std::size_t n = 0; n < h.size(); ++n)
    h.data()[n] =
        seed + 0.25f * static_cast<float>(n) - static_cast<float>((n * 7) % 11);
  Kokkos::deep_copy(v, h);
}

template <typename A, typename B>
int mismatches(const A& got, const B& want) {
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int e = 0; e < static_cast<int>(hw.extent(0)); ++e)
    for (int a = 0; a < static_cast<int>(hw.extent(1)); ++a)
      for (int b = 0; b < static_cast<int>(hw.extent(2)); ++b)
        for (int c = 0; c < static_cast<int>(hw.extent(3)); ++c)
          if (hg(e, a, b, c) != hw(e, a, b, c)) ++bad;
  return bad;
}

template <typename A, typename B>
int mismatches2(const A& got, const B& want) {
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int i = 0; i < static_cast<int>(hw.extent(0)); ++i)
    for (int j = 0; j < static_cast<int>(hw.extent(1)); ++j)
      if (hg(i, j) != hw(i, j)) ++bad;
  return bad;
}

bool synced() {
  return cudaGetLastError() == cudaSuccess &&
         cudaDeviceSynchronize() == cudaSuccess;
}

template <int TE, typename Policy, typename InView>
void single_stage_root(int E, Policy policy, int expected_league) {
  InView u("u", E, N, N, N);
  ViewR  cute_out("cute_out", E, N, N, N), team_out("team_out", E, N, N, N);
  fill(u, 1.0f);
  Kokkos::deep_copy(cute_out, -999.0f);

  auto g0      = make_level_graph<float, ES>(Map4<TE>{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  const auto out = g1.outputs(s);

  EXPECT_EQ(out.execute(policy, cute_out), expected_league);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, team_out);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches(cute_out, u), 0);
  EXPECT_EQ(mismatches(cute_out, team_out), 0);
}

}  // namespace

TEST(CuteLevelGraph, SingleStageRootCopiesTheInput) {
  single_stage_root<2, CutePolicyTag<>, ViewR>(8, CutePolicyTag<>{}, 4);
}

TEST(CuteLevelGraph, LayoutLeftInputToLayoutRightOutput) {
  single_stage_root<2, CutePolicyTag<>, View4<Kokkos::LayoutLeft>>(
      8, CutePolicyTag<>{}, 4);
}

TEST(CuteLevelGraph, IdleThreadsWithDefaultBlock) {
  single_stage_root<1, CutePolicyTag<>, ViewR>(6, CutePolicyTag<>{}, 6);
}

TEST(CuteLevelGraph, SmallerBlockStillCoversTheTile) {
  single_stage_root<1, CutePolicyTag<ES, 32>, ViewR>(6, CutePolicyTag<ES, 32>{},
                                                     6);
}

TEST(CuteLevelGraph, TwoMemberStageLevelBothRoots) {
  constexpr int E = 8;
  ViewR         u("u", E, N, N, N), w("w", E, N, N, N);
  ViewR         cu("cu", E, N, N, N), cw("cw", E, N, N, N);
  ViewR         tu("tu", E, N, N, N), tw("tw", E, N, N, N);
  fill(u, 1.0f);
  fill(w, -3.0f);

  auto g0           = make_level_graph<float, ES>(Map4<2>{});
  auto [g1, su, sw] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))),
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(w))));
  const auto out = g1.outputs(su, sw);

  EXPECT_EQ(out.execute(CutePolicyTag<>{}, cu, cw), E / 2);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, tu, tw);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches(cu, u), 0);
  EXPECT_EQ(mismatches(cw, w), 0);
  EXPECT_EQ(mismatches(cu, tu), 0);
  EXPECT_EQ(mismatches(cw, tw), 0);
}

TEST(CuteLevelGraph, TwoStageLevelsWithDifferentTiles) {
  constexpr int E = 12, B = 5, C = 3;
  using Map =
      LabelTiles<LabelTile<'e', 4>, LabelWhole<'b', B>, LabelWhole<'c', C>>;
  View2 a("a", E, B), b("b", E, C);
  View2 ca("ca", E, B), cb("cb", E, C), ta("ta", E, B), tb("tb", E, C);
  fill(a, 2.0f);
  fill(b, -1.0f);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, sa] =
      g0.add(make_stage_node(make_input_node(make_handle<'e', 'b'>(a))));
  auto [g2, sb] =
      g1.add(make_stage_node(make_input_node(make_handle<'e', 'c'>(b))));
  const auto out = g2.outputs(sa, sb);

  EXPECT_EQ(out.execute(CutePolicyTag<ES, 16>{}, ca, cb), E / 4);
  ASSERT_TRUE(synced());
  out.execute(TeamPolicyTag<ES>{}, ta, tb);
  ASSERT_TRUE(synced());

  EXPECT_EQ(mismatches2(ca, a), 0);
  EXPECT_EQ(mismatches2(cb, b), 0);
  EXPECT_EQ(mismatches2(ca, ta), 0);
  EXPECT_EQ(mismatches2(cb, tb), 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
