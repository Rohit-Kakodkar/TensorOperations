#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <utility>

using namespace TensorOperations;

namespace {

using ES = Kokkos::Cuda;

constexpr int N = 5, TE = 2, E = 7, EP = 8;

template <typename Layout>
using View4 = Kokkos::View<float****, Layout, ES>;
using ViewR = View4<Kokkos::LayoutRight>;
using ViewL = View4<Kokkos::LayoutLeft>;
using View2 = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using Count = Kokkos::View<int, ES>;

using Map =
    LabelTiles<LabelTile<'e', TE>, LabelWhole<'q', N>, LabelWhole<'a', N>,
               LabelWhole<'b', N>, LabelWhole<'c', N>>;

float value(int e, int a, int b, int c) {
  return std::sin(0.3f * e + 0.7f * a) + 0.1f * b - 0.2f * c + 1.0f;
}

template <typename V>
void fill_u(const V& v) {
  auto h = Kokkos::create_mirror_view(v);
  for (int e = 0; e < static_cast<int>(v.extent(0)); ++e)
    for (int a = 0; a < N; ++a)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c) h(e, a, b, c) = value(e, a, b, c);
  Kokkos::deep_copy(v, h);
}

void fill_h(const View2& v) {
  auto h = Kokkos::create_mirror_view(v);
  for (int q = 0; q < N; ++q)
    for (int a = 0; a < N; ++a) h(q, a) = 0.25f * q - 0.5f * a + 0.125f;
  Kokkos::deep_copy(v, h);
}

bool synced() { return cudaDeviceSynchronize() == cudaSuccess; }

struct Scale {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float v) const {
    return 2.0f * v + 0.5f;
  }
};

struct CheckedScale {
  Count                 bad;
  KOKKOS_FUNCTION float operator()(int e, int, int, int, float v) const {
    if (e >= E) Kokkos::atomic_inc(&bad());
    return 2.0f * v + 0.5f;
  }
};

struct CountSink {
  Count                calls, bad;
  ViewR                out;
  KOKKOS_FUNCTION void operator()(int e, int a, int b, int c, float v) const {
    Kokkos::atomic_inc(&calls());
    if (e >= E)
      Kokkos::atomic_inc(&bad());
    else
      out(e, a, b, c) = v;
  }
};

struct CheckedFn {
  Count                 bad;
  KOKKOS_FUNCTION float operator()(int e, int a, int b, int c) const {
    if (e >= E) Kokkos::atomic_inc(&bad());
    return static_cast<float>(e * 1000 + a * 100 + b * 10 + c);
  }
};

template <typename In>
auto contraction_graph(const View2& h, const In& u) {
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, sh] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(h))));
  auto [g2, su] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g3, c] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(sh, su));
  auto [g4, p] = g3.add(make_combine_node<'q', 'e', 'b', 'c'>(c, Scale{}));
  return g4.outputs(p);
}

template <typename Layout>
void contraction_matches_padded_run() {
  View2         h("h", N, N);
  View4<Layout> u("u", E, N, N, N);
  ViewR         up("up", EP, N, N, N);
  ViewR         got("got", N, E, N, N), want("want", N, EP, N, N);
  fill_h(h);
  fill_u(u);
  fill_u(up);
  EXPECT_EQ(contraction_graph(h, u).execute(CutePolicyTag<>{}, got), 4);
  ASSERT_TRUE(synced());
  EXPECT_EQ(contraction_graph(h, up).execute(CutePolicyTag<>{}, want), 4);
  ASSERT_TRUE(synced());
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int q = 0; q < N; ++q)
    for (int e = 0; e < E; ++e)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c)
          if (hg(q, e, b, c) != hw(q, e, b, c)) ++bad;
  EXPECT_EQ(bad, 0);
}

}  // namespace

TEST(CutePartialTile, StageRootLeavesTheRowBeyondTheViewAlone) {
  ViewR u("u", EP, N, N, N), backing("backing", EP, N, N, N);
  fill_u(u);
  Kokkos::deep_copy(backing, -7.0f);
  const auto rows = std::make_pair(0, E);
  auto ui = Kokkos::subview(u, rows, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  auto oi =
      Kokkos::subview(backing, rows, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(ui))));
  EXPECT_EQ(g1.outputs(s).execute(CutePolicyTag<>{}, oi), 4);
  ASSERT_TRUE(synced());

  auto hu  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, backing);
  int  bad = 0;
  for (int e = 0; e < EP; ++e)
    for (int a = 0; a < N; ++a)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c)
          if (ho(e, a, b, c) != (e < E ? hu(e, a, b, c) : -7.0f)) ++bad;
  EXPECT_EQ(bad, 0);
}

TEST(CutePartialTile, ContractionMatchesThePaddedRunLayoutRight) {
  contraction_matches_padded_run<Kokkos::LayoutRight>();
}

TEST(CutePartialTile, ContractionMatchesThePaddedRunLayoutLeft) {
  contraction_matches_padded_run<Kokkos::LayoutLeft>();
}

TEST(CutePartialTile,
     FunctionalInputAndCombineNeverSeeAnOverhangingCoordinate) {
  Count bad("bad");
  ViewR out("out", E, N, N, N);
  auto  g0 = make_level_graph<float, ES>(Map{});
  auto [g1, s] =
      g0.add(make_stage_node(make_functional_input_node<ES, 'e', 'a', 'b', 'c'>(
          Kokkos::LayoutRight{E, N, N, N}, CheckedFn{bad})));
  auto [g2, r] =
      g1.add(make_combine_node<'e', 'a', 'b', 'c'>(s, CheckedScale{bad}));
  EXPECT_EQ(g2.outputs(r).execute(CutePolicyTag<>{}, out), 4);
  ASSERT_TRUE(synced());

  int nbad = -1;
  Kokkos::deep_copy(nbad, bad);
  EXPECT_EQ(nbad, 0);
  auto ho    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  wrong = 0;
  for (int e = 0; e < E; ++e)
    for (int a = 0; a < N; ++a)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c)
          if (ho(e, a, b, c) != 2.0f * (e * 1000 + a * 100 + b * 10 + c) + 0.5f)
            ++wrong;
  EXPECT_EQ(wrong, 0);
}

TEST(CutePartialTile, SinkFiresOncePerValidPoint) {
  Count calls("calls"), bad("bad");
  ViewR u("u", E, N, N, N), out("out", E, N, N, N);
  fill_u(u);
  auto g0      = make_level_graph<float, ES>(Map{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto g2 = g1.add(
      make_combine_node<'e', 'a', 'b', 'c'>(s, CountSink{calls, bad, out}));
  EXPECT_EQ(g2.outputs().execute(CutePolicyTag<>{}), 4);
  ASSERT_TRUE(synced());

  int ncalls = -1, nbad = -1;
  Kokkos::deep_copy(ncalls, calls);
  Kokkos::deep_copy(nbad, bad);
  EXPECT_EQ(ncalls, E * N * N * N);
  EXPECT_EQ(nbad, 0);
}

TEST(CutePartialTile, TwoGriddedLabelsBothOverhang) {
  using Map2 = LabelTiles<LabelTile<'e', 2>, LabelTile<'a', 2>,
                          LabelWhole<'b', N>, LabelWhole<'c', N>>;
  ViewR u("u", E, N, N, N), out("out", E, N, N, N);
  fill_u(u);
  auto g0      = make_level_graph<float, ES>(Map2{});
  auto [g1, s] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(u))));
  auto [g2, r] = g1.add(make_combine_node<'e', 'a', 'b', 'c'>(s, Scale{}));
  EXPECT_EQ(g2.outputs(r).execute(CutePolicyTag<>{}, out), 12);
  ASSERT_TRUE(synced());

  auto hu  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int e = 0; e < E; ++e)
    for (int a = 0; a < N; ++a)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c)
          if (ho(e, a, b, c) != 2.0f * hu(e, a, b, c) + 0.5f) ++bad;
  EXPECT_EQ(bad, 0);
}

namespace {

using View1 = Kokkos::View<float*, ES>;

struct Scale1 {
  KOKKOS_FUNCTION float operator()(int, float v) const {
    return 2.0f * v + 0.5f;
  }
};

}  // namespace

TEST(CutePartialTile, RankOneTile) {
  constexpr int n = 10;
  View1         u("u", n), out("out", n);
  auto          hu = Kokkos::create_mirror_view(u);
  for (int e = 0; e < n; ++e) hu(e) = 0.25f * e;
  Kokkos::deep_copy(u, hu);

  auto g0      = make_level_graph<float, ES>(LabelTiles<LabelTile<'e', 4>>{});
  auto [g1, s] = g0.add(make_stage_node(make_input_node(make_handle<'e'>(u))));
  auto [g2, r] = g1.add(make_combine_node<'e'>(s, Scale1{}));
  EXPECT_EQ(g2.outputs(r).execute(CutePolicyTag<>{}, out), 3);
  ASSERT_TRUE(synced());

  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (int e = 0; e < n; ++e) EXPECT_EQ(ho(e), 2.0f * hu(e) + 0.5f) << e;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int r = RUN_ALL_TESTS();
  Kokkos::finalize();
  return r;
}
