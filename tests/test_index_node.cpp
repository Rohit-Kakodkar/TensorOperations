#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <type_traits>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

constexpr int N = 5, TE = 2, E = 8;

using Map = LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', N>,
                       LabelWhole<'j', N>, LabelWhole<'i', N>>;

template <typename Layout>
using IdxView = Kokkos::View<int****, Layout, ES>;
template <typename V, typename Layout = Kokkos::LayoutRight>
using ValView = Kokkos::View<V****, Layout, ES>;

int ival(int e, int k, int j, int i) {
  return 1000 * e + 97 * k + 13 * j + i + 7;
}
double uval(int e, int k, int j, int i) {
  return 0.2 + 0.11 * e - 0.07 * k + 0.13 * j - 0.03 * i +
         0.01 * (e + k) * (j + 1);
}

template <typename V>
void fill_index(const V& v) {
  auto h = Kokkos::create_mirror_view(v);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) h(e, k, j, i) = ival(e, k, j, i);
  Kokkos::deep_copy(v, h);
}

template <typename V>
void fill_values(const V& v) {
  using T = typename V::value_type;
  auto h  = Kokkos::create_mirror_view(v);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          h(e, k, j, i) = static_cast<T>(uval(e, k, j, i));
  Kokkos::deep_copy(v, h);
}

template <typename V>
int index_mismatches(const V& got) {
  auto h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  int  bad = 0;
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          if (h(e, k, j, i) != ival(e, k, j, i)) ++bad;
  return bad;
}

template <typename A, typename B>
int value_mismatches(const A& got, const B& want) {
  auto hg  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  auto hw  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, want);
  int  bad = 0;
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          if (hg(e, k, j, i) != hw(e, k, j, i)) ++bad;
  return bad;
}

template <typename V>
struct Scale {
  KOKKOS_INLINE_FUNCTION V operator()(int, int, int, int, V v) const {
    return V(2) * v + V(0.5);
  }
};

template <typename V>
struct WriteSink {
  ValView<V>                  out;
  KOKKOS_INLINE_FUNCTION void operator()(int e, int k, int j, int i,
                                         V v) const {
    out(e, k, j, i) = V(2) * v + V(0.5);
  }
};

using NodeLR = decltype(make_index_node<'e', 'k', 'j', 'i'>(
    std::declval<IdxView<Kokkos::LayoutRight>>()));
static_assert(Impl::has_node_tag_v<StagedTag, NodeLR>);
static_assert(std::is_same_v<NodeLR::value_type, int>);
static_assert(NodeLR::Rank == 4);
static_assert(std::is_same_v<Impl::lg_member_elem_t<NodeLR, float>, int>);
static_assert(std::is_same_v<Impl::lg_member_elem_t<NodeLR, double>, int>);

template <typename InLayout, typename OutLayout, typename V>
void index_root_copies() {
  IdxView<InLayout>  idx("idx", E, N, N, N);
  IdxView<OutLayout> out("out", E, N, N, N);
  fill_index(idx);
  Kokkos::deep_copy(out, -1);

  auto g0       = make_level_graph<V, ES>(Map{});
  auto [g1, ig] = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  static_assert(std::is_same_v<typename decltype(ig)::value_type, int>);

  EXPECT_EQ(g1.outputs(ig).execute(TeamPolicyTag<ES>{}, out), E / TE);
  Kokkos::fence();
  EXPECT_EQ(index_mismatches(out), 0);
}

template <typename V>
std::size_t int_tile_bytes() {
  return Impl::scratch_backing_t<V, ES>::shmem_size(
      Impl::slot_arena_step_of<V, int, ES>(TE * N * N * N));
}

}  // namespace

TEST(IndexNode, RootCopiesLayoutRight) {
  index_root_copies<Kokkos::LayoutRight, Kokkos::LayoutRight, float>();
}

TEST(IndexNode, RootCopiesLayoutLeftToLayoutRight) {
  index_root_copies<Kokkos::LayoutLeft, Kokkos::LayoutRight, float>();
}

TEST(IndexNode, RootCopiesLayoutRightToLayoutLeft) {
  index_root_copies<Kokkos::LayoutRight, Kokkos::LayoutLeft, float>();
}

TEST(IndexNode, RootCopiesInADoubleGraph) {
  index_root_copies<Kokkos::LayoutLeft, Kokkos::LayoutRight, double>();
}

TEST(IndexNode, IndexOnlyGraphSizesTheSlotAsInts) {
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  auto                         g0 = make_level_graph<double, ES>(Map{});
  auto [g1, ig]  = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  const auto out = g1.outputs(ig);
  EXPECT_EQ(out.scratch_bytes(), int_tile_bytes<double>());
  EXPECT_EQ(g1.slot_bytes(), int_tile_bytes<double>());

  ValView<double> u("u", E, N, N, N);
  auto [h1, su] = g0.add(
      make_stage_node(make_input_node(make_handle<'e', 'k', 'j', 'i'>(u))));
  EXPECT_LT(out.scratch_bytes(), h1.outputs(su).scratch_bytes());
}

TEST(IndexNode, UnreadIndexLevelLeavesFloatResultsBitwiseUnchanged) {
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  ValView<float>               u("u", E, N, N, N);
  ValView<float> with("with", E, N, N, N), without("w/o", E, N, N, N);
  fill_index(idx);
  fill_values(u);

  auto a0      = make_level_graph<float, ES>(Map{});
  auto [a1, s] = a0.add(
      make_stage_node(make_input_node(make_handle<'e', 'k', 'j', 'i'>(u))));
  auto [a2, r] =
      a1.add(make_combine_node<'e', 'k', 'j', 'i'>(s, Scale<float>{}));
  const auto out_a = a2.outputs(r);
  out_a.execute(TeamPolicyTag<ES>{}, without);

  auto b0       = make_level_graph<float, ES>(Map{});
  auto [b1, ig] = b0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  auto [b2, t]  = b1.add(
      make_stage_node(make_input_node(make_handle<'e', 'k', 'j', 'i'>(u))));
  auto [b3, q] =
      b2.add(make_combine_node<'e', 'k', 'j', 'i'>(t, Scale<float>{}));
  const auto out_b = b3.outputs(q);
  out_b.execute(TeamPolicyTag<ES>{}, with);
  Kokkos::fence();

  EXPECT_EQ(value_mismatches(with, without), 0);

  using LevelsB = typename decltype(b3)::levels_type;
  using Roots   = std::index_sequence<decltype(q)::SlotIdx>;
  static_assert(Impl::lg_slot_pool_v<LevelsB, Roots, decltype(ig)::SlotIdx> ==
                Impl::lg_slot_pool_v<LevelsB, Roots, decltype(t)::SlotIdx>);
  EXPECT_EQ(out_b.scratch_bytes(), out_a.scratch_bytes());
  constexpr std::size_t int_step =
      Impl::slot_arena_step_of<float, int, ES>(TE * N * N * N);
  EXPECT_EQ(b3.slot_bytes() - a2.slot_bytes(), int_step * sizeof(float));
}

template <typename V>
void live_index_beside_values() {
  IdxView<Kokkos::LayoutLeft>  idx("idx", E, N, N, N);
  IdxView<Kokkos::LayoutRight> iout("iout", E, N, N, N);
  ValView<V>                   u("u", E, N, N, N), got("got", E, N, N, N),
      want("want", E, N, N, N);
  fill_index(idx);
  fill_values(u);
  Kokkos::deep_copy(got, V(-7));

  auto g0       = make_level_graph<V, ES>(Map{});
  auto [g1, ig] = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  auto [g2, s]  = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'k', 'j', 'i'>(u))));
  auto g3 = g2.add(make_combine_node<'e', 'k', 'j', 'i'>(s, WriteSink<V>{got}));
  g3.outputs(ig).execute(TeamPolicyTag<ES>{}, iout);
  Kokkos::fence();

  using Levels = typename decltype(g3)::levels_type;
  using Roots  = std::index_sequence<decltype(ig)::SlotIdx>;
  static_assert(Impl::lg_slot_pool_v<Levels, Roots, decltype(ig)::SlotIdx> !=
                Impl::lg_slot_pool_v<Levels, Roots, decltype(s)::SlotIdx>);

  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  auto hw = Kokkos::create_mirror_view(want);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          hw(e, k, j, i) = V(2) * hu(e, k, j, i) + V(0.5);
  Kokkos::deep_copy(want, hw);

  EXPECT_EQ(index_mismatches(iout), 0);
  EXPECT_EQ(value_mismatches(got, want), 0);
}

TEST(IndexNode, LiveIndexSlotBesideFloatSlotsStaysDisjoint) {
  live_index_beside_values<float>();
}

TEST(IndexNode, LiveIndexSlotBesideDoubleSlotsStaysDisjoint) {
  live_index_beside_values<double>();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int r = RUN_ALL_TESTS();
  Kokkos::finalize();
  return r;
}
