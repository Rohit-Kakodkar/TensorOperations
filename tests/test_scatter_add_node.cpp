#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <tuple>
#include <type_traits>
#include <vector>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

constexpr int N = 5, TE = 2, E = 8, NGLOB = 211;

using Map = LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', N>,
                       LabelWhole<'j', N>, LabelWhole<'i', N>>;

template <typename Layout>
using IdxView = Kokkos::View<int****, Layout, ES>;
using ValView = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Vec     = Kokkos::View<float*, ES>;
using FieldL  = Kokkos::View<float**, Kokkos::LayoutLeft, ES>;
using FieldR  = Kokkos::View<float**, Kokkos::LayoutRight, ES>;

int gid(int e, int k, int j, int i) {
  return (97 * e + 31 * k + 7 * j + 3 * i * i + 5) % NGLOB;
}
float vval(int e, int k, int j, int i, int c) {
  return 0.25f *
         static_cast<float>((13 * e + 7 * k - 5 * j + 3 * i + 11 * c) % 29 - 9);
}

template <typename V>
void fill_index(const V& v) {
  auto h = Kokkos::create_mirror_view(v);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) h(e, k, j, i) = gid(e, k, j, i);
  Kokkos::deep_copy(v, h);
}

ValView make_values(int c) {
  ValView v("v", E, N, N, N);
  auto    h = Kokkos::create_mirror_view(v);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) h(e, k, j, i) = vval(e, k, j, i, c);
  Kokkos::deep_copy(v, h);
  return v;
}

template <typename F>
std::vector<float> reference(F value) {
  std::vector<float> r(NGLOB, 0.0f);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) r[gid(e, k, j, i)] += value(e, k, j, i);
  return r;
}

template <typename D>
int mismatches(const D& d, const std::vector<float>& want) {
  auto h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
  int  bad = 0;
  for (int g = 0; g < NGLOB; ++g)
    if (h(g) != want[g]) ++bad;
  return bad;
}

bool collides() {
  std::vector<int> hits(NGLOB, 0);
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) ++hits[gid(e, k, j, i)];
  for (int h : hits)
    if (h > 1) return true;
  return false;
}

struct Accessor {
  Vec                    v;
  KOKKOS_FUNCTION float& operator()(int g) const { return v(g); }
};

static_assert(Impl::atomic_target_v<Vec>);
static_assert(Impl::atomic_target_v<Accessor>);
static_assert(!Impl::atomic_target_v<Kokkos::View<const float*, ES>>);
static_assert(!Impl::atomic_target_v<FieldL>);

auto stage(const ValView& v) {
  return make_stage_node(make_input_node(make_handle<'e', 'k', 'j', 'i'>(v)));
}

template <bool Handle, typename IdxLayout, typename Dst>
void scatter_one(const IdxView<IdxLayout>& idx, const ValView& val, Dst dst) {
  auto g0 = make_level_graph<float, ES>(Map{});
  if constexpr (Handle) {
    auto [g1, ig, u] =
        g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx), stage(val));
    auto g2 = g1.add(make_scatter_add_node<'e', 'k', 'j', 'i'>(ig, dst, u));
    g2.outputs().execute(TeamPolicyTag<ES>{});
  } else {
    auto [g1, u] = g0.add(stage(val));
    auto g2 = g1.add(make_scatter_add_node<'e', 'k', 'j', 'i'>(idx, dst, u));
    g2.outputs().execute(TeamPolicyTag<ES>{});
  }
  Kokkos::fence();
}

template <typename IdxLayout>
void both_forms_match_reference() {
  ASSERT_TRUE(collides());
  IdxView<IdxLayout> idx("idx", E, N, N, N);
  fill_index(idx);
  const ValView val = make_values(0);
  const auto    want =
      reference([](int e, int k, int j, int i) { return vval(e, k, j, i, 0); });
  Vec a("a", NGLOB), b("b", NGLOB);
  scatter_one<false>(idx, val, a);
  scatter_one<true>(idx, val, b);
  EXPECT_EQ(mismatches(a, want), 0);
  EXPECT_EQ(mismatches(b, want), 0);
}

}  // namespace

TEST(ScatterAddNode, BothFormsMatchTheReferenceLayoutRightIndex) {
  both_forms_match_reference<Kokkos::LayoutRight>();
}

TEST(ScatterAddNode, BothFormsMatchTheReferenceLayoutLeftIndex) {
  both_forms_match_reference<Kokkos::LayoutLeft>();
}

TEST(ScatterAddNode, DestinationsCanBeColumnsOrAccessors) {
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  fill_index(idx);
  const ValView val = make_values(1);
  const auto    want =
      reference([](int e, int k, int j, int i) { return vval(e, k, j, i, 1); });
  FieldL fl("fl", NGLOB, 3);
  FieldR fr("fr", NGLOB, 3);
  Vec    v("v", NGLOB);
  scatter_one<true>(idx, val, Kokkos::subview(fl, Kokkos::ALL, 1));
  scatter_one<false>(idx, val, Kokkos::subview(fr, Kokkos::ALL, 2));
  scatter_one<true>(idx, val, Accessor{v});
  auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fl);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fr);
  EXPECT_EQ(mismatches(Kokkos::subview(hl, Kokkos::ALL, 1), want), 0);
  EXPECT_EQ(mismatches(Kokkos::subview(hr, Kokkos::ALL, 2), want), 0);
  EXPECT_EQ(mismatches(v, want), 0);
}

TEST(ScatterAddNode, ThreeScattersFromOneIndex) {
  IdxView<Kokkos::LayoutLeft> idx("idx", E, N, N, N);
  fill_index(idx);
  const ValView v0 = make_values(0), v1 = make_values(1), v2 = make_values(2);
  FieldL        f("f", NGLOB, 3);

  auto g0                   = make_level_graph<float, ES>(Map{});
  auto [g1, ig, u0, u1, u2] = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx),
                                     stage(v0), stage(v1), stage(v2));
  auto col = [&](int c) { return Kokkos::subview(f, Kokkos::ALL, c); };
  auto g2  = g1.add(make_scatter_add_node<'e', 'k', 'j', 'i'>(ig, col(0), u0),
                    make_scatter_add_node<'e', 'k', 'j', 'i'>(ig, col(1), u1),
                    make_scatter_add_node<'e', 'k', 'j', 'i'>(ig, col(2), u2));
  g2.outputs().execute(TeamPolicyTag<ES>{});
  Kokkos::fence();

  for (int c = 0; c < 3; ++c)
    EXPECT_EQ(mismatches(col(c), reference([c](int e, int k, int j, int i) {
                           return vval(e, k, j, i, c);
                         })),
              0)
        << "component " << c;
}

namespace {

struct TwicePlusOne {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float u) const {
    return 2.0f * u + 1.0f;
  }
};

}  // namespace

TEST(ScatterAddNode, GatherCombineScatterShareOneIndex) {
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  fill_index(idx);
  Vec  src("src", NGLOB), dst("dst", NGLOB);
  auto hs = Kokkos::create_mirror_view(src);
  for (int g = 0; g < NGLOB; ++g) hs(g) = 0.25f * static_cast<float>(g % 17);
  Kokkos::deep_copy(src, hs);

  auto g0       = make_level_graph<float, ES>(Map{});
  auto t1       = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  auto [g1, ig] = t1;
  auto [g2, u]  = g1.add(make_gather_node<'e', 'k', 'j', 'i'>(ig, src));
  auto t3 = g2.add(make_combine_node<'e', 'k', 'j', 'i'>(u, TwicePlusOne{}));
  auto [g3, r] = t3;
  auto g4      = g3.add(make_scatter_add_node<'e', 'k', 'j', 'i'>(ig, dst, r));

  using Levels            = typename std::decay_t<decltype(g4)>::levels_type;
  constexpr std::size_t I = std::tuple_element_t<1, decltype(t1)>::SlotIdx;
  constexpr std::size_t R = std::tuple_element_t<1, decltype(t3)>::SlotIdx;
  static_assert(Impl::lg_slot_pool_v<Levels, std::index_sequence<>, I> !=
                    Impl::lg_slot_pool_v<Levels, std::index_sequence<>, R>,
                "the index stays live through the scatter level, so it cannot "
                "share a buffer with the combine that feeds the scatter");

  g4.outputs().execute(TeamPolicyTag<ES>{});
  Kokkos::fence();
  EXPECT_EQ(mismatches(dst, reference([&](int e, int k, int j, int i) {
                         return 2.0f * hs(gid(e, k, j, i)) + 1.0f;
                       })),
            0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int r = RUN_ALL_TESTS();
  Kokkos::finalize();
  return r;
}
