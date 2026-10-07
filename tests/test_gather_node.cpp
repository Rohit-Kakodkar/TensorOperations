#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

constexpr int N = 5, TE = 2, E = 8, NGLOB = 211;

using Map = LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', N>,
                       LabelWhole<'j', N>, LabelWhole<'i', N>>;

template <typename Layout>
using IdxView = Kokkos::View<int****, Layout, ES>;
using Out     = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using Field   = Kokkos::View<float**, Kokkos::LayoutLeft, ES>;
using Vec     = Kokkos::View<float*, ES>;

int gid(int e, int k, int j, int i) {
  return (97 * e + 31 * k + 7 * j + 3 * i * i + 5) % NGLOB;
}
float fval(int g, int c) {
  return static_cast<float>(37 * g - 150 * c + g * c) * 0.125f;
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

Field make_field() {
  Field f("f", NGLOB, 3);
  auto  h = Kokkos::create_mirror_view(f);
  for (int g = 0; g < NGLOB; ++g)
    for (int c = 0; c < 3; ++c) h(g, c) = fval(g, c);
  Kokkos::deep_copy(f, h);
  return f;
}

auto column(const Field& f, int c) {
  return Kokkos::subview(f, Kokkos::ALL, c);
}

struct Ramp {
  KOKKOS_FUNCTION float operator()(int g) const { return 2.0f * g - 7.0f; }
};

template <typename F>
int ref_mismatches(const Out& got, F expect, float tol = 0.0f) {
  auto h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
  int  bad = 0;
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
          if (std::abs(h(e, k, j, i) - expect(gid(e, k, j, i))) >
              tol * (1.0f + std::abs(expect(gid(e, k, j, i)))))
            ++bad;
  return bad;
}

int bitwise_mismatches(const Out& a, const Out& b) {
  auto ha  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
  auto hb  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
  int  bad = 0;
  for (std::size_t n = 0; n < ha.size(); ++n)
    if (ha.data()[n] != hb.data()[n]) ++bad;
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

static_assert(Impl::int_indexable_v<Vec>);
static_assert(Impl::int_indexable_v<Ramp>);
static_assert(!Impl::int_indexable_v<Field>);

template <typename IdxLayout, bool Handle, typename Source>
Out gather_root(const IdxView<IdxLayout>& idx, Source source) {
  Out  out("out", E, N, N, N);
  auto g0 = make_level_graph<float, ES>(Map{});
  if constexpr (Handle) {
    auto [g1, ig] = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
    auto [g2, u] =
        g1.add(make_gather_node<'e', 'k', 'j', 'i'>(ig, std::move(source)));
    g2.outputs(u).execute(TeamPolicyTag<ES>{}, out);
  } else {
    auto [g1, u] =
        g0.add(make_gather_node<'e', 'k', 'j', 'i'>(idx, std::move(source)));
    g1.outputs(u).execute(TeamPolicyTag<ES>{}, out);
  }
  Kokkos::fence();
  return out;
}

template <typename IdxLayout>
void both_forms_match_reference() {
  ASSERT_TRUE(collides());
  IdxView<IdxLayout> idx("idx", E, N, N, N);
  fill_index(idx);
  const Field f = make_field();

  const Out view_form   = gather_root<IdxLayout, false>(idx, column(f, 1));
  const Out handle_form = gather_root<IdxLayout, true>(idx, column(f, 1));
  EXPECT_EQ(ref_mismatches(view_form, [](int g) { return fval(g, 1); }), 0);
  EXPECT_EQ(ref_mismatches(handle_form, [](int g) { return fval(g, 1); }), 0);
  EXPECT_EQ(bitwise_mismatches(view_form, handle_form), 0);
}

}  // namespace

TEST(GatherNode, BothFormsMatchTheReferenceLayoutRightIndex) {
  both_forms_match_reference<Kokkos::LayoutRight>();
}

TEST(GatherNode, BothFormsMatchTheReferenceLayoutLeftIndex) {
  both_forms_match_reference<Kokkos::LayoutLeft>();
}

TEST(GatherNode, SourcesCanBeAVectorOrAFunctor) {
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  fill_index(idx);
  Vec  v("v", NGLOB);
  auto hv = Kokkos::create_mirror_view(v);
  for (int g = 0; g < NGLOB; ++g) hv(g) = 0.5f * g + 3.0f;
  Kokkos::deep_copy(v, hv);

  for (const Out& o : {gather_root<Kokkos::LayoutRight, false>(idx, v),
                       gather_root<Kokkos::LayoutRight, true>(idx, v)})
    EXPECT_EQ(ref_mismatches(o, [](int g) { return 0.5f * g + 3.0f; }), 0);
  for (const Out& o : {gather_root<Kokkos::LayoutRight, false>(idx, Ramp{}),
                       gather_root<Kokkos::LayoutRight, true>(idx, Ramp{})})
    EXPECT_EQ(ref_mismatches(o, [](int g) { return 2.0f * g - 7.0f; }), 0);
}

namespace {

struct Mix3 {
  KOKKOS_FUNCTION float operator()(int, int, int, int, float a, float b,
                                   float c) const {
    return a - 2.0f * b + 0.25f * c;
  }
};

template <bool Handle>
Out three_gathers_then_combine(const IdxView<Kokkos::LayoutLeft>& idx,
                               const Field&                       f) {
  Out  out("out", E, N, N, N);
  auto g0 = make_level_graph<float, ES>(Map{});
  if constexpr (Handle) {
    auto t1       = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
    auto [g1, ig] = t1;
    auto t2 = g1.add(make_gather_node<'e', 'k', 'j', 'i'>(ig, column(f, 0)),
                     make_gather_node<'e', 'k', 'j', 'i'>(ig, column(f, 1)),
                     make_gather_node<'e', 'k', 'j', 'i'>(ig, column(f, 2)));
    auto [g2, u0, u1, u2] = t2;
    auto t3 = g2.add(make_combine_node<'e', 'k', 'j', 'i'>(u0, u1, u2, Mix3{}));
    auto [g3, r] = t3;
    const auto o = g3.outputs(r);

    using Levels = typename std::tuple_element_t<0, decltype(t3)>::levels_type;
    constexpr std::size_t R = std::tuple_element_t<1, decltype(t3)>::SlotIdx;
    constexpr std::size_t I = std::tuple_element_t<1, decltype(t1)>::SlotIdx;
    constexpr std::size_t U = std::tuple_element_t<1, decltype(t2)>::SlotIdx;
    using Roots             = std::index_sequence<R>;
    static_assert(Impl::lg_slot_pool_v<Levels, Roots, I> !=
                      Impl::lg_slot_pool_v<Levels, Roots, U>,
                  "the index is live while the gathers write their slots");
    static_assert(Impl::lg_slot_pool_v<Levels, Roots, I> ==
                      Impl::lg_slot_pool_v<Levels, Roots, R>,
                  "the index is dead after the gather level and can be reused");
    o.execute(TeamPolicyTag<ES>{}, out);
  } else {
    auto [g1, u0, u1, u2] =
        g0.add(make_gather_node<'e', 'k', 'j', 'i'>(idx, column(f, 0)),
               make_gather_node<'e', 'k', 'j', 'i'>(idx, column(f, 1)),
               make_gather_node<'e', 'k', 'j', 'i'>(idx, column(f, 2)));
    auto [g2, r] =
        g1.add(make_combine_node<'e', 'k', 'j', 'i'>(u0, u1, u2, Mix3{}));
    g2.outputs(r).execute(TeamPolicyTag<ES>{}, out);
  }
  Kokkos::fence();
  return out;
}

}  // namespace

TEST(GatherNode, ThreeGathersFromOneIndexFeedACombine) {
  IdxView<Kokkos::LayoutLeft> idx("idx", E, N, N, N);
  fill_index(idx);
  const Field f = make_field();
  const Out   a = three_gathers_then_combine<true>(idx, f);
  const Out   b = three_gathers_then_combine<false>(idx, f);
  EXPECT_EQ(ref_mismatches(
                a,
                [](int g) {
                  return fval(g, 0) - 2.0f * fval(g, 1) + 0.25f * fval(g, 2);
                },
                1e-6f),
            0);
  EXPECT_EQ(bitwise_mismatches(a, b), 0);
}

TEST(GatherNode, GatherFeedsAContraction) {
  using MapC =
      LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', N>, LabelWhole<'j', N>,
                 LabelWhole<'i', N>, LabelWhole<'p', N>>;
  IdxView<Kokkos::LayoutRight> idx("idx", E, N, N, N);
  fill_index(idx);
  const Field                                    f = make_field();
  Kokkos::View<float**, Kokkos::LayoutRight, ES> h("h", N, N);
  auto hh = Kokkos::create_mirror_view(h);
  for (int a = 0; a < N; ++a)
    for (int b = 0; b < N; ++b) hh(a, b) = 0.1f * a - 0.3f * b + 0.05f * a * b;
  Kokkos::deep_copy(h, hh);

  Out  out("out", E, N, N, N);
  auto g0       = make_level_graph<float, ES>(MapC{});
  auto [g1, ig] = g0.add(make_index_node<'e', 'k', 'j', 'i'>(idx));
  auto [g2, sh] =
      g1.add(make_stage_node(make_input_node(make_handle<'i', 'p'>(h))));
  auto [g3, u] = g2.add(make_gather_node<'e', 'k', 'j', 'i'>(ig, column(f, 2)));
  auto [g4, gx] = g3.add(make_contraction_node<'e', 'k', 'j', 'i'>(
      sh, u.template as<'e', 'k', 'j', 'p'>()));
  g4.outputs(gx).execute(TeamPolicyTag<ES>{}, out);
  Kokkos::fence();

  auto ho  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  int  bad = 0;
  for (int e = 0; e < E; ++e)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          float acc = 0.0f;
          for (int p = 0; p < N; ++p)
            acc += hh(i, p) * fval(gid(e, k, j, p), 2);
          if (std::abs(ho(e, k, j, i) - acc) > 1e-4f * (1.0f + std::abs(acc)))
            ++bad;
        }
  EXPECT_EQ(bad, 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int r = RUN_ALL_TESTS();
  Kokkos::finalize();
  return r;
}
