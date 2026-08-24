// ===========================================================================
// test_sink_combine.cpp — a combine that returns nothing and writes to global
// itself, so the graph executes with NO outputs.
//
// A rooted combine hands its result back to the library, which stores it into a
// designated output view. A SINK combine does the store itself -- typically a
// scatter through a mesh index map with Kokkos::atomic_add -- and returns void.
// Such a member contributes zero slots, and a graph whose last level is all
// sinks needs no output views at all.
//
// Three claims, in increasing order of what they would let through:
//
//   1. EQUIVALENCE. A sink that writes `2*v + 0.5` straight into a plain view
//      must produce BITWISE the same field as the identical graph written with
//      a real root. Bitwise, not within a tolerance: both paths run the same
//      combine fn over the same tile and write the same values to the same
//      coordinates; the only difference is who owns the store, so any
//      difference at all is a defect in the sink path.
//
//   2. THE SCATTER. The thing the feature is for: an atomic accumulate through
//      a many-to-one index map with deliberate collisions, checked against a
//      serial reference. Test 1 alone would pass with a plain `=` store,
//      because its map is the identity and has no collisions.
//
//   3. IT LOWERS SCRATCH. A root is pinned live to the end of the graph; a sink
//      is not. So dropping the roots for sinks must not RAISE the pool count,
//      and where the roots were the peak it lowers it. Asserted as a compile-
//      time num_pools comparison against the rooted form.
//
// EXTENTS ARE NON-SQUARE (the batch axis 'e' is 8, the rest 5) so a transposed
// coordinate cannot hide behind equal axes.
// ===========================================================================
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace TensorOperations;
using ES = Kokkos::DefaultExecutionSpace;

namespace {

constexpr int sN = 5, sTE = 2, sE = 8;

using ViewH = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewU = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using ViewC = Kokkos::View<float****, Kokkos::LayoutRight, ES>;
using ViewA = Kokkos::View<float*, Kokkos::LayoutRight, ES>;
using ViewM = Kokkos::View<int****, Kokkos::LayoutRight, ES>;

using Map =
    LabelTiles<LabelWhole<'q', sN>, LabelWhole<'a', sN>, LabelTile<'e', sTE>,
               LabelWhole<'b', sN>, LabelWhole<'c', sN>>;

float hval(int q, int a) { return 0.1f + 0.3f * q - 0.2f * a + 0.05f * q * a; }
float uval(int e, int a, int b, int c) {
  return 0.2f + 0.11f * e - 0.07f * a + 0.13f * b - 0.03f * c +
         0.01f * (e + a) * (b + 1);
}

// nglob is deliberately SMALLER than the coordinate count, so the map is
// many-to-one and distinct (q,e,b,c) collide -- which is what makes a scatter
// more than a permutation and what an atomic is for.
constexpr int sNglob = 37;
int           gid(int q, int e, int b, int c) {
  return (((q * sE + e) * sN + b) * sN + c) % sNglob;
}

// A rooted combine: hand the value back, the library stores it.
struct ScaleG {
  KOKKOS_INLINE_FUNCTION float operator()(int, int, int, int, float v) const {
    return 2.0f * v + 0.5f;
  }
};

// The same arithmetic, but the sink writes it itself into a plain 4D view at
// the global coordinate. Returns void: no output slot, no root.
struct WritePlainSink {
  ViewC                       out;
  KOKKOS_INLINE_FUNCTION void operator()(int q, int e, int b, int c,
                                         float v) const {
    out(q, e, b, c) = 2.0f * v + 0.5f;
  }
};

// The real thing: scatter-accumulate through the index map with collisions.
struct AccumSink {
  ViewA                       a;
  ViewM                       map;
  KOKKOS_INLINE_FUNCTION void operator()(int q, int e, int b, int c,
                                         float v) const {
    Kokkos::atomic_add(&a(map(q, e, b, c)), 2.0f * v + 0.5f);
  }
};

void fill_hu(ViewH Hd, ViewU Ud) {
  auto Hh = Kokkos::create_mirror_view(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int q = 0; q < sN; ++q)
    for (int a = 0; a < sN; ++a) Hh(q, a) = hval(q, a);
  for (int e = 0; e < sE; ++e)
    for (int a = 0; a < sN; ++a)
      for (int b = 0; b < sN; ++b)
        for (int c = 0; c < sN; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
  Kokkos::deep_copy(Hd, Hh);
  Kokkos::deep_copy(Ud, Uh);
}

TEST(SinkCombine, PlainValueSinkIsBitwiseIdenticalToARootedGraph) {
  ViewH Hd("H", sN, sN);
  ViewU Ud("U", sE, sN, sN, sN);
  ViewC Prooted("Prooted", sN, sE, sN, sN);
  ViewC Psink("Psink", sN, sE, sN, sN);
  fill_hu(Hd, Ud);

  // Rooted: combine -> output -> stored by the library into Prooted.
  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, h] =
        g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
    auto [g2, u] = g1.add(
        make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud))));
    auto [g3, ca] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(h, u));
    auto [g4, pa] = g3.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleG{}));
    g4.outputs(pa).execute(TeamPolicyTag2<ES>{}, Prooted);
    Kokkos::fence();
  }

  // Sink: combine writes Psink itself and returns nothing. No output view.
  {
    auto g0 = make_level_graph<float, ES>(Map{});
    auto [g1, h] =
        g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
    auto [g2, u] = g1.add(
        make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud))));
    auto [g3, ca] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(h, u));
    auto g4       = g3.add(
        make_combine_node<'q', 'e', 'b', 'c'>(ca, WritePlainSink{Psink}));
    g4.outputs().execute(TeamPolicyTag2<ES>{});
    Kokkos::fence();
  }

  auto Rh = Kokkos::create_mirror_view(Prooted);
  auto Sh = Kokkos::create_mirror_view(Psink);
  Kokkos::deep_copy(Rh, Prooted);
  Kokkos::deep_copy(Sh, Psink);

  int   mism  = 0;
  float scale = 0.0f;
  for (int q = 0; q < sN; ++q)
    for (int e = 0; e < sE; ++e)
      for (int b = 0; b < sN; ++b)
        for (int c = 0; c < sN; ++c) {
          scale = std::max(scale, std::abs(Rh(q, e, b, c)));
          if (Rh(q, e, b, c) != Sh(q, e, b, c)) ++mism;
        }
  EXPECT_GT(scale, 0.0f) << "void test: the rooted path produced all zeros";
  EXPECT_EQ(mism, 0)
      << "a sink writing plain values must match the rooted graph bit-for-bit";
}

TEST(SinkCombine, AtomicAccumulateThroughAnIndexMapMatchesReference) {
  ViewH Hd("H", sN, sN);
  ViewU Ud("U", sE, sN, sN, sN);
  ViewM Md("map", sN, sE, sN, sN);
  ViewA Ad("A", sNglob);
  fill_hu(Hd, Ud);
  Kokkos::deep_copy(Ad, 0.0f);

  auto Mh = Kokkos::create_mirror_view(Md);
  for (int q = 0; q < sN; ++q)
    for (int e = 0; e < sE; ++e)
      for (int b = 0; b < sN; ++b)
        for (int c = 0; c < sN; ++c) Mh(q, e, b, c) = gid(q, e, b, c);
  Kokkos::deep_copy(Md, Mh);

  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
  auto [g2, u] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud))));
  auto [g3, ca] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(h, u));
  auto g4 =
      g3.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, AccumSink{Ad, Md}));
  g4.outputs().execute(TeamPolicyTag2<ES>{});
  Kokkos::fence();

  auto Ah = Kokkos::create_mirror_view(Ad);
  Kokkos::deep_copy(Ah, Ad);

  // Serial reference: accumulate every (q,e,b,c) contribution into its node.
  std::vector<double> ref(sNglob, 0.0);
  for (int q = 0; q < sN; ++q)
    for (int e = 0; e < sE; ++e)
      for (int b = 0; b < sN; ++b)
        for (int c = 0; c < sN; ++c) {
          double g = 0.0;
          for (int a = 0; a < sN; ++a) g += hval(q, a) * uval(e, a, b, c);
          ref[gid(q, e, b, c)] += 2.0 * g + 0.5;
        }

  double max_err = 0.0, scale = 0.0;
  int    collided = 0;
  for (int n = 0; n < sNglob; ++n) {
    scale   = std::max(scale, std::abs(ref[n]));
    max_err = std::max(max_err, std::abs(ref[n] - Ah(n)));
  }
  // Prove the map really collided, or the atomic was never exercised.
  {
    std::vector<int> hits(sNglob, 0);
    for (int q = 0; q < sN; ++q)
      for (int e = 0; e < sE; ++e)
        for (int b = 0; b < sN; ++b)
          for (int c = 0; c < sN; ++c) ++hits[gid(q, e, b, c)];
    for (int n = 0; n < sNglob; ++n) collided += (hits[n] > 1);
  }
  EXPECT_GT(collided, 0) << "void test: the map never collided, so the atomic "
                            "was never actually contended";
  EXPECT_GT(scale, 0.0) << "void test: the reference is all zeros";
  EXPECT_LT(max_err, 1e-3) << "atomic scatter through the map != reference";
}

TEST(SinkCombine, DroppingTheRootForASinkDoesNotRaiseThePoolCount) {
  ViewH Hd("H", sN, sN);
  ViewU Ud("U", sE, sN, sN, sN);
  ViewC P0("P0", sN, sE, sN, sN), P1("P1", sN, sE, sN, sN),
      P2("P2", sN, sE, sN, sN);

  // Two graphs identical up to a THREE-member terminal level that all read the
  // same operand `ca`. The rooted form pins three output slots live to the end,
  // so at the terminal level `ca` plus the three roots coexist and become the
  // graph's peak; the sink form pins nothing, so the peak stays at the earlier
  // contraction level. num_pools is a compile-time property of outputs().
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
  auto [g2, u] = g1.add(
      make_stage_node(make_input_node(make_handle<'e', 'a', 'b', 'c'>(Ud))));
  auto [g3, ca] = g2.add(make_contraction_node<'q', 'e', 'b', 'c'>(h, u));

  auto gr = g3;
  auto [g4r, r0, r1, r2] =
      gr.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleG{}),
             make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleG{}),
             make_combine_node<'q', 'e', 'b', 'c'>(ca, ScaleG{}));
  auto gs = g3;
  auto g4s =
      gs.add(make_combine_node<'q', 'e', 'b', 'c'>(ca, WritePlainSink{P0}),
             make_combine_node<'q', 'e', 'b', 'c'>(ca, WritePlainSink{P1}),
             make_combine_node<'q', 'e', 'b', 'c'>(ca, WritePlainSink{P2}));

  constexpr std::size_t rooted_pools =
      decltype(g4r.outputs(r0, r1, r2))::num_pools;
  constexpr std::size_t sink_pools = decltype(g4s.outputs())::num_pools;

  static_assert(sink_pools <= rooted_pools,
                "a sink must not need MORE pools than the rooted form: a root "
                "is pinned to the end, a sink is not");
  EXPECT_LE(sink_pools, rooted_pools);
  // Here the three roots ARE the graph's peak, so dropping them for sinks
  // strictly lowers the count.
  EXPECT_LT(sink_pools, rooted_pools)
      << "rooted=" << rooted_pools << " sink=" << sink_pools;
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
