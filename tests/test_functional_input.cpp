// ===========================================================================
// test_functional_input.cpp — a leaf whose elements are COMPUTED, not
// addressed.
//
// A functional input exists for tensors that no stride describes. The
// motivating one is a spectral-element field: it lives in global-node order and
// is reached element-wise through a mesh index map, u(iglob(e,a,b), c). A hook
// cannot do this -- a hook fires after the load, so it can change a value but
// not redirect an address.
//
// Three claims, in increasing order of what they would let through:
//
//   1. EQUIVALENCE. A functional input that merely re-reads a plain view must
//      produce BITWISE the same answer as the plain input node. Stated bitwise
//      rather than within a tolerance because the two paths do identical
//      arithmetic on identical values; any difference at all is a real defect,
//      and a tolerance would hide exactly the small ones worth finding.
//
//   2. INDIRECTION. The thing the feature is for: a genuine gather through an
//      index map, against a hand-written reference. Test 1 alone would pass
//      with the functor's arguments silently permuted, because it re-reads a
//      view that is itself indexed in the same order.
//
//   3. THE GLOBAL COORDINATE. The functor must see the GLOBAL index, not the
//      tile-local one. This is the trap PR #34 fixed for combine functors, and
//      it is invisible with one team: tile 0's origin is zero, so a kernel that
//      forgets to fold in the origin is correct for exactly the first team and
//      wrong for every other. So fE/fTE = 3 teams, and the probe functor is
//      injective in 'e' -- a dropped origin does not shift the answer, it
//      duplicates team 0's into all three.
//
// EXTENTS ARE PAIRWISE DISTINCT (4/5/3) so that a transposed functor argument
// list cannot hide behind two equal axes -- the same discipline the relabel
// tests in test_level_graph.cpp use.
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

// 'a' is summed, so the output is (q,e,b). fE/fTE = 3 teams: see claim 3.
constexpr int fQ = 4, fA = 5, fB = 3, fTE = 2, fE = 6;

using ViewH   = Kokkos::View<float**, Kokkos::LayoutRight, ES>;
using ViewU   = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using ViewC   = Kokkos::View<float***, Kokkos::LayoutRight, ES>;
using ViewMap = Kokkos::View<int***, Kokkos::LayoutRight, ES>;
// Global-node storage, LayoutLeft so the node index is stride-1 -- the same
// shape sfpp_min's displacement field has.
using ViewGlobal = Kokkos::View<float**, Kokkos::LayoutLeft, ES>;

using Map = LabelTiles<LabelWhole<'q', fQ>, LabelWhole<'a', fA>,
                       LabelTile<'e', fTE>, LabelWhole<'b', fB>>;

float hval(int q, int a) { return 0.1f + 0.3f * q - 0.2f * a + 0.05f * q * a; }
float uval(int e, int a, int b) {
  return 0.2f + 0.11f * e - 0.07f * a + 0.13f * b + 0.01f * (e + a) * (b + 1);
}

// nglob is deliberately SMALLER than fE*fA*fB, so the map is many-to-one and
// distinct (e,a,b) share storage -- which is what a real mesh does, and what
// makes a gather more than a permutation.
constexpr int fNglob = 17;
constexpr int fComp  = 2;

int   gid(int e, int a, int b) { return (e * fA * fB + a * fB + b) % fNglob; }
float gval(int n, int c) { return 0.37f * n - 0.9f * c + 0.013f * n * c; }

// The functor under test: exactly the shape sfpp_min needs.
struct GatherThroughMap {
  ViewGlobal                   data;
  ViewMap                      map;
  int                          comp;
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int b) const {
    return data(map(e, a, b), comp);
  }
};

// A functional source that merely re-reads a plain view.
struct ReadThroughView {
  ViewU                        u;
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int b) const {
    return u(e, a, b);
  }
};

// Injective in every argument, and in 'e' with a stride larger than any
// (a,b) contribution, so a dropped tile origin cannot be mistaken for noise.
struct GlobalCoordProbe {
  KOKKOS_INLINE_FUNCTION float operator()(int e, int a, int b) const {
    return 1000.0f * e + 10.0f * a + 1.0f * b;
  }
};

void fill_operators(ViewH Hd) {
  auto Hh = Kokkos::create_mirror_view(Hd);
  for (int q = 0; q < fQ; ++q)
    for (int a = 0; a < fA; ++a) Hh(q, a) = hval(q, a);
  Kokkos::deep_copy(Hd, Hh);
}

// C[q,e,b] = sum_a H[q,a] * U[e,a,b], with U staged from `make_u_node()`.
template <typename MakeUNode>
void run_graph(ViewH Hd, ViewC Cd, MakeUNode make_u_node) {
  auto g0 = make_level_graph<float, ES>(Map{});
  auto [g1, h] =
      g0.add(make_stage_node(make_input_node(make_handle<'q', 'a'>(Hd))));
  auto [g2, u] = g1.add(make_stage_node(make_u_node()));
  auto [g3, c] = g2.add(make_contraction_node<'q', 'e', 'b'>(h, u));
  g3.outputs(c).execute(TeamPolicyTag<ES>{}, Cd);
  Kokkos::fence();
}

TEST(FunctionalInput, ReReadingAPlainViewIsBitwiseIdenticalToAnInputNode) {
  ViewH Hd("H", fQ, fA);
  ViewU Ud("U", fE, fA, fB);
  ViewC Cplain("Cplain", fQ, fE, fB);
  ViewC Cfunc("Cfunc", fQ, fE, fB);

  fill_operators(Hd);
  auto Uh = Kokkos::create_mirror_view(Ud);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) Uh(e, a, b) = uval(e, a, b);
  Kokkos::deep_copy(Ud, Uh);

  run_graph(Hd, Cplain,
            [&] { return make_input_node(make_handle<'e', 'a', 'b'>(Ud)); });
  run_graph(Hd, Cfunc, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        Kokkos::Array<int, 3>{fE, fA, fB}, ReadThroughView{Ud});
  });

  auto Ph = Kokkos::create_mirror_view(Cplain);
  auto Fh = Kokkos::create_mirror_view(Cfunc);
  Kokkos::deep_copy(Ph, Cplain);
  Kokkos::deep_copy(Fh, Cfunc);

  int   mismatches = 0;
  float scale      = 0.0f;
  for (int q = 0; q < fQ; ++q)
    for (int e = 0; e < fE; ++e)
      for (int b = 0; b < fB; ++b) {
        scale = std::max(scale, std::abs(Ph(q, e, b)));
        if (Ph(q, e, b) != Fh(q, e, b)) ++mismatches;
      }
  EXPECT_GT(scale, 0.0f) << "void test: the plain path produced all zeros";
  EXPECT_EQ(mismatches, 0)
      << "a functional input re-reading a plain view must be bitwise identical";
}

TEST(FunctionalInput, GathersThroughAnIndexMapAgainstReference) {
  ViewH      Hd("H", fQ, fA);
  ViewMap    Md("map", fE, fA, fB);
  ViewGlobal Gd("global", fNglob, fComp);
  ViewC      Cd("C", fQ, fE, fB);

  fill_operators(Hd);
  auto Mh = Kokkos::create_mirror_view(Md);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) Mh(e, a, b) = gid(e, a, b);
  Kokkos::deep_copy(Md, Mh);

  auto Gh = Kokkos::create_mirror_view(Gd);
  for (int n = 0; n < fNglob; ++n)
    for (int c = 0; c < fComp; ++c) Gh(n, c) = gval(n, c);
  Kokkos::deep_copy(Gd, Gh);

  constexpr int kComp = 1;  // not 0: a dropped component index reads gval(n,0)
  run_graph(Hd, Cd, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        Kokkos::Array<int, 3>{fE, fA, fB}, GatherThroughMap{Gd, Md, kComp});
  });

  auto Ch = Kokkos::create_mirror_view(Cd);
  Kokkos::deep_copy(Ch, Cd);

  double max_err = 0.0, scale = 0.0;
  for (int q = 0; q < fQ; ++q)
    for (int e = 0; e < fE; ++e)
      for (int b = 0; b < fB; ++b) {
        double ref = 0.0;
        for (int a = 0; a < fA; ++a)
          ref += hval(q, a) * gval(gid(e, a, b), kComp);
        scale   = std::max(scale, std::abs(ref));
        max_err = std::max(max_err, std::abs(ref - Ch(q, e, b)));
      }
  EXPECT_GT(scale, 0.0) << "void test: the reference is all zeros";
  EXPECT_LT(max_err, 1e-3) << "gathered input != reference";
}

// Order is a PERFORMANCE knob, not a semantic one, and this is the test that
// says so. The staging copy is `dst[coord] = sv[coord]` -- coordinate-indexed
// on both sides -- so permuting the order lanes visit coordinates in permutes
// who does what, never what gets written. Bitwise, not within a tolerance: the
// same values are written to the same places in a different sequence, so any
// difference at all is a defect.
//
// This is what makes the knob safe to tune blindly for coalescing.
TEST(FunctionalInput, TraversalOrderDoesNotChangeTheResult) {
  ViewH      Hd("H", fQ, fA);
  ViewMap    Md("map", fE, fA, fB);
  ViewGlobal Gd("global", fNglob, fComp);
  ViewC      Cright("Cright", fQ, fE, fB);
  ViewC      Cleft("Cleft", fQ, fE, fB);
  ViewC      Cperm("Cperm", fQ, fE, fB);

  fill_operators(Hd);
  auto Mh = Kokkos::create_mirror_view(Md);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) Mh(e, a, b) = gid(e, a, b);
  Kokkos::deep_copy(Md, Mh);
  auto Gh = Kokkos::create_mirror_view(Gd);
  for (int n = 0; n < fNglob; ++n)
    for (int c = 0; c < fComp; ++c) Gh(n, c) = gval(n, c);
  Kokkos::deep_copy(Gd, Gh);

  const Kokkos::Array<int, 3> ext{fE, fA, fB};
  const GatherThroughMap      fn{Gd, Md, 1};

  run_graph(Hd, Cright, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        DynamicTileLayoutRight<3>{ext}, fn);
  });
  run_graph(Hd, Cleft, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        DynamicTileLayoutLeft<3>{ext}, fn);
  });
  // An arbitrary permutation, and a 3-CYCLE rather than a swap: a swap is its
  // own inverse, so it cannot tell a gather order from a scatter one.
  run_graph(Hd, Cperm, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        StaticTileLayoutStride<StaticTile<fE, fA, fB>, 1, 2, 0>{}, fn);
  });

  auto Rh = Kokkos::create_mirror_view(Cright);
  auto Lh = Kokkos::create_mirror_view(Cleft);
  auto Ph = Kokkos::create_mirror_view(Cperm);
  Kokkos::deep_copy(Rh, Cright);
  Kokkos::deep_copy(Lh, Cleft);
  Kokkos::deep_copy(Ph, Cperm);

  int   left_diff = 0, perm_diff = 0;
  float scale = 0.0f;
  for (int q = 0; q < fQ; ++q)
    for (int e = 0; e < fE; ++e)
      for (int b = 0; b < fB; ++b) {
        scale = std::max(scale, std::abs(Rh(q, e, b)));
        if (Rh(q, e, b) != Lh(q, e, b)) ++left_diff;
        if (Rh(q, e, b) != Ph(q, e, b)) ++perm_diff;
      }
  EXPECT_GT(scale, 0.0f) << "void test: LayoutRight produced all zeros";
  EXPECT_EQ(left_diff, 0) << "LayoutLeft traversal changed the result";
  EXPECT_EQ(perm_diff, 0) << "permuted traversal changed the result";
}

// ...and the knob is actually CONNECTED. The test above cannot show that on
// its own: a node that ignored its order tag entirely would pass it three
// times over. So check both halves of the path -- the tag reaches the node's
// type, and make_tile_layout turns it into layouts that genuinely enumerate
// coordinates in different sequences.
TEST(FunctionalInput, TraversalOrderReachesTheTileLayout) {
  const Kokkos::Array<int, 3> ext{fE, fA, fB};
  auto                        right =
      make_functional_input_node<'e', 'a', 'b'>(ext, GlobalCoordProbe{});
  auto left = make_functional_input_node<'e', 'a', 'b'>(
      DynamicTileLayoutLeft<3>{ext}, GlobalCoordProbe{});
  static_assert(std::is_same_v<decltype(right)::order_tag, LayoutRight>,
                "extents-only must default to LayoutRight");
  static_assert(std::is_same_v<decltype(left)::order_tag, LayoutLeft>,
                "the order tag must reach the node type");

  // The tile the graph would resolve for these labels under Map.
  using TileT     = StaticTile<fTE, fA, fB>;
  const auto lr   = make_tile_layout(TileT{}, LayoutRight{});
  const auto ll   = make_tile_layout(TileT{}, LayoutLeft{});
  const int  size = lr.size();
  ASSERT_EQ(size, fTE * fA * fB);
  ASSERT_EQ(ll.size(), size);

  // Same coordinate SET, different sequence. Both halves matter: a layout that
  // visited a different set would be a bug, not a tuning choice.
  int differing = 0, seen_right = 0, seen_left = 0;
  for (int i = 0; i < size; ++i) {
    const auto cr = lr[i];
    const auto cl = ll[i];
    if (cr[0] != cl[0] || cr[1] != cl[1] || cr[2] != cl[2]) ++differing;
    seen_right += (cr[0] * fA + cr[1]) * fB + cr[2];
    seen_left += (cl[0] * fA + cl[1]) * fB + cl[2];
  }
  EXPECT_GT(differing, 0)
      << "LayoutLeft and LayoutRight enumerated the tile identically, so the "
         "order tag cannot be steering anything";
  EXPECT_EQ(seen_right, seen_left)
      << "the two orders visited different coordinate SETS, which is a bug "
         "rather than a different traversal";
}

// Kokkos's own layout tags are accepted, because a caller who already has a
// View thinks in those. Kokkos::LayoutRight{nE,nA,nB} carries extents as well
// as order, so it is a complete declaration -- but it is a DIFFERENT TYPE from
// this library's LayoutRight marker, and is converted at the factory boundary.
//
// Checked against the native spelling rather than against a reference: the
// claim is equivalence, and equivalence to the thing already tested is the
// strongest form of it. Both orders are covered, because a conversion that
// dropped the order entirely would still pass on Right alone.
TEST(FunctionalInput, AcceptsKokkosLayoutTagsEquivalentlyToNativeOnes) {
  ViewH      Hd("H", fQ, fA);
  ViewMap    Md("map", fE, fA, fB);
  ViewGlobal Gd("global", fNglob, fComp);
  ViewC      Cnative("Cnative", fQ, fE, fB);
  ViewC      Ckokkos("Ckokkos", fQ, fE, fB);

  fill_operators(Hd);
  auto Mh = Kokkos::create_mirror_view(Md);
  for (int e = 0; e < fE; ++e)
    for (int a = 0; a < fA; ++a)
      for (int b = 0; b < fB; ++b) Mh(e, a, b) = gid(e, a, b);
  Kokkos::deep_copy(Md, Mh);
  auto Gh = Kokkos::create_mirror_view(Gd);
  for (int n = 0; n < fNglob; ++n)
    for (int c = 0; c < fComp; ++c) Gh(n, c) = gval(n, c);
  Kokkos::deep_copy(Gd, Gh);

  const Kokkos::Array<int, 3> ext{fE, fA, fB};
  const GatherThroughMap      fn{Gd, Md, 1};

  for (int which = 0; which < 2; ++which) {
    if (which == 0) {
      run_graph(Hd, Cnative, [&] {
        return make_functional_input_node<'e', 'a', 'b'>(
            DynamicTileLayoutRight<3>{ext}, fn);
      });
      run_graph(Hd, Ckokkos, [&] {
        return make_functional_input_node<'e', 'a', 'b'>(
            Kokkos::LayoutRight{fE, fA, fB}, fn);
      });
    } else {
      run_graph(Hd, Cnative, [&] {
        return make_functional_input_node<'e', 'a', 'b'>(
            DynamicTileLayoutLeft<3>{ext}, fn);
      });
      run_graph(Hd, Ckokkos, [&] {
        return make_functional_input_node<'e', 'a', 'b'>(
            Kokkos::LayoutLeft{fE, fA, fB}, fn);
      });
    }

    auto Nh = Kokkos::create_mirror_view(Cnative);
    auto Kh = Kokkos::create_mirror_view(Ckokkos);
    Kokkos::deep_copy(Nh, Cnative);
    Kokkos::deep_copy(Kh, Ckokkos);

    int   diff  = 0;
    float scale = 0.0f;
    for (int q = 0; q < fQ; ++q)
      for (int e = 0; e < fE; ++e)
        for (int b = 0; b < fB; ++b) {
          scale = std::max(scale, std::abs(Nh(q, e, b)));
          if (Nh(q, e, b) != Kh(q, e, b)) ++diff;
        }
    EXPECT_GT(scale, 0.0f) << "void test: native path produced all zeros";
    EXPECT_EQ(diff, 0) << "Kokkos layout tag disagreed with the native one, "
                          "order index "
                       << which;
  }
}

// The Kokkos tag must reach the ORDER, not just the extents: a conversion that
// read dimension[] and then defaulted to LayoutRight would pass the runtime
// test above, since order is result-invariant by construction.
TEST(FunctionalInput, KokkosLayoutTagSelectsTheOrderNotJustTheExtents) {
  auto kr = make_functional_input_node<'e', 'a', 'b'>(
      Kokkos::LayoutRight{fE, fA, fB}, GlobalCoordProbe{});
  auto kl = make_functional_input_node<'e', 'a', 'b'>(
      Kokkos::LayoutLeft{fE, fA, fB}, GlobalCoordProbe{});
  static_assert(std::is_same_v<decltype(kr)::order_tag, LayoutRight>,
                "Kokkos::LayoutRight must map to LayoutRight");
  static_assert(std::is_same_v<decltype(kl)::order_tag, LayoutLeft>,
                "Kokkos::LayoutLeft must map to LayoutLeft");
  static_assert(
      std::is_same_v<decltype(kl)::layout_type, DynamicTileLayoutLeft<3>>,
      "the Kokkos tag must be converted to the native layout");
  // And the extents survived the conversion.
  EXPECT_EQ(kl.shape()[0], fE);
  EXPECT_EQ(kl.shape()[1], fA);
  EXPECT_EQ(kl.shape()[2], fB);
}

// Compile-time only, and deliberately so: the exec-space overload spells a
// label pack followed by an explicit type in the template argument list, which
// is the exact shape nvcc's EDG frontend rejects elsewhere in this header (see
// the note on functional_ret in Concept.hpp). An overload nothing instantiates
// is an overload nobody knows is broken, so name it here.
//
// DefaultHostExecutionSpace rather than DefaultExecutionSpace: on the CUDA
// build the two differ, so this distinguishes "the overload honoured my exec
// space" from "the default happened to be right". On the Serial build they
// coincide and the check degenerates to the default -- still compiled, still
// proving the overload resolves.
TEST(FunctionalInput, ExecSpaceOverloadHonoursTheRequestedSpace) {
  auto node = make_functional_input_node<Kokkos::DefaultHostExecutionSpace, 'e',
                                         'a', 'b'>(
      Kokkos::Array<int, 3>{fE, fA, fB}, GlobalCoordProbe{});
  static_assert(std::is_same_v<decltype(node)::exec_space,
                               Kokkos::DefaultHostExecutionSpace>,
                "the exec-space overload did not honour its exec space");
  static_assert(std::is_same_v<decltype(node)::value_type, float>,
                "value type must still be deduced from the functor");
  EXPECT_EQ(node.shape()[0], fE);
  EXPECT_EQ(node.shape()[2], fB);
}

TEST(FunctionalInput, FunctorSeesTheGlobalCoordinateNotTheTileLocalOne) {
  // Contract against the identity row of H so the contraction passes the staged
  // value through unchanged: C[q,e,b] = sum_a H[q,a] U[e,a,b] with H[q,a] =
  // (q == a) gives C[q,e,b] = U[e,q,b]. That keeps the probe's value visible in
  // the output instead of smeared across a sum.
  ViewH Hd("H", fQ, fA);
  ViewC Cd("C", fQ, fE, fB);

  auto Hh = Kokkos::create_mirror_view(Hd);
  for (int q = 0; q < fQ; ++q)
    for (int a = 0; a < fA; ++a) Hh(q, a) = (q == a) ? 1.0f : 0.0f;
  Kokkos::deep_copy(Hd, Hh);

  run_graph(Hd, Cd, [&] {
    return make_functional_input_node<'e', 'a', 'b'>(
        Kokkos::Array<int, 3>{fE, fA, fB}, GlobalCoordProbe{});
  });

  auto Ch = Kokkos::create_mirror_view(Cd);
  Kokkos::deep_copy(Ch, Cd);

  double max_err = 0.0;
  for (int q = 0; q < fQ; ++q)
    for (int e = 0; e < fE; ++e)
      for (int b = 0; b < fB; ++b) {
        const double ref = 1000.0 * e + 10.0 * q + 1.0 * b;
        max_err          = std::max(max_err, std::abs(ref - Ch(q, e, b)));
      }
  // A dropped origin makes team t report team 0's coordinates, so the error is
  // O(1000 * fTE) -- this bound is nowhere near it.
  EXPECT_LT(max_err, 1e-2)
      << "the functional source did not see the global coordinate: every team "
         "but the first would read tile 0";
}

}  // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
