// ===========================================================================
// bench_te.cpp — the TILE-WIDTH sweep for the level graph.
//
// TE is the number of elements one team owns. It is a COMPILE-TIME parameter
// of the graph (GMap carries LabelTile<'e', TE>), so every value is a fresh
// instantiation of the whole type tree. Putting it inside bench_new.cpp would
// multiply that harness's entire axis matrix by the number of TEs; this binary
// instead pins every other axis at the winning cell and varies only TE:
//
//   metrics/properties : layout_right_dynamic        (soa)
//   iglob layout       : LayoutRight
//   mesh numbering     : ix-fastest (the graph's own thread order)
//   loads              : once | redundant            (kept, so the load
//                                                     attribution stays
//                                                     self-consistent at the
//                                                     winning TE)
//
// The dummy is NOT swept: kExecChunk = 4 is its calibrated ground truth and
// bench.cpp is untouched.
//
// WHAT MOVES WITH TE, and why the answer is not obvious in advance:
//   * scratch scales LINEARLY with TE (~9 KB per element at NGLL=5), so TE
//     buys work per team at a direct cost in occupancy, and eventually hits
//     the shared-memory cap outright. new_footprint reports the request
//     without launching, so the ceiling is visible before the failure.
//   * work items per team = TE * 125, which changes how a given team size
//     divides the tile -- so TE and team size are NOT separable, and each TE
//     gets its own team sweep rather than inheriting TE=4's winner.
//   * the operator staging (hprime, hprimewgll) is per-TEAM, so its cost is
//     amortized over TE elements: larger TE dilutes it.
//
// TE must divide nspec: the team tier has no remainder path. nspec = 21344 =
// 2^5 * 23 * 29, so the powers of two up to 32 all divide it exactly.
// ===========================================================================
#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <kernel_new.hpp>
#include <layout.hpp>
#include <mesh.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace sfpp_min;

namespace {

using Off = LayoutRightDynamicOffset;

template <class Fn>
double best_ms(Fn&& fn, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) fn();
  Kokkos::fence();
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    Kokkos::Timer timer;
    fn();
    Kokkos::fence();
    best = std::min(best, timer.seconds());
  }
  return best * 1e3;
}

template <bool Keep, int TE>
int run_impl(int argc, char** argv) {
  const int  reps     = (argc > 1) ? std::atoi(argv[1]) : 5;
  const int  warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
  const int  team_arg = (argc > 3) ? std::atoi(argv[3]) : -1;
  const bool profile  = (argc > 4) && std::strlen(argv[4]) > 0;

  std::printf("kernel      : new (level graph), TE=%d, loads=%s\n", TE,
              Keep ? "redundant" : "once");
  std::printf("cell        : layout_right_dynamic soa iglob=ir num=xfast\n");

  const MeshDims d{60, 48, 9};
  const auto     set   = interior_elements(d);
  const int      nspec = set.nspec();
  if (nspec % TE != 0) {
    std::printf(
        "TE=%d does not divide nspec=%d; the team tier has no "
        "remainder path\n",
        TE, nspec);
    return 2;
  }

  IglobMapRight g(nspec);
  const int     nglob = renumber_ix_fastest(d, set, g);
  g.to_device();

  Metrics<Off>    m(nspec);
  Properties<Off> p(nspec);
  fill_curved(d, set, m);
  fill_properties(d, set, p);
  m.to_device();
  p.to_device();

  Fields f(nglob);
  for (int ig = 0; ig < nglob; ++ig)
    for (int c = 0; c < 3; ++c) {
      f.h_displacement(ig, c) =
          static_cast<real_t>(1e-3 * std::sin(0.0007 * ig + 1.3 * c));
      f.h_velocity(ig, c) =
          static_cast<real_t>(0.37 * std::sin(0.011 * ig + 1.7 * c) + 0.13);
    }
  f.to_device();
  Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));

  GlobalHPrime  hprime("hprime", NGLL, NGLL);
  GlobalWeights weights("weights", NGLL);
  {
    auto hh = Kokkos::create_mirror_view(hprime);
    auto hw = Kokkos::create_mirror_view(weights);
    for (int a = 0; a < NGLL; ++a) {
      hw(a) = static_cast<real_t>(gll::weight(a));
      for (int b = 0; b < NGLL; ++b)
        hh(a, b) = static_cast<real_t>(gll::hprime(a, b));
    }
    Kokkos::deep_copy(hprime, hh);
    Kokkos::deep_copy(weights, hw);
  }

  const GlobalHPrime hpwgll = make_hprimewgll(hprime, weights);
  Kokkos::fence();

  DummyKernelArgs args{make_accessor(m), make_accessor(p), g.map,
                       f.displacement,   f.velocity,       f.acceleration,
                       hprime,           weights,          nspec};

  // The footprint is answerable on the host. Print it BEFORE the launch so a
  // TE that overruns shared memory is diagnosed by its request rather than by
  // an opaque launch failure.
  const NewFootprint fp = new_footprint<Keep, TE>(args, hpwgll);
  std::printf(
      "scratch     : pooled %zu B  |  unpooled %zu B  |  %.0f B/element\n",
      fp.pooled, fp.unpooled, static_cast<double>(fp.pooled) / TE);

  const int league = new_stiffness<Keep, TE>(args, team_arg, hpwgll);
  Kokkos::fence();

  std::printf("mesh        : interior nspec = %d, nglob = %d\n", nspec, nglob);
  std::printf("teams       : %d (league) x %d elements = %d work items/team\n",
              league, TE, TE * kPointsPerElement);
  std::printf("team size   : %s\n",
              team_arg > 0 ? std::to_string(team_arg).c_str() : "Kokkos::AUTO");

  if (profile) {
    Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
    new_stiffness<Keep, TE>(args, team_arg, hpwgll);
    Kokkos::fence();
    std::printf("profile mode: one launch issued\n");
    return 0;
  }

  const double ms = best_ms(
      [&]() { new_stiffness<Keep, TE>(args, team_arg, hpwgll); }, warmup, reps);
  const double ns_per_element = ms * 1e6 / nspec;

  std::printf("time        : %.4f ms  |  %.2f ns/element  |  %.4f ns/point\n",
              ms, ns_per_element, ns_per_element / kPointsPerElement);
  return 0;
}

template <bool Keep>
int dispatch_te(int argc, char** argv, int te) {
  switch (te) {
    case 1:
      return run_impl<Keep, 1>(argc, argv);
    case 2:
      return run_impl<Keep, 2>(argc, argv);
    case 4:
      return run_impl<Keep, 4>(argc, argv);
    case 8:
      return run_impl<Keep, 8>(argc, argv);
    case 16:
      return run_impl<Keep, 16>(argc, argv);
    case 32:
      return run_impl<Keep, 32>(argc, argv);
    default:
      std::printf("unknown TE: %d (compiled for 1, 2, 4, 8, 16, 32)\n", te);
      return 2;
  }
}

int run(int argc, char** argv) {
  const int   te    = (argc > 5) ? std::atoi(argv[5]) : kExecChunk;
  const char* loads = (argc > 6) ? argv[6] : "once";
  return (std::strcmp(loads, "redundant") == 0)
             ? dispatch_te<true>(argc, argv, te)
             : dispatch_te<false>(argc, argv, te);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    rc = run(argc, argv);
  }
  Kokkos::finalize();
  return rc;
}
