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
#include <type_traits>

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

// The backend is fixed per BINARY, not per run: sfpp_min_bench_te is the team
// policy, sfpp_min_bench_te_cute (SFPP_MIN_BENCH_TE_CUTE) the CuTe policy.
// Instantiating both launches of the same graph in one translation unit makes
// nvcc stop inlining the team kernel's helpers (255 registers, ~15 KB stack,
// ~40x slower), so the team number is only trustworthy from a team-only TU.
struct TeamBackend {
  static constexpr const char* name = "team";

  template <bool Keep, int TE, class Args>
  static NewFootprint footprint(const Args& args, const GlobalHPrime& hw) {
    return new_footprint<Keep, TE>(
        args, TensorOperations::TeamPolicyTag<KernelES>{}, hw);
  }
  template <bool Keep, int TE, class Args>
  static int launch(const Args& args, int team_arg, const GlobalHPrime& hw) {
    return new_stiffness<Keep, TE>(
        args, TensorOperations::TeamPolicyTag<KernelES>{}, team_arg, hw);
  }
};

#if defined(SFPP_MIN_BENCH_TE_CUTE)
#if defined(SFPP_MIN_BENCH_ROW_OF_E)
template <int TE>
using BenchMmas = RowOfEMmas<TE>;
#else
template <int TE>
using BenchMmas = RowOfIMmas<TE>;
#endif

struct CuteBackend {
#if defined(SFPP_MIN_BENCH_ROW_OF_E)
  static constexpr const char* name = "cute (row of e)";
#else
  static constexpr const char* name = "cute (row of i)";
#endif

  template <bool Keep, int TE, class Args>
  static NewFootprint footprint(const Args& args, const GlobalHPrime& hw) {
    return new_footprint<Keep, TE, BenchMmas<TE>>(
        args, TensorOperations::CutePolicyTag<KernelES>{}, hw);
  }
  template <bool Keep, int TE, class Args>
  static int launch(const Args& args, int, const GlobalHPrime& hw) {
    return new_stiffness<Keep, TE, BenchMmas<TE>>(
        args, TensorOperations::CutePolicyTag<KernelES>{}, hw);
  }
};
#endif

template <class Backend, bool Keep, int TE>
int run_impl(int argc, char** argv) {
  const int  reps     = (argc > 1) ? std::atoi(argv[1]) : 5;
  const int  warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
  const int  team_arg = (argc > 3) ? std::atoi(argv[3]) : -1;
  const bool profile  = (argc > 4) && std::strlen(argv[4]) > 0;

  if (!std::is_same_v<Backend, TeamBackend> && team_arg > 0) {
    std::printf(
        "the CuTe block size comes from the graph's MMAs; team size "
        "(argv[3]) must be -1\n");
    return 2;
  }

  std::printf("kernel      : new (level graph), backend=%s, TE=%d, loads=%s\n",
              Backend::name, TE, Keep ? "redundant" : "once");
  std::printf("cell        : layout_right_dynamic soa iglob=ir num=xfast\n");

  const MeshDims d{60, 48, 9};
  const auto     set        = interior_elements(d);
  const int      nspec_real = set.nspec();
  const int      nspec      = (nspec_real + TE - 1) / TE * TE;

  IglobMapRight g(nspec);
  const int     nglob = renumber_ix_fastest(d, set, g);
  for (int e = nspec_real; e < nspec; ++e)
    for (int k = 0; k < NGLL; ++k)
      for (int j = 0; j < NGLL; ++j)
        for (int i = 0; i < NGLL; ++i) g.h_map(e, k, j, i) = nglob;
  g.to_device();

  Metrics<Off>    m(nspec);
  Properties<Off> p(nspec);
  fill_curved(d, set, m);
  fill_properties(d, set, p);
  m.to_device();
  p.to_device();

  Fields f(nglob + 1);
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
  const NewFootprint fp = Backend::template footprint<Keep, TE>(args, hpwgll);
  std::printf(
      "scratch     : pooled %zu B  |  unpooled %zu B  |  %.0f B/element\n",
      fp.pooled, fp.unpooled, static_cast<double>(fp.pooled) / TE);

  const int league = Backend::template launch<Keep, TE>(args, team_arg, hpwgll);
  Kokkos::fence();

  std::printf("mesh        : interior nspec = %d (padded to %d), nglob = %d\n",
              nspec_real, nspec, nglob);
  {
    auto   h   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     f.acceleration);
    double sum = 0.0, sum2 = 0.0;
    for (int ig = 0; ig < nglob; ++ig)
      for (int c = 0; c < 3; ++c) {
        sum += h(ig, c) * (1.0 + 0.001 * ((ig * 7 + c) % 13));
        sum2 += static_cast<double>(h(ig, c)) * h(ig, c);
      }
    std::printf("checksum    : weighted %.9e  |  l2^2 %.9e\n", sum, sum2);
  }
  std::printf("teams       : %d (league) x %d elements = %d work items/team\n",
              league, TE, TE * kPointsPerElement);
  if (std::is_same_v<Backend, TeamBackend>)
    std::printf("team size   : %s\n", team_arg > 0
                                          ? std::to_string(team_arg).c_str()
                                          : "Kokkos::AUTO");
  else
    std::printf("block size  : %d (from the graph's MMAs)\n", fp.threads);

  if (profile) {
    Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
    Backend::template launch<Keep, TE>(args, team_arg, hpwgll);
    Kokkos::fence();
    std::printf("profile mode: one launch issued\n");
    return 0;
  }

  const double ms = best_ms(
      [&]() { Backend::template launch<Keep, TE>(args, team_arg, hpwgll); },
      warmup, reps);
  const double ns_per_element = ms * 1e6 / nspec_real;

  std::printf("time        : %.4f ms  |  %.2f ns/element  |  %.4f ns/point\n",
              ms, ns_per_element, ns_per_element / kPointsPerElement);
  return 0;
}

template <class Backend, bool Keep, int TE>
int run_te(int argc, char** argv) {
  constexpr int threads = NGLL * NGLL * TE;
  if constexpr (!std::is_same_v<Backend, TeamBackend> && threads > 1024) {
    std::printf(
        "TE=%d needs %d threads per block at NGLL=%d; a CuTe block has at "
        "most 1024\n",
        TE, threads, NGLL);
    return 2;
  } else {
    return run_impl<Backend, Keep, TE>(argc, argv);
  }
}

template <class Backend, bool Keep>
int dispatch_te(int argc, char** argv, int te) {
  switch (te) {
    case 1:
      return run_te<Backend, Keep, 1>(argc, argv);
    case 2:
      return run_te<Backend, Keep, 2>(argc, argv);
#if defined(SFPP_MIN_BENCH_ROW_OF_E)
    case 3:
      return run_te<Backend, Keep, 3>(argc, argv);
#endif
    case 4:
      return run_te<Backend, Keep, 4>(argc, argv);
    case 5:
      return run_te<Backend, Keep, 5>(argc, argv);
    case 8:
      return run_te<Backend, Keep, 8>(argc, argv);
    case 16:
      return run_te<Backend, Keep, 16>(argc, argv);
    case 32:
      return run_te<Backend, Keep, 32>(argc, argv);
    default:
      std::printf("unknown TE: %d (compiled for 1, 2, 4, 5, 8, 16, 32)\n", te);
      return 2;
  }
}

template <class Backend>
int dispatch_loads(int argc, char** argv, int te, const char* loads) {
  return (std::strcmp(loads, "redundant") == 0)
             ? dispatch_te<Backend, true>(argc, argv, te)
             : dispatch_te<Backend, false>(argc, argv, te);
}

int run(int argc, char** argv) {
  const int   te    = (argc > 5) ? std::atoi(argv[5]) : kExecChunk;
  const char* loads = (argc > 6) ? argv[6] : "once";
#if defined(SFPP_MIN_BENCH_TE_CUTE)
  return dispatch_loads<CuteBackend>(argc, argv, te, loads);
#else
  return dispatch_loads<TeamBackend>(argc, argv, te, loads);
#endif
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
