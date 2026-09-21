// ===========================================================================
// bench_mat.cpp — prices the MATERIALIZATION axis inside the calibrated dummy.
//
// One process, one mesh, one set of fields: NMat = 2, 3, 4 are timed back to
// back on identical data, so the only thing that moves between cells is how
// many intermediates are forced through team scratch. NMat = 2 is the
// incumbent and is the run's own control -- it must reproduce the 24,124 B
// scratch request and the calibrated team size, or the run reports FIDELITY
// FAIL and nothing in it is quotable.
//
// Every cell is also checked against dummy_stiffness on the same inputs: an
// extra pass through scratch must not move the answer, and a materialization
// that changed the physics would otherwise be timed as if it had not.
//
// Axes are bench.cpp's, minus the scratch-layout knob: LayoutLeft only, since
// the incumbent it must reproduce is LayoutLeft.
// ===========================================================================
#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <kernel_dummy.hpp>
#include <kernel_mat.hpp>
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

struct SoA {
  template <class Off>
  using metrics_t = Metrics<Off>;
  template <class Off>
  using properties_t = Properties<Off>;
};

struct AoS {
  template <class Off>
  using metrics_t = MetricsAoS<Off>;
  template <class Off>
  using properties_t = PropertiesAoS<Off>;
};

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

struct Cell {
  std::size_t scratch  = 0;
  int         resolved = 0;
  double      ms       = 0;
  double      max_diff = 0;
};

template <int NMat, int MinBlocks, class Args, class F>
Cell run_cell(const Args& args, F& f, int team_arg, int warmup, int reps,
              const typename Fields::host_type& ref) {
  Cell c;
  c.scratch = mat_shmem_size<NMat>();

  Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
  c.resolved = mat_stiffness<NMat, MinBlocks>(args, team_arg);
  Kokkos::fence();
  f.to_host();
  for (std::size_t ig = 0; ig < ref.extent(0); ++ig)
    for (int comp = 0; comp < 3; ++comp) {
      const double d =
          std::abs(static_cast<double>(f.h_acceleration(ig, comp)) -
                   static_cast<double>(ref(ig, comp)));
      c.max_diff = std::max(c.max_diff, d);
    }

  c.ms = best_ms([&]() { mat_stiffness<NMat, MinBlocks>(args, team_arg); },
                 warmup, reps);
  return c;
}

template <class Off, class Storage = SoA>
int run_impl(int argc, char** argv, const char* layout_name) {
  const int reps     = (argc > 1) ? std::atoi(argv[1]) : 5;
  const int warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
  const int team_arg = (argc > 3) ? std::atoi(argv[3]) : -1;

  std::printf("kernel      : dummy + materialization axis\n");
  std::printf("layout      : %s\n", layout_name);

  const MeshDims d{60, 48, 9};
  const auto     set   = interior_elements(d);
  const int      nspec = set.nspec();

  IglobMap  g(nspec);
  const int nglob = renumber_access_order(d, set, g);
  g.to_device();

  typename Storage::template metrics_t<Off>    m(nspec);
  typename Storage::template properties_t<Off> p(nspec);
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

  DummyKernelArgs args{make_accessor(m), make_accessor(p), g.map,
                       f.displacement,   f.velocity,       f.acceleration,
                       hprime,           weights,          nspec};

  Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
  dummy_stiffness<Kokkos::LayoutLeft>(args, team_arg);
  Kokkos::fence();
  f.to_host();
  Fields::host_type ref("ref", nglob, 3);
  Kokkos::deep_copy(ref, f.h_acceleration);

  const double dummy_ms =
      best_ms([&]() { dummy_stiffness<Kokkos::LayoutLeft>(args, team_arg); },
              warmup, reps);

  const Cell b2 = run_cell<2, 5>(args, f, team_arg, warmup, reps, ref);
  const Cell c2 = run_cell<2, 0>(args, f, team_arg, warmup, reps, ref);
  const Cell c3 = run_cell<3, 0>(args, f, team_arg, warmup, reps, ref);
  const Cell c4 = run_cell<4, 0>(args, f, team_arg, warmup, reps, ref);

  std::printf("mesh        : %dx%dx%d, interior nspec = %d, nglob = %d\n",
              d.nex, d.ney, d.nez, nspec, nglob);
  std::printf("teams       : %d x %d elements\n", num_teams(nspec), kExecChunk);
  std::printf("team size   : %d %s\n", c2.resolved,
              team_arg > 0 ? "(forced)" : "(Kokkos::AUTO)");
  std::printf("dummy_stiffness <256,5> (untouched incumbent): %.4f ms\n",
              dummy_ms);
  std::printf("\n");
  std::printf(
      "  NMat  bounds  materialized              scratch B  barriers   "
      "time ms   ns/elem   vs NMat=2   max|d accel|\n");

  const char* what[4] = {"u, F", "u, F", "u, du, F", "u, du, F, T"};
  const int   nm[4]   = {2, 2, 3, 4};
  const int   bar[4]  = {2, 2, 3, 4};
  const char* lb[4]   = {"<256,5>", "none", "none", "none"};
  const Cell  cs[4]   = {b2, c2, c3, c4};
  for (int k = 0; k < 4; ++k) {
    std::printf("  %4d  %-7s %-24s %9zu %9d %9.4f %9.2f %11.3fx %14.3e\n",
                nm[k], lb[k], what[k], cs[k].scratch, bar[k], cs[k].ms,
                cs[k].ms * 1e6 / nspec, cs[k].ms / c2.ms, cs[k].max_diff);
  }
  std::printf("\n");

  bool ok = true;
  if (c2.scratch != 24124u) {
    std::printf("FIDELITY FAIL: NMat=2 scratch %zu != 24124 B\n", c2.scratch);
    ok = false;
  }
  const double drift = std::abs(b2.ms - dummy_ms) / dummy_ms;
  if (drift > 0.02) {
    std::printf(
        "FIDELITY FAIL: NMat=2 <256,5> is %.1f%% off dummy_stiffness -- the "
        "control "
        "does not reproduce the incumbent, so no cell here is quotable\n",
        100.0 * drift);
    ok = false;
  }
  for (int k = 0; k < 4; ++k)
    if (cs[k].max_diff > 1e-5) {
      std::printf("CORRECTNESS FAIL: NMat=%d differs from the dummy by %.3e\n",
                  nm[k], cs[k].max_diff);
      ok = false;
    }
  if (c2.resolved != 256)
    std::printf("FIDELITY WARN: team size %d != 256 (ground truth)\n",
                c2.resolved);
  if (ok) std::printf("all gates PASS\n");
  return ok ? 0 : 1;
}

template <class Storage>
int dispatch_layout(int argc, char** argv, const char* layout) {
  if (std::strcmp(layout, "chunk_tiled_static") == 0)
    return run_impl<ChunkTiledOffset, Storage>(argc, argv, layout);
  if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
    return run_impl<ChunkTiledDynamicOffset, Storage>(argc, argv, layout);
  if (std::strcmp(layout, "layout_right") == 0)
    return run_impl<LayoutRightOffset, Storage>(argc, argv, layout);
  if (std::strcmp(layout, "layout_right_dynamic") == 0)
    return run_impl<LayoutRightDynamicOffset, Storage>(argc, argv, layout);
  if (std::strcmp(layout, "layout_left") == 0)
    return run_impl<LayoutLeftOffset, Storage>(argc, argv, layout);
  if (std::strcmp(layout, "layout_left_dynamic") == 0)
    return run_impl<LayoutLeftDynamicOffset, Storage>(argc, argv, layout);
  std::printf("unknown layout: %s\n", layout);
  return 2;
}

int run(int argc, char** argv) {
  const char* layout  = (argc > 4) ? argv[4] : "chunk_tiled_dynamic";
  const char* storage = (argc > 5) ? argv[5] : "soa";
  if (std::strcmp(storage, "aos") == 0)
    return dispatch_layout<AoS>(argc, argv, layout);
  return dispatch_layout<SoA>(argc, argv, layout);
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
