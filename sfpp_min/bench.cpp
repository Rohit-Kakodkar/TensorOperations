#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <kernel_dummy.hpp>
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

template <class Off, class Storage = SoA,
          class ScratchLayout = Kokkos::LayoutLeft>
int run_impl(int argc, char** argv, const char* layout_name) {
  const int  reps     = (argc > 1) ? std::atoi(argv[1]) : 5;
  const int  warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
  const int  team_arg = (argc > 3) ? std::atoi(argv[3]) : -1;
  const bool profile  = (argc > 4) && std::strlen(argv[4]) > 0;

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

  DummyKernelArgs args{make_accessor(m), make_accessor(p), g.map,
                       f.displacement,   f.velocity,       f.acceleration,
                       hprime,           weights,          nspec};

  const int resolved = dummy_stiffness<ScratchLayout>(args, team_arg);
  Kokkos::fence();

  const std::size_t scratch = dummy_shmem_size<ScratchLayout>();
  const std::size_t launch_shmem =
      scratch + 8u * (static_cast<std::size_t>(resolved) + 2u);

  std::printf("mesh        : %dx%dx%d, interior nspec = %d, nglob = %d\n",
              d.nex, d.ney, d.nez, nspec, nglob);
  std::printf("teams       : %d x %d elements\n", num_teams(nspec), kExecChunk);
  std::printf("team size   : %d %s\n", resolved,
              team_arg > 0 ? "(forced)" : "(Kokkos::AUTO)");
  std::printf("scratch req : %zu B  -> predicted launch shmem %zu B\n", scratch,
              launch_shmem);

  // The three ways to silently break fidelity -- tidied-away redundant loads,
  // the wrong scratch layout, an unpinned team size -- all make this kernel
  // FASTER. Review does not catch that, so assert it here.
  bool ok = true;
  if (scratch != 24124u) {
    std::printf("FIDELITY FAIL: scratch request %zu != 24124 B\n", scratch);
    ok = false;
  }
  if (resolved != 256) {
    std::printf(
        "FIDELITY WARN: team size %d != 256 (ground truth); the "
        "register/scratch footprint has drifted\n",
        resolved);
  }
  if (launch_shmem != 26188u) {
    std::printf("FIDELITY WARN: predicted launch shmem %zu != 26188 B\n",
                launch_shmem);
  }

  if (profile) {
    Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
    dummy_stiffness<ScratchLayout>(args, team_arg);
    Kokkos::fence();
    std::printf("profile mode: one launch issued\n");
    return ok ? 0 : 1;
  }

  const double ms = best_ms(
      [&]() { dummy_stiffness<ScratchLayout>(args, team_arg); }, warmup, reps);
  const double ns_per_element = ms * 1e6 / nspec;
  const double ns_per_point   = ns_per_element / kPointsPerElement;

  std::printf("time        : %.4f ms  |  %.2f ns/element  |  %.4f ns/point\n",
              ms, ns_per_element, ns_per_point);
  std::printf(
      "ground truth: 19.49 ns/element (A100 PCIe, ncu-locked clocks "
      "excluded) -- compare only within one surface\n");
  return ok ? 0 : 1;
}

int run(int argc, char** argv) {
  const char* layout  = (argc > 5) ? argv[5] : "chunk_tiled_dynamic";
  const char* storage = (argc > 6) ? argv[6] : "soa";
  const char* scratch = (argc > 7) ? argv[7] : "ll";

  const std::string label = std::string(layout) + " " + storage + " " + scratch;
  const char*       tag   = label.c_str();

  const bool aos = std::strcmp(storage, "aos") == 0;
  const bool lr  = std::strcmp(scratch, "lr") == 0;

  // SoA x scratch-LayoutLeft: the full Sprint 3 offset plane (the incumbent is
  // chunk_tiled_dynamic in this plane).
  if (!aos && !lr) {
    if (std::strcmp(layout, "chunk_tiled_static") == 0)
      return run_impl<ChunkTiledOffset, SoA, Kokkos::LayoutLeft>(argc, argv,
                                                                 tag);
    if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
      return run_impl<ChunkTiledDynamicOffset, SoA, Kokkos::LayoutLeft>(
          argc, argv, tag);
    if (std::strcmp(layout, "layout_right") == 0)
      return run_impl<LayoutRightOffset, SoA, Kokkos::LayoutLeft>(argc, argv,
                                                                  tag);
    if (std::strcmp(layout, "layout_right_dynamic") == 0)
      return run_impl<LayoutRightDynamicOffset, SoA, Kokkos::LayoutLeft>(
          argc, argv, tag);
    if (std::strcmp(layout, "layout_left") == 0)
      return run_impl<LayoutLeftOffset, SoA, Kokkos::LayoutLeft>(argc, argv,
                                                                 tag);
    if (std::strcmp(layout, "layout_left_dynamic") == 0)
      return run_impl<LayoutLeftDynamicOffset, SoA, Kokkos::LayoutLeft>(
          argc, argv, tag);
  }

  // SoA x scratch-LayoutRight: the scratch axis, measured at the incumbent
  // offset only.
  if (!aos && lr) {
    if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
      return run_impl<ChunkTiledDynamicOffset, SoA, Kokkos::LayoutRight>(
          argc, argv, tag);
    std::printf("scratch=lr is measured only for chunk_tiled_dynamic\n");
    return 2;
  }

  // AoS x scratch-LayoutLeft: the container axis, at static and dynamic extent.
  if (aos && !lr) {
    if (std::strcmp(layout, "chunk_tiled_static") == 0)
      return run_impl<ChunkTiledOffset, AoS, Kokkos::LayoutLeft>(argc, argv,
                                                                 tag);
    if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
      return run_impl<ChunkTiledDynamicOffset, AoS, Kokkos::LayoutLeft>(
          argc, argv, tag);
    std::printf(
        "storage=aos is measured only for chunk_tiled_static and "
        "chunk_tiled_dynamic\n");
    return 2;
  }

  // AoS x scratch-LayoutRight: the AoS-scratch interaction, at the incumbent
  // offset only.
  if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
    return run_impl<ChunkTiledDynamicOffset, AoS, Kokkos::LayoutRight>(
        argc, argv, tag);
  std::printf(
      "storage=aos scratch=lr is measured only for chunk_tiled_dynamic\n");
  return 2;
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
