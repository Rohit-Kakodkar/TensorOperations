// ===========================================================================
// bench_new.cpp — wall-clock for the LevelGraph kernel, on the SAME interior
// mesh, access renumbering and best-of-N harness bench.cpp uses for the dummy,
// so the two numbers compare within one surface.
//
// Two axes the dummy bench does not have, and one it has that this one drops:
//
//   * loads = once | redundant. The formulation saving and the load-once saving
//     are reported separately (plan's two-variant decision); this selects which
//     variant runs. `once` is the kernel as written; `redundant` reissues the
//     dummy's dead global loads so a warp-inst delta has its own opcode alibi.
//
//   * hprimewgll is precomputed ONCE here and handed to new_stiffness, so the
//     device build of hw(p,r)=hprime(p,r)*w(p) never lands in the timed region.
//
//   * iglob = ir | il (argv[8], DEFAULT ir). The mesh index map's memory order.
//     The graph's threads walk ix fastest (the staging level inherits the
//     functional input's LayoutRight tile order), so LayoutRight puts ix at
//     stride 1; LayoutLeft -- the incumbent, and what bench.cpp keeps -- puts
//     ispec there instead and gives every lane of a warp its own sector. `il`
//     is kept here only as the in-run CONTROL for that claim.
//
//     NOTE what this does NOT fix: renumber_access_order numbers iglob VALUES
//     fastest along the element axis, so the u(iglob,comp) dereference is still
//     strided for this kernel. This axis moves the index read, not the indirect
//     read behind it.
//
//   * numbering = xfast | acc (argv[9], DEFAULT xfast). The mesh NUMBERING,
//     which sets iglob's VALUES and therefore how the u(iglob,comp) gather and
//     the atomic scatter land. `xfast` numbers ix innermost, in the graph's own
//     thread order; `acc` is the dummy's incumbent numbering (element axis
//     innermost), kept as the in-run CONTROL. A runtime branch, not a template
//     axis -- it changes values, not types, so it costs no instantiation.
//
//   * NO scratch-layout axis. The graph is LayoutRight-on-scratch, full stop --
//     that asymmetry against the dummy's LayoutLeft is a REPORTED divergence
//     (Sprint 4 priced it at +1.65%), not a benchmark knob. `lr` is rejected.
//
// The dummy bench's `scratch != 24124` FATAL guard is calibration for the dummy
// and cannot apply here. It is replaced by the new kernel's own footprint
// (pooled + unpooled scratch, from new_footprint, no launch) printed for the
// GATE C report -- expected higher than the dummy's 24,124, which is the point.
// LaunchBounds<256,5> is deliberately NOT inherited; team size is swept via
// argv.
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

template <bool Keep, class Off, class Storage, class IglobLayout>
int run_impl(int argc, char** argv, const char* tag) {
  const int  reps     = (argc > 1) ? std::atoi(argv[1]) : 5;
  const int  warmup   = (argc > 2) ? std::atoi(argv[2]) : 2;
  const int  team_arg = (argc > 3) ? std::atoi(argv[3]) : -1;
  const bool profile  = (argc > 4) && std::strlen(argv[4]) > 0;

  std::printf("kernel      : new (level graph), loads=%s\n",
              Keep ? "redundant" : "once");
  std::printf("cell        : %s\n", tag);

  const MeshDims d{60, 48, 9};
  const auto     set   = interior_elements(d);
  const int      nspec = set.nspec();

  // The mesh NUMBERING is a runtime choice: it changes iglob's VALUES, not any
  // type, so it costs no instantiation. `xfast` numbers in the graph's own
  // thread order; `acc` is the dummy's incumbent numbering, kept as the
  // control.
  const char* numbering = (argc > 9) ? argv[9] : "xfast";
  const bool  num_acc   = std::strcmp(numbering, "acc") == 0;
  if (!num_acc && std::strcmp(numbering, "xfast") != 0) {
    std::printf("unknown numbering: %s (expected xfast | acc)\n", numbering);
    return 2;
  }
  IglobMapT<IglobLayout> g(nspec);
  const int              nglob = num_acc ? renumber_access_order(d, set, g)
                                         : renumber_ix_fastest(d, set, g);
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

  // Out of the timed region: the divergence operator, built once.
  const GlobalHPrime hpwgll = make_hprimewgll(hprime, weights);
  Kokkos::fence();

  DummyKernelArgs args{make_accessor(m), make_accessor(p), g.map,
                       f.displacement,   f.velocity,       f.acceleration,
                       hprime,           weights,          nspec};

  const NewFootprint fp = new_footprint<Keep>(args, hpwgll);

  const int league = new_stiffness<Keep>(args, team_arg, hpwgll);
  Kokkos::fence();

  std::printf("mesh        : %dx%dx%d, interior nspec = %d, nglob = %d\n",
              d.nex, d.ney, d.nez, nspec, nglob);
  std::printf("teams       : %d (league) x %d elements\n", league, kExecChunk);
  std::printf("team size   : %s\n",
              team_arg > 0 ? std::to_string(team_arg).c_str()
                           : "Kokkos::AUTO (resolved size from ncu/launch)");
  std::printf(
      "scratch     : pooled %zu B (launch request)  |  unpooled %zu B  "
      "|  dummy incumbent 24124 B\n",
      fp.pooled, fp.unpooled);

  if (profile) {
    Kokkos::deep_copy(f.acceleration, static_cast<real_t>(0));
    new_stiffness<Keep>(args, team_arg, hpwgll);
    Kokkos::fence();
    std::printf("profile mode: one launch issued\n");
    return 0;
  }

  const double ms = best_ms(
      [&]() { new_stiffness<Keep>(args, team_arg, hpwgll); }, warmup, reps);
  const double ns_per_element = ms * 1e6 / nspec;
  const double ns_per_point   = ns_per_element / kPointsPerElement;

  std::printf("time        : %.4f ms  |  %.2f ns/element  |  %.4f ns/point\n",
              ms, ns_per_element, ns_per_point);
  std::printf(
      "ground truth: 19.49 ns/element (A100 PCIe, dummy, ncu-locked clocks "
      "excluded) -- compare only within one surface\n");
  return 0;
}

template <bool Keep, class Storage, class IglobLayout>
int dispatch_layout(int argc, char** argv, const char* layout,
                    const char* tag) {
  if (std::strcmp(layout, "chunk_tiled_static") == 0)
    return run_impl<Keep, ChunkTiledOffset, Storage, IglobLayout>(argc, argv,
                                                                  tag);
  if (std::strcmp(layout, "chunk_tiled_dynamic") == 0)
    return run_impl<Keep, ChunkTiledDynamicOffset, Storage, IglobLayout>(
        argc, argv, tag);
  if (std::strcmp(layout, "layout_right") == 0)
    return run_impl<Keep, LayoutRightOffset, Storage, IglobLayout>(argc, argv,
                                                                   tag);
  if (std::strcmp(layout, "layout_right_dynamic") == 0)
    return run_impl<Keep, LayoutRightDynamicOffset, Storage, IglobLayout>(
        argc, argv, tag);
  if (std::strcmp(layout, "layout_left") == 0)
    return run_impl<Keep, LayoutLeftOffset, Storage, IglobLayout>(argc, argv,
                                                                  tag);
  if (std::strcmp(layout, "layout_left_dynamic") == 0)
    return run_impl<Keep, LayoutLeftDynamicOffset, Storage, IglobLayout>(
        argc, argv, tag);
  std::printf("unknown layout: %s\n", layout);
  return 2;
}

template <bool Keep, class IglobLayout>
int dispatch_storage(int argc, char** argv, const char* layout,
                     const char* storage, const char* tag) {
  if (std::strcmp(storage, "aos") == 0) {
    if (std::strcmp(layout, "chunk_tiled_static") == 0 ||
        std::strcmp(layout, "chunk_tiled_dynamic") == 0)
      return dispatch_layout<Keep, AoS, IglobLayout>(argc, argv, layout, tag);
    std::printf(
        "storage=aos is measured only for chunk_tiled_static and "
        "chunk_tiled_dynamic\n");
    return 2;
  }
  return dispatch_layout<Keep, SoA, IglobLayout>(argc, argv, layout, tag);
}

template <class IglobLayout>
int dispatch_loads(int argc, char** argv, const char* layout,
                   const char* storage, bool redundant, const char* tag) {
  return redundant ? dispatch_storage<true, IglobLayout>(argc, argv, layout,
                                                         storage, tag)
                   : dispatch_storage<false, IglobLayout>(argc, argv, layout,
                                                          storage, tag);
}

int run(int argc, char** argv) {
  const char* layout  = (argc > 5) ? argv[5] : "chunk_tiled_dynamic";
  const char* storage = (argc > 6) ? argv[6] : "soa";
  const char* loads   = (argc > 7) ? argv[7] : "once";
  // iglob layout. `ir` is the DEFAULT here and `il` is the incumbent: the
  // graph's threads walk ix fastest, so a LayoutRight map puts that axis at
  // stride 1. The dummy is the mirror image (element-fastest threads, and
  // renumber_access_order numbers for that), so bench.cpp keeps LayoutLeft and
  // is not given this knob at all. `il` exists here as the in-run CONTROL.
  const char* iglob = (argc > 8) ? argv[8] : "ir";

  const bool redundant = std::strcmp(loads, "redundant") == 0;

  const char* numbering = (argc > 9) ? argv[9] : "xfast";

  const std::string label = std::string(layout) + " " + storage + " " + loads +
                            " iglob=" + iglob + " num=" + numbering;
  const char*       tag   = label.c_str();

  if (std::strcmp(iglob, "ir") == 0)
    return dispatch_loads<Kokkos::LayoutRight>(argc, argv, layout, storage,
                                               redundant, tag);
  if (std::strcmp(iglob, "il") == 0)
    return dispatch_loads<Kokkos::LayoutLeft>(argc, argv, layout, storage,
                                              redundant, tag);
  std::printf("unknown iglob layout: %s (expected ir | il)\n", iglob);
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
