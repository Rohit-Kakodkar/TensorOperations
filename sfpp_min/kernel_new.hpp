#pragma once

// ============================================================================
// THE NEW KERNEL. The same 3D elastic stiffness operator kernel_dummy.hpp
// transcribes by hand, expressed instead as a TensorOperations LevelGraph over
// the SAME data structures (DummyKernelArgs, the SoA/AoS accessors, every
// offset policy). It reuses kernel_dummy.hpp's point loads and no-op consumers
// verbatim, so the two kernels are handed nvcc the SAME per-point load and
// dead-code-elimination decisions and a counter delta is attributable to the
// library's formulation, not to a rewritten physics stream.
//
// The two things the library did not have before this sprint, both at the
// boundary between the graph and the mesh:
//
//   * GATHER. The displacement is read u(iglob(e,k,j,i), c) -- an indirection
//     through the mesh index map, not an affine address. That is a functional
//     input (PR A): the field answers operator()(e,k,j,i) and the library never
//     learns what iglob is.
//
//   * SCATTER. The result is atomic_add(&acceleration(iglob,c), -res). That is
//     a sink combine (PR B): the terminal member writes to global itself and
//     returns void, so the graph executes with NO output views.
//
// FORMULATION DIVERGENCES from the dummy, reported not hidden (they are the
// point of the measurement):
//
//   * hprimewgll. The dummy folds the summed-axis quadrature weight into the
//     divergence loop (`F * hp(l,ix) * weights_g(l)`). The graph folds it into
//     the operator ONCE, hprimewgll(p,r) = hprime(p,r)*weight(p), so the back-
//     contraction is a plain hw . F. That is fewer multiplies per point -- a
//     structural FFMA difference the dummy's inline weight cannot have.
//
//   * scratch layout. The library is LayoutRight-on-scratch; the dummy is
//     LayoutLeft (SPECFEM++'s chunk_ndim_view). Sprint 4 priced that axis at
//     +1.65%.
//
//   * the tail check. The dummy carries SPECFEM++'s per-work-item is_end()
//     (num_elements = min(nspec-base,4)); the graph's tile map divides 21344
//     evenly by TE and has none, so it does slightly LESS work.
//
// KeepRedundantLoads reproduces the dummy's redundant per-point global loads --
// the second metric load, the dead displacement re-read, and the divergence-
// stage property/velocity/metric loads that feed only no-ops -- so the two
// variants bracket the load-once saving with its own opcode alibi.
// ============================================================================

#include <config.hpp>
#include <data.hpp>
#include <kernel_dummy.hpp>

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/LevelPlan.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace sfpp_min {

using KernelES = Kokkos::DefaultExecutionSpace;

// The gather the library never looks inside: a displacement value at a grid
// coordinate comes from the mesh index map. The coordinate is GLOBAL (the
// functional input folds the tile origin in), so iglob is indexed directly.
template <typename IglobView>
struct GatherDisplacement {
  Fields::view_type             u;
  IglobView                     iglob;
  int                           comp;
  KOKKOS_INLINE_FUNCTION real_t operator()(int e, int k, int j, int i) const {
    return u(iglob(e, k, j, i), comp);
  }
};

// Chain rule, Hooke stress and F = T*J for all nine (direction, component)
// pairs from one metric/material load. The nine gradients arrive as operands
// (x* = df/dxi, y* = df/deta, z* = df/dgamma, each per component); the metric
// and material fields ride in the closure and are read through the dummy's own
// load_metric/load_property, which is what preserves the offset arithmetic
// under test. The stress is spelled EXACTLY as the dummy spells it (l2m/lam on
// the diagonal, mu on the off-diagonal) so FFMA/pt stays a control.
template <bool KeepRedundantLoads, typename MetricsAcc, typename PropertiesAcc,
          typename IglobView>
struct Integrand9 {
  MetricsAcc        metrics;
  PropertiesAcc     properties;
  Fields::view_type displacement;
  IglobView         iglob;

  KOKKOS_FUNCTION Kokkos::Array<real_t, 9> operator()(
      int e, int k, int j, int i, real_t x0, real_t x1, real_t x2, real_t y0,
      real_t y1, real_t y2, real_t z0, real_t z1, real_t z2) const {
    const real_t df_dxi[3]    = {x0, x1, x2};
    const real_t df_deta[3]   = {y0, y1, y2};
    const real_t df_dgamma[3] = {z0, z1, z2};

    // Metric load #2 (10 comps) feeds F; the chain rule uses grad_metric. Under
    // load-once they are the same 10-comp load; the dummy loads a separate
    // 9-comp grad_metric first, so KeepRedundantLoads mirrors that order.
    PointMetric grad_metric;
    PointMetric point_metric;
    if constexpr (KeepRedundantLoads) {
      load_metric(metrics, e, k, j, i, grad_metric, false);
      load_metric(metrics, e, k, j, i, point_metric, true);
    } else {
      load_metric(metrics, e, k, j, i, point_metric, true);
      grad_metric = point_metric;
    }

    real_t du[3][3];
    for (int c = 0; c < 3; ++c) {
      du[c][0] = grad_metric.xix * df_dxi[c] + grad_metric.etax * df_deta[c] +
                 grad_metric.gammax * df_dgamma[c];
      du[c][1] = grad_metric.xiy * df_dxi[c] + grad_metric.etay * df_deta[c] +
                 grad_metric.gammay * df_dgamma[c];
      du[c][2] = grad_metric.xiz * df_dxi[c] + grad_metric.etaz * df_deta[c] +
                 grad_metric.gammaz * df_dgamma[c];
    }

    PointProperty point_property;
    load_property(properties, e, k, j, i, point_property);

    // The dummy re-reads displacement from GLOBAL here even though it is
    // staged, to feed the cosserat no-op. Reproduced only under
    // KeepRedundantLoads.
    real_t point_displacement[3] = {0, 0, 0};
    if constexpr (KeepRedundantLoads) {
      const int ig = iglob(e, k, j, i);
      for (int c = 0; c < 3; ++c) point_displacement[c] = displacement(ig, c);
    }

    const real_t l2m = point_property.lambdaplus2mu();
    const real_t lam = point_property.lambda();
    const real_t mu  = point_property.mu;

    real_t T[3][3];
    T[0][0] = l2m * du[0][0] + lam * (du[1][1] + du[2][2]);
    T[1][1] = l2m * du[1][1] + lam * (du[0][0] + du[2][2]);
    T[2][2] = l2m * du[2][2] + lam * (du[0][0] + du[1][1]);
    T[0][1] = mu * (du[0][1] + du[1][0]);
    T[1][0] = T[0][1];
    T[0][2] = mu * (du[0][2] + du[2][0]);
    T[2][0] = T[0][2];
    T[1][2] = mu * (du[1][2] + du[2][1]);
    T[2][1] = T[1][2];

    compute_attenuation(T);
    compute_cosserat_stress(point_property, point_displacement, T);

    real_t F0[3], F1[3], F2[3];
    for (int c = 0; c < 3; ++c) {
      F0[c] = point_metric.jacobian *
              (T[c][0] * point_metric.xix + T[c][1] * point_metric.xiy +
               T[c][2] * point_metric.xiz);
      F1[c] = point_metric.jacobian *
              (T[c][0] * point_metric.etax + T[c][1] * point_metric.etay +
               T[c][2] * point_metric.etaz);
      F2[c] = point_metric.jacobian *
              (T[c][0] * point_metric.gammax + T[c][1] * point_metric.gammay +
               T[c][2] * point_metric.gammaz);
    }
    return {F0[0], F0[1], F0[2], F1[0], F1[1], F1[2], F2[0], F2[1], F2[2]};
  }
};

// The terminal pair. What used to be one sink is now TWO combine nodes:
//
//   AccelFromDivergence  9 divergence operands -> r0, r1, r2   (three slots)
//   ScatterAccel         r0, r1, r2            -> atomic_add    (no slot)
//
// WHY SPLIT SOMETHING THAT ALREADY WORKED. A sink level's iteration is driven
// by its OPERAND 0 (LevelGraph.hpp, lg_run_combine_level: no output slot
// exists, so the tile comes from evs.get<0>().iter_view()). When operand 0 was
// tx0 -- a CONTRACTION result -- that view is stored in the contraction's
// CANONICAL order, freeA ++ freeB = (i, e, k, j), so the memory-order decode
// peeled j first: consecutive threads differed in j, and i, the axis iglob is
// numbered along, changed only every 100 threads. The atomics therefore
// scattered, and RED sectors sat at 1.73/pt against the dummy's 1.08 while
// every ordinary load had already been coalesced.
//
// A COMBINE result is stored in DECLARED order instead (lg_member_decl_tile
// applies permC only for ContractionTag). So making the sink's operand 0 a
// combine output -- r0 -- is what puts i back on the fastest axis. That is the
// whole point of the split: it is not a fusion decision, it is a decision about
// WHICH NODE KIND SITS IN THE OPERAND-0 SLOT, because that is what picks the
// thread map for the level.
//
// The price is explicit: three more slots and one more barrier. It is worth it
// only if coalesced atomics beat that, which is a measurement, not an argument.
template <typename WeightsView>
struct AccelFromDivergence {
  WeightsView w;

  KOKKOS_FUNCTION Kokkos::Array<real_t, 3> operator()(
      int, int k, int j, int i, real_t tx0, real_t tx1, real_t tx2, real_t te0,
      real_t te1, real_t te2, real_t tg0, real_t tg1, real_t tg2) const {
    const real_t wjk = w(j) * w(k);
    const real_t wik = w(i) * w(k);
    const real_t wij = w(i) * w(j);
    return {-(wjk * tx0 + wik * te0 + wij * tg0),
            -(wjk * tx1 + wik * te1 + wij * tg1),
            -(wjk * tx2 + wik * te2 + wij * tg2)};
  }
};

// The scatter, now reading three ready values. iglob is still read ONCE and the
// three atomics still land in one member. The redundant-load block moved here
// with the atomics it precedes, so the statement ORDER the dummy has --
// accel, then damping/boundary, then atomic_add -- is unchanged by the split.
template <bool KeepRedundantLoads, typename MetricsAcc, typename PropertiesAcc,
          typename IglobView>
struct ScatterAccel {
  Fields::view_type a;
  IglobView         iglob;
  GlobalWeights     w;
  Fields::view_type velocity;
  MetricsAcc        metrics;
  PropertiesAcc     properties;

  KOKKOS_FUNCTION void operator()(int e, int k, int j, int i, real_t r0,
                                  real_t r1, real_t r2) const {
    const int ig = iglob(e, k, j, i);

    real_t accel[3] = {r0, r1, r2};

    if constexpr (KeepRedundantLoads) {
      PointProperty div_property;
      load_property(properties, e, k, j, i, div_property);
      real_t point_velocity[3];
      for (int c = 0; c < 3; ++c) point_velocity[c] = velocity(ig, c);
      PointMetric div_metric;
      load_metric(metrics, e, k, j, i, div_metric, true);
      const real_t factor = w(i) * w(j) * w(k) * div_metric.jacobian;
      compute_damping_force(factor, div_property, point_velocity, accel);
      apply_boundary_conditions(div_property, point_velocity, accel);
    }

    Kokkos::atomic_add(&a(ig, 0), accel[0]);
    Kokkos::atomic_add(&a(ig, 1), accel[1]);
    Kokkos::atomic_add(&a(ig, 2), accel[2]);
  }
};

// hprimewgll(p,r) = hprime(p,r)*weight(p): the summed-axis quadrature weight
// the dummy carries INSIDE its divergence loop, folded here into the operator
// ONCE. Built on device from the two GLL views. The benchmark calls this before
// the timed region and passes the result to new_stiffness so it stays out of
// the measurement; tests let new_stiffness build it.
inline GlobalHPrime make_hprimewgll(GlobalHPrime  hprime,
                                    GlobalWeights weights) {
  GlobalHPrime hw("sfpp_min::hprimewgll", NGLL, NGLL);
  Kokkos::parallel_for(
      "sfpp_min::build_hprimewgll",
      Kokkos::RangePolicy<KernelES>(0, NGLL * NGLL),
      KOKKOS_LAMBDA(const int n) {
        const int p = n / NGLL;
        const int r = n % NGLL;
        hw(p, r)    = hprime(p, r) * weights(p);
      });
  return hw;
}

// The one place the graph is built. new_stiffness launches it; new_footprint
// queries its scratch without launching -- both go through here, so the
// footprint reported at GATE C is the SAME graph the launch requests, never a
// hand-copied upper bound that can drift from it.
template <bool KeepRedundantLoads, int TE, typename MetricsAcc,
          typename PropertiesAcc, typename IglobView>
auto build_new_graph(
    const DummyKernelArgs<MetricsAcc, PropertiesAcc, IglobView>& args,
    GlobalHPrime                                                 hw) {
  using namespace TensorOperations;
  using ES   = KernelES;
  using GMap = LabelTiles<LabelTile<'e', TE>, LabelWhole<'k', NGLL>,
                          LabelWhole<'j', NGLL>, LabelWhole<'i', NGLL>,
                          LabelWhole<'p', NGLL>, LabelWhole<'r', NGLL>>;

  const Kokkos::Array<int, 4> ext{args.nspec, NGLL, NGLL, NGLL};

  const Integrand9<KeepRedundantLoads, MetricsAcc, PropertiesAcc, IglobView>
      integrand{args.metrics, args.properties, args.displacement, args.iglob};
  const AccelFromDivergence<GlobalWeights> to_accel{args.weights};
  const ScatterAccel<KeepRedundantLoads, MetricsAcc, PropertiesAcc, IglobView>
      sink{args.acceleration, args.iglob,   args.weights,
           args.velocity,     args.metrics, args.properties};

  auto g0           = make_level_graph<real_t, ES>(GMap{});
  auto [g1, h, hwn] = g0.add(
      make_stage_node(make_input_node(make_handle<'r', 'p'>(args.hprime))),
      make_stage_node(make_input_node(make_handle<'p', 'r'>(hw))));
  auto [g2, u0, u1, u2] = g1.add(
      make_stage_node(make_functional_input_node<'e', 'k', 'j', 'i'>(
          ext,
          GatherDisplacement<IglobView>{args.displacement, args.iglob, 0})),
      make_stage_node(make_functional_input_node<'e', 'k', 'j', 'i'>(
          ext,
          GatherDisplacement<IglobView>{args.displacement, args.iglob, 1})),
      make_stage_node(make_functional_input_node<'e', 'k', 'j', 'i'>(
          ext,
          GatherDisplacement<IglobView>{args.displacement, args.iglob, 2})));

  auto gx = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'i', 'p'>(), uu.template as<'e', 'k', 'j', 'p'>());
  };
  auto ge = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'j', 'p'>(), uu.template as<'e', 'k', 'p', 'i'>());
  };
  auto gg = [&](auto uu) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        h.template as<'k', 'p'>(), uu.template as<'e', 'p', 'j', 'i'>());
  };
  auto [g3, gx0, gx1, gx2, ge0, ge1, ge2, gg0, gg1, gg2] = g2.add(
      gx(u0), gx(u1), gx(u2), ge(u0), ge(u1), ge(u2), gg(u0), gg(u1), gg(u2));

  auto [g4, fx0, fx1, fx2, fe0, fe1, fe2, fg0, fg1, fg2] =
      g3.add(make_combine_node<'e', 'k', 'j', 'i'>(gx0, gx1, gx2, ge0, ge1, ge2,
                                                   gg0, gg1, gg2, integrand));

  auto dvx = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hwn.template as<'p', 'i'>(), f.template as<'e', 'k', 'j', 'p'>());
  };
  auto dve = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hwn.template as<'p', 'j'>(), f.template as<'e', 'k', 'p', 'i'>());
  };
  auto dvg = [&](auto f) {
    return make_contraction_node<'e', 'k', 'j', 'i'>(
        hwn.template as<'p', 'k'>(), f.template as<'e', 'p', 'j', 'i'>());
  };
  auto [g5, tx0, tx1, tx2, te0, te1, te2, tg0, tg1, tg2] =
      g4.add(dvx(fx0), dvx(fx1), dvx(fx2), dve(fe0), dve(fe1), dve(fe2),
             dvg(fg0), dvg(fg1), dvg(fg2));

  auto [g6, r0, r1, r2] = g5.add(make_combine_node<'e', 'k', 'j', 'i'>(
      tx0, tx1, tx2, te0, te1, te2, tg0, tg1, tg2, to_accel));

  auto g7 = g6.add(make_combine_node<'e', 'k', 'j', 'i'>(r0, r1, r2, sink));

  using Plan = LevelPlan<std::decay_t<decltype(g7.levels)>>;
  static_assert(Plan::num_levels == 7,
                "two operator/stage levels then five compute levels: the "
                "terminal scatter is its own level so its operand 0 is a "
                "COMBINE result, in declared order, and the atomics walk i");
  static_assert(Plan::num_slots == 2 + 3 + 9 + 9 + 9 + 3 + 0,
                "three accel slots; the terminal sink still contributes none");
  return g7;
}

// The scratch the launch requests. `pooled` is what set_scratch_size gets
// (after liveness pooling); `unpooled` is one buffer per slot. The launch shmem
// ncu reports is `pooled` plus Kokkos's per-team overhead. No launch here.
struct NewFootprint {
  std::size_t pooled   = 0;
  std::size_t unpooled = 0;
};

template <bool KeepRedundantLoads = false, int TE = kExecChunk,
          typename MetricsAcc, typename PropertiesAcc, typename IglobView>
NewFootprint new_footprint(
    const DummyKernelArgs<MetricsAcc, PropertiesAcc, IglobView>& args,
    GlobalHPrime                                                 hw) {
  auto g6 = build_new_graph<KeepRedundantLoads, TE>(args, hw);
  return {g6.outputs().scratch_bytes(), g6.outputs().slot_bytes()};
}

// Signature shape matches dummy_stiffness so make_args and every offset policy
// work unchanged. hprimewgll is optional: the benchmark precomputes it once and
// passes it so it stays out of the timed region; tests let it build here.
template <bool KeepRedundantLoads = false, int TE = kExecChunk,
          typename MetricsAcc, typename PropertiesAcc, typename IglobView>
int new_stiffness(
    const DummyKernelArgs<MetricsAcc, PropertiesAcc, IglobView>& args,
    int team_size = -1, GlobalHPrime hprimewgll = GlobalHPrime{}) {
  using namespace TensorOperations;
  using ES = KernelES;

  GlobalHPrime hw = hprimewgll;
  if (hw.extent(0) == 0) hw = make_hprimewgll(args.hprime, args.weights);

  auto g6 = build_new_graph<KeepRedundantLoads, TE>(args, hw);
  return g6.outputs().team_size(team_size).execute(TeamPolicyTag2<ES>{});
}

}  // namespace sfpp_min
