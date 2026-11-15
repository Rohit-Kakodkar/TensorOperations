// ===========================================================================
// kernel_case.hpp -- scaffolding shared by test_kernel.cpp (the dummy) and
// test_kernel_new.cpp (the level graph): one mesh, one field, one serial
// oracle and one 1e-4 bound, so both kernels are judged identically.
// ===========================================================================
#pragma once

#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <kernel_dummy.hpp>
#include <layout.hpp>
#include <mesh.hpp>
#include <reference.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace sfpp_min::test {

inline constexpr MeshDims kD{4, 3, 2};

inline constexpr double kB[3][3] = {{3.0e-4, -1.7e-4, 0.9e-4},
                                    {2.1e-4, 1.3e-4, -2.6e-4},
                                    {-0.8e-4, 2.9e-4, 1.1e-4}};

template <typename Off>
struct Case {
  ElementSet       set;
  IglobMap         g;
  int              nglob = 0;
  Metrics<Off>     m;
  Properties<Off>  p;
  Coordinates<Off> c;
  Fields           f;
};

// ix_fastest selects the level graph's own mesh numbering (renumber_ix_fastest)
// instead of the dummy's. It changes iglob's VALUES, so the oracle -- which
// reads the same map -- must still agree exactly.
template <typename Off>
Case<Off> make_case(bool ix_fastest = false) {
  Case<Off> k;
  k.set              = all_elements(kD);
  k.g                = IglobMap(k.set.nspec());
  k.nglob            = ix_fastest ? renumber_ix_fastest(kD, k.set, k.g)
                                  : renumber_access_order(kD, k.set, k.g);
  k.m                = Metrics<Off>(k.set.nspec());
  k.p                = Properties<Off>(k.set.nspec());
  k.c                = Coordinates<Off>(k.set.nspec());
  k.f                = Fields(k.nglob);
  const Geometry geo = make_geometry(kD);
  fill_affine(kD, k.set, geo, k.m, k.c);
  fill_properties(kD, k.set, k.p);
  return k;
}

template <typename Off>
void set_linear_field(Case<Off>& k) {
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const int    ig   = k.g.h_map(ispec, iz, iy, ix);
          const double x[3] = {k.c.x.host(ispec, iz, iy, ix),
                               k.c.y.host(ispec, iz, iy, ix),
                               k.c.z.host(ispec, iz, iy, ix)};
          for (int comp = 0; comp < 3; ++comp) {
            double v = 0.0;
            for (int j = 0; j < 3; ++j) v += kB[comp][j] * x[j];
            k.f.h_displacement(ig, comp) = static_cast<real_t>(v);
          }
        }
}

template <typename Off>
void set_constant_field(Case<Off>& k, const double c[3]) {
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      k.f.h_displacement(ig, comp) = static_cast<real_t>(c[comp]);
}

// The divergence callback loads velocity to feed a no-op. Fill it with
// non-trivial, non-symmetric data so a surviving load reads real values.
template <typename Off>
void set_velocity(Case<Off>& k) {
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      k.f.h_velocity(ig, comp) =
          static_cast<real_t>(0.37 * std::sin(0.011 * ig + 1.7 * comp) + 0.13);
}

template <typename Off>
Case<Off> make_linear_case(bool ix_fastest = false) {
  Case<Off> k = make_case<Off>(ix_fastest);
  set_linear_field(k);
  set_velocity(k);
  return k;
}

struct GllViews {
  GlobalHPrime  hprime;
  GlobalWeights weights;
};

inline GllViews make_gll_views() {
  GllViews v{GlobalHPrime("hprime", NGLL, NGLL),
             GlobalWeights("weights", NGLL)};
  auto     hh = Kokkos::create_mirror_view(v.hprime);
  auto     hw = Kokkos::create_mirror_view(v.weights);
  for (int a = 0; a < NGLL; ++a) {
    hw(a) = static_cast<real_t>(gll::weight(a));
    for (int b = 0; b < NGLL; ++b)
      hh(a, b) = static_cast<real_t>(gll::hprime(a, b));
  }
  Kokkos::deep_copy(v.hprime, hh);
  Kokkos::deep_copy(v.weights, hw);
  return v;
}

// Uploads the fields and zeroes acceleration, which every kernel accumulates
// into. The caller uploads the metrics, properties and map it passes.
template <typename Off, typename M, typename P, typename Map>
auto make_args(Case<Off>& k, const GllViews& q, const M& m, const P& p,
               const Map& iglob) {
  k.f.to_device();
  Kokkos::deep_copy(k.f.acceleration, static_cast<real_t>(0));
  return DummyKernelArgs{make_accessor(m), make_accessor(p), iglob,
                         k.f.displacement, k.f.velocity,     k.f.acceleration,
                         q.hprime,         q.weights,        k.set.nspec()};
}

template <typename Off>
auto make_args(Case<Off>& k, const GllViews& q) {
  k.m.to_device();
  k.p.to_device();
  k.g.to_device();
  return make_args(k, q, k.m, k.p, k.g.map);
}

// The AoS containers interleave the ten metric and three property components
// into one buffer each; the numeric result must be storage-invariant.
template <typename Off>
void fill_aos(Case<Off>& k, MetricsAoS<Off>& m, PropertiesAoS<Off>& p) {
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          m.xix.host(ispec, iz, iy, ix)    = k.m.xix.host(ispec, iz, iy, ix);
          m.xiy.host(ispec, iz, iy, ix)    = k.m.xiy.host(ispec, iz, iy, ix);
          m.xiz.host(ispec, iz, iy, ix)    = k.m.xiz.host(ispec, iz, iy, ix);
          m.etax.host(ispec, iz, iy, ix)   = k.m.etax.host(ispec, iz, iy, ix);
          m.etay.host(ispec, iz, iy, ix)   = k.m.etay.host(ispec, iz, iy, ix);
          m.etaz.host(ispec, iz, iy, ix)   = k.m.etaz.host(ispec, iz, iy, ix);
          m.gammax.host(ispec, iz, iy, ix) = k.m.gammax.host(ispec, iz, iy, ix);
          m.gammay.host(ispec, iz, iy, ix) = k.m.gammay.host(ispec, iz, iy, ix);
          m.gammaz.host(ispec, iz, iy, ix) = k.m.gammaz.host(ispec, iz, iy, ix);
          m.jacobian.host(ispec, iz, iy, ix) =
              k.m.jacobian.host(ispec, iz, iy, ix);
          p.kappa.host(ispec, iz, iy, ix) = k.p.kappa.host(ispec, iz, iy, ix);
          p.mu.host(ispec, iz, iy, ix)    = k.p.mu.host(ispec, iz, iy, ix);
          p.rho.host(ispec, iz, iy, ix)   = k.p.rho.host(ispec, iz, iy, ix);
        }
  m.to_device();
  p.to_device();
}

struct ErrorReport {
  double worst_abs = 0.0;
  double scale     = 0.0;
  double relative() const {
    return scale > 0.0 ? worst_abs / scale : worst_abs;
  }
};

template <typename Off>
void fetch(Case<Off>& k) {
  Kokkos::fence();
  k.f.to_host();
}

// Fences, copies acceleration back and compares it with the double-precision
// serial oracle, which reads the same iglob map and sums the same
// contributions.
template <typename Off>
ErrorReport compare_to_oracle(Case<Off>& k) {
  fetch(k);
  const ReferenceOutput want =
      reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob);
  ErrorReport e;
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int c = 0; c < 3; ++c) {
      const double got = k.f.h_acceleration(ig, c);
      const double ref = want.a(c, ig);
      e.worst_abs      = std::max(e.worst_abs, std::abs(got - ref));
      e.scale          = std::max(e.scale, std::abs(ref));
    }
  return e;
}

inline void expect_matches_oracle(const ErrorReport& e,
                                  const std::string& name) {
  std::printf(
      "[ INFO     ] %s: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      name.c_str(), e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name;
}

// A rigid-body translation is in the operator's null space exactly, for any
// metric field, because hprime's row sums vanish. On device the residual is not
// zero but float roundoff, so it is stated as a RATIO against the magnitude the
// same launch produces on the linear field: the null-space residual must be
// negligible next to normal output, while a transposed hprime puts it at the
// order of the output, i.e. a ratio near 1. Returns that ratio.
template <typename Launch>
double rigid_body_ratio(Launch launch) {
  auto         k    = make_case<ChunkTiledOffset>();
  const double c[3] = {1.3e-3, -0.7e-3, 2.1e-3};
  set_constant_field(k, c);
  set_velocity(k);
  launch(make_args(k, make_gll_views()));
  fetch(k);
  double residual = 0.0;
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      residual =
          std::max(residual,
                   std::abs(static_cast<double>(k.f.h_acceleration(ig, comp))));

  auto k2 = make_linear_case<ChunkTiledOffset>();
  launch(make_args(k2, make_gll_views()));
  const double scale = compare_to_oracle(k2).scale;
  std::printf(
      "[ INFO     ] rigid body: residual = %.3e, operator output "
      "scale = %.3e, ratio = %.3e\n",
      residual, scale, residual / scale);
  EXPECT_GT(scale, 0.0) << "no output to compare against; test is void";
  return residual / scale;
}

}  // namespace sfpp_min::test
