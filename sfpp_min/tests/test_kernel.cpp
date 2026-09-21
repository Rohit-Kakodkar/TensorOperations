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

#include <cmath>
#include <cstdio>
#include <vector>

using namespace sfpp_min;

namespace {

constexpr MeshDims kD{4, 3, 2};

constexpr double kB[3][3] = {{3.0e-4, -1.7e-4, 0.9e-4},
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

template <typename Off>
Case<Off> make_case() {
  Case<Off> k;
  k.set              = all_elements(kD);
  k.g                = IglobMap(k.set.nspec());
  k.nglob            = renumber_access_order(kD, k.set, k.g);
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

struct GllViews {
  GlobalHPrime  hprime;
  GlobalWeights weights;
};

GllViews make_gll_views() {
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

template <typename Off>
auto make_args(Case<Off>& k, const GllViews& q) {
  k.m.to_device();
  k.p.to_device();
  k.g.to_device();
  k.f.to_device();
  Kokkos::deep_copy(k.f.acceleration, static_cast<real_t>(0));
  return DummyKernelArgs{
      make_accessor(k.m), make_accessor(k.p), k.g.map,
      k.f.displacement,   k.f.velocity,       k.f.acceleration,
      q.hprime,           q.weights,          k.set.nspec()};
}

struct ErrorReport {
  double worst_abs = 0.0;
  double scale     = 0.0;
  double relative() const {
    return scale > 0.0 ? worst_abs / scale : worst_abs;
  }
};

template <typename ScratchLayout = Kokkos::LayoutLeft, typename Off>
ErrorReport run_and_compare(Case<Off>& k) {
  const GllViews q    = make_gll_views();
  auto           args = make_args(k, q);
  dummy_stiffness<ScratchLayout>(args);
  Kokkos::fence();
  k.f.to_host();

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

template <typename Off>
ErrorReport run_and_compare_aos(Case<Off>& k) {
  MetricsAoS<Off>    m(k.set.nspec());
  PropertiesAoS<Off> p(k.set.nspec());
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

  const GllViews q = make_gll_views();
  k.g.to_device();
  k.f.to_device();
  Kokkos::deep_copy(k.f.acceleration, static_cast<real_t>(0));
  DummyKernelArgs args{make_accessor(m), make_accessor(p), k.g.map,
                       k.f.displacement, k.f.velocity,     k.f.acceleration,
                       q.hprime,         q.weights,        k.set.nspec()};
  dummy_stiffness(args);
  Kokkos::fence();
  k.f.to_host();

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

}  // namespace

TEST(SfppMinKernel, MatchesSerialOracleStaticExtents) {
  auto k = make_case<ChunkTiledOffset>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare(k);
  std::printf(
      "[ INFO     ] static extents: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << "oracle produced an all-zero field; test is void";
  EXPECT_LT(e.relative(), 1e-4);
}

TEST(SfppMinKernel, MatchesSerialOracleDynamicExtents) {
  auto k = make_case<ChunkTiledDynamicOffset>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare(k);
  std::printf(
      "[ INFO     ] dynamic extents: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0);
  EXPECT_LT(e.relative(), 1e-4);
}

template <typename Off>
void expect_layout_invariant(const char* name) {
  auto k = make_case<Off>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare(k);
  std::printf(
      "[ INFO     ] %s: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      name, e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name;
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutRight) {
  expect_layout_invariant<PlainOffset>("LayoutRight (PlainOffset)");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutRightDynamic) {
  expect_layout_invariant<LayoutRightDynamicOffset>("LayoutRight dynamic");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutLeft) {
  expect_layout_invariant<LayoutLeftOffset>("LayoutLeft");
}

TEST(SfppMinKernel, MatchesSerialOracleLayoutLeftDynamic) {
  expect_layout_invariant<LayoutLeftDynamicOffset>("LayoutLeft dynamic");
}

// The AoS container interleaves the ten metric and three property components
// into one buffer; the numeric result must be storage-invariant.
template <typename Off>
void expect_aos_invariant(const char* name) {
  auto k = make_case<Off>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare_aos(k);
  std::printf(
      "[ INFO     ] %s: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      name, e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name;
}

TEST(SfppMinKernel, MatchesSerialOracleAoSStaticExtents) {
  expect_aos_invariant<ChunkTiledOffset>("AoS static extents");
}

TEST(SfppMinKernel, MatchesSerialOracleAoSDynamicExtents) {
  expect_aos_invariant<ChunkTiledDynamicOffset>("AoS dynamic extents");
}

// Scratch LayoutRight is a reshuffle of team-scratch storage; the numeric
// result must be scratch-layout-invariant.
TEST(SfppMinKernel, MatchesSerialOracleScratchLayoutRight) {
  auto k = make_case<ChunkTiledDynamicOffset>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare<Kokkos::LayoutRight>(k);
  std::printf(
      "[ INFO     ] scratch LayoutRight: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0);
  EXPECT_LT(e.relative(), 1e-4);
}

// A rigid-body translation is in the operator's null space exactly, for any
// metric field, because hprime's row sums vanish. On device the residual is not
// zero but float roundoff, so the bound cannot be the double-precision floor
// Sprint 1 used for the oracle. It is stated instead as a RATIO against the
// magnitude the same operator produces on a non-trivial field of comparable
// amplitude: the null-space residual must be negligible compared to normal
// output. A transposed hprime -- the break this test exists to catch -- puts
// the residual at the same order as the output, i.e. a ratio near 1.
TEST(SfppMinKernel, RigidBodyTranslationIsAtTheFloatRoundoffFloor) {
  auto         k    = make_case<ChunkTiledOffset>();
  const double c[3] = {1.3e-3, -0.7e-3, 2.1e-3};
  set_constant_field(k, c);
  set_velocity(k);

  const GllViews q    = make_gll_views();
  auto           args = make_args(k, q);
  dummy_stiffness(args);
  Kokkos::fence();
  k.f.to_host();

  double residual = 0.0;
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      residual =
          std::max(residual,
                   std::abs(static_cast<double>(k.f.h_acceleration(ig, comp))));

  auto k2 = make_case<ChunkTiledOffset>();
  set_linear_field(k2);
  set_velocity(k2);
  const ErrorReport ref          = run_and_compare(k2);
  const double      output_scale = ref.scale;

  const double ratio = residual / output_scale;
  std::printf(
      "[ INFO     ] rigid body: residual = %.3e, operator output "
      "scale = %.3e, ratio = %.3e\n",
      residual, output_scale, ratio);
  EXPECT_GT(output_scale, 0.0) << "no output to compare against; test is void";
  EXPECT_LT(ratio, 1e-5);
}

TEST(SfppMinKernel, ResolvedTeamSizeAndScratchMatchTheGroundTruth) {
  std::printf("[ INFO     ] dummy_shmem_size() = %zu B\n", dummy_shmem_size());
  EXPECT_EQ(dummy_shmem_size(), 24124u)
      << "scratch request must be 6008 + 18008 + 108; the launch adds "
         "8*(team_size+2) on top to reach the measured 26188 B";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
