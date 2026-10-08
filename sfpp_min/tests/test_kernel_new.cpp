// ===========================================================================
// test_kernel_new.cpp — the LevelGraph kernel against the same serial oracle,
// scaffolding and bounds shared with test_kernel.cpp.
//
// Every correctness claim test_kernel.cpp makes of the dummy is made here of
// new_stiffness: the oracle match at 1e-4 across all six offset policies, SoA
// and AoS storage, and -- the one that catches a transposed operator -- the
// rigid-body null test. The new axes are the two the dummy does not have:
//
//   * the gather and the scatter. Correctness of both is implied by the oracle
//     match -- a wrong iglob on either end moves the answer -- but the atomic
//     accumulate is the reason acceleration is zeroed before each launch and
//     compared against reference_stiffness, which sums the same contributions.
// ===========================================================================
#include <access_model.hpp>
#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <kernel_new.hpp>
#include <layout.hpp>
#include <mesh.hpp>
#include <reference.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace sfpp_min;

namespace {

// Correctness is TE-invariant: the graph computes the same field at any tile
// width. The ground-truth TE is kExecChunk (4), and at TE=4 the pooled scratch
// is ~36 KB -- fine under CUDA's ~227 KB opt-in cap, but over the host
// backend's 32 KB per-team limit. So the host runs the correctness sweep at
// TE=2 (pooled ~18 KB) and the device at the ground-truth TE=4. The 36 KB
// footprint itself is a GATE C finding, measured on CUDA, not a thing this test
// asserts.
constexpr int kTestTE =
    std::is_same_v<KernelES, Kokkos::DefaultHostExecutionSpace> ? 2
                                                                : kExecChunk;

// Every test runs once per backend the build has: the team policy always, and
// the CuTe policy when the build is configured with TENSOR_OPS_CUTLASS_DIR.
using TeamPolicy = TensorOperations::TeamPolicyTag<KernelES>;
#if defined(TENSOR_OPS_ENABLE_CUTE)
using CutePolicy = TensorOperations::CutePolicyTag<KernelES>;
#endif

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

template <int TE, GatherMode G, ScatterMode Sc, typename Args>
void launch_modes(const Args& args, TeamPolicy policy) {
  new_stiffness<TE, G, Sc>(args, policy);
}
#if defined(TENSOR_OPS_ENABLE_CUTE)
template <int TE, GatherMode G, ScatterMode Sc, typename Args>
void launch_modes(const Args& args, CutePolicy policy) {
  using Mmas = std::conditional_t<Sc == ScatterMode::IndexSlot, RowOfEMmas<TE>,
                                  RowOfIMmas<TE>>;
  new_stiffness<TE, Mmas, G, Sc>(args, policy);
}
#endif

template <typename Policy, int TE = kTestTE,
          GatherMode  G  = GatherMode::Functional,
          ScatterMode Sc = ScatterMode::Functional, typename Off>
ErrorReport run_and_compare(Case<Off>& k) {
  const GllViews q    = make_gll_views();
  auto           args = make_args(k, q);
  launch_modes<TE, G, Sc>(args, Policy{});
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

// The same run over a LayoutRight index map. iglob holds the SAME integers in
// both cases -- only the container's memory order changes -- so this must land
// on the serial oracle exactly as the LayoutLeft run does. A transposed
// subscript would still produce a plausible-looking field, which is why the
// oracle rather than a self-comparison is the judge.
template <typename Policy, GatherMode G = GatherMode::Functional,
          ScatterMode Sc = ScatterMode::Functional, typename Off>
ErrorReport run_and_compare_iglob_right(Case<Off>& k) {
  IglobMapRight gr(k.set.nspec());
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix)
          gr.h_map(ispec, iz, iy, ix) = k.g.h_map(ispec, iz, iy, ix);
  gr.to_device();

  const GllViews q = make_gll_views();
  k.m.to_device();
  k.p.to_device();
  k.f.to_device();
  Kokkos::deep_copy(k.f.acceleration, static_cast<real_t>(0));
  DummyKernelArgs args{make_accessor(k.m), make_accessor(k.p), gr.map,
                       k.f.displacement,   k.f.velocity,       k.f.acceleration,
                       q.hprime,           q.weights,          k.set.nspec()};
  launch_modes<kTestTE, G, Sc>(args, Policy{});
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

template <typename Policy, typename Off>
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
  new_stiffness<kTestTE>(args, Policy{});
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

template <typename Policy, typename Off>
void expect_layout_invariant(const char* name) {
  auto k = make_case<Off>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare<Policy>(k);
  std::printf(
      "[ INFO     ] %s: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      name, e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name;
}

template <typename Policy>
class SfppMinKernelNew : public ::testing::Test {};

struct PolicyName {
  template <typename Policy>
  static std::string GetName(int) {
    return std::is_same_v<Policy, TeamPolicy> ? "Team" : "Cute";
  }
};

#if defined(TENSOR_OPS_ENABLE_CUTE)
using Policies = ::testing::Types<TeamPolicy, CutePolicy>;
#else
using Policies = ::testing::Types<TeamPolicy>;
#endif
TYPED_TEST_SUITE(SfppMinKernelNew, Policies, PolicyName);

}  // namespace

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleStaticExtents) {
  expect_layout_invariant<TypeParam, ChunkTiledOffset>("static extents");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleDynamicExtents) {
  expect_layout_invariant<TypeParam, ChunkTiledDynamicOffset>(
      "dynamic extents");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleLayoutRight) {
  expect_layout_invariant<TypeParam, PlainOffset>("LayoutRight (PlainOffset)");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleLayoutRightDynamic) {
  expect_layout_invariant<TypeParam, LayoutRightDynamicOffset>(
      "LayoutRight dynamic");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleLayoutLeft) {
  expect_layout_invariant<TypeParam, LayoutLeftOffset>("LayoutLeft");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleLayoutLeftDynamic) {
  expect_layout_invariant<TypeParam, LayoutLeftDynamicOffset>(
      "LayoutLeft dynamic");
}

template <typename Policy, typename Off>
void expect_aos_invariant(const char* name) {
  auto k = make_case<Off>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare_aos<Policy>(k);
  std::printf(
      "[ INFO     ] %s: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      name, e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name;
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleAoSStaticExtents) {
  expect_aos_invariant<TypeParam, ChunkTiledOffset>("AoS static extents");
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleAoSDynamicExtents) {
  expect_aos_invariant<TypeParam, ChunkTiledDynamicOffset>(
      "AoS dynamic extents");
}

// A rigid-body translation is in the operator's null space exactly, because
// hprime's row sums vanish. On device the residual is float roundoff, not the
// double floor. A transposed hprime -- or a transposed hprimewgll, which the
// graph builds itself -- puts the residual at the order of the output, ratio
// near 1. This is the one test that catches either transpose.
TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleWithLayoutRightIglob) {
  auto k = make_case<ChunkTiledDynamicOffset>();
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare_iglob_right<TypeParam>(k);
  std::printf(
      "[ INFO     ] LayoutRight iglob: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << "LayoutRight iglob: oracle produced zero field";
  EXPECT_LT(e.relative(), 1e-4) << "LayoutRight iglob";
}

template <typename Policy, GatherMode G, ScatterMode Sc>
void expect_mode_oracle(const char* name) {
  for (int right = 0; right < 2; ++right) {
    auto k = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
    set_linear_field(k);
    set_velocity(k);
    const ErrorReport e = right ? run_and_compare_iglob_right<Policy, G, Sc>(k)
                                : run_and_compare<Policy, kTestTE, G, Sc>(k);
    std::printf("[ INFO     ] %s (iglob %s): relative = %.3e\n", name,
                right ? "LayoutRight" : "LayoutLeft", e.relative());
    EXPECT_GT(e.scale, 0.0) << name << ": oracle produced a zero field";
    EXPECT_LT(e.relative(), 1e-4) << name << " right=" << right;
  }
}

TYPED_TEST(SfppMinKernelNew, GatherFromTheIglobViewMatchesTheOracle) {
  expect_mode_oracle<TypeParam, GatherMode::IndexView, ScatterMode::Functional>(
      "gather iglob view");
}

TYPED_TEST(SfppMinKernelNew, GatherFromTheIndexSlotMatchesTheOracle) {
  expect_mode_oracle<TypeParam, GatherMode::IndexSlot, ScatterMode::Functional>(
      "gather index slot");
}

TYPED_TEST(SfppMinKernelNew, ScatterToTheIglobViewMatchesTheOracle) {
  expect_mode_oracle<TypeParam, GatherMode::Functional, ScatterMode::IndexView>(
      "scatter iglob view");
  expect_mode_oracle<TypeParam, GatherMode::IndexSlot, ScatterMode::IndexView>(
      "gather index slot + scatter iglob view");
}

#if defined(TENSOR_OPS_ENABLE_CUTE)
TYPED_TEST(SfppMinKernelNew, ScatterFromTheIndexSlotMatchesTheOracle) {
  expect_mode_oracle<TypeParam, GatherMode::Functional, ScatterMode::IndexSlot>(
      "scatter index slot");
  expect_mode_oracle<TypeParam, GatherMode::IndexSlot, ScatterMode::IndexSlot>(
      "gather + scatter index slot");
}
#endif

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleWithIxFastestNumbering) {
  auto k = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare<TypeParam>(k);
  std::printf(
      "[ INFO     ] ix-fastest numbering: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << "ix-fastest: oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << "ix-fastest numbering";
}

// The two axes together: the graph's numbering AND a LayoutRight index map.
TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleIxFastestWithLayoutRightIglob) {
  auto k = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare_iglob_right<TypeParam>(k);
  std::printf(
      "[ INFO     ] ix-fastest + LayoutRight iglob: max|diff| = %.3e, "
      "scale = %.3e, relative = %.3e\n",
      e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << "ix-fastest + LR iglob: all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << "ix-fastest + LayoutRight iglob";
}

// TE-INVARIANCE, asserted rather than assumed. The graph's tile width changes
// the slot tile shapes and therefore the whole scratch pooling plan, so "the
// same field at any tile width" is a claim about code that differs, not a
// tautology. The TE sweep picked TE=1 as the winner, and until this test
// existed nothing had checked TE=1 against the oracle at all.
template <typename Policy, int TE>
void expect_te_invariant() {
  auto k = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  set_linear_field(k);
  set_velocity(k);
  const ErrorReport e = run_and_compare<Policy, TE>(k);
  std::printf(
      "[ INFO     ] TE=%d: max|diff| = %.3e, scale = %.3e, relative = %.3e\n",
      TE, e.worst_abs, e.scale, e.relative());
  EXPECT_GT(e.scale, 0.0) << "TE=" << TE
                          << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << "TE=" << TE;
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleAtTileWidthOne) {
  expect_te_invariant<TypeParam, 1>();
}

TYPED_TEST(SfppMinKernelNew, MatchesSerialOracleAtTileWidthTwo) {
  expect_te_invariant<TypeParam, 2>();
}

TYPED_TEST(SfppMinKernelNew, RigidBodyTranslationIsAtTheFloatRoundoffFloor) {
  auto         k    = make_case<ChunkTiledOffset>();
  const double c[3] = {1.3e-3, -0.7e-3, 2.1e-3};
  set_constant_field(k, c);
  set_velocity(k);

  const GllViews q    = make_gll_views();
  auto           args = make_args(k, q);
  new_stiffness<kTestTE>(args, TypeParam{});
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
  const ErrorReport ref          = run_and_compare<TypeParam>(k2);
  const double      output_scale = ref.scale;

  const double ratio = residual / output_scale;
  std::printf(
      "[ INFO     ] rigid body: residual = %.3e, operator output "
      "scale = %.3e, ratio = %.3e\n",
      residual, output_scale, ratio);
  EXPECT_GT(output_scale, 0.0) << "no output to compare against; test is void";
  // Float roundoff in the derivative sums grows with the order (hprime's
  // corner entries are DEG(DEG+1)/4: 5 at NGLL = 5, 14 at NGLL = 8 -- measured
  // ratio 1.7e-5 there). A transposed operator puts the ratio near 1, so a
  // 1e-4 bound at higher orders still separates the two by four decades.
  EXPECT_LT(ratio, NGLL == 5 ? 1e-5 : 1e-4);
}

#if defined(TENSOR_OPS_ENABLE_CUTE)
// The two backends run the same graph, so they must agree far tighter than
// either agrees with the double-precision oracle. Only the atomic scatter's
// accumulation order differs, which puts the floor at float roundoff.
TEST(SfppMinKernelNewCute, MatchesTheTeamBackend) {
  auto kt = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  set_linear_field(kt);
  set_velocity(kt);
  const GllViews qt = make_gll_views();
  new_stiffness<kTestTE>(make_args(kt, qt), TeamPolicy{});
  Kokkos::fence();
  kt.f.to_host();

  auto kc = make_case<ChunkTiledDynamicOffset>(/*ix_fastest=*/true);
  set_linear_field(kc);
  set_velocity(kc);
  const GllViews qc = make_gll_views();
  new_stiffness<kTestTE>(make_args(kc, qc), CutePolicy{});
  Kokkos::fence();
  kc.f.to_host();

  double worst = 0.0, scale = 0.0;
  for (int ig = 0; ig < kt.nglob; ++ig)
    for (int c = 0; c < 3; ++c) {
      const double t = kt.f.h_acceleration(ig, c);
      scale          = std::max(scale, std::abs(t));
      worst = std::max(worst, std::abs(t - kc.f.h_acceleration(ig, c)));
    }
  const double relative = scale > 0.0 ? worst / scale : worst;
  std::printf(
      "[ INFO     ] cute vs team: max|diff| = %.3e, scale = %.3e, "
      "relative = %.3e\n",
      worst, scale, relative);
  EXPECT_GT(scale, 0.0) << "void test: the team backend produced a zero field";
  EXPECT_LT(relative, 1e-5);
}

// Each TE is a different RowOfI thread map (NGLL*NGLL*TE threads, element
// slowest), so every contraction and register-driven combine partitions
// differently. The field must not move.
template <int         TE, typename Off, typename Mmas = RowOfIMmas<TE>,
          GatherMode  G  = GatherMode::Functional,
          ScatterMode Sc = ScatterMode::Functional>
void expect_cute_oracle(const char* name) {
  auto k = make_case<Off>(/*ix_fastest=*/true);
  set_linear_field(k);
  set_velocity(k);
  const GllViews q = make_gll_views();
  new_stiffness<TE, Mmas, G, Sc>(make_args(k, q), CutePolicy{});
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
  std::printf("[ INFO     ] %s TE=%d: relative = %.3e\n", name, TE,
              e.relative());
  EXPECT_GT(e.scale, 0.0) << name << ": oracle produced an all-zero field";
  EXPECT_LT(e.relative(), 1e-4) << name << " TE=" << TE;
}

TEST(SfppMinKernelNewCute, MatchesTheOracleAtEveryTE) {
  expect_cute_oracle<1, LayoutRightDynamicOffset>("LR");
  expect_cute_oracle<2, LayoutRightDynamicOffset>("LR");
  expect_cute_oracle<4, LayoutRightDynamicOffset>("LR");
  if constexpr (NGLL == 5)
    expect_cute_oracle<8, LayoutRightDynamicOffset>("LR");
  expect_cute_oracle<4, ChunkTiledDynamicOffset>("chunk");
  expect_cute_oracle<4, LayoutLeftDynamicOffset>("LL");
}

TEST(SfppMinKernelNewCute, RowOfEMatchesTheOracleAtEveryTE) {
  expect_cute_oracle<1, LayoutRightDynamicOffset, RowOfEMmas<1>>("row-of-e LR");
  expect_cute_oracle<2, LayoutRightDynamicOffset, RowOfEMmas<2>>("row-of-e LR");
  expect_cute_oracle<3, LayoutRightDynamicOffset, RowOfEMmas<3>>("row-of-e LR");
  expect_cute_oracle<4, LayoutRightDynamicOffset, RowOfEMmas<4>>("row-of-e LR");
  if constexpr (NGLL == 5)
    expect_cute_oracle<8, LayoutRightDynamicOffset, RowOfEMmas<8>>(
        "row-of-e LR");
  expect_cute_oracle<4, ChunkTiledDynamicOffset, RowOfEMmas<4>>(
      "row-of-e chunk");
  expect_cute_oracle<4, LayoutLeftDynamicOffset, RowOfEMmas<4>>("row-of-e LL");
}

TEST(SfppMinKernelNewCute, RowOfEGatherModesMatchTheOracle) {
  expect_cute_oracle<1, LayoutRightDynamicOffset, RowOfEMmas<1>,
                     GatherMode::IndexView>("row-of-e iglob view");
  expect_cute_oracle<4, LayoutRightDynamicOffset, RowOfEMmas<4>,
                     GatherMode::IndexView>("row-of-e iglob view");
  expect_cute_oracle<1, LayoutRightDynamicOffset, RowOfEMmas<1>,
                     GatherMode::IndexSlot>("row-of-e index slot");
  expect_cute_oracle<4, LayoutRightDynamicOffset, RowOfEMmas<4>,
                     GatherMode::IndexSlot>("row-of-e index slot");
  expect_cute_oracle<4, LayoutLeftDynamicOffset, RowOfEMmas<4>,
                     GatherMode::IndexSlot>("row-of-e index slot LL");
}

TEST(SfppMinKernelNewCute, RowOfEScatterFromTheIndexSlotMatchesTheOracle) {
  expect_cute_oracle<1, LayoutRightDynamicOffset, RowOfEMmas<1>,
                     GatherMode::Functional, ScatterMode::IndexSlot>(
      "row-of-e scatter index slot");
  expect_cute_oracle<4, LayoutRightDynamicOffset, RowOfEMmas<4>,
                     GatherMode::IndexSlot, ScatterMode::IndexSlot>(
      "row-of-e gather + scatter index slot");
  expect_cute_oracle<4, LayoutLeftDynamicOffset, RowOfEMmas<4>,
                     GatherMode::IndexSlot, ScatterMode::IndexSlot>(
      "row-of-e gather + scatter index slot LL");
  expect_cute_oracle<2, LayoutRightDynamicOffset, RowOfEMmas<2>,
                     GatherMode::IndexSlot, ScatterMode::IndexSlot>(
      "row-of-e gather + scatter index slot TE2");
}

template <typename Plan, std::size_t... Ks>
constexpr bool all_in_register(std::index_sequence<Ks...>) {
  return (Plan::template in_register<Ks>() && ...);
}

TEST(SfppMinKernelNewCute, RowOfECombinesReadEveryContractionInRegisters) {
  auto           k    = make_case<LayoutRightDynamicOffset>();
  const GllViews q    = make_gll_views();
  const auto     args = make_args(k, q);
  const auto     hw   = make_hprimewgll(args.hprime, args.weights);
  using G = decltype(build_new_graph<kTestTE, RowOfEMmas<kTestTE>>(args, hw));
  using Levels    = typename G::levels_type;
  constexpr int N = TensorOperations::Impl::lg_cute_num_threads_v<Levels>;
  using Integrand =
      TensorOperations::Impl::lg_cute_combine_plan<Levels, 3, 0, N>;
  using ToAccel = TensorOperations::Impl::lg_cute_combine_plan<Levels, 5, 0, N>;
  static_assert(N == NGLL * NGLL * NGLL);
  static_assert(Integrand::aligned_driven &&
                all_in_register<Integrand>(std::make_index_sequence<9>{}));
  static_assert(ToAccel::aligned_driven &&
                all_in_register<ToAccel>(std::make_index_sequence<9>{}));
  const NewFootprint row_e =
      new_footprint<kTestTE, RowOfEMmas<kTestTE>>(args, CutePolicy{}, hw);
  const NewFootprint row_i = new_footprint<kTestTE>(args, CutePolicy{}, hw);
  std::printf("[ INFO     ] smem pooled: row-of-e %zu B, row-of-i %zu B\n",
              row_e.pooled, row_i.pooled);
  EXPECT_LT(row_e.pooled, row_i.pooled);
  EXPECT_EQ(row_e.threads, NGLL * NGLL * NGLL);
}

// The CuTe launch fits the default dynamic shared memory without an opt-in,
// pooling buys something, and the RowOfI MMA gives one thread per row.
TEST(SfppMinKernelNewCute, FootprintIsPooledAndFitsTheDefaultSharedMemory) {
  auto               k    = make_case<ChunkTiledDynamicOffset>();
  const GllViews     q    = make_gll_views();
  const auto         args = make_args(k, q);
  const auto         hw   = make_hprimewgll(args.hprime, args.weights);
  const NewFootprint fp   = new_footprint<kTestTE>(args, CutePolicy{}, hw);
  std::printf(
      "[ INFO     ] cute TE=%d: pooled %zu B, unpooled %zu B, block %d\n",
      kTestTE, fp.pooled, fp.unpooled, fp.threads);
  EXPECT_GT(fp.pooled, 0u);
  EXPECT_LE(fp.pooled, fp.unpooled);
  if constexpr (NGLL == 5) EXPECT_LE(fp.pooled, std::size_t{48 * 1024});
  EXPECT_EQ(fp.threads, NGLL * NGLL * kTestTE);
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  const int rc = RUN_ALL_TESTS();
  Kokkos::finalize();
  return rc;
}
