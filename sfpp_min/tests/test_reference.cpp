#include <config.hpp>
#include <data.hpp>
#include <geometry.hpp>
#include <gll.hpp>
#include <mesh.hpp>
#include <reference.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace sfpp_min;

namespace {

using Off = ChunkTiledOffset;

constexpr MeshDims kD{4, 3, 2};

constexpr double kB[3][3] = {{3.0e-4, -1.7e-4, 0.9e-4},
                             {2.1e-4, 1.3e-4, -2.6e-4},
                             {-0.8e-4, 2.9e-4, 1.1e-4}};

struct Case {
  ElementSet       set;
  IglobMap         g;
  int              nglob = 0;
  Metrics<Off>     m;
  Properties<Off>  p;
  Coordinates<Off> c;
  Fields           f;
};

Case make_case(bool affine, bool uniform_material) {
  Case k;
  k.set   = all_elements(kD);
  k.g     = IglobMap(k.set.nspec());
  k.nglob = renumber_access_order(kD, k.set, k.g);
  k.m     = Metrics<Off>(k.set.nspec());
  k.p     = Properties<Off>(k.set.nspec());
  k.c     = Coordinates<Off>(k.set.nspec());
  k.f     = Fields(k.nglob);
  if (affine) {
    const Geometry geo = make_geometry(kD);
    fill_affine(kD, k.set, geo, k.m, k.c);
  } else {
    fill_curved(kD, k.set, k.m);
  }
  if (uniform_material)
    fill_properties_uniform(k.set, k.p, 2300.0, 2800.0, 1500.0);
  else
    fill_properties(kD, k.set, k.p);
  return k;
}

void set_constant_field(Case& k, const double c[3]) {
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int comp = 0; comp < 3; ++comp)
      k.f.h_displacement(ig, comp) = static_cast<real_t>(c[comp]);
}

void set_linear_field(Case& k) {
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

std::vector<char> interior_node_flags(const Case& k) {
  std::vector<char> on_boundary(k.nglob, 0);
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec) {
    int ex, ey, ez;
    grid_element_coords(kD, k.set.to_grid[ispec], ex, ey, ez);
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const int  gx = DEG * ex + ix, gy = DEG * ey + iy, gz = DEG * ez + iz;
          const bool edge = gx == 0 || gx == kD.gnx() - 1 || gy == 0 ||
                            gy == kD.gny() - 1 || gz == 0 || gz == kD.gnz() - 1;
          if (edge) on_boundary[k.g.h_map(ispec, iz, iy, ix)] = 1;
        }
  }
  return on_boundary;
}

}  // namespace

TEST(SfppMinReference, RigidBodyTranslationGivesZeroForce) {
  Case         k    = make_case(false, false);
  const double c[3] = {0.37, -0.81, 0.55};
  set_constant_field(k, c);
  const auto out =
      reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob, true);

  double max_du = 0.0;
  for (double v : out.du) max_du = std::max(max_du, std::abs(v));
  double max_a = 0.0;
  for (double v : out.accel) max_a = std::max(max_a, std::abs(v));

  double scale = 0.0;
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const double l2m = k.p.kappa.host(ispec, iz, iy, ix) +
                             (4.0 / 3.0) * k.p.mu.host(ispec, iz, iy, ix);
          scale            = std::max(
              scale, l2m * std::abs(k.m.jacobian.host(ispec, iz, iy, ix)));
        }
  scale *= 0.81;

  std::printf(
      "[ INFO     ] rigid body: max|du/dx| = %.3e   max|accel| = %.3e "
      "  scale = %.3e   ratio = %.2e\n",
      max_du, max_a, scale, max_a / scale);
  EXPECT_LT(max_du, 1e-14);
  EXPECT_LT(max_a, 1e-13 * scale);
}

TEST(SfppMinReference, ConstantStrainReproducesTheImposedGradient) {
  Case k = make_case(true, false);
  set_linear_field(k);
  const auto out =
      reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob, true);

  double worst = 0.0;
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int pt = 0; pt < kPointsPerElement; ++pt)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
          worst =
              std::max(worst, std::abs(out.du_at(ispec, pt, c, d) - kB[c][d]));
  std::printf("[ INFO     ] constant strain: max|du/dx - B| = %.3e\n", worst);
  EXPECT_LT(worst, 1e-8);
}

TEST(SfppMinReference, ConstantStrainStressMatchesHookeComputedIndependently) {
  Case k = make_case(true, false);
  set_linear_field(k);
  const auto out =
      reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob, true);

  double worst_rel = 0.0;
  for (int ispec = 0; ispec < k.set.nspec(); ++ispec)
    for (int iz = 0; iz < NGLL; ++iz)
      for (int iy = 0; iy < NGLL; ++iy)
        for (int ix = 0; ix < NGLL; ++ix) {
          const int    pt  = (iz * NGLL + iy) * NGLL + ix;
          const double kap = k.p.kappa.host(ispec, iz, iy, ix);
          const double mu  = k.p.mu.host(ispec, iz, iy, ix);
          const double l2m = kap + (4.0 / 3.0) * mu;
          const double lam = l2m - 2.0 * mu;
          const double tr  = kB[0][0] + kB[1][1] + kB[2][2];
          double       E[3][3];
          for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d)
              E[c][d] = (c == d) ? (lam * tr + 2.0 * mu * kB[c][c])
                                 : mu * (kB[c][d] + kB[d][c]);
          for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d) {
              const double got = out.stress_at(ispec, pt, c, d);
              worst_rel = std::max(worst_rel, std::abs(got - E[c][d]) /
                                                  std::abs(l2m * tr + 1.0));
            }
        }
  std::printf(
      "[ INFO     ] constant strain: max relative stress error = %.3e\n",
      worst_rel);
  EXPECT_LT(worst_rel, 1e-4)
      << "floor is float storage of u: |u|*eps*max|hprime|*sqrt(NGLL)*|M| "
         "~ 1.4e-8 absolute on du, ~5e-5 relative; a wrong kappa->lambda "
         "constant shifts l2m by 19%, so 1e-4 keeps full power";
}

TEST(SfppMinReference, ConstantStrainUniformMaterialGivesZeroInteriorForce) {
  Case k = make_case(true, true);
  set_linear_field(k);
  const auto out   = reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob);
  const auto onbnd = interior_node_flags(k);

  double max_int = 0.0, max_bnd = 0.0;
  int    n_int = 0;
  for (int ig = 0; ig < k.nglob; ++ig)
    for (int c = 0; c < 3; ++c) {
      const double v = std::abs(out.a(c, ig));
      if (onbnd[ig]) {
        max_bnd = std::max(max_bnd, v);
      } else {
        max_int = std::max(max_int, v);
        if (c == 0) ++n_int;
      }
    }
  std::printf(
      "[ INFO     ] uniform material: max|f| interior = %.3e over %d "
      "nodes, boundary = %.3e\n",
      max_int, n_int, max_bnd);
  EXPECT_GT(n_int, 100);
  EXPECT_GT(max_bnd, 1.0);
  EXPECT_LT(max_int, 1e-4 * max_bnd)
      << "cancellation is limited by the same float-storage floor on T; "
         "a 1% weight error would leave ~1e4 here, not ~7";
}

TEST(SfppMinReference, ResultIsIndependentOfStorageLayout) {
  Case k = make_case(true, false);
  set_linear_field(k);
  const auto a = reference_stiffness(k.set, k.m, k.p, k.g, k.f, k.nglob);

  Metrics<PlainOffset>     m2(k.set.nspec());
  Properties<PlainOffset>  p2(k.set.nspec());
  Coordinates<PlainOffset> c2(k.set.nspec());
  const Geometry           geo = make_geometry(kD);
  fill_affine(kD, k.set, geo, m2, c2);
  fill_properties(kD, k.set, p2);
  const auto b = reference_stiffness(k.set, m2, p2, k.g, k.f, k.nglob);

  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < a.accel.size(); ++i) {
    worst = std::max(worst, std::abs(a.accel[i] - b.accel[i]));
    scale = std::max(scale, std::abs(a.accel[i]));
  }
  std::printf("[ INFO     ] layout invariance: max|diff| = %.3e (scale %.3e)\n",
              worst, scale);
  EXPECT_LT(worst, 1e-12 * scale);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
