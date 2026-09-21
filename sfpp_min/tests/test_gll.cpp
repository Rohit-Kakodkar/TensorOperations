#include <gll.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace sfpp_min;

namespace {

double ipow(double x, int k) {
  double r = 1.0;
  for (int i = 0; i < k; ++i) r *= x;
  return r;
}

double device_worst_row_sum() {
  Kokkos::View<double*> worst("worst", 1);
  Kokkos::parallel_for(
      "gll_device_rowsums", Kokkos::RangePolicy<>(0, NGLL),
      KOKKOS_LAMBDA(const int point) {
        double s = 0.0;
        for (int poly = 0; poly < NGLL; ++poly) s += gll::hprime(point, poly);
        Kokkos::atomic_max(&worst(0), Kokkos::abs(s));
      });
  Kokkos::fence();
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), worst);
  return h(0);
}

double simple_hprime(int point, int poly) {
  if (point == 0 && poly == 0) return -DEG * (DEG + 1.0) * 0.25;
  if (point == DEG && poly == DEG) return DEG * (DEG + 1.0) * 0.25;
  if (point == poly) return 0.0;
  const double xp = gll::node(point);
  const double xq = gll::node(poly);
  return gll::pnleg(xp) / (gll::pnleg(xq) * (xp - xq));
}

}  // namespace

TEST(SfppMinGll, WeightsSumToTwo) {
  double s = 0.0;
  for (int i = 0; i < NGLL; ++i) s += gll::weight(i);
  EXPECT_NEAR(s, 2.0, 1e-14);
}

TEST(SfppMinGll, NodesAreSymmetricWithExactEndpoints) {
  EXPECT_DOUBLE_EQ(gll::node(0), -1.0);
  EXPECT_DOUBLE_EQ(gll::node(DEG), 1.0);
  EXPECT_DOUBLE_EQ(gll::node(2), 0.0);
  for (int i = 0; i < NGLL; ++i)
    EXPECT_NEAR(gll::node(i), -gll::node(DEG - i), 1e-15) << "i=" << i;
  EXPECT_NEAR(gll::node(3), std::sqrt(3.0 / 7.0), 1e-15);
}

TEST(SfppMinGll, RowSumsVanish) {
  for (int point = 0; point < NGLL; ++point) {
    double s = 0.0;
    for (int poly = 0; poly < NGLL; ++poly) s += gll::hprime(point, poly);
    EXPECT_NEAR(s, 0.0, 1e-13) << "point=" << point;
  }
}

TEST(SfppMinGll, ColumnSumsDoNotVanish) {
  double worst = 0.0;
  for (int poly = 0; poly < NGLL; ++poly) {
    double s = 0.0;
    for (int point = 0; point < NGLL; ++point) s += gll::hprime(point, poly);
    worst = std::max(worst, std::abs(s));
  }
  EXPECT_GT(worst, 0.1) << "the rigid-body transpose check depends on this "
                           "asymmetry; a symmetric hprime would disarm it";
  EXPECT_NEAR(worst, 5.625, 1e-10);
}

TEST(SfppMinGll, DifferentiationIsExactThroughDegreeFour) {
  for (int k = 0; k <= DEG; ++k) {
    for (int point = 0; point < NGLL; ++point) {
      double d = 0.0;
      for (int poly = 0; poly < NGLL; ++poly)
        d += gll::hprime(point, poly) * ipow(gll::node(poly), k);
      const double exact = (k == 0) ? 0.0 : k * ipow(gll::node(point), k - 1);
      EXPECT_NEAR(d, exact, 1e-12) << "k=" << k << " point=" << point;
    }
  }
}

TEST(SfppMinGll, QuadratureIsExactThroughDegreeSevenAndNotEight) {
  for (int k = 0; k <= 7; ++k) {
    double q = 0.0;
    for (int i = 0; i < NGLL; ++i) q += gll::weight(i) * ipow(gll::node(i), k);
    const double exact = (k % 2) ? 0.0 : 2.0 / (k + 1);
    EXPECT_NEAR(q, exact, 1e-13) << "k=" << k;
  }
  double q8 = 0.0;
  for (int i = 0; i < NGLL; ++i) q8 += gll::weight(i) * ipow(gll::node(i), 8);
  EXPECT_GT(std::abs(q8 - 2.0 / 9.0), 1e-3);
}

TEST(SfppMinGll, FullFormulaMatchesSimpleFormulaAtGllNodes) {
  for (int point = 0; point < NGLL; ++point)
    for (int poly = 0; poly < NGLL; ++poly)
      EXPECT_NEAR(gll::hprime(point, poly), simple_hprime(point, poly), 1e-13)
          << "point=" << point << " poly=" << poly;
}

TEST(SfppMinGll, TablesAreReachableFromDeviceCode) {
  EXPECT_LT(device_worst_row_sum(), 1e-13);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
