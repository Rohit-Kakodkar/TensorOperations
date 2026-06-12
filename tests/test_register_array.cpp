#include <Kokkos_Core.hpp>
#include <TensorOperations/RegisterArray.hpp>
#include <gtest/gtest.h>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Metadata: rank / size / extent for 1D, 2D, 3D tiles.
// ---------------------------------------------------------------------------
TEST(RegisterArrayTest, Metadata) {
  using R1 = RegisterArray<float, 5>;
  static_assert(R1::rank == 1);
  static_assert(R1::size == 5);
  EXPECT_EQ(R1::extent(0), 5);

  using R2 = RegisterArray<float, 3, 4>;
  static_assert(R2::rank == 2);
  static_assert(R2::size == 12);
  EXPECT_EQ(R2::extent(0), 3);
  EXPECT_EQ(R2::extent(1), 4);

  using R3 = RegisterArray<double, 2, 3, 4>;
  static_assert(R3::rank == 3);
  static_assert(R3::size == 24);
  EXPECT_EQ(R3::extent(0), 2);
  EXPECT_EQ(R3::extent(1), 3);
  EXPECT_EQ(R3::extent(2), 4);
}

// ---------------------------------------------------------------------------
// Row-major layout: data_ must be laid out so (i,j) -> i*ncols + j, and the
// runtime operator() must agree with the compile-time at<>() accessor.
// ---------------------------------------------------------------------------
TEST(RegisterArrayTest, RowMajorLayout) {
  RegisterArray<int, 3, 4> r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) r(i, j) = i * 4 + j;

  // Backing storage is row-major contiguous 0,1,2,...,11
  for (std::size_t k = 0; k < r.size; ++k)
    EXPECT_EQ(r.data_[k], static_cast<int>(k));

  // Read-back via operator()
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) EXPECT_EQ(r(i, j), i * 4 + j);

  // Compile-time accessor agrees with runtime accessor
  EXPECT_EQ((r.at<0, 0>()), r(0, 0));
  EXPECT_EQ((r.at<1, 2>()), r(1, 2));
  EXPECT_EQ((r.at<2, 3>()), r(2, 3));
}

TEST(RegisterArrayTest, ThreeDLayout) {
  RegisterArray<int, 2, 3, 4> r{};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) r(i, j, k) = (i * 3 + j) * 4 + k;

  for (std::size_t k = 0; k < r.size; ++k)
    EXPECT_EQ(r.data_[k], static_cast<int>(k));

  EXPECT_EQ((r.at<1, 2, 3>()), r(1, 2, 3));
}

TEST(RegisterArrayTest, Fill) {
  RegisterArray<float, 2, 2> r{};
  r.fill(7.5f);
  for (std::size_t k = 0; k < r.size; ++k) EXPECT_FLOAT_EQ(r.data_[k], 7.5f);
}

// ---------------------------------------------------------------------------
// In-kernel use: each iteration builds a register-blocked 2x2 accumulator,
// runs an outer-product update over a contracted loop, and writes its tile to
// a global output View. Mirrors the register-staging pattern of a contraction.
//
//   C(m,n) = sum_k A(m,k) * B(k,n),  with A(m,k)=m+k, B(k,n)=k+n, K=3
// ---------------------------------------------------------------------------
TEST(RegisterArrayTest, InKernelOuterProduct) {
  constexpr int M = 2, N = 2, K = 3;
  using View2 = Kokkos::View<int**, Kokkos::LayoutRight, Kokkos::HostSpace>;
  View2 out("out", M, N);

  Kokkos::parallel_for(
      "register_blocked_tile",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(int) {
        RegisterArray<int, M, N> acc{};
        acc.fill(0);

        // operand slivers staged in registers per k-step
        for (int k = 0; k < K; ++k) {
          RegisterArray<int, M> a{};
          RegisterArray<int, N> b{};
          for (int m = 0; m < M; ++m) a(m) = m + k;  // A(m,k)
          for (int n = 0; n < N; ++n) b(n) = k + n;  // B(k,n)

          for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) acc(m, n) += a(m) * b(n);
        }

        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) out(m, n) = acc(m, n);
      });
  Kokkos::fence();

  // Reference result
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int expected = 0;
      for (int k = 0; k < K; ++k) expected += (m + k) * (k + n);
      EXPECT_EQ(out(m, n), expected);
    }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
