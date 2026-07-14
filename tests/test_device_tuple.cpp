#include <Kokkos_Core.hpp>
#include <TensorOperations/DeviceTuple.hpp>
#include <gtest/gtest.h>

#include <type_traits>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Compile-time surface: tuple_element_t, tuple_size(_v), ::size.
// ---------------------------------------------------------------------------
namespace {
using T3 = DeviceTuple<int, float, double>;
static_assert(T3::size == 3);
static_assert(tuple_size_v<T3> == 3);
static_assert(std::is_same_v<tuple_element_t<0, T3>, int>);
static_assert(std::is_same_v<tuple_element_t<1, T3>, float>);
static_assert(std::is_same_v<tuple_element_t<2, T3>, double>);

// Duplicate element types stay unambiguous (index-keyed leaves).
using Tdup = DeviceTuple<int, int>;
static_assert(tuple_size_v<Tdup> == 2);
static_assert(std::is_same_v<tuple_element_t<0, Tdup>, int>);
static_assert(std::is_same_v<tuple_element_t<1, Tdup>, int>);
}  // namespace

TEST(DeviceTupleTest, HostGetFreeAndMember) {
  DeviceTuple<int, float, double> t(7, 2.5f, 3.25);

  // free get<I>
  EXPECT_EQ(get<0>(t), 7);
  EXPECT_FLOAT_EQ(get<1>(t), 2.5f);
  EXPECT_DOUBLE_EQ(get<2>(t), 3.25);

  // member get<I>
  EXPECT_EQ(t.get<0>(), 7);
  EXPECT_FLOAT_EQ(t.get<1>(), 2.5f);

  // mutate through a reference
  get<0>(t)  = 42;
  t.get<1>() = 8.0f;
  EXPECT_EQ(get<0>(t), 42);
  EXPECT_FLOAT_EQ(t.get<1>(), 8.0f);
}

TEST(DeviceTupleTest, DuplicateTypesDistinctByIndex) {
  DeviceTuple<int, int> t(3, 9);
  EXPECT_EQ(get<0>(t), 3);
  EXPECT_EQ(get<1>(t), 9);
  get<1>(t) = 100;
  EXPECT_EQ(get<0>(t), 3);  // untouched
  EXPECT_EQ(get<1>(t), 100);
}

// Kernel body as a named functor: an extended __host__ __device__ lambda
// (KOKKOS_LAMBDA) may not appear inside GoogleTest's private TestBody under
// nvcc.
struct TupleRoundTripKernel {
  Kokkos::View<double*> out;
  KOKKOS_FUNCTION void  operator()(int) const {
    DeviceTuple<int, float, double> t(2, 4.0f, 8.0);
    out(0) = static_cast<double>(get<0>(t));   // free get on device
    out(1) = static_cast<double>(t.get<1>());  // member get on device
    out(2) = get<2>(t);
  }
};

TEST(DeviceTupleTest, DeviceRoundTrip) {
  // Build a tuple on device inside a kernel, read each element via get<I>, and
  // write into a view; verify on host. Exercises device-side construction +
  // indexing (the whole point of a device tuple).
  Kokkos::View<double*> out("out", 3);
  Kokkos::parallel_for("device_tuple_roundtrip", Kokkos::RangePolicy<>(0, 1),
                       TupleRoundTripKernel{out});
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  EXPECT_DOUBLE_EQ(h(0), 2.0);
  EXPECT_DOUBLE_EQ(h(1), 4.0);
  EXPECT_DOUBLE_EQ(h(2), 8.0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
