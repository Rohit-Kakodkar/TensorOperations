#include <gtest/gtest.h>
#include <TensorOperations/Layout.hpp>

using namespace TensorOperations;

// ---------------------------------------------------------------------------
// Impl::filled_array — constexpr fill used internally by layout helpers.
// ---------------------------------------------------------------------------
TEST(LayoutTest, FilledArray) {
  constexpr auto a = Impl::filled_array<3>(1);
  static_assert(a.size() == 3);
  static_assert(a[0] == 1 && a[1] == 1 && a[2] == 1);

  constexpr auto z = Impl::filled_array<2>(0);
  static_assert(z[0] == 0 && z[1] == 0);
  SUCCEED();
}
