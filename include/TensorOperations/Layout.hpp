#pragma once
#include <array>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

namespace Impl {

// constexpr std::array filled with a single value (no <algorithm> on device).
template <int Rank>
constexpr std::array<int, Rank> filled_array(int v) {
  std::array<int, Rank> a{};
  for (int i = 0; i < Rank; ++i) a[i] = v;
  return a;
}

// Extracts the Kokkos memory layout of a View-like T (has ::array_layout).
// Falls back to LayoutRight for plain TensorLike grid types (which the rest of
// the library already treats as row-major). Mirrors exec_space_of /
// value_type_of in NodeHandle.hpp.
template <typename T, typename = void>
struct layout_of {
  using type = Kokkos::LayoutRight;
};
template <typename T>
struct layout_of<T, std::void_t<typename T::array_layout>> {
  using type = typename T::array_layout;
};
template <typename T>
using layout_of_t = typename layout_of<T>::type;

template <typename T>
inline constexpr bool is_layout_left_v =
    std::is_same_v<layout_of_t<T>, Kokkos::LayoutLeft>;
template <typename T>
inline constexpr bool is_layout_stride_v =
    std::is_same_v<layout_of_t<T>, Kokkos::LayoutStride>;

}  // namespace Impl

}  // namespace TensorOperations
