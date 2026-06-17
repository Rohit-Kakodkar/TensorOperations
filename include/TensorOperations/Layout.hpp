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

// ---------------------------------------------------------------------------
// StridedLayout — the slot -> global-address rule for register-tier staging.
//
// A staging evaluator maps each compile-time register slot's local multi-index
// `local[d]` to a global element of the input tensor as:
//
//     global[d] = tile_offset[d] + local[d] * stride[d]
//
// This decouples the *block shape* (the Tiling) from the *access pattern* (this
// layout). The strides are runtime, which is residency-safe: they only enter
// the runtime global read index, never the compile-time register write index.
//
//   stride[d] == 1            -> contiguous sub-block load (default).
//   stride[d] == #threads(d)  -> staggered / coalesced load: across a thread
//                                group, each staging step reads consecutive
//                                global addresses.
//
// Thread identity is folded into `tile_offset` by the launcher, so the layout
// only needs to carry the per-mode strides.
// ---------------------------------------------------------------------------
template <int Rank>
struct StridedLayout {
  static constexpr int  rank   = Rank;
  std::array<int, Rank> stride = Impl::filled_array<Rank>(1);

  constexpr int operator[](int d) const { return stride[d]; }
};

}  // namespace TensorOperations
