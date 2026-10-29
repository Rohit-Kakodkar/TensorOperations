#pragma once
#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// ---------------------------------------------------------------------------
// StridedAlias<V, ExecSpace, Trailing...>
//
// A non-owning LayoutRight tensor over a raw pointer: a runtime leading extent
// and compile-time trailing extents. It exists because Kokkos::View stops at
// rank 8 while a graph output frame may not -- an element stiffness block
// indexed (element, row component, row point x3, column component, column
// point x3) is rank 9. It satisfies TensorLike (rank, extent, stride, data,
// operator() with one int per axis) and publishes the typedefs the tiling
// code reads (value_type, array_layout, execution_space), so it is accepted
// wherever a Kokkos::View output is.
// ---------------------------------------------------------------------------
template <typename V, typename ExecSpace, int... Trailing>
struct StridedAlias {
  static_assert(((Trailing > 0) && ...),
                "StridedAlias: trailing extents must be positive");

  using value_type      = V;
  using array_layout    = Kokkos::LayoutRight;
  using execution_space = ExecSpace;
  using memory_space    = typename ExecSpace::memory_space;
  static constexpr int rank = 1 + static_cast<int>(sizeof...(Trailing));

  V*  ptr_;
  int leading_;

  KOKKOS_FUNCTION V* data() const noexcept { return ptr_; }

  KOKKOS_FUNCTION int extent(int d) const noexcept {
    constexpr int t[] = {Trailing...};
    return d == 0 ? leading_ : t[d - 1];
  }

  // LayoutRight: stride(d) = product of the extents after d.
  KOKKOS_FUNCTION std::ptrdiff_t stride(int d) const noexcept {
    constexpr int  t[] = {Trailing...};
    std::ptrdiff_t s   = 1;
    for (int k = static_cast<int>(sizeof...(Trailing)) - 1; k >= d; --k)
      s *= t[k];
    return s;
  }

  template <typename... I>
    requires(sizeof...(I) == static_cast<std::size_t>(rank))
  KOKKOS_FUNCTION V& operator()(I... i) const {
    const int idx[] = {static_cast<int>(i)...};
    std::ptrdiff_t off = 0;
    for (int d = 0; d < rank; ++d) off = off * extent(d) + idx[d];
    return ptr_[off];
  }
};

/// make_strided_alias<ES, E1, ..., En>(ptr, leading)
template <typename ExecSpace, int... Trailing, typename V>
KOKKOS_FUNCTION StridedAlias<V, ExecSpace, Trailing...> make_strided_alias(
    V* ptr, int leading) {
  return {ptr, leading};
}

}  // namespace TensorOperations
