#pragma once
#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace TensorOperations {

// ---------------------------------------------------------------------------
// RegisterArray — a thread-private, compile-time-sized, register-resident
// multidimensional tile.
//
// Used as the RangePolicy staging tier (the register counterpart of team
// scratch). Storage is a Kokkos::Array so it is usable inside KOKKOS_FUNCTION
// kernels on host and device. Indexing is row-major (LayoutRight) to match the
// rest of the library (see Impl::KokkosViewN).
//
// Register-residency caveat: the backing array only stays in registers when the
// index expression folds to a compile-time constant. In a hot loop, callers
// must use unrolled / compile-time indices — the at<Idx...>() accessor exists
// for exactly that and guarantees a constant offset. The runtime operator()
// accessor is general and convenient but a non-constant index will spill the
// array to local memory on GPU.
// ---------------------------------------------------------------------------
template <typename T, int... Extents>
struct RegisterArray {
  static constexpr int         rank = sizeof...(Extents);
  static constexpr std::size_t size = (static_cast<std::size_t>(Extents) * ...);
  using value_type                  = T;

  static_assert(rank > 0, "RegisterArray requires at least one extent");
  static_assert(((Extents > 0) && ...), "all extents must be positive");

  Kokkos::Array<T, size> data_;

  // -- compile-time shape --------------------------------------------------
  static constexpr std::array<std::size_t, rank> extents_{
      static_cast<std::size_t>(Extents)...};

  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr int extent(int k) { return static_cast<int>(extents_[k]); }

  // Row-major strides: stride[rank-1] = 1, stride[k] = stride[k+1]*extent(k+1).
  static constexpr std::array<std::size_t, rank> strides_ = [] {
    std::array<std::size_t, rank> e{static_cast<std::size_t>(Extents)...};
    std::array<std::size_t, rank> s{};
    s[rank - 1] = 1;
    for (int k = rank - 2; k >= 0; --k) s[k] = s[k + 1] * e[k + 1];
    return s;
  }();

  // -- runtime-index accessor (general) ------------------------------------
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr T& operator()(I... idx) {
    static_assert(sizeof...(I) == rank,
                  "number of indices must match RegisterArray rank");
    return data_[offset_of(static_cast<std::size_t>(idx)...)];
  }

  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr const T& operator()(I... idx) const {
    static_assert(sizeof...(I) == rank,
                  "number of indices must match RegisterArray rank");
    return data_[offset_of(static_cast<std::size_t>(idx)...)];
  }

  // -- compile-time-index accessor (guarantees register residency) ---------
  template <int... Idx>
  KOKKOS_FORCEINLINE_FUNCTION constexpr T& at() {
    static_assert(sizeof...(Idx) == rank,
                  "number of indices must match RegisterArray rank");
    return data_[constexpr_offset<Idx...>()];
  }

  template <int... Idx>
  KOKKOS_FORCEINLINE_FUNCTION constexpr const T& at() const {
    static_assert(sizeof...(Idx) == rank,
                  "number of indices must match RegisterArray rank");
    return data_[constexpr_offset<Idx...>()];
  }

  // -- helpers -------------------------------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void fill(T v) {
    for (std::size_t k = 0; k < size; ++k) data_[k] = v;
  }

 private:
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t offset_of(I... idx) const {
    std::array<std::size_t, rank> ix{static_cast<std::size_t>(idx)...};
    std::size_t                   flat = 0;
    for (int k = 0; k < rank; ++k) flat += ix[k] * strides_[k];
    return flat;
  }

  template <int... Idx>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t constexpr_offset() {
    std::array<std::size_t, rank> ix{static_cast<std::size_t>(Idx)...};
    std::size_t                   flat = 0;
    for (int k = 0; k < rank; ++k) flat += ix[k] * strides_[k];
    return flat;
  }
};

}  // namespace TensorOperations
