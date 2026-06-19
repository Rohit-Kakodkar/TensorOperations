#pragma once
#include <TensorOperations/TileLayout.hpp>
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
//
// Layout: encode/decode logic lives in StaticTileLayout<Extents...> (exposed
// as layout_type). strides_ and extents_ are forwarded aliases so existing
// callers continue to compile without change.
// ---------------------------------------------------------------------------
template <typename T, int... Extents>
struct RegisterArray {
  using layout_type = StaticTileLayout<Extents...>;

  static constexpr int         rank = layout_type::rank;
  static constexpr std::size_t size = layout_type::size;

  // Forwarded aliases — preserve existing call sites in Range.hpp.
  static constexpr auto extents_ = layout_type::extents_;
  static constexpr auto strides_ = layout_type::strides_;

  using value_type = T;

  static_assert(rank > 0, "RegisterArray requires at least one extent");

  Kokkos::Array<T, size> data_;

  // -- compile-time shape --------------------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr int extent(int k) noexcept { return layout_type::extent(k); }

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

  // -- array-indexed accessors (coordinate as Kokkos::Array) ---------------
  // Delegates to operator() — same register-residency caveats apply.
  KOKKOS_FORCEINLINE_FUNCTION T& operator[](Kokkos::Array<int, rank> idx) {
    return [&]<std::size_t... D>(std::index_sequence<D...>) -> T& {
      return (*this)(idx[D]...);
    }(std::make_index_sequence<rank>{});
  }

  KOKKOS_FORCEINLINE_FUNCTION const T& operator[](
      Kokkos::Array<int, rank> idx) const {
    return [&]<std::size_t... D>(std::index_sequence<D...>) -> const T& {
      return (*this)(idx[D]...);
    }(std::make_index_sequence<rank>{});
  }

  // -- helpers -------------------------------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void fill(T v) {
    for (std::size_t k = 0; k < size; ++k) data_[k] = v;
  }

 private:
  // Delegate multi-index → flat encode to the layout.
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t offset_of(I... idx) const {
    return layout_type{}.flat(static_cast<int>(idx)...);
  }

  // Compile-time-index variant: all Idx are template ints so the compiler
  // constant-folds layout_type{}.flat() to a single integer.
  template <int... Idx>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t constexpr_offset() {
    return layout_type{}.flat(Idx...);
  }
};

}  // namespace TensorOperations
