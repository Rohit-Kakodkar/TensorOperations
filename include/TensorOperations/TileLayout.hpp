#pragma once
#include <TensorOperations/Macros.hpp>
#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace TensorOperations {

namespace Impl {

template <int N>
struct Index {
  int data[N];

  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr Index(I... idx) noexcept
      : data{static_cast<int>(idx)...} {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr Index(
      const Kokkos::Array<int, N>& idx) noexcept {
    for (int d = 0; d < N; ++d) data[d] = idx[d];
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr int& operator[](int d) noexcept {
    return data[d];
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr const int& operator[](
      int d) const noexcept {
    return data[d];
  }

  template <int D>
  KOKKOS_FORCEINLINE_FUNCTION constexpr int get() const noexcept {
    return data[D];
  }

  template <int D>
  KOKKOS_FORCEINLINE_FUNCTION constexpr void set(int v) noexcept {
    data[D] = v;
  }
};

}  // namespace Impl

// ---------------------------------------------------------------------------
// TileLayout types — encode/decode between a flat index and an N-dimensional
// tile coordinate, and satisfy the View<ViewType, Layout> interface.
//
// Four specialisations covering static/dynamic extents × row/column major:
//
//   StaticTileLayoutRight<int... Extents>   — compile-time, row-major (C)
//   StaticTileLayoutLeft<int... Extents>    — compile-time, column-major (F)
//   DynamicTileLayoutRight<int Rank>        — runtime, row-major (C)
//   DynamicTileLayoutLeft<int Rank>         — runtime, column-major (F)
//
// All four expose the View Layout interface:
//   rank                                   — static constexpr int
//   extent(d)                              — per-dimension extent
//   stride(d)                              — per-dimension stride
//   base_offset()                          — always 0
//   size()                                 — total element count
//   offset(index_sequence, args)           — flat offset from multi-index tuple
//   operator[](int)                        — flat → Kokkos::Array<int, rank>
//
// Static variants also expose:
//   num_elements                           — static constexpr std::size_t total
//   extents_, strides_                     — static constexpr arrays
//
// Factories make_tile_layout(StaticTile<E...>) and
// make_tile_layout(DynamicTile<N>) live in Tiling.hpp (which includes this
// header) to avoid a circular dependency.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// StaticTileLayoutRight — compile-time extents, row-major (rightmost fastest)
// ---------------------------------------------------------------------------
template <int... Extents>
struct StaticTileLayoutRight {
  static constexpr int         rank = sizeof...(Extents);
  static constexpr std::size_t num_elements =
      (static_cast<std::size_t>(Extents) * ...);

  static_assert(rank > 0 && ((Extents > 0) && ...),
                "StaticTileLayoutRight requires at least one positive extent");

  template <int I>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int extent() noexcept {
    constexpr int e[] = {Extents...};
    return e[I];
  }

  template <int I>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int stride() noexcept {
    constexpr int e[] = {Extents...};
    int           s   = 1;
    for (int i = rank - 1; i > I; --i) s *= e[i];
    return s;
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent(int k) noexcept {
    constexpr int e[] = {Extents...};
    return e[k];
  }
  // Row-major stride: product of extents to the right of k
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride(int k) noexcept {
    constexpr int e[] = {Extents...};
    int           s   = 1;
    for (int i = rank - 1; i > k; --i) s *= e[i];
    return s;
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int base_offset() noexcept {
    return 0;
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int size() noexcept {
    return static_cast<int>(num_elements);
  }

  // flat → multi-index (decode): independent per-dim, exposes compile-time
  // divisors
  KOKKOS_FORCEINLINE_FUNCTION auto operator[](int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<rank>{});
  }

  // multi-index → flat offset (encode)
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t flat(
      I... idx) const noexcept {
    return flat_impl(std::index_sequence_for<I...>{}, idx...);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset(
      const Impl::Index<rank>& coord) noexcept {
    return flat_offset_impl_(coord, std::make_index_sequence<rank>{});
  }

 private:
  template <std::size_t... Ds, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ds...>, I... idx) noexcept {
    return (
        (static_cast<std::size_t>(idx) * static_cast<std::size_t>(stride(Ds))) +
        ...);
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static auto decode_impl(
      int linear, std::index_sequence<Ds...>) noexcept {
    return Impl::Index<rank>{
        static_cast<int>((linear / stride(Ds)) % extent(Ds))...};
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset_impl_(
      const Impl::Index<rank>& coord, std::index_sequence<Ds...>) noexcept {
    return (base_offset() + ... +
            (coord.template get<static_cast<int>(Ds)>() *
             stride<static_cast<int>(Ds)>()));
  }
};

// ---------------------------------------------------------------------------
// StaticTileLayoutLeft — compile-time extents, column-major (leftmost fastest)
// ---------------------------------------------------------------------------
template <int... Extents>
struct StaticTileLayoutLeft {
  static constexpr int         rank = sizeof...(Extents);
  static constexpr std::size_t num_elements =
      (static_cast<std::size_t>(Extents) * ...);

  static_assert(rank > 0 && ((Extents > 0) && ...),
                "StaticTileLayoutLeft requires at least one positive extent");

  template <int I>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int extent() noexcept {
    constexpr int e[] = {Extents...};
    return e[I];
  }

  template <int I>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int stride() noexcept {
    constexpr int e[] = {Extents...};
    int           s   = 1;
    for (int i = 0; i < I; ++i) s *= e[i];
    return s;
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent(int k) noexcept {
    constexpr int e[] = {Extents...};
    return e[k];
  }
  // Column-major stride: product of extents to the left of k
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride(int k) noexcept {
    constexpr int e[] = {Extents...};
    int           s   = 1;
    for (int i = 0; i < k; ++i) s *= e[i];
    return s;
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int base_offset() noexcept {
    return 0;
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int size() noexcept {
    return static_cast<int>(num_elements);
  }

  // flat → multi-index (decode): same formula as Right, strides encode the
  // order
  KOKKOS_FORCEINLINE_FUNCTION auto operator[](int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<rank>{});
  }

  // multi-index → flat offset (encode)
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t flat(
      I... idx) const noexcept {
    return flat_impl(std::index_sequence_for<I...>{}, idx...);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset(
      const Impl::Index<rank>& coord) noexcept {
    return flat_offset_impl_(coord, std::make_index_sequence<rank>{});
  }

 private:
  template <std::size_t... Ds, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ds...>, I... idx) noexcept {
    return (
        (static_cast<std::size_t>(idx) * static_cast<std::size_t>(stride(Ds))) +
        ...);
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static auto decode_impl(
      int linear, std::index_sequence<Ds...>) noexcept {
    return Impl::Index<rank>{
        static_cast<int>((linear / stride(Ds)) % extent(Ds))...};
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset_impl_(
      const Impl::Index<rank>& coord, std::index_sequence<Ds...>) noexcept {
    return (base_offset() + ... +
            (coord.template get<static_cast<int>(Ds)>() *
             stride<static_cast<int>(Ds)>()));
  }
};

// ---------------------------------------------------------------------------
// DynamicTileLayoutRight — runtime extents, row-major (rightmost fastest)
// ---------------------------------------------------------------------------
template <int Rank>
struct DynamicTileLayoutRight {
  static constexpr int rank = Rank;

  Kokkos::Array<int, Rank>         extents_;
  Kokkos::Array<std::size_t, Rank> strides_;  // precomputed row-major strides

  KOKKOS_FUNCTION DynamicTileLayoutRight() : extents_{}, strides_{} {}

  KOKKOS_FUNCTION explicit DynamicTileLayoutRight(Kokkos::Array<int, Rank> ext)
      : extents_(ext) {
    strides_[Rank - 1] = 1;
    for (int k = Rank - 2; k >= 0; --k)
      strides_[k] = strides_[k + 1] * static_cast<std::size_t>(ext[k + 1]);
  }

  KOKKOS_FUNCTION int extent(int k) const noexcept { return extents_[k]; }
  KOKKOS_FUNCTION int stride(int k) const noexcept {
    return static_cast<int>(strides_[k]);
  }
  KOKKOS_FUNCTION static constexpr int base_offset() noexcept { return 0; }
  KOKKOS_FUNCTION int                  size() const noexcept {
    int s = 1;
    for (int k = 0; k < Rank; ++k) s *= extents_[k];
    return s;
  }

  // flat → multi-index (decode): peel from rightmost (row-major)
  KOKKOS_FUNCTION auto operator[](int linear) const noexcept {
    Kokkos::Array<int, Rank> idx{};
    for (int d = Rank - 1; d >= 0; --d) {
      idx[d] = linear % extents_[d];
      linear /= extents_[d];
    }
    return Impl::Index<Rank>{idx};
  }

  // multi-index → flat offset (encode)
  KOKKOS_FUNCTION std::size_t flat(
      Kokkos::Array<int, Rank> idx) const noexcept {
    std::size_t f = 0;
    for (int k = 0; k < Rank; ++k)
      f += static_cast<std::size_t>(idx[k]) * strides_[k];
    return f;
  }

  KOKKOS_FUNCTION int flat_offset(
      const Impl::Index<Rank>& coord) const noexcept {
    int off = base_offset();
    for (int d = 0; d < Rank; ++d)
      off += coord[d] * static_cast<int>(strides_[d]);
    return off;
  }
};

// ---------------------------------------------------------------------------
// DynamicTileLayoutLeft — runtime extents, column-major (leftmost fastest)
// ---------------------------------------------------------------------------
template <int Rank>
struct DynamicTileLayoutLeft {
  static constexpr int rank = Rank;

  Kokkos::Array<int, Rank> extents_;
  Kokkos::Array<std::size_t, Rank>
      strides_;  // precomputed column-major strides

  KOKKOS_FUNCTION DynamicTileLayoutLeft() : extents_{}, strides_{} {}

  KOKKOS_FUNCTION explicit DynamicTileLayoutLeft(Kokkos::Array<int, Rank> ext)
      : extents_(ext) {
    strides_[0] = 1;
    for (int k = 1; k < Rank; ++k)
      strides_[k] = strides_[k - 1] * static_cast<std::size_t>(ext[k - 1]);
  }

  KOKKOS_FUNCTION int extent(int k) const noexcept { return extents_[k]; }
  KOKKOS_FUNCTION int stride(int k) const noexcept {
    return static_cast<int>(strides_[k]);
  }
  KOKKOS_FUNCTION static constexpr int base_offset() noexcept { return 0; }
  KOKKOS_FUNCTION int                  size() const noexcept {
    int s = 1;
    for (int k = 0; k < Rank; ++k) s *= extents_[k];
    return s;
  }

  // flat → multi-index (decode): peel from leftmost (column-major)
  KOKKOS_FUNCTION auto operator[](int linear) const noexcept {
    Kokkos::Array<int, Rank> idx{};
    for (int d = 0; d < Rank; ++d) {
      idx[d] = linear % extents_[d];
      linear /= extents_[d];
    }
    return Impl::Index<Rank>{idx};
  }

  // multi-index → flat offset (encode)
  KOKKOS_FUNCTION std::size_t flat(
      Kokkos::Array<int, Rank> idx) const noexcept {
    std::size_t f = 0;
    for (int k = 0; k < Rank; ++k)
      f += static_cast<std::size_t>(idx[k]) * strides_[k];
    return f;
  }

  KOKKOS_FUNCTION int flat_offset(
      const Impl::Index<Rank>& coord) const noexcept {
    int off = base_offset();
    for (int d = 0; d < Rank; ++d)
      off += coord[d] * static_cast<int>(strides_[d]);
    return off;
  }
};

}  // namespace TensorOperations
