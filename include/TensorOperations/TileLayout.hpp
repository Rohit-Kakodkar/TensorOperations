#pragma once
#include <TensorOperations/Macros.hpp>
#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace TensorOperations {

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

  static constexpr std::array<int, rank> extents_{Extents...};

  // Row-major strides: strides_[rank-1]=1,
  // strides_[k]=strides_[k+1]*extents_[k+1]
  static constexpr std::array<std::size_t, rank> strides_ = [] {
    std::array<std::size_t, rank> s{};
    s[rank - 1] = 1;
    for (int k = rank - 2; k >= 0; --k)
      s[k] = s[k + 1] * static_cast<std::size_t>(extents_[k + 1]);
    return s;
  }();

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent(int k) noexcept {
    return extents_[k];
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride(int k) noexcept {
    return static_cast<int>(strides_[k]);
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int base_offset() noexcept {
    return 0;
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int size() noexcept {
    return static_cast<int>(num_elements);
  }

  // View Layout: flat offset from a tuple of rank indices
  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION int offset(std::index_sequence<Is...>,
                                         auto& args) const noexcept {
    return ((static_cast<int>(std::get<Is>(args)) *
             static_cast<int>(strides_[Is])) +
            ...);
  }

  // flat → multi-index (decode): independent per-dim, exposes compile-time
  // divisors
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, rank> operator[](
      int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<rank>{});
  }

  // multi-index → flat offset (encode)
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t flat(
      I... idx) const noexcept {
    return flat_impl(std::index_sequence_for<I...>{}, idx...);
  }

 private:
  template <std::size_t... Ds, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ds...>, I... idx) noexcept {
    return ((static_cast<std::size_t>(idx) * strides_[Ds]) + ...);
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, rank> decode_impl(
      int linear, std::index_sequence<Ds...>) noexcept {
    return Kokkos::Array<int, rank>{static_cast<int>(
        (linear / static_cast<int>(strides_[Ds])) % extents_[Ds])...};
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

  static constexpr std::array<int, rank> extents_{Extents...};

  // Column-major strides: strides_[0]=1,
  // strides_[k]=strides_[k-1]*extents_[k-1]
  static constexpr std::array<std::size_t, rank> strides_ = [] {
    std::array<std::size_t, rank> s{};
    s[0] = 1;
    for (int k = 1; k < rank; ++k)
      s[k] = s[k - 1] * static_cast<std::size_t>(extents_[k - 1]);
    return s;
  }();

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent(int k) noexcept {
    return extents_[k];
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride(int k) noexcept {
    return static_cast<int>(strides_[k]);
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int base_offset() noexcept {
    return 0;
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int size() noexcept {
    return static_cast<int>(num_elements);
  }

  // View Layout: flat offset from a tuple of rank indices
  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION int offset(std::index_sequence<Is...>,
                                         auto& args) const noexcept {
    return ((static_cast<int>(std::get<Is>(args)) *
             static_cast<int>(strides_[Is])) +
            ...);
  }

  // flat → multi-index (decode): same formula as Right, strides encode the
  // order
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, rank> operator[](
      int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<rank>{});
  }

  // multi-index → flat offset (encode)
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t flat(
      I... idx) const noexcept {
    return flat_impl(std::index_sequence_for<I...>{}, idx...);
  }

 private:
  template <std::size_t... Ds, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ds...>, I... idx) noexcept {
    return ((static_cast<std::size_t>(idx) * strides_[Ds]) + ...);
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, rank> decode_impl(
      int linear, std::index_sequence<Ds...>) noexcept {
    return Kokkos::Array<int, rank>{static_cast<int>(
        (linear / static_cast<int>(strides_[Ds])) % extents_[Ds])...};
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

  // View Layout: flat offset from a tuple of rank indices
  template <std::size_t... Is>
  KOKKOS_FUNCTION int offset(std::index_sequence<Is...>,
                             auto& args) const noexcept {
    return ((static_cast<int>(std::get<Is>(args)) *
             static_cast<int>(strides_[Is])) +
            ...);
  }

  // flat → multi-index (decode): peel from rightmost (row-major)
  KOKKOS_FUNCTION Kokkos::Array<int, Rank> operator[](
      int linear) const noexcept {
    Kokkos::Array<int, Rank> idx{};
    for (int d = Rank - 1; d >= 0; --d) {
      idx[d] = linear % extents_[d];
      linear /= extents_[d];
    }
    return idx;
  }

  // multi-index → flat offset (encode)
  KOKKOS_FUNCTION std::size_t flat(
      Kokkos::Array<int, Rank> idx) const noexcept {
    std::size_t f = 0;
    for (int k = 0; k < Rank; ++k)
      f += static_cast<std::size_t>(idx[k]) * strides_[k];
    return f;
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

  // View Layout: flat offset from a tuple of rank indices
  template <std::size_t... Is>
  KOKKOS_FUNCTION int offset(std::index_sequence<Is...>,
                             auto& args) const noexcept {
    return ((static_cast<int>(std::get<Is>(args)) *
             static_cast<int>(strides_[Is])) +
            ...);
  }

  // flat → multi-index (decode): peel from leftmost (column-major)
  KOKKOS_FUNCTION Kokkos::Array<int, Rank> operator[](
      int linear) const noexcept {
    Kokkos::Array<int, Rank> idx{};
    for (int d = 0; d < Rank; ++d) {
      idx[d] = linear % extents_[d];
      linear /= extents_[d];
    }
    return idx;
  }

  // multi-index → flat offset (encode)
  KOKKOS_FUNCTION std::size_t flat(
      Kokkos::Array<int, Rank> idx) const noexcept {
    std::size_t f = 0;
    for (int k = 0; k < Rank; ++k)
      f += static_cast<std::size_t>(idx[k]) * strides_[k];
    return f;
  }
};

}  // namespace TensorOperations
