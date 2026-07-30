#pragma once
#include <TensorOperations/Macros.hpp>
#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace TensorOperations {

struct LayoutRight {};
struct LayoutLeft {};

// Forward declaration — defined in Tiling.hpp, which includes this header.
// Required so StaticLayout can specialise on StaticTile<E...> without a
// circular include.
template <int... E>
struct StaticTile;

// Forward declaration — defined in DeviceTuple.hpp, deliberately NOT included
// here (it pulls in all of Kokkos_Core, and this header carries only
// Kokkos_Array / Kokkos_Macros). The nested StaticLayout below uses it purely
// as a type-level bracket grouping one integer_sequence per mode; it is never
// instantiated as an object.
template <typename... Ts>
struct DeviceTuple;

// ---------------------------------------------------------------------------
// StaticLayout<Extents, Strides, Order> — the one generic static layout.
//
//   Extents = StaticTile<E...>                   logical extent per dimension
//   Strides = std::integer_sequence<int, S...>   memory stride per dimension
//   Order   = std::integer_sequence<int, O...>   Order[j] = dimension at
//                                                memory-order position j
//                                                (fastest-varying first)
//
// Three wrapper types rather than three bare packs: only one template parameter
// pack may be deduced per specialisation. Defined below, after the Impl helpers
// it needs; declared here so those helpers can name it.
// ---------------------------------------------------------------------------
template <typename Extents, typename Strides, typename Order>
struct StaticLayout;

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

// ---------------------------------------------------------------------------
// StaticTileLayoutBase<Extents...>
//
// Shared compile-time members for StaticTileLayoutRight, StaticTileLayoutLeft,
// and StaticTileLayoutStride: rank, num_elements, extent(), size(),
// base_offset(). Stride computation differs between layouts and stays in the
// derived classes.
// ---------------------------------------------------------------------------
template <int... Extents>
struct StaticTileLayoutBase {
  static constexpr bool        is_static = true;
  static constexpr int         rank      = sizeof...(Extents);
  static constexpr std::size_t num_elements =
      (static_cast<std::size_t>(Extents) * ...);

  static_assert(rank > 0 && ((Extents > 0) && ...),
                "StaticTileLayout requires at least one positive extent");

  template <int I>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int extent() noexcept {
    constexpr int e[] = {Extents...};
    return e[I];
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent(int k) noexcept {
    constexpr int e[] = {Extents...};
    return e[k];
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int base_offset() noexcept {
    return 0;
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int size() noexcept {
    return static_cast<int>(num_elements);
  }
};

// ---------------------------------------------------------------------------
// Compile-time memory orders and packed strides — the ingredients that turn
// StaticLayout into the row-major, column-major, and arbitrary-order layouts
// below.
// ---------------------------------------------------------------------------

// order_is_permutation<Rank, O...>() — is O... a permutation of [0, Rank)?
// Guards StaticLayout's Order parameter; a repeated or out-of-range dimension
// would silently corrupt every decode.
template <int Rank, int... O>
constexpr bool order_is_permutation() noexcept {
  constexpr int ord[]      = {O...};
  bool          seen[Rank] = {};
  for (int j = 0; j < Rank; ++j) {
    if (ord[j] < 0 || ord[j] >= Rank || seen[ord[j]]) return false;
    seen[ord[j]] = true;
  }
  return true;
}

// Type-level memory orders: {N-1,...,0} for LayoutRight (rightmost dimension
// fastest) and {0,...,N-1} for LayoutLeft. The function-style siblings
// Impl::right_order_seq / left_order_seq in TiledLayout.hpp build the same
// sequences as values; these are needed as types, to name a base class.
template <int N, typename Js>
struct RightOrder;
template <int N, std::size_t... J>
struct RightOrder<N, std::index_sequence<J...>> {
  using type = std::integer_sequence<int, (N - 1 - static_cast<int>(J))...>;
};
template <int N>
using right_order_t = typename RightOrder<N, std::make_index_sequence<N>>::type;

template <int N, typename Js>
struct LeftOrder;
template <int N, std::size_t... J>
struct LeftOrder<N, std::index_sequence<J...>> {
  using type = std::integer_sequence<int, static_cast<int>(J)...>;
};
template <int N>
using left_order_t = typename LeftOrder<N, std::make_index_sequence<N>>::type;

// PackedStrideAt<StaticTile<E...>, Order>::get(d) — the *dense* stride of
// dimension d: the product of the extents that vary faster than d in memory
// order Order. This is the stride of a gap-free tile; a StaticLayout may
// declare wider strides (padding), in which case the two differ.
//
// A separate class, not a member of StaticLayout, because packed_strides_t
// below calls get() in a template argument — legal only for an already
// complete class.
template <typename Tile, typename OrderSeq>
struct PackedStrideAt;

template <int... E, int... O>
struct PackedStrideAt<StaticTile<E...>, std::integer_sequence<int, O...>> {
  static constexpr int rank = sizeof...(E);

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int get(int d) noexcept {
    constexpr int e[]   = {E...};
    constexpr int ord[] = {O...};
    int           s     = 1;
    for (int j = 0; j < rank; ++j) {
      if (ord[j] == d) return s;
      s *= e[ord[j]];
    }
    return 0;  // unreachable for valid d in [0, rank)
  }
};

template <typename Tile, typename OrderSeq, typename Js>
struct PackedStrides;
template <typename Tile, typename OrderSeq, std::size_t... J>
struct PackedStrides<Tile, OrderSeq, std::index_sequence<J...>> {
  using type = std::integer_sequence<int, PackedStrideAt<Tile, OrderSeq>::get(
                                              static_cast<int>(J))...>;
};

// The gap-free strides of Tile in memory order OrderSeq, as a template
// argument for StaticLayout.
template <typename Tile, typename OrderSeq>
using packed_strides_t = typename PackedStrides<
    Tile, OrderSeq,
    std::make_index_sequence<PackedStrideAt<Tile, OrderSeq>::rank>>::type;

// static_layout_for_t<Tile, OrderSeq> — the packed StaticLayout for a tile laid
// out in a given memory order. Base class of all three named static layouts.
template <typename Tile, typename OrderSeq>
using static_layout_for_t =
    StaticLayout<Tile, packed_strides_t<Tile, OrderSeq>, OrderSeq>;

// ---------------------------------------------------------------------------
// Nested-layout helpers — the machinery behind the hierarchical StaticLayout
// specialisation, where one logical *mode* is backed by several memory
// *leaves*. Kept here (not as members) for the same reason PackedStrideAt is:
// their results are used in template arguments, which requires a complete type.
// ---------------------------------------------------------------------------

// The I-th value of an int pack, as a constant expression. (Tiling.hpp has an
// identical pack_at, but it includes this header, not the other way round.)
template <std::size_t I, int... V>
KOKKOS_FORCEINLINE_FUNCTION constexpr int value_at() noexcept {
  constexpr int a[] = {V...};
  return a[I];
}

// The M-th type of a type pack. Recursion depth is the mode count (2-6), so
// the cost is nil; DeviceTuple's tuple_element_t would need the complete type.
template <std::size_t M, typename... Ts>
struct TypeAt;
template <typename T0, typename... Ts>
struct TypeAt<0, T0, Ts...> {
  using type = T0;
};
template <std::size_t M, typename T0, typename... Ts>
struct TypeAt<M, T0, Ts...> : TypeAt<M - 1, Ts...> {};
template <std::size_t M, typename... Ts>
using type_at_t = typename TypeAt<M, Ts...>::type;

// Concatenate integer_sequences left to right: the flattened leaf view of a
// nested layout's extents (or order).
template <typename... Seqs>
struct ConcatSeq {
  using type = std::integer_sequence<int>;
};
template <int... A>
struct ConcatSeq<std::integer_sequence<int, A...>> {
  using type = std::integer_sequence<int, A...>;
};
template <int... A, int... B, typename... Rest>
struct ConcatSeq<std::integer_sequence<int, A...>,
                 std::integer_sequence<int, B...>, Rest...>
    : ConcatSeq<std::integer_sequence<int, A..., B...>, Rest...> {};
template <typename... Seqs>
using concat_seq_t = typename ConcatSeq<Seqs...>::type;

// integer_sequence -> StaticTile, so flattened leaf extents can be fed to the
// existing PackedStrideAt (StaticTile need only be declared, never completed).
template <typename Seq>
struct SeqToTile;
template <int... E>
struct SeqToTile<std::integer_sequence<int, E...>> {
  using type = StaticTile<E...>;
};
template <typename Seq>
using seq_to_tile_t = typename SeqToTile<Seq>::type;

// order_is_permutation, reached through a sequence rather than a loose pack.
template <typename OrdSeq>
struct OrderSeqIsPermutation;
template <int... O>
struct OrderSeqIsPermutation<std::integer_sequence<int, O...>> {
  static constexpr bool value =
      order_is_permutation<static_cast<int>(sizeof...(O)), O...>();
};

// ---------------------------------------------------------------------------
// Mode<ExtSeq, StrSeq> — one logical mode of a nested StaticLayout: `arity`
// memory leaves whose extents multiply to `mode_size`.
//
// A mode index i in [0, mode_size) splits into leaf coordinates
//   c_l = (i / divisor(l)) % extent_at(l),
// leaf 0 fastest-varying, and contributes sum_l c_l * stride_at(l) to the
// memory offset.
//
// Two shortcuts fall out of compile-time constants and matter a lot:
//   • leaf 0 has divisor 1, so it needs no division;
//   • the last leaf needs no modulo (i < mode_size bounds the top quotient).
// Together they make an ARITY-1 MODE COMPILE TO EXACTLY `i * stride`, i.e.
// identical to the flat layout — the nested form costs nothing when it
// degenerates.
// ---------------------------------------------------------------------------
template <typename ExtSeq, typename StrSeq>
struct Mode;

template <int... E, int... S>
struct Mode<std::integer_sequence<int, E...>,
            std::integer_sequence<int, S...>> {
  static constexpr int arity     = static_cast<int>(sizeof...(E));
  static constexpr int mode_size = (E * ... * 1);

  static_assert(sizeof...(E) == sizeof...(S),
                "nested StaticLayout: a mode's extents and strides must have "
                "the same number of leaves");
  static_assert(arity > 0, "nested StaticLayout: a mode needs at least 1 leaf");
  static_assert(((E > 0) && ...),
                "nested StaticLayout requires positive extents");
  static_assert(((S > 0) && ...),
                "nested StaticLayout requires positive strides");

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int extent_at(int l) noexcept {
    constexpr int e[] = {E...};
    return e[l];
  }
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride_at(int l) noexcept {
    constexpr int s[] = {S...};
    return s[l];
  }
  // Within-mode packed divisor for leaf l: the product of the extents before
  // it.
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int divisor(int l) noexcept {
    constexpr int e[] = {E...};
    int           p   = 1;
    for (int k = 0; k < l; ++k) p *= e[k];
    return p;
  }

  // Leaf coordinate L of mode index i. The constexpr locals guarantee the
  // divisor is a compile-time constant, so the `/` lowers to multiply-high plus
  // shift rather than a real integer division (expensive on GPU).
  template <int L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int leaf_coord(int i) noexcept {
    constexpr int p = divisor(L);
    constexpr int e = extent_at(L);
    const int     q = (p == 1) ? i : i / p;
    if constexpr (L + 1 == arity)
      return q;  // i < mode_size, so q is already in [0, e)
    else
      return q % e;
  }

  // The memory offset this mode contributes for mode index i.
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t offset(
      int i) noexcept {
    return offset_impl(i, std::make_index_sequence<sizeof...(E)>{});
  }

 private:
  template <std::size_t... L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t offset_impl(
      int i, std::index_sequence<L...>) noexcept {
    return ((static_cast<std::size_t>(leaf_coord<static_cast<int>(L)>(i)) *
             static_cast<std::size_t>(stride_at(static_cast<int>(L)))) +
            ... + std::size_t{0});
  }
};

// ---------------------------------------------------------------------------
// DynamicTileLayoutBase<Rank>
//
// Shared runtime members for DynamicTileLayoutRight and DynamicTileLayoutLeft:
// the extents_ / strides_ arrays, extent(), stride(), base_offset(), size(),
// flat(), and flat_offset(). Both directions use the same strides_ field and
// the same access formulas — they differ only in how strides_ is initialized
// (constructor direction) and how operator[] peels dimensions. Those two pieces
// stay in the derived classes.
// ---------------------------------------------------------------------------
template <int Rank>
struct DynamicTileLayoutBase {
  static constexpr bool is_static = false;
  static constexpr int  rank      = Rank;

  Kokkos::Array<int, Rank>         extents_;
  Kokkos::Array<std::size_t, Rank> strides_;

  KOKKOS_FUNCTION DynamicTileLayoutBase() : extents_{}, strides_{} {}

 protected:
  // Protected so derived constructors can supply pre-built strides.
  KOKKOS_FUNCTION DynamicTileLayoutBase(Kokkos::Array<int, Rank>         ext,
                                        Kokkos::Array<std::size_t, Rank> str)
      : extents_(ext), strides_(str) {}

 public:
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

}  // namespace Impl

// ---------------------------------------------------------------------------
// TileLayout types — encode/decode between a flat index and an N-dimensional
// tile coordinate, and satisfy the View<ViewType, Layout> interface.
//
// One generic compile-time layout, three named packed specialisations of it,
// and two runtime layouts:
//
//   StaticLayout<StaticTile<E...>,                 — compile-time extents,
//                integer_sequence<int, S...>,        strides, and memory order
//                integer_sequence<int, O...>>
//   StaticLayout<DeviceTuple<ExtSeqs...>,          — nested (hierarchical):
//                DeviceTuple<StrSeqs...>,            one logical mode backed by
//                DeviceTuple<OrdSeqs...>>            several memory leaves
//   StaticTileLayoutRight<int... Extents>          — packed, row-major (C)
//   StaticTileLayoutLeft<int... Extents>           — packed, column-major (F)
//   StaticTileLayoutStride<StaticTile<E...>,       — packed, arbitrary memory
//                          int... Order>             order (Order[j] = dim
//                                                    index fastest at j=0)
//   DynamicTileLayoutRight<int Rank>               — runtime, row-major (C)
//   DynamicTileLayoutLeft<int Rank>                — runtime, column-major (F)
//
// The three named static layouts are thin derived classes of StaticLayout that
// supply the packed (gap-free) strides for their memory order; they differ from
// each other only in that order. They stay distinct class templates rather than
// aliases because Tiling.hpp and TiledLayout.hpp overload on
// StaticTileLayoutRight<Extents...> etc., and deduction cannot see through an
// alias whose stride arguments are computed.
//
// All of them expose the View Layout interface:
//   rank                                   — static constexpr int
//   extent(d)                              — per-dimension extent
//   stride(d)                              — per-dimension stride
//   base_offset()                          — always 0
//   size()                                 — total element count
//   flat(I... idx)                         — multi-index → flat offset
//   flat_offset(Index<rank>)               — same, from an Index value
//   operator[](int)                        — flat → Impl::Index<rank>
//
// Static variants also expose:
//   num_elements                           — static constexpr std::size_t total
//   extent<I>(), stride<I>()               — compile-time per-dimension queries
//   mode_arity<M>()                        — leaves backing mode M (1 if flat)
//   mode_extents<M>(), mode_strides<M>()   — that mode's leaves, as an
//                                            integer_sequence
//   mode_extent<M,L>(), mode_stride<M,L>() — one leaf of one mode
//
// The mode_* family is spelled identically on the flat and nested static
// layouts, so leaf-walking code spans both; a flat layout is the degenerate
// case where every mode has exactly one leaf. Note stride(d) is the exception:
// the nested layout omits it, because a multi-leaf mode has no single stride.
//
// Factories make_tile_layout(StaticTile<E...>, ...) and
// make_tile_layout(DynamicTile<N>) live in Tiling.hpp (which includes this
// header) to avoid a circular dependency.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// StaticLayout — compile-time extents, strides, and memory order.
//
// Strides are an explicit template pack, so a StaticLayout need not be
// gap-free: extents {4,8} with strides {9,1} is a 4x8 tile with a row pitch of
// 9. Order is still required, because with arbitrary strides the memory order
// can no longer be recovered from them cheaply, and decode needs it.
//
// All strides, extents, flat(), flat_offset(), and decode are compile-time
// evaluated — no runtime arithmetic on the stride/extent arrays.
// ---------------------------------------------------------------------------
template <int... Extents, int... Strides, int... Order>
struct StaticLayout<StaticTile<Extents...>,
                    std::integer_sequence<int, Strides...>,
                    std::integer_sequence<int, Order...>>
    : Impl::StaticTileLayoutBase<Extents...> {
  using base = Impl::StaticTileLayoutBase<Extents...>;
  using base::base_offset;
  using base::extent;
  using base::num_elements;
  using base::rank;
  using base::size;

  static_assert(sizeof...(Strides) == rank,
                "StaticLayout: Strides must have exactly rank elements");
  static_assert(sizeof...(Order) == rank,
                "StaticLayout: Order must have exactly rank elements");
  static_assert(((Strides > 0) && ...),
                "StaticLayout requires positive strides");
  static_assert(Impl::order_is_permutation<rank, Order...>(),
                "StaticLayout: Order must be a permutation of [0, rank)");

  template <int D>
  KOKKOS_FORCEINLINE_FUNCTION static consteval int stride() noexcept {
    constexpr int s[] = {Strides...};
    return s[D];
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int stride(int k) noexcept {
    constexpr int s[] = {Strides...};
    return s[k];
  }

  // --- per-mode (leaf-level) accessors -------------------------------------
  // Spelled identically on the nested specialisation below, so code that walks
  // a layout's leaves works against either. A flat layout is the degenerate
  // case: every mode has exactly one leaf, whose extent and stride are the
  // mode's own.
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_arity() noexcept {
    return 1;
  }
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto mode_extents() noexcept {
    return std::integer_sequence<
        int, Impl::value_at<static_cast<std::size_t>(M), Extents...>()>{};
  }
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto mode_strides() noexcept {
    return std::integer_sequence<
        int, Impl::value_at<static_cast<std::size_t>(M), Strides...>()>{};
  }
  template <int M, int L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_extent() noexcept {
    static_assert(L == 0, "flat StaticLayout: every mode has exactly one leaf");
    return Impl::value_at<static_cast<std::size_t>(M), Extents...>();
  }
  template <int M, int L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_stride() noexcept {
    static_assert(L == 0, "flat StaticLayout: every mode has exactly one leaf");
    return Impl::value_at<static_cast<std::size_t>(M), Strides...>();
  }

  // flat → multi-index (decode): coord[d] = (linear / packed_stride(d)) %
  // extent(d). Independent per-dim, exact integer division on compile-time
  // divisors (no reciprocal needed). Divides by the *packed* strides, not
  // stride(d): a linear index runs over [0, size()), which counts elements, so
  // a padded layout's pad slots must not enter the decode.
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
  using packed_ = Impl::PackedStrideAt<StaticTile<Extents...>,
                                       std::integer_sequence<int, Order...>>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int packed_stride(
      int d) noexcept {
    return packed_::get(d);
  }

  template <std::size_t... Ds, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ds...>, I... idx) noexcept {
    return ((static_cast<std::size_t>(idx) *
             static_cast<std::size_t>(stride(static_cast<int>(Ds)))) +
            ...);
  }

  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION static auto decode_impl(
      int linear, std::index_sequence<Ds...>) noexcept {
    return Impl::Index<rank>{
        static_cast<int>((linear / packed_stride(static_cast<int>(Ds))) %
                         extent(static_cast<int>(Ds)))...};
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
// StaticLayout<DeviceTuple<ExtSeqs...>,   — nested (hierarchical) layout: one
//              DeviceTuple<StrSeqs...>,     integer_sequence per MODE, each
//              DeviceTuple<OrdSeqs...>>     holding that mode's LEAVES.
//
// A mode is one logical axis backed by several memory leaves, so a rank-2
// layout can describe memory that is really 4-dimensional:
//
//   extents ((5,5),(5,5))  strides ((1,25),(5,125))  order ((0,2),(1,3))
//
//   rank == 2, extent(0) == extent(1) == 25, num_elements == 625
//   flat(i,j) = (i%5)*1 + ((i/5)%5)*25 + (j%5)*5 + ((j/5)%5)*125
//
// i.e. a column-major (5,5,5,5) tensor indexed (i0,j0,i1,j1) presented as a
// 25x25 matrix — a reshape neither StaticTileLayoutLeft nor Right can express.
//
// INDEXING IS ONE SCALAR PER MODE: flat(i, j) with i, j in [0, 25). Splitting
// a mode index into its leaves happens internally, on compile-time divisors,
// so View::operator() and Impl::Index need no changes.
//
// Order values are GLOBAL leaf indices (over the flattened leaves), nested to
// mirror the extents' shape: ((0,2),(1,3)) flattens to 0,2,1,3 — the memory
// order of strides 1, 5, 25, 125.
//
// The collapsed accessors (rank, extent(d), extent<I>(), size(),
// num_elements) come from StaticTileLayoutBase over the per-mode products, so
// every existing consumer keeps working unchanged. Per-leaf data is reached
// through mode_extents<M>() / mode_strides<M>() — deliberately NOT named
// extent<I>(), so that spelling keeps one meaning across both specialisations.
//
// stride(int d) is NOT provided: a multi-leaf mode has no single stride. This
// type therefore does not satisfy the TensorLike concept.
// ---------------------------------------------------------------------------
template <typename... ExtSeqs, typename... StrSeqs, typename... OrdSeqs>
struct StaticLayout<DeviceTuple<ExtSeqs...>, DeviceTuple<StrSeqs...>,
                    DeviceTuple<OrdSeqs...>>
    : Impl::StaticTileLayoutBase<Impl::Mode<ExtSeqs, StrSeqs>::mode_size...> {
  using base =
      Impl::StaticTileLayoutBase<Impl::Mode<ExtSeqs, StrSeqs>::mode_size...>;
  using base::base_offset;
  using base::extent;  // collapsed: extent(d) and extent<I>() both -> int
  using base::num_elements;
  using base::rank;
  using base::size;

  template <std::size_t M>
  using mode_t = Impl::Mode<Impl::type_at_t<M, ExtSeqs...>,
                            Impl::type_at_t<M, StrSeqs...>>;

  static_assert(sizeof...(ExtSeqs) == sizeof...(StrSeqs) &&
                    sizeof...(ExtSeqs) == sizeof...(OrdSeqs),
                "nested StaticLayout: extents, strides and order must have the "
                "same number of modes");
  static_assert(((ExtSeqs::size() == OrdSeqs::size()) && ...),
                "nested StaticLayout: Order's nesting must mirror Extents'");
  static_assert(
      Impl::OrderSeqIsPermutation<Impl::concat_seq_t<OrdSeqs...>>::value,
      "nested StaticLayout: the flattened Order must be a permutation of "
      "[0, total_leaves)");

  // --- per-mode (leaf-level) accessors -------------------------------------
  // mode_extents/mode_strides return the mode's integer_sequence itself: every
  // value is compile-time, and the sequence TYPE is what a future reshape or
  // subview factory needs in order to build a new layout type.
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_arity() noexcept {
    return mode_t<static_cast<std::size_t>(M)>::arity;
  }
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto mode_extents() noexcept {
    return Impl::type_at_t<static_cast<std::size_t>(M), ExtSeqs...>{};
  }
  template <int M>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto mode_strides() noexcept {
    return Impl::type_at_t<static_cast<std::size_t>(M), StrSeqs...>{};
  }
  template <int M, int L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_extent() noexcept {
    return mode_t<static_cast<std::size_t>(M)>::extent_at(L);
  }
  template <int M, int L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_stride() noexcept {
    return mode_t<static_cast<std::size_t>(M)>::stride_at(L);
  }

  // --- encode / decode -----------------------------------------------------

  // multi-index -> flat offset (encode), one scalar index per mode.
  template <typename... I>
  KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t flat(
      I... idx) const noexcept {
    return flat_impl(std::index_sequence_for<I...>{}, idx...);
  }

  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset(
      const Impl::Index<rank>& coord) noexcept {
    return flat_offset_impl_(coord, std::make_index_sequence<rank>{});
  }

  // flat -> multi-index (decode). Recovers each leaf coordinate through the
  // leaf's *packed* stride — a linear index runs over [0, size()), which counts
  // elements, so a padded layout's gaps must not enter — then recombines the
  // leaves of a mode into that mode's collapsed index.
  KOKKOS_FORCEINLINE_FUNCTION auto operator[](int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<rank>{});
  }

 private:
  // The flattened leaf view, so the EXISTING PackedStrideAt computes the dense
  // per-leaf memory strides that decode needs — no new stride-walking logic.
  using flat_packed_ =
      Impl::PackedStrideAt<Impl::seq_to_tile_t<Impl::concat_seq_t<ExtSeqs...>>,
                           Impl::concat_seq_t<OrdSeqs...>>;

  // Global leaf index of (mode M, leaf L): the arities of all modes before M,
  // plus L.
  template <std::size_t M, std::size_t L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int global_leaf() noexcept {
    constexpr int a[] = {Impl::Mode<ExtSeqs, StrSeqs>::arity...};
    // Signed loop counter: `m < M` on an unsigned m warns for mode 0 (M == 0).
    int g = 0;
    for (int m = 0; m < static_cast<int>(M); ++m) g += a[m];
    return g + static_cast<int>(L);
  }

  template <std::size_t... Ms, typename... I>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t flat_impl(
      std::index_sequence<Ms...>, I... idx) noexcept {
    return (mode_t<Ms>::offset(static_cast<int>(idx)) + ... + std::size_t{0});
  }

  template <std::size_t... Ms>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int flat_offset_impl_(
      const Impl::Index<rank>& coord, std::index_sequence<Ms...>) noexcept {
    return (base_offset() + ... +
            static_cast<int>(mode_t<Ms>::offset(
                coord.template get<static_cast<int>(Ms)>())));
  }

  // Leaf L of mode M's contribution to that mode's collapsed index.
  template <std::size_t M, std::size_t L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int leaf_contrib(
      int linear) noexcept {
    constexpr int p = flat_packed_::get(global_leaf<M, L>());
    constexpr int e = mode_t<M>::extent_at(static_cast<int>(L));
    constexpr int d = mode_t<M>::divisor(static_cast<int>(L));
    return ((linear / p) % e) * d;
  }

  template <std::size_t M, std::size_t... L>
  KOKKOS_FORCEINLINE_FUNCTION static constexpr int mode_of(
      int linear, std::index_sequence<L...>) noexcept {
    return (leaf_contrib<M, L>(linear) + ... + 0);
  }

  template <std::size_t... Ms>
  KOKKOS_FORCEINLINE_FUNCTION static auto decode_impl(
      int linear, std::index_sequence<Ms...>) noexcept {
    return Impl::Index<rank>{
        mode_of<Ms>(linear, std::make_index_sequence<mode_t<Ms>::arity>{})...};
  }
};

// ---------------------------------------------------------------------------
// StaticTileLayoutRight — compile-time extents, row-major (rightmost fastest)
// ---------------------------------------------------------------------------
template <int... Extents>
struct StaticTileLayoutRight
    : Impl::static_layout_for_t<
          StaticTile<Extents...>,
          Impl::right_order_t<static_cast<int>(sizeof...(Extents))>> {};

// ---------------------------------------------------------------------------
// StaticTileLayoutLeft — compile-time extents, column-major (leftmost fastest)
// ---------------------------------------------------------------------------
template <int... Extents>
struct StaticTileLayoutLeft
    : Impl::static_layout_for_t<
          StaticTile<Extents...>,
          Impl::left_order_t<static_cast<int>(sizeof...(Extents))>> {};

// ---------------------------------------------------------------------------
// StaticTileLayoutStride — compile-time extents, arbitrary memory order
//
// Order[j] = dimension index at memory-order position j (fastest first).
// stride(Order[j]) = product of extents[Order[0]] * ... * extents[Order[j-1]]
//
// Reduces to StaticTileLayoutRight when Order = {N-1,...,0} and to
// StaticTileLayoutLeft when Order = {0,...,N-1} — literally: all three derive
// from the same StaticLayout for a given order. Useful for permuted operand
// tiles where the memory order is known at compile time.
// ---------------------------------------------------------------------------

template <typename ExtTile, int... Order>
struct StaticTileLayoutStride;  // primary declaration; specialised below

template <int... Extents, int... Order>
struct StaticTileLayoutStride<StaticTile<Extents...>, Order...>
    : Impl::static_layout_for_t<StaticTile<Extents...>,
                                std::integer_sequence<int, Order...>> {};

// ---------------------------------------------------------------------------
// DynamicTileLayoutRight — runtime extents, row-major (rightmost fastest)
// ---------------------------------------------------------------------------
template <int Rank>
struct DynamicTileLayoutRight : Impl::DynamicTileLayoutBase<Rank> {
  using Base = Impl::DynamicTileLayoutBase<Rank>;
  using Base::extents_;
  using Base::strides_;

  KOKKOS_FUNCTION DynamicTileLayoutRight() : Base() {}

  KOKKOS_FUNCTION explicit DynamicTileLayoutRight(Kokkos::Array<int, Rank> ext)
      : Base() {
    this->extents_           = ext;
    this->strides_[Rank - 1] = 1;
    for (int k = Rank - 2; k >= 0; --k)
      this->strides_[k] =
          this->strides_[k + 1] * static_cast<std::size_t>(ext[k + 1]);
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
};

// ---------------------------------------------------------------------------
// DynamicTileLayoutLeft — runtime extents, column-major (leftmost fastest)
// ---------------------------------------------------------------------------
template <int Rank>
struct DynamicTileLayoutLeft : Impl::DynamicTileLayoutBase<Rank> {
  using Base = Impl::DynamicTileLayoutBase<Rank>;
  using Base::extents_;
  using Base::strides_;

  KOKKOS_FUNCTION DynamicTileLayoutLeft() : Base() {}

  KOKKOS_FUNCTION explicit DynamicTileLayoutLeft(Kokkos::Array<int, Rank> ext)
      : Base() {
    this->extents_    = ext;
    this->strides_[0] = 1;
    for (int k = 1; k < Rank; ++k)
      this->strides_[k] =
          this->strides_[k - 1] * static_cast<std::size_t>(ext[k - 1]);
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
};

}  // namespace TensorOperations
