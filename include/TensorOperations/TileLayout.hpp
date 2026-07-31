#pragma once
#include <TensorOperations/Macros.hpp>
#include <array>
#include <cstddef>
#include <utility>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

// TENSOR_OPS_VERIFY_RESHAPE — compile-time static_assert that a generic
// reshape's emitted strides reproduce the source's element→offset map, element
// for element. Nothing else checks that: StaticLayout's own permutation and
// positivity static_asserts all pass on a wrong plan, so a planner bug would
// surface as silently wrong numerics rather than a compile error.
//
// The check is O(num_elements) constexpr work per instantiation, so it is a
// debug-build check: on when NDEBUG is absent or Kokkos' bounds checking is on
// (the same switch Tiling.hpp and TiledLayout.hpp already use), off in a
// release build. Define it to 0 or 1 on the command line to override either
// way.
//
// Lives here rather than in Macros.hpp because it reads a Kokkos macro, and
// Macros.hpp is included before <Kokkos_Macros.hpp> by both its consumers.
#if !defined(TENSOR_OPS_VERIFY_RESHAPE)
#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK) || !defined(NDEBUG)
#define TENSOR_OPS_VERIFY_RESHAPE 1
#else
#define TENSOR_OPS_VERIFY_RESHAPE 0
#endif
#endif

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
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator[](
      int linear) const noexcept {
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
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto decode_impl(
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
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto operator[](
      int linear) const noexcept {
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
  KOKKOS_FORCEINLINE_FUNCTION static constexpr auto decode_impl(
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

namespace Impl {

// ---------------------------------------------------------------------------
// Generic reshape: planner and materializer.
//
// Reinterprets a flat StaticLayout — packed or padded, any memory order — under
// a new shape and a new memory order, with no data movement. The result is a
// flat StaticLayout when every new mode maps to a single contiguous run of the
// source, and the nested (hierarchical) StaticLayout when one does not.
//
// SEMANTICS (numpy, stated over MEMORY order on both sides): the k-th element
// of the source in its own memory-order enumeration is the k-th element of the
// result in its memory-order enumeration. That reduces to the existing packed
// reshape(Right, tile) / reshape(Left, tile) overloads when the source and
// target orders match — which is exactly why the target order has to be given
// explicitly: Right lists new extents slowest-first and Left fastest-first, so
// no rule inferred from the extents alone reproduces both.
//
// EVERYTHING HERE IS COMPILE-TIME ONLY and deliberately NOT KOKKOS_FUNCTION.
// Marking a constexpr __host__ function __host__ __device__ trips nvcc warning
// #20013 when it is called from device-side constant evaluation, and would need
// --expt-relaxed-constexpr. A plan never exists at runtime, so host-only is
// both correct and warning-free. Do not "fix" this by adding KOKKOS_FUNCTION.
//
// The user-facing entry points are the reshape() overloads in Tiling.hpp and
// TiledLayout.hpp.
// ---------------------------------------------------------------------------

// Fixed capacities for the ragged plan below. A plan is a compile-time value,
// so it cannot allocate; exceeding either bound is reported as CapacityExceeded
// rather than silently truncated.
//
// Modes is 16 rather than 8 because tile_layout() reshapes a rank-N source into
// 2N modes: at 8 it would have capped tiling at rank 4, where the hand-rolled
// interleaving it replaced had no rank limit at all. With 16 leaves the real
// ceiling is a rank-8 source, whose tiling needs exactly 16 of each.
inline constexpr int kReshapeMaxModes  = 16;
inline constexpr int kReshapeMaxLeaves = 16;

enum class ReshapeStatus : int {
  Ok = 0,
  CountMismatch,     // product(new extents) != source element count
  BadOrder,          // the new order is not a permutation of [0, new rank)
                     // — including having the wrong number of entries
  NotFactorable,     // a new mode cannot be carved from the source's stream
  CapacityExceeded,  // more modes / leaves than the plan can hold
};

// ---------------------------------------------------------------------------
// ReshapePlan — the ragged descriptor the materializer turns into a layout
// type.
//
// `nmodes` logical modes, each backed by `arity[m]` memory leaves. Leaves are
// flattened MODE-MAJOR into ext/str (mode 0's leaves first), so mode m's leaves
// occupy [base(m), base(m) + arity[m]). `ord` holds global leaf indices listed
// in ascending MEMORY position — the flat form the nested StaticLayout wants.
// ---------------------------------------------------------------------------
struct ReshapePlan {
  ReshapeStatus status  = ReshapeStatus::Ok;
  int           nmodes  = 0;
  int           nleaves = 0;
  int           arity[kReshapeMaxModes]{};
  int           ext[kReshapeMaxLeaves]{};
  int           str[kReshapeMaxLeaves]{};
  int           ord[kReshapeMaxLeaves]{};

  constexpr bool ok() const noexcept { return status == ReshapeStatus::Ok; }

  // Global index of mode m's first leaf.
  constexpr int base(int m) const noexcept {
    int b = 0;
    for (int i = 0; i < m; ++i) b += arity[i];
    return b;
  }

  // True when the result collapses to a flat layout: one leaf per mode.
  constexpr bool all_arity_one() const noexcept {
    for (int m = 0; m < nmodes; ++m)
      if (arity[m] != 1) return false;
    return true;
  }
};

constexpr int reshape_gcd(int a, int b) noexcept {
  while (b) {
    const int t = a % b;
    a           = b;
    b           = t;
  }
  return a;
}

// order_is_permutation over an array rather than a pack, for the runtime-shaped
// arrays the planner works with.
template <int N>
constexpr bool reshape_order_is_permutation(const int (&O)[N]) noexcept {
  bool seen[N] = {};
  for (int j = 0; j < N; ++j) {
    if (O[j] < 0 || O[j] >= N || seen[O[j]]) return false;
    seen[O[j]] = true;
  }
  return true;
}

// ---------------------------------------------------------------------------
// make_reshape_plan — greedy split/merge factorization, one constexpr pass.
//
//   E/S/O  source extents, strides, and memory order (O[j] = the dimension at
//          memory position j, fastest-varying first)
//   F/NO   new extents, and the new memory order (NO[jj] = the new mode at
//          memory position jj, fastest first)
//
// Walks the source's memory-order mixed-radix stream and carves each new mode
// out of it in memory order. Each carve takes gcd(what the mode still needs,
// what is left of the current source dimension) as one leaf; a gcd of 1, or
// running out of source, means the target shape is not an affine
// reinterpretation of the source and the plan is rejected rather than
// approximated.
// ---------------------------------------------------------------------------
template <int N, int M, int MO>
constexpr ReshapePlan make_reshape_plan(const int (&E)[N], const int (&S)[N],
                                        const int (&O)[N], const int (&F)[M],
                                        const int (&NO)[MO]) noexcept {
  ReshapePlan p{};
  p.nmodes = M;

  if (M > kReshapeMaxModes) return ReshapePlan{ReshapeStatus::CapacityExceeded};

  // NO is validated FIRST and by the planner itself, not only by the caller's
  // static_assert: the loop below indexes per-mode arrays by NO[jj], and a
  // reshape's return type is computed before its body's static_asserts ever
  // run, so a wrong-length or out-of-range order would be a constexpr
  // out-of-bounds read — a hard error at the wrong place, with none of the
  // intended diagnostics.
  if (MO != M) return ReshapePlan{ReshapeStatus::BadOrder};
  if (!reshape_order_is_permutation(NO))
    return ReshapePlan{ReshapeStatus::BadOrder};

  // Checked up front so a mismatch reports as itself rather than as whatever
  // the carve loop happens to trip over first.
  {
    long long src = 1, dst = 1;
    for (int d = 0; d < N; ++d) src *= E[d];
    for (int m = 0; m < M; ++m) dst *= F[m];
    if (src != dst) return ReshapePlan{ReshapeStatus::CountMismatch};
  }

  // The source's memory-order stream: extents and strides, fastest axis first.
  int se[N]{}, ss[N]{};
  for (int j = 0; j < N; ++j) {
    se[j] = E[O[j]];
    ss[j] = S[O[j]];
  }

  // Emission log, in memory order: which (mode, leaf) each emission was.
  int em_mode[kReshapeMaxLeaves]{}, em_leaf[kReshapeMaxLeaves]{};
  int nem = 0;

  // Per-mode leaf staging.
  int mode_ext[kReshapeMaxModes][kReshapeMaxLeaves]{};
  int mode_str[kReshapeMaxModes][kReshapeMaxLeaves]{};

  int i = 0, rem_e = (N > 0 ? se[0] : 1), rem_s = (N > 0 ? ss[0] : 1);

  for (int jj = 0; jj < M; ++jj) {
    const int m    = NO[jj];
    int       need = F[m];

    if (need == 1) {  // degenerate mode: one leaf of extent 1
      mode_ext[m][0] = 1;
      mode_str[m][0] = 1;
      p.arity[m]     = 1;
      if (nem >= kReshapeMaxLeaves)
        return ReshapePlan{ReshapeStatus::CapacityExceeded};
      em_mode[nem] = m;
      em_leaf[nem] = 0;
      ++nem;
      continue;
    }

    while (need > 1) {
      if (rem_e == 1) {  // current source dimension exhausted -> advance
        ++i;
        if (i >= N) return ReshapePlan{ReshapeStatus::NotFactorable};
        rem_e = se[i];
        rem_s = ss[i];
        continue;
      }
      const int g = reshape_gcd(need, rem_e);
      if (g == 1) return ReshapePlan{ReshapeStatus::NotFactorable};

      const int l = p.arity[m];

      // Coalesce with the mode's previous leaf when this carve continues it
      // contiguously: a leaf (e0, s0) followed by (g, s0*e0) addresses exactly
      // the same elements as the single leaf (e0*g, s0). Without this, any
      // reshape that merges whole source dimensions -- collapsing a packed
      // rank-3 tile to a matrix, say -- would emit one leaf per source
      // dimension and land on the nested layout, when the flat one describes it
      // exactly. That costs stride(d), the TensorLike concept, and a divide and
      // modulo per access, all for nothing.
      //
      // Safe because a mode's leaves are emitted consecutively (each mode is
      // carved once, to completion), so merging two of them cannot disturb any
      // other mode's memory position; and the merged run is contiguous, so it
      // keeps the position of the earlier leaf. No new emission is logged.
      if (l > 0 && rem_s == mode_str[m][l - 1] * mode_ext[m][l - 1]) {
        mode_ext[m][l - 1] *= g;
        rem_s *= g;
        rem_e /= g;
        need /= g;
        continue;
      }

      if (l >= kReshapeMaxLeaves || nem >= kReshapeMaxLeaves)
        return ReshapePlan{ReshapeStatus::CapacityExceeded};
      mode_ext[m][l] = g;
      mode_str[m][l] = rem_s;
      p.arity[m]     = l + 1;
      em_mode[nem]   = m;
      em_leaf[nem]   = l;
      ++nem;

      // The rest of this source dimension carries on at a stride g times wider,
      // which is what lets a later mode pick up where this one stopped.
      rem_s *= g;
      rem_e /= g;
      need /= g;
    }
  }

  // Every source element must have been consumed. Trailing extent-1 dimensions
  // are fine; anything else means the carve stopped short.
  for (int j = i + 1; j < N; ++j)
    if (se[j] != 1) return ReshapePlan{ReshapeStatus::NotFactorable};
  if (rem_e != 1) return ReshapePlan{ReshapeStatus::NotFactorable};

  // Flatten the staged leaves MODE-MAJOR.
  p.nleaves = nem;
  int w     = 0;
  for (int m = 0; m < M; ++m)
    for (int l = 0; l < p.arity[m]; ++l) {
      p.ext[w] = mode_ext[m][l];
      p.str[w] = mode_str[m][l];
      ++w;
    }

  // Order: the global leaf index of each emission, in memory order.
  for (int q = 0; q < nem; ++q) p.ord[q] = p.base(em_mode[q]) + em_leaf[q];

  return p;
}

// ---------------------------------------------------------------------------
// ReshapePlanFor — carries a plan as a TYPE.
//
// A tag functor rather than a class-type non-type template parameter. Both
// forms compile under nvcc 13.2 for device, but this one does not require
// ReshapePlan to remain a structural type, so the three 16-int arrays never end
// up frozen into the mangled name of every intermediate alias.
// ---------------------------------------------------------------------------
template <typename SrcExt, typename SrcStr, typename SrcOrd, typename NewExt,
          typename NewOrd>
struct ReshapePlanFor;

template <int... E, int... S, int... O, int... F, int... NO>
struct ReshapePlanFor<StaticTile<E...>, std::integer_sequence<int, S...>,
                      std::integer_sequence<int, O...>, StaticTile<F...>,
                      std::integer_sequence<int, NO...>> {
  constexpr ReshapePlan operator()() const noexcept {
    constexpr int e[]  = {E...};
    constexpr int s[]  = {S...};
    constexpr int o[]  = {O...};
    constexpr int f[]  = {F...};
    constexpr int no[] = {NO...};
    return make_reshape_plan(e, s, o, f, no);
  }
};

// --- materializer ----------------------------------------------------------

enum class ReshapeField : int { Ext = 0, Str = 1, Ord = 2 };

constexpr int reshape_field(const ReshapePlan& p, ReshapeField f,
                            int i) noexcept {
  return f == ReshapeField::Ext
             ? p.ext[i]
             : (f == ReshapeField::Str ? p.str[i] : p.ord[i]);
}

// Mode M's leaves for one field, as an integer_sequence — the two-level pack
// expansion whose INNER length varies per mode.
//
// ext/str are flattened mode-major, so mode M's leaves are at base(M)+L. `ord`
// is a FLAT memory-position list and is sliced by those same arity groups,
// which looks wrong and is not: the nested StaticLayout only requires that the
// CONCATENATION of OrdSeqs be the memory-order list of global leaf indices (it
// consumes concat_seq_t<OrdSeqs...>, see flat_packed_ above), and that the
// grouping mirror the extents' arity shape. Slicing by arity satisfies both.
template <typename Fn, ReshapeField F, int M, typename Ls>
struct ReshapeModeSeq;
template <typename Fn, ReshapeField F, int M, std::size_t... L>
struct ReshapeModeSeq<Fn, F, M, std::index_sequence<L...>> {
  static constexpr ReshapePlan P = Fn{}();
  using type =
      std::integer_sequence<int, reshape_field(
                                     P, F, P.base(M) + static_cast<int>(L))...>;
};

template <typename Fn, ReshapeField F, int M>
using reshape_mode_seq_t = typename ReshapeModeSeq<
    Fn, F, M,
    std::make_index_sequence<static_cast<std::size_t>(Fn{}().arity[M])>>::type;

// A mode with more than one leaf: the nested specialisation.
template <typename Fn, typename Ms>
struct ReshapeNested;
template <typename Fn, std::size_t... Ms>
struct ReshapeNested<Fn, std::index_sequence<Ms...>> {
  using type = StaticLayout<
      DeviceTuple<
          reshape_mode_seq_t<Fn, ReshapeField::Ext, static_cast<int>(Ms)>...>,
      DeviceTuple<
          reshape_mode_seq_t<Fn, ReshapeField::Str, static_cast<int>(Ms)>...>,
      DeviceTuple<
          reshape_mode_seq_t<Fn, ReshapeField::Ord, static_cast<int>(Ms)>...>>;
};

// Every mode arity 1: global leaf index == mode index, so ext/str ARE the
// per-dimension extents and strides and ord IS the dimension order. Collapsing
// to the flat specialisation is not about speed (an arity-1 nested mode already
// compiles to exactly i*stride) — it is so the result keeps stride(d) and
// satisfies the TensorLike concept whenever it possibly can.
//
// A packed result additionally recovers the NAMED layout —
// StaticTileLayoutRight or StaticTileLayoutLeft rather than the StaticLayout
// they derive from. Without that, this overload would not be a drop-in superset
// of the packed two-argument ones: tile_layout() (and therefore tile_view()) is
// overloaded on the four named types only, so handing it a base-class result is
// a compile error, and the whole register-blocked GEMM path runs through
// tile_view. Returning the same type the two-argument overload returns for the
// same reshape also keeps the two families from disagreeing about what they
// describe.
template <typename Fn, typename Ms>
struct ReshapeFlat;
template <typename Fn, std::size_t... Ms>
struct ReshapeFlat<Fn, std::index_sequence<Ms...>> {
  static constexpr ReshapePlan P = Fn{}();

  static constexpr int rank_ = static_cast<int>(sizeof...(Ms));
  using tile_t               = StaticTile<P.ext[Ms]...>;
  using str_t                = std::integer_sequence<int, P.str[Ms]...>;
  using ord_t                = std::integer_sequence<int, P.ord[Ms]...>;

  // Packed means: the declared strides are the gap-free ones for this order.
  static constexpr bool packed_ =
      std::is_same_v<str_t, packed_strides_t<tile_t, ord_t>>;

  using type = std::conditional_t<
      packed_ && std::is_same_v<ord_t, right_order_t<rank_>>,
      StaticTileLayoutRight<P.ext[Ms]...>,
      std::conditional_t<packed_ && std::is_same_v<ord_t, left_order_t<rank_>>,
                         StaticTileLayoutLeft<P.ext[Ms]...>,
                         StaticLayout<tile_t, str_t, ord_t>>>;
};

template <typename T>
struct ReshapeIdentity {
  using type = T;
};

// Stand-in return type for a rejected plan. The reshape overload's body
// supplies the real diagnostic through a status-specific static_assert; this
// only has to be a well-formed type, never a used one. Without it, a rejected
// plan would reach StaticTileLayoutBase with zero modes and fail there instead,
// burying the intended message under a cascade.
using ReshapeErrorLayout =
    StaticLayout<StaticTile<1>, std::integer_sequence<int, 1>,
                 std::integer_sequence<int, 0>>;

// conditional_t selects between the metafunctions themselves; naming one does
// not instantiate it, only the chosen ::type does. That is what keeps a
// rejected plan from instantiating a layout it cannot describe.
template <typename Fn>
struct ReshapeResult {
  static constexpr ReshapePlan P = Fn{}();
  using modes = std::make_index_sequence<static_cast<std::size_t>(P.nmodes)>;
  using type  = typename std::conditional_t<
      !P.ok(), ReshapeIdentity<ReshapeErrorLayout>,
      std::conditional_t<P.all_arity_one(), ReshapeFlat<Fn, modes>,
                         ReshapeNested<Fn, modes>>>::type;
};

template <typename Fn>
using reshape_result_t = typename ReshapeResult<Fn>::type;

// A plan's status as a constant, reachable without CALLING anything.
//
// reshape() is KOKKOS_FORCEINLINE_FUNCTION, i.e. __host__ __device__, and the
// planner is deliberately host-only (see the note at the top of this section).
// Invoking it from reshape()'s body — even in a static_assert, where it is only
// ever constant-evaluated — trips nvcc warning #20013 and would need
// --expt-relaxed-constexpr. Reading a variable template instead does not: the
// evaluation happens here, at namespace scope, in host context.
template <typename Fn>
inline constexpr ReshapeStatus reshape_status_v = Fn{}().status;

template <typename Fn>
inline constexpr bool reshape_ok_v = reshape_status_v<Fn> == ReshapeStatus::Ok;

// The layout reshape(src, tile, order) returns, named directly from the source
// and target parameters. This is the spelling the reshape overloads use for
// their trailing return type.
template <typename SrcExt, typename SrcStr, typename SrcOrd, typename NewExt,
          typename NewOrd>
using reshaped_layout_t =
    reshape_result_t<ReshapePlanFor<SrcExt, SrcStr, SrcOrd, NewExt, NewOrd>>;

// --- verification (see TENSOR_OPS_VERIFY_RESHAPE at the top of this file) ----

// The memory offset of the k-th element of a layout's memory-order enumeration,
// computed straight from (extents, strides, order). Deliberately independent of
// StaticLayout's own encode/decode, so comparing a result against it is an
// oracle rather than a tautology.
template <int N>
constexpr int reshape_stream_offset(const int (&E)[N], const int (&S)[N],
                                    const int (&O)[N], int k) noexcept {
  int off = 0;
  for (int j = 0; j < N; ++j) {  // fastest axis first
    const int d = O[j];
    off += (k % E[d]) * S[d];
    k /= E[d];
  }
  return off;
}

// Does Result reproduce the source's element→offset map, element for element?
// Exercises the real runtime path — Result's decode and Mode::offset — not just
// the plan arrays.
template <typename Result, typename SrcExt, typename SrcStr, typename SrcOrd>
struct ReshapeReproducesSource;

template <typename Result, int... E, int... S, int... O>
struct ReshapeReproducesSource<Result, StaticTile<E...>,
                               std::integer_sequence<int, S...>,
                               std::integer_sequence<int, O...>> {
  static constexpr bool check() noexcept {
    constexpr int e[] = {E...};
    constexpr int s[] = {S...};
    constexpr int o[] = {O...};
    const int     n   = static_cast<int>(Result::num_elements);
    for (int k = 0; k < n; ++k)
      if (Result::flat_offset(Result{}[k]) != reshape_stream_offset(e, s, o, k))
        return false;
    return true;
  }
  static constexpr bool value = check();
};

// Runs the check only for an accepted plan. A rejected plan's result type is
// the ReshapeErrorLayout placeholder, which of course does not reproduce
// anything — checking it would bury the status-specific diagnostic under a
// second, wrong one. Partial specialisation rather than `||` because a
// disjunction would still instantiate the right-hand ::value.
template <bool Ok, typename Result, typename SrcExt, typename SrcStr,
          typename SrcOrd>
struct ReshapeVerifyIf {
  static constexpr bool value = true;
};
template <typename Result, typename SrcExt, typename SrcStr, typename SrcOrd>
struct ReshapeVerifyIf<true, Result, SrcExt, SrcStr, SrcOrd>
    : ReshapeReproducesSource<Result, SrcExt, SrcStr, SrcOrd> {};

}  // namespace Impl

}  // namespace TensorOperations
