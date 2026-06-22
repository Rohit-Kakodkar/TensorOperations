#pragma once
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include <TensorOperations/Tiling.hpp>

namespace TensorOperations {

// Forward declaration — layouts hold no friend references; all offset methods
// are public, so View needs no special access.
template <typename ViewType, typename Layout>
struct View;

// ---------------------------------------------------------------------------
// Impl helpers — pure arithmetic, unchanged.
// ---------------------------------------------------------------------------

namespace Impl {

// Selection sort returning dimension indices ordered by ascending stride
// (memory order): order[0] is the fastest-varying dim. Peeling dims in this
// order during delinearize makes consecutive flat indices walk contiguous
// memory. Shared by compute_tiled_layout and the subview factory.
template <int N>
KOKKOS_FUNCTION Kokkos::Array<int, N> argsort_by_stride(
    Kokkos::Array<int, N> strides) noexcept {
  Kokkos::Array<int, N> order{};
  for (int i = 0; i < N; ++i) order[i] = i;
  for (int i = 0; i < N; ++i) {
    int m = i;
    for (int j = i + 1; j < N; ++j)
      if (strides[order[j]] < strides[order[m]]) m = j;
    if (m != i) {
      const int t = order[i];
      order[i]    = order[m];
      order[m]    = t;
    }
  }
  return order;
}

// Per-dimension reciprocal 1/extent, for divide-free (reciprocal-multiply)
// decode on device.
template <int N>
KOKKOS_FUNCTION Kokkos::Array<float, N> reciprocals(
    Kokkos::Array<int, N> extents) noexcept {
  Kokkos::Array<float, N> inv{};
  for (int d = 0; d < N; ++d) inv[d] = 1.0f / static_cast<float>(extents[d]);
  return inv;
}

template <int N>
struct TiledLayoutResult {
  Kokkos::Array<int, N>   orig_extents;
  Kokkos::Array<int, N>   outer_extents;
  Kokkos::Array<int, N>   outer_strides;
  Kokkos::Array<int, N>   inner_strides;
  Kokkos::Array<int, N>   inner_order;
  Kokkos::Array<float, N> inner_inv_extents;
};

template <int N>
KOKKOS_FUNCTION TiledLayoutResult<N> compute_tiled_layout(
    Kokkos::Array<int, N> extents, Kokkos::Array<int, N> strides,
    Kokkos::Array<int, N> tile_sizes) {
  TiledLayoutResult<N>  r{};
  Kokkos::Array<int, N> inner_extents{};  // tile size clamped to orig extent
  for (int d = 0; d < N; ++d) {
    const int T       = tile_sizes[d] < extents[d] ? tile_sizes[d] : extents[d];
    inner_extents[d]  = T;
    r.orig_extents[d] = extents[d];
    r.outer_extents[d] = (extents[d] + T - 1) / T;
    r.outer_strides[d] = T * strides[d];
    r.inner_strides[d] = strides[d];
  }
  r.inner_order       = argsort_by_stride<N>(r.inner_strides);
  r.inner_inv_extents = reciprocals<N>(inner_extents);
  return r;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// TiledLayout<N, TileLayoutT>
//
// All layout parameters for an N-dimensional tiled view: outer tile counts,
// outer and inner (element) strides, and original extents, plus the inner-tile
// extents/decode delegated to a TileLayoutT (StaticTileLayoutRight /
// DynamicTileLayoutRight from TileLayout.hpp). Mind the naming: *TileLayout*
// encodes one tile's inner coordinates; *TiledLayout* (this) is the full
// tiled-view layout that embeds one.
//
// Public interface:
//   extent(d)      — outer tile count for d < N; clamped tile size for d >= N
//   stride(d)      — outer stride for d < N; element stride for d >= N
//   operator[](i)  — row-major delinearize: flat inner index → N-D tile coord
//                    (delegated to the embedded TileLayoutT)
//   offset(index_sequence<Is...>, args) — full 2N-index flat offset
//   base_offset()  — always 0 (tiled views start at the backing data pointer)
//   size()         — product of all 2N extents
//   check_bounds(args) — debug bounds check over all N outer/inner index pairs
// ---------------------------------------------------------------------------

template <int N, typename TileLayoutT>
struct TiledLayout {
 public:
  static constexpr int rank = 2 * N;
  using tile_layout_t       = TileLayoutT;

  KOKKOS_FUNCTION TiledLayout(tile_layout_t                     tile_layout,
                              const Impl::TiledLayoutResult<N>& r) noexcept
      : tile_layout_(tile_layout),
        orig_extents_(r.orig_extents),
        outer_extents_(r.outer_extents),
        outer_strides_(r.outer_strides),
        inner_strides_(r.inner_strides),
        inner_order_(r.inner_order),
        inner_inv_extents_(r.inner_inv_extents) {}

  // extent(d): outer tile count for d < N; clamped tile size for d >= N.
  KOKKOS_FUNCTION int extent(int d) const noexcept {
    if (d < N) return outer_extents_[d];
    const int T = tile_layout_.extent(d - N);
    const int E = orig_extents_[d - N];
    return T < E ? T : E;
  }

  // stride(d): outer stride for d < N; element (inner) stride for d >= N.
  KOKKOS_FUNCTION int stride(int d) const noexcept {
    return d < N ? outer_strides_[d] : inner_strides_[d - N];
  }

  // Delinearize: row-major flat inner index → N-D inner tile coordinate.
  KOKKOS_FUNCTION Kokkos::Array<int, N> operator[](int linear) const noexcept {
    return tile_layout_[linear];
  }

  // Host-precomputed memory-order permutation and reciprocals, forwarded by
  // subview_tile into a SubviewLayout (see SubviewLayout::operator[]).
  KOKKOS_FUNCTION const Kokkos::Array<int, N>& inner_order() const noexcept {
    return inner_order_;
  }
  KOKKOS_FUNCTION const Kokkos::Array<float, N>& inner_inv_extents()
      const noexcept {
    return inner_inv_extents_;
  }

  KOKKOS_FUNCTION static constexpr int base_offset() noexcept { return 0; }

  KOKKOS_FUNCTION int size() const noexcept {
    int s = 1;
    for (int d = 0; d < 2 * N; ++d) s *= extent(d);
    return s;
  }

  // Bounds check over all N outer/inner index pairs; no-op without debug flag.
  // Takes the 2N-index coordinate array directly (not a tuple): nvcc
  // miscompiles the std::get/fold tuple path on device (see
  // View::flat_offset_).
  KOKKOS_FUNCTION void check_bounds(
      const Kokkos::Array<int, 2 * N>& coord) const {
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    for (int i = 0; i < N; ++i)
      if (coord[i] * tile_layout_.extent(i) + coord[N + i] >= orig_extents_[i])
        Kokkos::abort("TiledView: out-of-bounds access");
#else
    (void)coord;
#endif
  }

 private:
  tile_layout_t           tile_layout_;
  Kokkos::Array<int, N>   orig_extents_;
  Kokkos::Array<int, N>   outer_extents_;
  Kokkos::Array<int, N>   outer_strides_;
  Kokkos::Array<int, N>   inner_strides_;
  Kokkos::Array<int, N>   inner_order_;
  Kokkos::Array<float, N> inner_inv_extents_;
};

// ---------------------------------------------------------------------------
// SubviewLayout<N>
//
// Layout parameters for a non-owning slice: flat base offset into the
// backing storage, per-dimension extents, and per-dimension strides.
//
// Public interface:
//   extent(d)         — extent of free dimension d
//   stride(d)         — stride of free dimension d
//   size()            — product of all extents
//   unit_stride_dim() — first free dim with unit stride, or -1
//   base_offset()     — flat offset of element [0,...,0] from backing data
//   offset(index_sequence<Is...>, args) — full flat offset from backing data
//   operator[](i)     — memory-aligned delinearize: peels the smallest-stride
//                       (fastest-varying) dimension first so consecutive flat
//                       indices walk contiguous memory (coalesced reads),
//                       regardless of input layout. Divide-free: uses the
//                       precomputed reciprocals (see ctor).
//
// Non-overlapping strides assumed.
// ---------------------------------------------------------------------------

template <int N>
struct SubviewLayout {
 public:
  static constexpr int rank = N;

  KOKKOS_FUNCTION SubviewLayout(int base, Kokkos::Array<int, N> ext,
                                Kokkos::Array<int, N>   str,
                                Kokkos::Array<int, N>   order,
                                Kokkos::Array<float, N> inv_ext) noexcept
      : base_offset_(base),
        extents_(ext),
        strides_(str),
        order_(order),
        inv_extents_(inv_ext) {}

  KOKKOS_FUNCTION int extent(int d) const noexcept { return extents_[d]; }
  KOKKOS_FUNCTION int stride(int d) const noexcept { return strides_[d]; }

  KOKKOS_FUNCTION int size() const noexcept {
    int s = 1;
    for (int d = 0; d < N; ++d) s *= extents_[d];
    return s;
  }

  // Index of the first free dimension with unit stride (the contiguous "run"
  // direction), or -1 if none. Used by the staging fast path to walk that
  // dimension innermost with sequential memory reads.
  KOKKOS_FUNCTION int unit_stride_dim() const noexcept {
    for (int d = 0; d < N; ++d)
      if (strides_[d] == 1) return d;
    return -1;
  }

  KOKKOS_FUNCTION int base_offset() const noexcept { return base_offset_; }

  // Memory-aligned delinearize: flat index → N-D coordinate, peeling the
  // smallest-stride dimension first so consecutive flat indices walk contiguous
  // memory. Divide/modulo are replaced by a reciprocal multiply (see ctor) for
  // fixed-cost, divergence-free decode on device.
  KOKKOS_FUNCTION Kokkos::Array<int, N> operator[](int linear) const noexcept {
    Kokkos::Array<int, N> idx{};
    for (int j = 0; j < N; ++j) {
      const int d = order_[j];
      const int e = extents_[d];
      int q = static_cast<int>(static_cast<float>(linear) * inv_extents_[d]);
      int r = linear - q * e;
      if (r < 0) {
        r += e;
        --q;
      } else if (r >= e) {
        r -= e;
        ++q;
      }
      idx[d] = r;
      linear = q;
    }
    return idx;
  }

 private:
  int                   base_offset_;
  Kokkos::Array<int, N> extents_;
  Kokkos::Array<int, N> strides_;
  Kokkos::Array<int, N> order_;  // dims by stride ascending (memory order)
  Kokkos::Array<float, N>
      inv_extents_;  // 1/extents_[d], for divide-free decode
};

// ---------------------------------------------------------------------------
// View<ViewType, Layout>
//
// Generic view backed by a ViewType (any type exposing .data()) and indexed
// by a Layout that provides:
//   rank                                — static constexpr int
//   extent(d)                           — logical extent of dimension d
//   stride(d)                           — stride of dimension d
//   base_offset()                       — flat offset of [0,...,0] from data()
//   size()                              — product of all extents
//   operator[](int)                     — flat → multi-index delinearize
//
// The element offset for a multi-index is computed by View as
// base_offset() + Σ coord[d]·stride(d) (see View::flat_offset_), so a layout
// only needs base_offset()/stride(d), not a variadic offset() helper.
//
// TiledLayout<N,TL>  → rank = 2N  (outer tile coords then inner offsets)
// SubviewLayout<N>   → rank = N   (free dimensions of a subview)
// ---------------------------------------------------------------------------

template <typename ViewType, typename Layout>
struct View {
  static constexpr int rank = Layout::rank;
  using value_type          = typename ViewType::value_type;
  using layout_t            = Layout;

  ViewType backing_;
  Layout   layout_;

  KOKKOS_FUNCTION int extent(int d) const noexcept { return layout_.extent(d); }
  KOKKOS_FUNCTION int stride(int d) const noexcept { return layout_.stride(d); }
  KOKKOS_FUNCTION value_type* data() const noexcept {
    return backing_.data() + layout_.base_offset();
  }
  KOKKOS_FUNCTION const Layout& layout() const noexcept { return layout_; }
  KOKKOS_FUNCTION int           size() const noexcept { return layout_.size(); }

  // unit_stride_dim() forwarded to layout when available (SubviewLayout only).
  KOKKOS_FUNCTION int unit_stride_dim() const noexcept
    requires requires { layout_.unit_stride_dim(); }
  {
    return layout_.unit_stride_dim();
  }

  // Flat element offset from backing_.data() for a multi-index. Every layout's
  // offset() equals base_offset() + Σ coord[d]·stride(d), so this stride loop
  // is equivalent. It is used instead of the layout's variadic
  // offset(index_sequence, tuple) helper because nvcc (CUDA 13.x) miscompiles
  // that std::forward_as_tuple/std::get/fold path for these views on device,
  // yielding a wrong offset (garbage reads, faulting or misplaced writes).
  KOKKOS_FUNCTION int flat_offset_(
      const Kokkos::Array<int, rank>& coord) const {
    int off = layout_.base_offset();
    for (int d = 0; d < rank; ++d) off += coord[d] * layout_.stride(d);
    return off;
  }

  template <typename... Idx>
    requires(sizeof...(Idx) == rank)
  KOKKOS_FUNCTION value_type& operator()(Idx... idx) const {
    const Kokkos::Array<int, rank> coord{static_cast<int>(idx)...};
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    if constexpr (requires { layout_.check_bounds(coord); })
      layout_.check_bounds(coord);
#endif
    return backing_.data()[flat_offset_(coord)];
  }

  KOKKOS_FUNCTION value_type& operator[](Kokkos::Array<int, rank> idx) const {
    return backing_.data()[flat_offset_(idx)];
  }
};

// ---------------------------------------------------------------------------
// TiledView<ViewType, Tile>
//
// Convenience alias: a View backed by TiledLayout<N, TileLayoutT>, where N and
// TileLayoutT are derived from Tile (StaticTile<E...> or DynamicTile<N>).
// operator() accepts 2N arguments: N outer tile coordinates then N inner
// offsets.
// ---------------------------------------------------------------------------

template <typename ViewType, typename Tile>
using TiledView =
    View<ViewType, TiledLayout<Tile::rank, decltype(make_tile_layout(
                                               std::declval<Tile>()))>>;

// ---------------------------------------------------------------------------
// Subview<ViewType, FreeRank>
//
// Convenience alias: a View backed by SubviewLayout<FreeRank>.
// FreeRank = number of Kokkos::ALL dimensions in the subview call.
// backing_ is stored by value (Kokkos::View has shallow-copy semantics).
// ---------------------------------------------------------------------------

template <typename ViewType, int FreeRank>
using Subview = View<ViewType, SubviewLayout<FreeRank>>;

// ---------------------------------------------------------------------------
// ScratchView<ValueType, ExecSpace, Layout>
//
// Convenience alias: a View backed by a Layout and allocated in the scratch
// memory space of an execution space. Used for tiled staging and reduction
// accumulation.
// ---------------------------------------------------------------------------

template <typename ValueType, typename ExecSpace, typename Layout>
using ScratchView =
    View<Kokkos::View<ValueType*, typename ExecSpace::scratch_memory_space,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
         Layout>;

// ---------------------------------------------------------------------------
// Impl helpers for subview
// ---------------------------------------------------------------------------

namespace Impl {

template <typename T>
inline constexpr bool is_all_v = std::is_same_v<std::decay_t<T>, Kokkos::ALL_t>;

template <typename... Slices>
inline constexpr int free_rank_v = static_cast<int>((is_all_v<Slices> + ...));

}  // namespace Impl

// ---------------------------------------------------------------------------
// tile_view — factory: Kokkos::View × Tile → TiledView
//
// One overload for both tile kinds: the inner TileLayout is selected by
// make_tile_layout(tile), and tile.extent(d) reads tile sizes uniformly
// (StaticTile's extent is static constexpr, DynamicTile's a member).
// ---------------------------------------------------------------------------

template <typename ViewType, typename Tile>
KOKKOS_FUNCTION auto tile_view(const ViewType& view, Tile tile)
    -> TiledView<ViewType, Tile> {
  constexpr int N = Tile::rank;
  using LT        = typename TiledView<ViewType, Tile>::layout_t;

  Kokkos::Array<int, N> extents{}, strides{}, tsizes{};
  for (int d = 0; d < N; ++d) {
    extents[d] = static_cast<int>(view.extent(d));
    strides[d] = static_cast<int>(view.stride(d));
    tsizes[d]  = tile.extent(d);
  }

  const auto r = Impl::compute_tiled_layout<N>(extents, strides, tsizes);
  return {view, LT{make_tile_layout(tile), r}};
}

// ---------------------------------------------------------------------------
// subview — factory: TiledView × mixed (int|Kokkos::ALL) → Subview
//
// One slice argument per dimension of the TiledView (2N total).
// int  → fix that dimension, contributing int*stride to base_offset.
// ALL  → keep that dimension free in the returned Subview.
// ---------------------------------------------------------------------------

template <typename ViewType, int N, typename TL, typename... Slices>
KOKKOS_FUNCTION auto subview(const View<ViewType, TiledLayout<N, TL>>& tv,
                             Slices... slices) {
  constexpr int Full = View<ViewType, TiledLayout<N, TL>>::rank;
  static_assert(sizeof...(Slices) == Full,
                "subview: one slice argument per tiled dimension required");
  constexpr int Free = Impl::free_rank_v<Slices...>;

  int                      base_offset = 0;
  Kokkos::Array<int, Free> extents{};
  Kokkos::Array<int, Free> strides{};
  int                      free_d = 0;

  auto slice_tuple = std::forward_as_tuple(slices...);
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (
        [&]<std::size_t I>() {
          auto& s       = std::get<I>(slice_tuple);
          using S       = std::decay_t<decltype(s)>;
          const int dim = static_cast<int>(I);
          if constexpr (Impl::is_all_v<S>) {
            extents[free_d] = tv.extent(dim);
            strides[free_d] = tv.stride(dim);
            ++free_d;
          } else {
            base_offset += static_cast<int>(s) * tv.stride(dim);
          }
        }.template operator()<Is>(),
        ...);
  }(std::make_index_sequence<Full>{});

  const auto order       = Impl::argsort_by_stride<Free>(strides);
  const auto inv_extents = Impl::reciprocals<Free>(extents);

  return Subview<ViewType, Free>{
      tv.backing_,
      SubviewLayout<Free>{base_offset, extents, strides, order, inv_extents}};
}

// ---------------------------------------------------------------------------
// subview_tile — factory: TiledView × outer coordinate array → Subview
//
// Fixes all N outer dims; keeps all N inner dims free.
// The returned Subview has FreeRank == N (all inner dimensions are free).
// order_ and inv_extents_ are forwarded directly from TiledLayout — no
// per-call sorting or reciprocal computation.
// ---------------------------------------------------------------------------

template <typename ViewType, int N, typename TL>
KOKKOS_FUNCTION auto subview_tile(
    const View<ViewType, TiledLayout<N, TL>>&       tv,
    Kokkos::Array<int, static_cast<std::size_t>(N)> outer_idx) {
  int base_offset = 0;
  for (int d = 0; d < N; ++d) base_offset += outer_idx[d] * tv.stride(d);
  Kokkos::Array<int, N> extents{}, strides{};
  for (int d = 0; d < N; ++d) {
    extents[d] = tv.extent(N + d);
    strides[d] = tv.stride(N + d);
  }
  return Subview<ViewType, N>{
      tv.backing_,
      SubviewLayout<N>{base_offset, extents, strides, tv.layout_.inner_order(),
                       tv.layout_.inner_inv_extents()}};
}

// ---------------------------------------------------------------------------
// reshape — reinterpret a View under a new shape given by a Tile.
//
// Delegates to reshape(layout, tile) (defined in Tiling.hpp) to produce the
// new layout, then wraps it with the same backing storage. The requires clause
// constrains the overload to layout types that have a reshape overload,
// producing a clear compile error for unsupported layouts (SubviewLayout,
// TiledLayout) rather than a cryptic substitution failure.
//
//   reshape(View<VT, StaticTileLayoutRight<32>>{...}, StaticTile<4, 8>{})
//       -> View<VT, StaticTileLayoutRight<4, 8>>
//   reshape(View<VT, DynamicTileLayoutLeft<1>>{...},  DynamicTile<2>{{4,8}})
//       -> View<VT, DynamicTileLayoutLeft<2>>
// ---------------------------------------------------------------------------

template <typename ViewType, typename Layout, typename Tile>
  requires(requires(Layout l, Tile t) { reshape(l, t); })
KOKKOS_FUNCTION auto reshape(const View<ViewType, Layout>& view,
                             const Tile&                   tile)
    -> View<ViewType, decltype(reshape(view.layout(), tile))> {
  return {view.backing_, reshape(view.layout(), tile)};
}

}  // namespace TensorOperations
