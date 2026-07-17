#pragma once
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include <TensorOperations/Permute.hpp>
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
KOKKOS_FUNCTION constexpr Kokkos::Array<float, N> reciprocals(
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

  KOKKOS_FUNCTION int flat_offset(
      const Impl::Index<2 * N>& coord) const noexcept {
    int off = base_offset();
    for (int d = 0; d < 2 * N; ++d) off += coord[d] * stride(d);
    return off;
  }

  // Bounds check over all N outer/inner index pairs; no-op without debug flag.
  // Takes the 2N-index coordinate array directly (not a tuple): nvcc
  // miscompiles the std::get/fold tuple path on device (see
  // View::flat_offset_).
  KOKKOS_FUNCTION void check_bounds(const Impl::Index<2 * N>& coord) const {
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

  KOKKOS_FUNCTION constexpr int extent(int d) const noexcept {
    return extents_[d];
  }
  KOKKOS_FUNCTION constexpr int stride(int d) const noexcept {
    return strides_[d];
  }

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

  KOKKOS_FUNCTION constexpr int base_offset() const noexcept {
    return base_offset_;
  }

  KOKKOS_FUNCTION int flat_offset(const Impl::Index<N>& coord) const noexcept {
    int off = base_offset_;
    for (int d = 0; d < N; ++d) off += coord[d] * strides_[d];
    return off;
  }

  // Memory-aligned delinearize: flat index → N-D coordinate, peeling the
  // smallest-stride dimension first so consecutive flat indices walk contiguous
  // memory. Divide/modulo are replaced by a reciprocal multiply (see ctor) for
  // fixed-cost, divergence-free decode on device.
  KOKKOS_FUNCTION Impl::Index<N> operator[](int linear) const noexcept {
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
    return Impl::Index<N>{idx};
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
// OrderedSubviewLayout<N, int... Order>
//
// Like SubviewLayout<N> but with the memory-order permutation baked in as
// compile-time template parameters. When all array subscripts in operator[]
// and flat_offset are compile-time constants, CUDA can keep every per-element
// array in registers instead of spilling to per-thread local memory.
//
// Order[j] = the logical dimension at memory-order position j (fastest first).
// For LayoutRight/C order: Order = {N-1, ..., 1, 0}
// For LayoutLeft/Fortran:  Order = {0, 1, ..., N-1}
// ---------------------------------------------------------------------------

template <int N, int... Order>
struct OrderedSubviewLayout {
  static_assert(sizeof...(Order) == N, "Order must have exactly N elements");
  static constexpr int rank = N;

  KOKKOS_FUNCTION constexpr OrderedSubviewLayout(
      int base, Kokkos::Array<int, N> ext, Kokkos::Array<int, N> str,
      Kokkos::Array<float, N> inv_ext) noexcept
      : base_offset_(base),
        extents_(ext),
        strides_(str),
        inv_extents_(inv_ext) {}

  KOKKOS_FUNCTION constexpr int extent(int d) const noexcept {
    return extents_[d];
  }
  KOKKOS_FUNCTION constexpr int stride(int d) const noexcept {
    return strides_[d];
  }

  KOKKOS_FUNCTION int size() const noexcept {
    int s = 1;
    for (int d = 0; d < N; ++d) s *= extents_[d];
    return s;
  }

  KOKKOS_FUNCTION int unit_stride_dim() const noexcept {
    constexpr int arr[] = {Order...};
    return arr[0];  // Order[0] is the fastest-varying dimension
  }

  KOKKOS_FUNCTION constexpr int base_offset() const noexcept {
    return base_offset_;
  }

  KOKKOS_FUNCTION int flat_offset(const Impl::Index<N>& coord) const noexcept {
    return flat_offset_impl_(coord, std::make_index_sequence<N>{});
  }

  // Memory-aligned delinearize: compile-time Order... ensures every array
  // subscript is a compile-time constant → all per-element arrays stay in
  // registers (no local memory spill).
  KOKKOS_FUNCTION Impl::Index<N> operator[](int linear) const noexcept {
    return decode_impl(linear, std::make_index_sequence<N>{});
  }

 private:
  template <std::size_t... Ds>
  KOKKOS_FORCEINLINE_FUNCTION int flat_offset_impl_(
      const Impl::Index<N>& coord, std::index_sequence<Ds...>) const noexcept {
    return (base_offset_ + ... +
            (coord.template get<static_cast<int>(Ds)>() *
             strides_[static_cast<int>(Ds)]));
  }

  template <std::size_t... J>
  KOKKOS_FORCEINLINE_FUNCTION Impl::Index<N> decode_impl(
      int linear, std::index_sequence<J...>) const noexcept {
    Impl::Index<N> idx{};
    // Non-template lambda + pack expansion: J is a compile-time constant at
    // each expansion step, so constexpr d = arr[J] is also compile-time.
    // Every array access in the lambda body therefore uses a compile-time
    // index, allowing CUDA to keep idx.data and the extents/inv_extents in
    // registers.
    (
        [&] {
          constexpr int arr[] = {Order...};
          constexpr int d     = arr[static_cast<int>(J)];
          const int     e     = extents_[d];
          int           q =
              static_cast<int>(static_cast<float>(linear) * inv_extents_[d]);
          int r = linear - q * e;
          if (r < 0) {
            r += e;
            --q;
          } else if (r >= e) {
            r -= e;
            ++q;
          }
          idx.data[d] = r;  // compile-time d → register write
          linear      = q;
        }(),
        ...);
    return idx;
  }

  int                     base_offset_;
  Kokkos::Array<int, N>   extents_;
  Kokkos::Array<int, N>   strides_;
  Kokkos::Array<float, N> inv_extents_;
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

  // Delegates to layout_.flat_offset(). Kept as a separate method rather than
  // inlining a fold/variadic path because nvcc (CUDA 13.x) miscompiles
  // std::forward_as_tuple/std::get folds on device, yielding wrong offsets.
  KOKKOS_FUNCTION int flat_offset_(const Impl::Index<rank>& coord) const {
    return layout_.flat_offset(coord);
  }

  template <typename... Idx>
    requires(sizeof...(Idx) == rank)
  KOKKOS_FUNCTION value_type& operator()(Idx... idx) const {
    const Impl::Index<rank> coord{static_cast<int>(idx)...};
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    if constexpr (requires { layout_.check_bounds(coord); })
      layout_.check_bounds(coord);
#endif
    return backing_.data()[flat_offset_(coord)];
  }

  KOKKOS_FUNCTION value_type& operator[](Impl::Index<rank> idx) const {
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
    View<ViewType,
         TiledLayout<Tile::rank, decltype(make_tile_layout(std::declval<Tile>(),
                                                           LayoutRight{}))>>;

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
// GlobalView<ViewType, Layout>
//
// Semantic alias for View<ViewType, Layout> used when ViewType is backed by
// global (not scratch) memory — e.g. an operand's native TensorHandle view,
// reinterpreted under a tile's Layout with no data movement. Purely
// documentary: distinguishes "a View over global memory" from ScratchView at
// a call site; it is not a distinct type.
// ---------------------------------------------------------------------------

template <typename ViewType, typename Layout>
using GlobalView = View<ViewType, Layout>;

// ---------------------------------------------------------------------------
// Impl helpers for subview_tile
// ---------------------------------------------------------------------------

namespace Impl {

// Unpack integer_sequence<int, Order...> into OrderedSubviewLayout template
// args.  std::size_t M matches Kokkos::Array's size_t size param to avoid
// template deduction failure (int vs size_t).
template <typename VT, std::size_t M, int... Order>
KOKKOS_FUNCTION auto make_ordered_subview(
    VT backing, int base, Kokkos::Array<int, M> ext, Kokkos::Array<int, M> str,
    Kokkos::Array<float, M> inv, std::integer_sequence<int, Order...>) {
  static_assert(M == sizeof...(Order));
  constexpr int N = static_cast<int>(M);
  return View<VT, OrderedSubviewLayout<N, Order...>>{
      backing, OrderedSubviewLayout<N, Order...>{base, ext, str, inv}};
}

// Compile-time order sequences for LayoutRight {N-1,...,0} and LayoutLeft
// {0,...,N-1}.
template <int N, std::size_t... J>
constexpr auto right_order_seq(std::index_sequence<J...>) {
  return std::integer_sequence<int, (N - 1 - static_cast<int>(J))...>{};
}
template <int N, std::size_t... J>
constexpr auto left_order_seq(std::index_sequence<J...>) {
  return std::integer_sequence<int, static_cast<int>(J)...>{};
}

// Layout-only sibling of make_ordered_subview: unpack integer_sequence<int,
// Order...> into OrderedSubviewLayout template args and return the layout (no
// backing / View). Used by reorder_layout.
template <std::size_t M, int... Order>
KOKKOS_FUNCTION constexpr auto make_ordered_subview_layout(
    int base, Kokkos::Array<int, M> ext, Kokkos::Array<int, M> str,
    Kokkos::Array<float, M> inv, std::integer_sequence<int, Order...>) {
  static_assert(M == sizeof...(Order));
  constexpr int N = static_cast<int>(M);
  return OrderedSubviewLayout<N, Order...>{base, ext, str, inv};
}

// Remap a compile-time memory order under a gather permutation:
// NewOrder[j] = inverse(Perm)[Order[j]], the new position of old dim Order[j]
// (inverse_perm_array comes from the einsum plan helpers in Permute.hpp).
template <int... Order, typename PermSeq>
constexpr auto reorder_order_seq(std::integer_sequence<int, Order...>,
                                 PermSeq) {
  constexpr auto inv = inverse_perm_array<PermSeq>();
  return std::integer_sequence<int, inv[Order]...>{};
}

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
  return {view, LT{make_tile_layout(tile, LayoutRight{}), r}};
}

// ---------------------------------------------------------------------------
// tile_view overload for View<ViewType, TileLayout> — same backing data,
// 2N-dimensional tiled layout via tile_layout(src_layout, tile).
//
// Distinct from the Kokkos::View overload above: that one creates a
// TiledLayout (complex outer/inner structure). This one produces a flat
// StaticTileLayout or DynamicTileLayout from tile_layout(), preserving all
// compile-time information when both the source layout and tile are static.
//
// Enabled only when tile_layout(Layout, Tile) is valid, i.e. Layout is one
// of the four TileLayout types (not TiledLayout, SubviewLayout, etc.).
// ---------------------------------------------------------------------------

template <typename ViewType, typename Layout, typename Tile>
  requires(requires(Layout l, Tile t) { tile_layout(l, t); })
KOKKOS_FUNCTION auto tile_view(const View<ViewType, Layout>& view, Tile tile)
    -> View<ViewType, decltype(tile_layout(view.layout(), tile))> {
  return {view.backing_, tile_layout(view.layout(), tile)};
}

// ---------------------------------------------------------------------------
// subview_tile — factory: TiledView × outer coordinate array → Subview
//
// Three overloads, resolved on ViewType::array_layout:
//   LayoutRight  → OrderedSubviewLayout with compile-time order {N-1,...,0}
//   LayoutLeft   → OrderedSubviewLayout with compile-time order {0,...,N-1}
//   everything else → runtime SubviewLayout (LayoutStride or unknown)
//
// The two specialised overloads eliminate local memory spill: every array
// subscript in OrderedSubviewLayout::operator[] is a compile-time constant, so
// CUDA keeps idx.data and the extents/inv_extents in registers.
// ---------------------------------------------------------------------------

namespace Impl {
// Shared setup for all subview_tile overloads: flat base offset of the
// selected outer tile plus the inner extents and strides.
template <int N, typename VT, typename TL>
KOKKOS_FUNCTION auto subview_tile_params(
    const View<VT, TiledLayout<N, TL>>&                    tv,
    const Kokkos::Array<int, static_cast<std::size_t>(N)>& outer_idx) {
  struct Params {
    int                   base;
    Kokkos::Array<int, N> ext;
    Kokkos::Array<int, N> str;
  };
  Params p{0, {}, {}};
  for (int d = 0; d < N; ++d) p.base += outer_idx[d] * tv.stride(d);
  for (int d = 0; d < N; ++d) {
    p.ext[d] = tv.extent(N + d);
    p.str[d] = tv.stride(N + d);
  }
  return p;
}
}  // namespace Impl

// LayoutRight specialization.
template <typename VT, int N, typename TL>
  requires std::is_same_v<typename VT::array_layout, Kokkos::LayoutRight>
KOKKOS_FUNCTION auto subview_tile(
    const View<VT, TiledLayout<N, TL>>&             tv,
    Kokkos::Array<int, static_cast<std::size_t>(N)> outer_idx) {
  const auto p = Impl::subview_tile_params(tv, outer_idx);
  return Impl::make_ordered_subview(
      tv.backing_, p.base, p.ext, p.str, Impl::reciprocals<N>(p.ext),
      Impl::right_order_seq<N>(std::make_index_sequence<N>{}));
}

// LayoutLeft specialization.
template <typename VT, int N, typename TL>
  requires std::is_same_v<typename VT::array_layout, Kokkos::LayoutLeft>
KOKKOS_FUNCTION auto subview_tile(
    const View<VT, TiledLayout<N, TL>>&             tv,
    Kokkos::Array<int, static_cast<std::size_t>(N)> outer_idx) {
  const auto p = Impl::subview_tile_params(tv, outer_idx);
  return Impl::make_ordered_subview(
      tv.backing_, p.base, p.ext, p.str, Impl::reciprocals<N>(p.ext),
      Impl::left_order_seq<N>(std::make_index_sequence<N>{}));
}

// Generic fallback for LayoutStride / unknown layouts: runtime SubviewLayout.
template <typename VT, int N, typename TL>
KOKKOS_FUNCTION auto subview_tile(
    const View<VT, TiledLayout<N, TL>>&             tv,
    Kokkos::Array<int, static_cast<std::size_t>(N)> outer_idx) {
  const auto p = Impl::subview_tile_params(tv, outer_idx);
  return Subview<VT, N>{
      tv.backing_,
      SubviewLayout<N>{p.base, p.ext, p.str, tv.layout_.inner_order(),
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

// ---------------------------------------------------------------------------
// reorder_layout — logically permute (transpose) a subview's dimensions.
//
// Gather convention: Perm[i] is the source dimension that becomes new
// dimension i, so new.extent(i) == src.extent(Perm[i]) and likewise for
// stride. base_offset is unchanged (same first element, same backing data).
// The permutation is compile-time (std::integer_sequence) for both layout
// kinds so the OrderedSubviewLayout overload can bake the new memory order into
// template params.
//
// Type-preserving: SubviewLayout<N>              -> SubviewLayout<N>
//                  OrderedSubviewLayout<N,Ord..> ->
//                  OrderedSubviewLayout<N,New..>
// ---------------------------------------------------------------------------

template <int N, int... Perm>
KOKKOS_FUNCTION auto reorder_layout(const SubviewLayout<N>& src,
                                    std::integer_sequence<int, Perm...>)
    -> SubviewLayout<N> {
  static_assert(sizeof...(Perm) == N,
                "reorder_layout: permutation must have N elements");
  constexpr int         perm[] = {Perm...};
  Kokkos::Array<int, N> ext{}, str{};
  for (int i = 0; i < N; ++i) {
    ext[i] = src.extent(perm[i]);
    str[i] = src.stride(perm[i]);
  }
  // Strides are only relabelled, so the memory order is recomputed from the
  // reordered strides (order_ indexes the new dimensions).
  const auto order = Impl::argsort_by_stride<N>(str);
  const auto inv   = Impl::reciprocals<N>(ext);
  return SubviewLayout<N>{src.base_offset(), ext, str, order, inv};
}

template <int N, int... Order, int... Perm>
KOKKOS_FUNCTION constexpr auto reorder_layout(
    const OrderedSubviewLayout<N, Order...>& src,
    std::integer_sequence<int, Perm...>      perm) {
  static_assert(sizeof...(Perm) == N,
                "reorder_layout: permutation must have N elements");
  constexpr int         p[] = {Perm...};
  Kokkos::Array<int, N> ext{}, str{};
  for (int i = 0; i < N; ++i) {
    ext[i] = src.extent(p[i]);
    str[i] = src.stride(p[i]);
  }
  const auto inv = Impl::reciprocals<N>(ext);
  return Impl::make_ordered_subview_layout(
      src.base_offset(), ext, str, inv,
      Impl::reorder_order_seq(std::integer_sequence<int, Order...>{}, perm));
}

// ---------------------------------------------------------------------------
// reorder_view — View-level wrapper: reorder the layout, keep the same backing
// storage. Mirrors the reshape(View, ...) overload. The requires clause
// constrains the overload to layouts with a reorder_layout (SubviewLayout /
// OrderedSubviewLayout), SFINAE-ing out others (TiledLayout, TileLayout).
// ---------------------------------------------------------------------------

template <typename ViewType, typename Layout, int... Perm>
  requires(requires(Layout l) {
    reorder_layout(l, std::integer_sequence<int, Perm...>{});
  })
KOKKOS_FUNCTION auto reorder_view(const View<ViewType, Layout>&       view,
                                  std::integer_sequence<int, Perm...> perm)
    -> View<ViewType, decltype(reorder_layout(view.layout(), perm))> {
  return {view.backing_, reorder_layout(view.layout(), perm)};
}

}  // namespace TensorOperations
