#pragma once
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include <TensorOperations/Tiling.hpp>

namespace TensorOperations {

// Forward declarations for friend declarations in layout structs.
template <typename ViewType, typename Tile>
struct TiledView;

template <typename ViewType, int FreeRank>
struct Subview;

// ---------------------------------------------------------------------------
// Impl helpers — pure arithmetic, unchanged.
// ---------------------------------------------------------------------------

namespace Impl {

template <int N>
struct TiledLayoutResult {
  Kokkos::Array<int, N>   orig_extents;
  Kokkos::Array<int, N>   outer_extents;
  Kokkos::Array<int, N>   outer_strides;
  Kokkos::Array<int, N>   inner_strides;
  Kokkos::Array<int, N>   inner_order;
  Kokkos::Array<int, N>   outer_order;
  Kokkos::Array<float, N> inner_inv_extents;
  Kokkos::Array<float, N> outer_inv_extents;
};

template <int N>
KOKKOS_FUNCTION TiledLayoutResult<N> compute_tiled_layout(
    Kokkos::Array<int, N> extents, Kokkos::Array<int, N> strides,
    Kokkos::Array<int, N> tile_sizes) {
  TiledLayoutResult<N> r{};
  for (int d = 0; d < N; ++d) {
    const int T       = tile_sizes[d] < extents[d] ? tile_sizes[d] : extents[d];
    r.orig_extents[d] = extents[d];
    r.outer_extents[d] = (extents[d] + T - 1) / T;
    r.outer_strides[d] = T * strides[d];
    r.inner_strides[d] = strides[d];
  }
  for (int i = 0; i < N; ++i) r.inner_order[i] = i;
  for (int i = 0; i < N; ++i) {
    int m = i;
    for (int j = i + 1; j < N; ++j)
      if (r.inner_strides[r.inner_order[j]] < r.inner_strides[r.inner_order[m]])
        m = j;
    if (m != i) {
      const int t      = r.inner_order[i];
      r.inner_order[i] = r.inner_order[m];
      r.inner_order[m] = t;
    }
  }
  for (int i = 0; i < N; ++i) r.outer_order[i] = i;
  for (int i = 0; i < N; ++i) {
    int m = i;
    for (int j = i + 1; j < N; ++j)
      if (r.outer_strides[r.outer_order[j]] < r.outer_strides[r.outer_order[m]])
        m = j;
    if (m != i) {
      const int t      = r.outer_order[i];
      r.outer_order[i] = r.outer_order[m];
      r.outer_order[m] = t;
    }
  }
  for (int d = 0; d < N; ++d) {
    const int T = tile_sizes[d] < extents[d] ? tile_sizes[d] : extents[d];
    r.inner_inv_extents[d] = 1.0f / static_cast<float>(T);
  }
  for (int d = 0; d < N; ++d)
    r.outer_inv_extents[d] = 1.0f / static_cast<float>(r.outer_extents[d]);
  return r;
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// TiledLayout<N, InnerExtents>
//
// All layout parameters for an N-dimensional tiled view: outer tile counts,
// outer and inner (element) strides, original extents, and inner tile extents.
//
// Public interface:
//   extent(d)      — outer tile count for d < N; clamped tile size for d >= N
//   stride(d)      — outer stride for d < N; element stride for d >= N
//   operator[](i)  — row-major delinearize: flat inner index → N-D tile coord
//
// All data members are private; TiledView is a friend to access the offset
// and bounds-check helpers used inside operator().
// ---------------------------------------------------------------------------

template <int N, typename InnerExtents>
struct TiledLayout {
 public:
  using inner_extents_t = InnerExtents;

  KOKKOS_FUNCTION TiledLayout(inner_extents_t inner, Kokkos::Array<int, N> orig,
                              Kokkos::Array<int, N>   outer_ext,
                              Kokkos::Array<int, N>   outer_str,
                              Kokkos::Array<int, N>   inner_str,
                              Kokkos::Array<int, N>   inner_ord,
                              Kokkos::Array<int, N>   outer_ord,
                              Kokkos::Array<float, N> inner_inv_ext,
                              Kokkos::Array<float, N> outer_inv_ext) noexcept
      : inner_extents_(inner),
        orig_extents_(orig),
        outer_extents_(outer_ext),
        outer_strides_(outer_str),
        inner_strides_(inner_str),
        inner_order_(inner_ord),
        outer_order_(outer_ord),
        inner_inv_extents_(inner_inv_ext),
        outer_inv_extents_(outer_inv_ext) {}

  // extent(d): outer tile count for d < N; clamped tile size for d >= N.
  KOKKOS_FUNCTION int extent(int d) const noexcept {
    if (d < N) return outer_extents_[d];
    const int T = static_cast<int>(inner_extents_.extent(d - N));
    const int E = orig_extents_[d - N];
    return T < E ? T : E;
  }

  // stride(d): outer stride for d < N; element (inner) stride for d >= N.
  KOKKOS_FUNCTION int stride(int d) const noexcept {
    return d < N ? outer_strides_[d] : inner_strides_[d - N];
  }

  // Delinearize: row-major flat inner index → N-D inner tile coordinate.
  // Non-overlapping strides assumed.
  KOKKOS_FUNCTION Kokkos::Array<int, N> operator[](int linear) const noexcept {
    Kokkos::Array<int, N> idx{};
    for (int d = N - 1; d >= 0; --d) {
      idx[d] = linear % static_cast<int>(inner_extents_.extent(d));
      linear /= static_cast<int>(inner_extents_.extent(d));
    }
    return idx;
  }

 private:
  inner_extents_t       inner_extents_;
  Kokkos::Array<int, N> orig_extents_;
  Kokkos::Array<int, N> outer_extents_;
  Kokkos::Array<int, N> outer_strides_;
  Kokkos::Array<int, N> inner_strides_;

 public:
  Kokkos::Array<int, N>   inner_order_;
  Kokkos::Array<int, N>   outer_order_;
  Kokkos::Array<float, N> inner_inv_extents_;
  Kokkos::Array<float, N> outer_inv_extents_;

  template <std::size_t... Is>
  KOKKOS_FUNCTION int offset_(std::index_sequence<Is...>, auto& args) const {
    return ((static_cast<int>(std::get<Is>(args)) * outer_strides_[Is] +
             static_cast<int>(std::get<N + Is>(args)) * inner_strides_[Is]) +
            ...);
  }

  template <std::size_t I>
  KOKKOS_FUNCTION void check_one_(auto& args) const {
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    if (static_cast<int>(std::get<I>(args)) *
                static_cast<int>(inner_extents_.extent(I)) +
            static_cast<int>(std::get<N + I>(args)) >=
        orig_extents_[I])
      Kokkos::abort("TiledView: out-of-bounds access");
#else
    (void)args;
#endif
  }

  template <std::size_t... Is>
  KOKKOS_FUNCTION void check_bounds_(std::index_sequence<Is...>,
                                     auto& args) const {
    (check_one_<Is>(args), ...);
  }

  template <typename V, typename T>
  friend struct TiledView;
};

// ---------------------------------------------------------------------------
// Primary template — undefined; two specialisations below.
// ---------------------------------------------------------------------------

template <typename ViewType, typename Tile>
struct TiledView;

// ---------------------------------------------------------------------------
// TiledView — StaticTile specialisation
//
// operator() accepts 2N arguments: (t0,…,tN-1, r0,…,rN-1).
// All layout state lives in layout_; backing_ owns only the raw pointer.
// ---------------------------------------------------------------------------

template <typename ViewType, int... TileExtents>
struct TiledView<ViewType, StaticTile<TileExtents...>> {
  static constexpr int N = sizeof...(TileExtents);
  static_assert(N == static_cast<int>(ViewType::rank),
                "StaticTile rank must equal View rank");
  static constexpr int rank = 2 * N;
  using value_type          = typename ViewType::value_type;

  using inner_extents_t =
      Kokkos::extents<int, static_cast<std::size_t>(TileExtents)...>;
  using layout_t = TiledLayout<N, inner_extents_t>;

  ViewType backing_;
  layout_t layout_;

  KOKKOS_FUNCTION int extent(int d) const noexcept { return layout_.extent(d); }
  KOKKOS_FUNCTION int stride(int d) const noexcept { return layout_.stride(d); }
  KOKKOS_FUNCTION value_type* data() const noexcept { return backing_.data(); }

  template <typename... Indices>
    requires(sizeof...(Indices) == 2 * N)
  KOKKOS_FUNCTION value_type& operator()(Indices... idx) const {
    auto args = std::forward_as_tuple(idx...);
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    layout_.check_bounds_(std::make_index_sequence<N>{}, args);
#endif
    return backing_
        .data()[layout_.offset_(std::make_index_sequence<N>{}, args)];
  }
};

// ---------------------------------------------------------------------------
// TiledView — DynamicTile specialisation
// ---------------------------------------------------------------------------

template <typename ViewType, int N>
struct TiledView<ViewType, DynamicTile<N>> {
  static_assert(N == static_cast<int>(ViewType::rank),
                "DynamicTile rank must equal View rank");
  static constexpr int rank = 2 * N;
  using value_type          = typename ViewType::value_type;

  using inner_extents_t = Kokkos::dextents<int, N>;
  using layout_t        = TiledLayout<N, inner_extents_t>;

  ViewType backing_;
  layout_t layout_;

  KOKKOS_FUNCTION int extent(int d) const noexcept { return layout_.extent(d); }
  KOKKOS_FUNCTION int stride(int d) const noexcept { return layout_.stride(d); }
  KOKKOS_FUNCTION value_type* data() const noexcept { return backing_.data(); }

  template <typename... Indices>
    requires(sizeof...(Indices) == 2 * N)
  KOKKOS_FUNCTION value_type& operator()(Indices... idx) const {
    auto args = std::forward_as_tuple(idx...);
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
    layout_.check_bounds_(std::make_index_sequence<N>{}, args);
#endif
    return backing_
        .data()[layout_.offset_(std::make_index_sequence<N>{}, args)];
  }
};

// ---------------------------------------------------------------------------
// SubviewLayout<N>
//
// Layout parameters for a non-owning slice: flat base offset into the
// backing storage, per-dimension extents, and per-dimension strides.
//
// Public interface:
//   extent(d)              — extent of free dimension d
//   stride(d)              — stride of free dimension d
//   operator[](i)          — row-major delinearize: flat index → N-D coordinate
//   delinearize_ordered(i) — memory-aligned delinearize: peels the smallest-
//                            stride (fastest-varying) dimension first, so that
//                            consecutive flat indices walk contiguous memory
//                            (coalesced reads) regardless of input layout.
//
// Non-overlapping strides assumed.
// ---------------------------------------------------------------------------

template <int N>
struct SubviewLayout {
 public:
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

  template <std::size_t... Is>
  KOKKOS_FUNCTION int offset_(std::index_sequence<Is...>, auto& args) const {
    return ((static_cast<int>(std::get<Is>(args)) * strides_[Is]) + ...);
  }

  template <typename V, int FR>
  friend struct Subview;
};

// ---------------------------------------------------------------------------
// Subview — a non-owning slice of a TiledView
//
// Stores the backing ViewType by reference for zero-copy overhead. The
// caller is responsible for keeping the source TiledView alive.
//
// FreeRank = number of Kokkos::ALL dimensions in the subview call.
// ---------------------------------------------------------------------------

template <typename ViewType, int FreeRank>
struct Subview {
  static constexpr int rank = FreeRank;
  using value_type          = typename ViewType::value_type;
  using layout_t            = SubviewLayout<FreeRank>;

  const ViewType& backing_;
  layout_t        layout_;

  KOKKOS_FUNCTION int extent(int d) const noexcept { return layout_.extent(d); }
  KOKKOS_FUNCTION int stride(int d) const noexcept { return layout_.stride(d); }
  KOKKOS_FUNCTION int unit_stride_dim() const noexcept {
    return layout_.unit_stride_dim();
  }
  KOKKOS_FUNCTION value_type* data() const noexcept {
    return backing_.data() + layout_.base_offset_;
  }
  KOKKOS_FUNCTION const layout_t& layout() const noexcept { return layout_; }

  KOKKOS_FUNCTION value_type& operator[](
      Kokkos::Array<int, FreeRank> idx) const {
    return [&]<std::size_t... D>(std::index_sequence<D...>) -> value_type& {
      return (*this)(idx[D]...);
    }(std::make_index_sequence<FreeRank>{});
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == FreeRank)
  KOKKOS_FUNCTION value_type& operator()(Indices... idx) const {
    auto args = std::forward_as_tuple(idx...);
    return backing_
        .data()[layout_.base_offset_ +
                layout_.offset_(std::make_index_sequence<FreeRank>{}, args)];
  }
};

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
// ---------------------------------------------------------------------------

template <typename ViewType, int... TileExtents>
KOKKOS_FUNCTION auto tile_view(const ViewType& view, StaticTile<TileExtents...>)
    -> TiledView<ViewType, StaticTile<TileExtents...>> {
  constexpr int N = sizeof...(TileExtents);
  using TV        = TiledView<ViewType, StaticTile<TileExtents...>>;
  using LE        = typename TV::inner_extents_t;
  using LT        = typename TV::layout_t;

  Kokkos::Array<int, N> extents{}, strides{}, tsizes{};
  for (int d = 0; d < N; ++d) {
    extents[d] = static_cast<int>(view.extent(d));
    strides[d] = static_cast<int>(view.stride(d));
    tsizes[d]  = StaticTile<TileExtents...>::extent(d);
  }

  const auto r = Impl::compute_tiled_layout<N>(extents, strides, tsizes);
  return {view, LT{LE{}, r.orig_extents, r.outer_extents, r.outer_strides,
                   r.inner_strides, r.inner_order, r.outer_order,
                   r.inner_inv_extents, r.outer_inv_extents}};
}

template <typename ViewType, int N>
KOKKOS_FUNCTION auto tile_view(const ViewType& view, DynamicTile<N> tile)
    -> TiledView<ViewType, DynamicTile<N>> {
  using TV = TiledView<ViewType, DynamicTile<N>>;
  using LT = typename TV::layout_t;

  Kokkos::Array<int, N> extents{}, strides{}, tsizes{};
  for (int d = 0; d < N; ++d) {
    extents[d] = static_cast<int>(view.extent(d));
    strides[d] = static_cast<int>(view.stride(d));
    tsizes[d]  = tile.extent(d);
  }
  const auto r = Impl::compute_tiled_layout<N>(extents, strides, tsizes);

  auto inner_ext = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return Kokkos::dextents<int, N>{tsizes[Is]...};
  }(std::make_index_sequence<N>{});

  return {view, LT{inner_ext, r.orig_extents, r.outer_extents, r.outer_strides,
                   r.inner_strides, r.inner_order, r.outer_order,
                   r.inner_inv_extents, r.outer_inv_extents}};
}

// ---------------------------------------------------------------------------
// subview — factory: TiledView × mixed (int|Kokkos::ALL) → Subview
//
// One slice argument per dimension of the TiledView (2N total).
// int  → fix that dimension, contributing int*stride to base_offset.
// ALL  → keep that dimension free in the returned Subview.
// ---------------------------------------------------------------------------

template <typename ViewType, typename Tile, typename... Slices>
auto subview(const TiledView<ViewType, Tile>& tv, Slices... slices) {
  using TV           = TiledView<ViewType, Tile>;
  constexpr int Full = TV::rank;
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

  Kokkos::Array<int, Free>   order{};
  Kokkos::Array<float, Free> inv_extents{};
  for (int i = 0; i < Free; ++i) order[i] = i;
  for (int i = 0; i < Free; ++i) {
    int m = i;
    for (int j = i + 1; j < Free; ++j)
      if (strides[order[j]] < strides[order[m]]) m = j;
    if (m != i) {
      const int t = order[i];
      order[i]    = order[m];
      order[m]    = t;
    }
  }
  for (int d = 0; d < Free; ++d)
    inv_extents[d] = 1.0f / static_cast<float>(extents[d]);

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

template <typename ViewType, typename Tile>
KOKKOS_FUNCTION auto subview_tile(
    const TiledView<ViewType, Tile>&                        tv,
    Kokkos::Array<int, TiledView<ViewType, Tile>::rank / 2> outer_idx) {
  constexpr int N           = TiledView<ViewType, Tile>::rank / 2;
  int           base_offset = 0;
  for (int d = 0; d < N; ++d) base_offset += outer_idx[d] * tv.stride(d);
  Kokkos::Array<int, N> extents{}, strides{};
  for (int d = 0; d < N; ++d) {
    extents[d] = tv.extent(N + d);
    strides[d] = tv.stride(N + d);
  }
  return Subview<ViewType, N>{
      tv.backing_,
      SubviewLayout<N>{base_offset, extents, strides, tv.layout_.inner_order_,
                       tv.layout_.inner_inv_extents_}};
}

}  // namespace TensorOperations
