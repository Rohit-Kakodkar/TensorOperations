#pragma once

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <cute/tensor.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace TensorOperations {

namespace Impl {

template <typename T, typename ModesSeq, std::size_t... Is>
KOKKOS_FUNCTION auto make_cute_tensor(const TensorHandle<T, ModesSeq>& h,
                                      std::index_sequence<Is...>) {
  return cute::make_tensor(
      cute::make_gmem_ptr(h.data()),
      cute::make_layout(cute::make_shape(static_cast<int>(h.extent(Is))...),
                        cute::make_stride(static_cast<int>(h.stride(Is))...)));
}

template <typename T, typename ModesSeq>
struct CuteHandle {
  static constexpr int Rank = TensorHandle<T, ModesSeq>::Rank;
  using modes_seq           = ModesSeq;
  using tensor_type         = decltype(make_cute_tensor(
      std::declval<const TensorHandle<T, ModesSeq>&>(),
      std::make_index_sequence<Rank>{}));

  tensor_type tensor;
};

template <typename T, typename ModesSeq>
KOKKOS_FUNCTION CuteHandle<T, ModesSeq> make_cute_handle(
    const TensorHandle<T, ModesSeq>& h) {
  return {make_cute_tensor(
      h, std::make_index_sequence<TensorHandle<T, ModesSeq>::Rank>{})};
}

template <typename Fn, typename CoordIter, typename V>
struct CuteFunctionalIterator {
  using value_type   = V;
  using element_type = V;
  using reference    = V;

  Fn        fn;
  CoordIter coord;

  KOKKOS_FUNCTION V operator*() const {
    const auto c = *coord;
    return call(c, std::make_index_sequence<decltype(cute::rank(c))::value>{});
  }

  template <typename C>
  KOKKOS_FUNCTION auto operator+(const C& c) const {
    auto next = coord + c;
    return CuteFunctionalIterator<Fn, decltype(next), V>{fn, next};
  }

  template <typename C>
  KOKKOS_FUNCTION V operator[](const C& c) const {
    return *(*this + c);
  }

 private:
  template <typename Coord, std::size_t... Is>
  KOKKOS_FUNCTION V call(const Coord& c, std::index_sequence<Is...>) const {
    return fn(static_cast<int>(cute::get<Is>(c))...);
  }
};

template <typename V, typename Fn, std::size_t R, std::size_t... Is>
KOKKOS_FUNCTION auto make_cute_functional_tensor(
    const Fn& fn, const Kokkos::Array<int, R>& shape,
    std::index_sequence<Is...>) {
  const auto id = cute::make_identity_tensor(cute::make_shape(shape[Is]...));
  return cute::make_tensor(
      CuteFunctionalIterator<Fn, decltype(id.data()), V>{fn, id.data()},
      id.layout());
}

template <typename V, typename Fn, int R>
using cute_functional_tensor_t = decltype(make_cute_functional_tensor<V>(
    std::declval<const Fn&>(),
    std::declval<const Kokkos::Array<int, static_cast<std::size_t>(R)>&>(),
    std::make_index_sequence<static_cast<std::size_t>(R)>{}));

}  // namespace Impl

template <typename ES, typename Storage, int R, typename HookOp>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<IntermTag, Storage, std::integral_constant<int, R>, ES, HookOp>,
    void> {
  static_assert(cute::is_tensor<Storage>::value,
                "CuTe value evaluator: storage must be a cute::Tensor");

 public:
  using node_type   = NodeHandle<IntermTag, Storage,
                                 std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = void;
  static constexpr int Rank = R;
  using storage_type        = Storage;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;

  KOKKOS_FUNCTION explicit Evaluator(node_type n) : node_(n) {}

  KOKKOS_FUNCTION const node_type& node() const { return node_; }

 private:
  node_type node_;
};

template <typename ES, typename Storage, typename HookOp = NoHook>
KOKKOS_FUNCTION auto make_cute_interm_node(Storage tensor, HookOp hook = {}) {
  return NodeHandle<IntermTag, Storage,
                    std::integral_constant<int, Storage::rank>, ES, HookOp>{
      std::move(tensor), std::move(hook)};
}

namespace Impl {

template <typename ES, typename Storage, typename HookOp>
KOKKOS_FUNCTION auto make_cute_value_evaluator(Storage tile, HookOp hook) {
  auto node = make_cute_interm_node<ES>(std::move(tile), std::move(hook));
  return Evaluator<CutePolicyTag<ES>, decltype(node), void>(node);
}

}  // namespace Impl

struct FragmentTag {};

template <typename Frag, typename Coords, typename TileShape, typename IntRank,
          typename ExecSpace, typename HookOp>
struct NodeHandle<FragmentTag, Frag, Coords, TileShape, IntRank, ExecSpace,
                  HookOp> {
  static_assert(cute::is_tensor<Frag>::value && cute::is_rmem<Frag>::value,
                "fragment node: storage must be a register cute::Tensor");
  static_assert(cute::is_tensor<Coords>::value,
                "fragment node: coordinates must be a cute::Tensor");
  static_assert(std::is_same_v<decltype(cute::shape(std::declval<Frag>())),
                               decltype(cute::shape(std::declval<Coords>()))>,
                "fragment node: fragment and coordinates must share a shape");
  static_assert(cute::is_static<TileShape>::value &&
                    cute::rank_v<TileShape> == IntRank::value,
                "fragment node: the tile shape must be static and flat with "
                "one mode per node mode");

  using node_tag            = FragmentTag;
  static constexpr int Rank = IntRank::value;
  using storage_type        = Frag;
  using coords_type         = Coords;
  using tile_shape_type     = TileShape;
  using value_type          = typename Frag::value_type;
  using exec_space          = ExecSpace;
  using hook_type           = HookOp;

  Frag                         frag_;
  Coords                       coords_;
  [[no_unique_address]] HookOp hook_op;
};

template <typename ES, int R, typename TileShape, typename Frag,
          typename Coords, typename HookOp = NoHook>
KOKKOS_FUNCTION auto make_cute_fragment_node(Frag frag, Coords coords,
                                             HookOp hook = {}) {
  return NodeHandle<FragmentTag, Frag, Coords, TileShape,
                    std::integral_constant<int, R>, ES, HookOp>{
      std::move(frag), std::move(coords), std::move(hook)};
}

template <typename ES, typename Frag, typename Coords, typename TileShape,
          int R, typename HookOp>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<FragmentTag, Frag, Coords, TileShape,
                           std::integral_constant<int, R>, ES, HookOp>,
                void> {
 public:
  using node_type    = NodeHandle<FragmentTag, Frag, Coords, TileShape,
                                  std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag   = CutePolicyTag<ES>;
  using tiling_type  = void;
  using storage_type = Frag;
  using coords_type  = Coords;
  using tile_shape_type     = TileShape;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  static constexpr int Rank = R;

  KOKKOS_FUNCTION explicit Evaluator(node_type n) : node_(n) {}

  KOKKOS_FUNCTION const node_type& node() const { return node_; }

 private:
  node_type node_;
};

namespace Impl {

template <typename ES, int R, typename TileShape, typename Frag,
          typename Coords, typename HookOp>
KOKKOS_FUNCTION auto make_cute_fragment_value_evaluator(Frag   frag,
                                                        Coords coords,
                                                        HookOp hook) {
  auto node = make_cute_fragment_node<ES, R, TileShape>(
      std::move(frag), std::move(coords), std::move(hook));
  return Evaluator<CutePolicyTag<ES>, decltype(node), void>(node);
}

template <typename ES, int R, typename TileShape, typename Frags,
          typename Coords, std::size_t... Ms>
KOKKOS_FUNCTION auto make_cute_fragment_value_evaluators(
    const Frags& frags, const Coords& coords, std::index_sequence<Ms...>) {
  using result_t =
      decltype(make_cute_fragment_value_evaluator<ES, R, TileShape>(
          frags[0], coords, NoHook{}));
  return Kokkos::Array<result_t, sizeof...(Ms)>{
      make_cute_fragment_value_evaluator<ES, R, TileShape>(frags[Ms], coords,
                                                           NoHook{})...};
}

}  // namespace Impl

template <typename ES, TensorLike T, typename ModesSeq, typename HookOp,
          typename Tiler>
class Evaluator<CutePolicyTag<ES>, NodeHandle<InputTag, T, ModesSeq, HookOp>,
                Tiler> {
 public:
  using node_type           = NodeHandle<InputTag, T, ModesSeq, HookOp>;
  using policy_tag          = CutePolicyTag<ES>;
  using tiling_type         = Tiler;
  using exec_space          = ES;
  static constexpr int Rank = node_type::Rank;

  static_assert(cute::rank_v<Tiler> == Rank,
                "CuTe input: tiler rank must equal the node's rank");

  KOKKOS_FUNCTION Evaluator(node_type n, Tiler t)
      : handle_(Impl::make_cute_handle(n.handle)),
        tiler_(t),
        hook_(n.hook_op) {}

  template <typename Coord>
  KOKKOS_FUNCTION auto operator()(const Coord& coord) const {
    return Impl::make_cute_value_evaluator<ES>(
        cute::local_tile(handle_.tensor, tiler_, coord), hook_);
  }

 private:
  Impl::CuteHandle<T, ModesSeq> handle_;
  Tiler                         tiler_;
  [[no_unique_address]] HookOp  hook_;
};

template <typename ES, typename Fn, typename ModesSeq, typename ValueType,
          typename NodeES, typename Layout, typename HookOp, typename Tiler>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<FunctionalTag, Fn, ModesSeq, ValueType, NodeES, Layout, HookOp>,
    Tiler> {
 public:
  using node_type   = NodeHandle<FunctionalTag, Fn, ModesSeq, ValueType, NodeES,
                                 Layout, HookOp>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = Tiler;
  using exec_space  = ES;
  static constexpr int Rank = node_type::Rank;

  static_assert(std::is_same_v<NodeES, ES>,
                "CuTe functional input: the node's execution space must be the "
                "policy's");
  static_assert(cute::rank_v<Tiler> == Rank,
                "CuTe functional input: tiler rank must equal the node's rank");

  KOKKOS_FUNCTION Evaluator(node_type n, Tiler t)
      : tensor_(
            Impl::make_cute_functional_tensor<typename node_type::result_type>(
                n.fn_, n.shape(), std::make_index_sequence<Rank>{})),
        tiler_(t),
        hook_(n.hook_op) {}

  template <typename Coord>
  KOKKOS_FUNCTION auto operator()(const Coord& coord) const {
    return Impl::make_cute_value_evaluator<ES>(
        cute::local_tile(tensor_, tiler_, coord), hook_);
  }

 private:
  Impl::cute_functional_tensor_t<typename node_type::result_type, Fn, Rank>
                               tensor_;
  Tiler                        tiler_;
  [[no_unique_address]] HookOp hook_;
};

template <typename ThrLayout>
struct CuteThreadTag {
  ThrLayout thr_layout;
  int       thr_idx;
};

template <typename ThrLayout>
struct CuteSmemLoadTag : CuteThreadTag<ThrLayout> {};

template <typename ThrLayout>
struct CuteStoreTag : CuteThreadTag<ThrLayout> {};

template <typename TileShape, typename Part>
struct CuteStagedTag {
  Part part;
};

template <typename TileShape, typename Part, typename IdxFrag>
struct CuteGatherTag {
  Part    part;
  IdxFrag idx;
};

template <typename AEval, typename BEval, typename TiledMma>
struct CuteContractTag {
  AEval    a;
  BEval    b;
  TiledMma mma;
  int      thr_idx;
};

template <typename TiledMma, int FreeA>
struct CuteMmaPartitioner {
  TiledMma mma;
  int      thr_idx;

  KOKKOS_FUNCTION bool active() const {
    return thr_idx < static_cast<int>(cute::size(mma));
  }

  template <typename Tensor>
  KOKKOS_FUNCTION auto operator()(const Tensor& t) const {
    constexpr int R = decltype(cute::rank(t))::value;
    static_assert(FreeA > 0 && FreeA < R,
                  "CuTe MMA partitioner: C needs at least one free mode from "
                  "each of A and B");
    return mma.get_slice(thr_idx).partition_C(
        cute::group_modes<1, 1 + R - FreeA>(cute::group_modes<0, FreeA>(t)));
  }
};

template <typename ThrLayout>
struct CuteThreadPartitioner {
  ThrLayout thr_layout;
  int       thr_idx;

  KOKKOS_FUNCTION bool active() const {
    return thr_idx < static_cast<int>(cute::size(thr_layout));
  }

  template <typename Tensor>
  KOKKOS_FUNCTION auto operator()(const Tensor& t) const {
    return cute::local_partition(t, thr_layout, thr_idx);
  }
};

template <typename TV>
struct CuteTVPartitioner {
  TV  tv;
  int thr_idx;

  KOKKOS_FUNCTION bool active() const {
    return thr_idx < static_cast<int>(cute::size<0>(tv));
  }

  template <typename Tensor>
  KOKKOS_FUNCTION auto operator()(const Tensor& t) const {
    return cute::composition(t, tv)(thr_idx, cute::_);
  }
};

template <typename Partitioner>
struct CuteFragmentStoreTag {
  Partitioner part;
};

namespace Impl {

template <typename Tile>
struct cute_shape_of;

template <int... Es>
struct cute_shape_of<StaticTile<Es...>> {
  using type = cute::Shape<cute::Int<Es>...>;
};

template <typename Tile>
using cute_shape_of_t = typename cute_shape_of<Tile>::type;

template <typename TileShape, std::size_t... Is>
constexpr std::array<int, sizeof...(Is)> cute_static_extents(
    std::index_sequence<Is...>) {
  return {static_cast<int>(
      decltype(cute::get<Is>(std::declval<TileShape>()))::value)...};
}

template <typename TileShape, int... Order>
constexpr std::array<int, sizeof...(Order)> cute_thr_extents(
    int num_threads, std::integer_sequence<int, Order...>) {
  constexpr std::size_t R = sizeof...(Order);
  const auto            ext =
      cute_static_extents<TileShape>(std::make_index_sequence<R>{});
  const int          order[] = {Order...};
  std::array<int, R> thr{};
  for (std::size_t d = 0; d < R; ++d) thr[d] = 1;
  int budget = num_threads;
  for (std::size_t k = 0; k < R; ++k) {
    const int m = order[k];
    int       d = ext[m] < budget ? ext[m] : budget;
    while (ext[m] % d != 0) --d;
    thr[m] = d;
    budget /= d;
  }
  return thr;
}

template <int... Order, std::size_t R>
constexpr std::array<int, R> cute_thr_strides(
    const std::array<int, R>& thr, std::integer_sequence<int, Order...>) {
  const int          order[] = {Order...};
  std::array<int, R> st{};
  int                s = 1;
  for (std::size_t k = 0; k < R; ++k) {
    st[order[k]] = s;
    s *= thr[order[k]];
  }
  return st;
}

template <typename TileShape, typename Order, int NumThreads>
inline constexpr auto cute_thr_ext_v =
    cute_thr_extents<TileShape>(NumThreads, Order{});

template <typename TileShape, typename Order, int NumThreads>
inline constexpr auto cute_thr_stride_v =
    cute_thr_strides(cute_thr_ext_v<TileShape, Order, NumThreads>, Order{});

template <typename TileShape, typename Order, int NumThreads, typename Is>
struct CuteThrLayout;

template <typename TileShape, typename Order, int NumThreads, std::size_t... Is>
struct CuteThrLayout<TileShape, Order, NumThreads, std::index_sequence<Is...>> {
  using type = cute::Layout<
      cute::Shape<
          cute::Int<cute_thr_ext_v<TileShape, Order, NumThreads>[Is]>...>,
      cute::Stride<
          cute::Int<cute_thr_stride_v<TileShape, Order, NumThreads>[Is]>...>>;
};

template <typename TileShape, typename Order, int NumThreads>
using cute_thr_layout_t = typename CuteThrLayout<
    TileShape, Order, NumThreads,
    std::make_index_sequence<cute::rank_v<TileShape>>>::type;

template <int R, typename ArrayLayout>
struct view_contiguity {
  static_assert(std::is_same_v<ArrayLayout, Kokkos::LayoutLeft> ||
                    std::is_same_v<ArrayLayout, Kokkos::LayoutRight>,
                "CuTe backend: a staged View must be LayoutLeft or LayoutRight "
                "so its contiguous mode is known at compile time");
  template <std::size_t... Is>
  static auto make(std::index_sequence<Is...>) -> std::conditional_t<
      std::is_same_v<ArrayLayout, Kokkos::LayoutLeft>,
      std::integer_sequence<int, static_cast<int>(Is)...>,
      std::integer_sequence<int, static_cast<int>(R - 1 - Is)...>>;
  using type = decltype(make(std::make_index_sequence<R>{}));
};

template <int R, typename ArrayLayout>
using view_contiguity_t = typename view_contiguity<R, ArrayLayout>::type;

template <int R, typename OrderTag>
struct order_contiguity {
  static_assert(std::is_same_v<OrderTag, LayoutLeft> ||
                    std::is_same_v<OrderTag, LayoutRight>,
                "CuTe backend: a functional input's order must be LayoutLeft, "
                "LayoutRight or an explicit fastest-first permutation");
  using type = view_contiguity_t<
      R, std::conditional_t<std::is_same_v<OrderTag, LayoutLeft>,
                            Kokkos::LayoutLeft, Kokkos::LayoutRight>>;
};

template <int R, int... Ord>
struct order_contiguity<R, std::integer_sequence<int, Ord...>> {
  static_assert(sizeof...(Ord) == R,
                "CuTe backend: order must name every mode once");
  using type = std::integer_sequence<int, Ord...>;
};

template <int R, typename OrderTag>
using order_contiguity_t = typename order_contiguity<R, OrderTag>::type;

template <typename ES, typename Storage, int R, typename HookOp,
          typename ThrLayout, typename Tag>
class CuteThreadTileEvaluator {
  static_assert(cute::is_tensor<Storage>::value,
                "CuTe thread tile: storage must be a cute::Tensor");
  static_assert(cute::is_static<ThrLayout>::value,
                "CuTe thread tile: thread layout must be static");
  static_assert(cute::rank_v<ThrLayout> == R,
                "CuTe thread tile: thread layout rank must equal the tile's "
                "rank");
  static_assert(std::is_base_of_v<CuteThreadTag<ThrLayout>, Tag>,
                "CuTe thread tile: tag must derive from CuteThreadTag");

 public:
  using node_type   = NodeHandle<IntermTag, Storage,
                                 std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = Tag;
  static constexpr int Rank = R;
  using storage_type        = Storage;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;

  KOKKOS_FUNCTION CuteThreadTileEvaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

 protected:
  KOKKOS_FUNCTION CuteThreadPartitioner<ThrLayout> partitioner() const {
    return {tag_.thr_layout, tag_.thr_idx};
  }

  template <typename Tensor>
  KOKKOS_FUNCTION auto partition(const Tensor& t) const {
    return partitioner()(t);
  }

  node_type   node_;
  tiling_type tag_;
};

template <typename Op, typename Coord, typename V, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION void apply_cute_hook(const Op& op, const Coord& c,
                                                 V& v,
                                                 std::index_sequence<Is...>) {
  op(static_cast<int>(cute::get<Is>(c))..., v);
}

template <typename Part, typename Tensor, typename Frag>
inline constexpr bool partitions_like_v =
    std::is_same_v<decltype(cute::shape(std::declval<const Part&>()(
                       std::declval<const Tensor&>()))),
                   decltype(cute::shape(std::declval<Frag>()))>;

template <int R, typename Src, typename Part, typename Tiler, typename Coord,
          typename T, typename ModesSeq, typename HookOp, int... Perm>
KOKKOS_FUNCTION void cute_store_global(const Src& src, const Part& part,
                                       const Tiler& tiler, const Coord& coord,
                                       const TensorHandle<T, ModesSeq>& out,
                                       std::integer_sequence<int, Perm...>,
                                       const HookOp& hook) {
  static_assert(sizeof...(Perm) == R,
                "store permutation must have one entry per output mode");
  static_assert(TensorHandle<T, ModesSeq>::Rank == R,
                "CuTe store: output rank must equal the tile's rank");

  const auto g = make_cute_handle(out).tensor;
  const auto gc =
      cute::make_tensor(g.data(), cute::select<Perm...>(g.layout()));
  if (!part.active()) return;
  auto gp = part(cute::local_tile(gc, tiler, coord));

  if constexpr (std::is_same_v<HookOp, NoHook>) {
    cute::copy(src, gp);
  } else {
    const auto cp = part(cute::local_tile(
        cute::make_identity_tensor(cute::shape(gc)), tiler, coord));
    for (int i = 0; i < static_cast<int>(cute::size(src)); ++i) {
      auto v = src(i);
      apply_cute_hook(hook, cp(i), v, std::make_index_sequence<R>{});
      gp(i) = v;
    }
  }
}

}  // namespace Impl

template <typename ES, typename Storage, int R, typename HookOp,
          typename ThrLayout>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<IntermTag, Storage, std::integral_constant<int, R>, ES, HookOp>,
    CuteSmemLoadTag<ThrLayout>>
    : public Impl::CuteThreadTileEvaluator<ES, Storage, R, HookOp, ThrLayout,
                                           CuteSmemLoadTag<ThrLayout>> {
  using base = Impl::CuteThreadTileEvaluator<ES, Storage, R, HookOp, ThrLayout,
                                             CuteSmemLoadTag<ThrLayout>>;

 public:
  using base::base;

  template <typename SrcEval>
  KOKKOS_FUNCTION auto operator=(const SrcEval& src) const {
    static_assert(SrcEval::Rank == R,
                  "staged source and destination must have equal rank");

    const auto dst = this->node_.storage_;
    if (this->partitioner().active())
      cute::copy(this->partition(src.node().storage_), this->partition(dst));

    return Impl::make_cute_value_evaluator<ES>(dst, src.node().hook_op);
  }
};

template <typename ES, typename Operand, typename ModesSeq, typename NodeTile,
          typename TV, typename TileShape, typename Part>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<StagedTag, Operand, ModesSeq, NodeTile, TV>,
                CuteStagedTag<TileShape, Part>> {
 public:
  using node_type   = NodeHandle<StagedTag, Operand, ModesSeq, NodeTile, TV>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = CuteStagedTag<TileShape, Part>;
  using value_type  = typename node_type::value_type;
  using exec_space  = ES;
  using modes_seq   = typename node_type::modes_seq;
  static constexpr int Rank = node_type::Rank;

  static_assert(cute::is_static<TileShape>::value &&
                    cute::rank_v<TileShape> == Rank,
                "CuTe staged: the tile shape must be static with one mode per "
                "operand mode");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  static constexpr int NumOut = node_type::NumOut;

  template <typename Coord>
  KOKKOS_FUNCTION auto operator()(const Coord& coord) const {
    const auto src =
        make_evaluator<CutePolicyTag<ES>>(node_.operand_, TileShape{})(coord);
    const Part part   = tag_.part;
    const auto coords = part(cute::make_identity_tensor(TileShape{}));
    if constexpr (NumOut == 1) {
      auto frag = cute::make_tensor<value_type>(cute::shape(coords));
      if (part.active()) cute::copy(part(src.node().storage_), frag);
      return Impl::make_cute_fragment_value_evaluator<ES, Rank, TileShape>(
          frag, coords, src.node().hook_op);
    } else {
      using frag_t =
          decltype(cute::make_tensor<value_type>(cute::shape(coords)));
      Kokkos::Array<frag_t, NumOut> frags;
      if (part.active()) {
        const auto sp = part(src.node().storage_);
        CUTE_UNROLL
        for (int v = 0; v < static_cast<int>(cute::size(coords)); ++v) {
          const auto r = sp(v);
          CUTE_UNROLL
          for (int o = 0; o < NumOut; ++o) frags[o](v) = r[o];
        }
      }
      return Impl::make_cute_fragment_value_evaluators<ES, Rank, TileShape>(
          frags, coords, std::make_index_sequence<NumOut>{});
    }
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

template <typename ES, typename IdxSlot, typename Source, typename OpModes,
          typename ModesSeq, typename NodeTile, typename TV, typename TileShape,
          typename Part, typename IdxFrag>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<StagedTag, NodeHandle<GatherTag, IdxSlot, Source, OpModes>,
               ModesSeq, NodeTile, TV>,
    CuteGatherTag<TileShape, Part, IdxFrag>> {
 public:
  using node_type =
      NodeHandle<StagedTag, NodeHandle<GatherTag, IdxSlot, Source, OpModes>,
                 ModesSeq, NodeTile, TV>;
  using policy_tag          = CutePolicyTag<ES>;
  using tiling_type         = CuteGatherTag<TileShape, Part, IdxFrag>;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;
  using modes_seq           = typename node_type::modes_seq;
  static constexpr int Rank = node_type::Rank;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  KOKKOS_FUNCTION auto operator()() const {
    const auto coords = tag_.part(cute::make_identity_tensor(TileShape{}));
    auto       frag   = cute::make_tensor<value_type>(cute::shape(coords));
    static_assert(decltype(cute::size(frag))::value ==
                      decltype(cute::size(std::declval<IdxFrag>()))::value,
                  "CuTe gather: the index fragment must hold one index per "
                  "gathered value");
    if (tag_.part.active()) {
      CUTE_UNROLL
      for (int v = 0; v < static_cast<int>(cute::size(frag)); ++v)
        frag(v) = static_cast<value_type>(
            node_.operand_.source_(static_cast<int>(tag_.idx(v))));
    }
    return Impl::make_cute_fragment_value_evaluator<ES, Rank, TileShape>(
        frag, coords, NoHook{});
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

template <typename ES, typename Storage, int R, typename HookOp,
          typename ThrLayout>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<IntermTag, Storage, std::integral_constant<int, R>, ES, HookOp>,
    CuteStoreTag<ThrLayout>>
    : public Impl::CuteThreadTileEvaluator<ES, Storage, R, HookOp, ThrLayout,
                                           CuteStoreTag<ThrLayout>> {
  using base = Impl::CuteThreadTileEvaluator<ES, Storage, R, HookOp, ThrLayout,
                                             CuteStoreTag<ThrLayout>>;

 public:
  using base::base;

  template <typename Coord, typename T, typename ModesSeq, typename PermSeq>
  KOKKOS_FUNCTION void operator()(const Coord&                     coord,
                                  const TensorHandle<T, ModesSeq>& out,
                                  PermSeq                          perm) const {
    const auto part = this->partitioner();
    Impl::cute_store_global<R>(part(this->node_.storage_), part,
                               cute::shape(this->node_.storage_), coord, out,
                               perm, this->node_.hook_op);
  }
};

namespace Impl {

template <typename Layout, typename T, T... P>
KOKKOS_FUNCTION constexpr auto select_seq(const Layout& l,
                                          std::integer_sequence<T, P...>) {
  return cute::select<static_cast<int>(P)...>(l);
}

template <int Shift, int N, std::size_t... I>
auto rotate_seq_impl(std::index_sequence<I...>)
    -> std::integer_sequence<int, static_cast<int>((I + Shift) % N)...>;

template <int Shift, int N>
using rotate_seq_t =
    decltype(rotate_seq_impl<Shift, N>(std::make_index_sequence<N>{}));

template <int Free, int NumK, typename Tensor>
KOKKOS_FUNCTION auto group_free_contracted(const Tensor& t) {
  return cute::group_modes<1, 1 + NumK>(cute::group_modes<0, Free>(t));
}

}  // namespace Impl

template <typename Part, typename PermSeq>
struct CutePermutedPartitioner {
  Part part;

  KOKKOS_FUNCTION bool active() const { return part.active(); }

  template <typename Tensor>
  KOKKOS_FUNCTION auto operator()(const Tensor& t) const {
    return part(
        cute::make_tensor(t.data(), Impl::select_seq(t.layout(), PermSeq{})));
  }
};

template <typename ES, typename NA, typename NB, typename IntCRank, typename S,
          typename HookOp, typename CModesSeq, typename PermCSeq, typename Mma,
          typename AEval, typename BEval, typename TiledMma>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                           CModesSeq, PermCSeq, Mma>,
                CuteContractTag<AEval, BEval, TiledMma>> {
 public:
  using node_type  = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                                CModesSeq, PermCSeq, Mma>;
  using policy_tag = CutePolicyTag<ES>;
  using tiling_type = CuteContractTag<AEval, BEval, TiledMma>;
  using value_type  = S;
  using exec_space  = ES;

  static constexpr int RankC = node_type::Rank;
  static constexpr int NumK  = node_type::NumContracted;
  static constexpr int RankA = NA::Rank;
  static constexpr int RankB = NB::Rank;
  static constexpr int FreeA = RankA - NumK;
  static constexpr int FreeB = RankB - NumK;
  static constexpr int Rank  = RankC;

 private:
  using a_storage_t = typename AEval::storage_type;
  using b_storage_t = typename BEval::storage_type;
  using permA_t     = Impl::node_permA_t<node_type>;
  using permB_t     = Impl::node_permB_t<node_type>;

  static_assert(cute::is_smem<a_storage_t>::value &&
                    cute::is_smem<b_storage_t>::value,
                "CuTe contraction: operands must be shared-memory tensors");
  static_assert(cute::is_static<typename a_storage_t::layout_type>::value &&
                    cute::is_static<typename b_storage_t::layout_type>::value,
                "CuTe contraction: operand layouts must be static");
  static_assert(a_storage_t::rank == RankA && b_storage_t::rank == RankB,
                "CuTe contraction: operand tensor ranks must equal the "
                "operand nodes' ranks");
  static_assert(FreeA + FreeB == RankC,
                "CuTe contraction: free-mode counts must sum to the output "
                "rank");
  static_assert(FreeA > 0 && FreeB > 0 && NumK > 0,
                "CuTe contraction: needs a free mode on each operand and at "
                "least one contracted mode");
  static_assert(std::is_same_v<typename TiledMma::ValTypeC, S>,
                "CuTe contraction: the MMA accumulator type must be the "
                "node's scalar");

  using a_canon_t = decltype(Impl::select_seq(
      std::declval<typename a_storage_t::layout_type>(), permA_t{}));
  using b_canon_t = decltype(Impl::select_seq(
      Impl::select_seq(std::declval<typename b_storage_t::layout_type>(),
                       permB_t{}),
      Impl::rotate_seq_t<NumK, RankB>{}));

  static_assert(std::is_same_v<decltype(cute::take<FreeA, RankA>(
                                   cute::shape(std::declval<a_canon_t>()))),
                               decltype(cute::take<FreeB, RankB>(
                                   cute::shape(std::declval<b_canon_t>())))>,
                "CuTe contraction: A's and B's contracted extents must match");

 public:
  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  KOKKOS_FUNCTION auto operator()() const {
    const auto a  = tag_.a.node().storage_;
    const auto b  = tag_.b.node().storage_;
    const auto sA = Impl::group_free_contracted<FreeA, NumK>(
        cute::make_tensor(a.data(), Impl::select_seq(a.layout(), permA_t{})));
    const auto sB = Impl::group_free_contracted<FreeB, NumK>(cute::make_tensor(
        b.data(), Impl::select_seq(Impl::select_seq(b.layout(), permB_t{}),
                                   Impl::rotate_seq_t<NumK, RankB>{})));

    const auto thr = tag_.mma.get_slice(tag_.thr_idx);
    const auto idC = cute::make_identity_tensor(
        cute::make_shape(cute::shape<0>(sA), cute::shape<0>(sB)));
    const auto cC   = thr.partition_C(idC);
    auto       frag = thr.partition_fragment_C(idC);
    cute::clear(frag);
    if (tag_.thr_idx < static_cast<int>(cute::size(tag_.mma))) {
      const auto tA = thr.partition_A(sA);
      const auto tB = thr.partition_B(sB);
      auto       rA = thr.partition_fragment_A(sA);
      auto       rB = thr.partition_fragment_B(sB);
      cute::copy(tA, rA);
      CUTE_UNROLL
      for (int k = 0; k < static_cast<int>(cute::size<2>(rA)); ++k) {
        cute::copy(tB(cute::_, cute::_, k), rB(cute::_, cute::_, k));
        cute::gemm(tag_.mma, frag, rA(cute::_, cute::_, k),
                   rB(cute::_, cute::_, k), frag);
      }
    }

    return Impl::make_cute_fragment_value_evaluator<
        ES, RankC, decltype(cute::flatten(cute::shape(idC)))>(frag, cC,
                                                              node_.hook_op);
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

namespace Impl {

template <typename Eval>
inline constexpr bool is_cute_fragment_eval_v =
    has_node_tag_v<FragmentTag, typename Eval::node_type>;

template <typename... OpEvals>
inline constexpr int combine_rank_v =
    std::tuple_element_t<0, std::tuple<OpEvals...>>::Rank;

template <typename Eval>
constexpr bool cute_combine_operand_ok() {
  if constexpr (is_cute_fragment_eval_v<Eval>)
    return true;
  else
    return cute::is_tensor<typename Eval::storage_type>::value &&
           cute::is_smem<typename Eval::storage_type>::value &&
           cute::is_static<typename Eval::storage_type::layout_type>::value;
}

}  // namespace Impl

template <typename Coords, typename TileShape, typename... OpEvals>
struct CuteCombineTag {
  DeviceTuple<OpEvals...>                              ops;
  Kokkos::Array<int, Impl::combine_rank_v<OpEvals...>> origin{};
  bool                                                 active = true;
  Coords                                               coords;
};

template <typename TileShape, typename Part, typename Origin,
          typename... OpEvals>
KOKKOS_FUNCTION auto make_cute_combine_tag(const Part&   part,
                                           const Origin& origin,
                                           const OpEvals&... ops) {
  const auto coords = part(cute::make_identity_tensor(TileShape{}));
  return CuteCombineTag<std::decay_t<decltype(coords)>, TileShape, OpEvals...>{
      DeviceTuple<OpEvals...>(ops...), origin, part.active(), coords};
}

template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename Coords, typename TileShape, typename... OpEvals>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                           IntNumOut, Ops...>,
                CuteCombineTag<Coords, TileShape, OpEvals...>> {
 public:
  using node_type   = NodeHandle<CombineTag, CombineFn, IntCRank, S, ES,
                                 CModesSeq, IntNumOut, Ops...>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = CuteCombineTag<Coords, TileShape, OpEvals...>;
  using value_type  = S;
  using exec_space  = ES;

  static constexpr int Rank   = node_type::Rank;
  static constexpr int NumOps = node_type::NumOps;
  static constexpr int NumOut = node_type::NumOut;

 private:
  template <std::size_t K>
  using op_eval_t = std::tuple_element_t<K, std::tuple<OpEvals...>>;
  template <std::size_t K>
  using op_node_t = std::tuple_element_t<K, std::tuple<Ops...>>;
  template <std::size_t K>
  using op_perm_t =
      Impl::label_perm_seq_t<CModesSeq, typename op_node_t<K>::modes_seq>;

  template <std::size_t K>
  static constexpr bool operand_fits() {
    using E = op_eval_t<K>;
    if constexpr (Impl::is_cute_fragment_eval_v<E>)
      return std::is_same_v<typename op_node_t<K>::modes_seq, CModesSeq> &&
             std::is_same_v<typename E::tile_shape_type, TileShape> &&
             decltype(cute::size(
                 std::declval<typename E::storage_type>()))::value ==
                 decltype(cute::size(std::declval<Coords>()))::value;
    else
      return std::is_same_v<
          decltype(cute::flatten(cute::shape(Impl::select_seq(
              std::declval<typename E::storage_type::layout_type>(),
              op_perm_t<K>{})))),
          TileShape>;
  }

  template <std::size_t... Ks>
  static constexpr bool operands_fit(std::index_sequence<Ks...>) {
    return (operand_fits<Ks>() && ...);
  }

  static_assert(sizeof...(OpEvals) == NumOps,
                "CuTe combine: one operand evaluator per node operand");
  static_assert(((OpEvals::Rank == Rank) && ...),
                "CuTe combine: every operand must have the output's rank");
  static_assert((Impl::cute_combine_operand_ok<OpEvals>() && ...),
                "CuTe combine: an operand must be a fragment node or a "
                "shared-memory cute::Tensor with a static layout");
  static_assert(operands_fit(std::make_index_sequence<NumOps>{}),
                "CuTe combine: a fragment operand must be labelled in the "
                "output's order and hold one value per assigned coordinate; a "
                "shared-memory operand must present the tile's extents in the "
                "output's order");

 public:
  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : fn_(n.fn),
        ops_(t.ops),
        origin_(t.origin),
        active_(t.active),
        coords_(t.coords) {}

  KOKKOS_FUNCTION auto operator()() const {
    constexpr int nv = decltype(cute::size(std::declval<Coords>()))::value;
    if constexpr (NumOut == 0) {
      if (active_) {
        CUTE_UNROLL
        for (int v = 0; v < nv; ++v) {
          const auto oc = cute::flatten(coords_(v));
          if constexpr (Impl::is_scatter_frag_fn_v<CombineFn>)
            fn_.add(v, read<0>(v, oc));
          else
            Impl::apply_combine(
                fn_, global_index(oc, std::make_index_sequence<Rank>{}),
                gather(v, oc, std::make_index_sequence<NumOps>{}));
        }
      }
    } else {
      using frag_t = decltype(cute::make_tensor<S>(cute::shape(coords_)));
      Kokkos::Array<frag_t, NumOut> outs;
      if (active_) {
        CUTE_UNROLL
        for (int v = 0; v < nv; ++v) {
          const auto oc = cute::flatten(coords_(v));
          const auto r  = Impl::as_output_array<S>(Impl::apply_combine(
              fn_, global_index(oc, std::make_index_sequence<Rank>{}),
              gather(v, oc, std::make_index_sequence<NumOps>{})));
          CUTE_UNROLL
          for (int m = 0; m < NumOut; ++m) outs[m](v) = r[m];
        }
      }
      return Impl::make_cute_fragment_value_evaluators<ES, Rank, TileShape>(
          outs, coords_, std::make_index_sequence<NumOut>{});
    }
  }

 private:
  template <std::size_t K, typename Coord>
  KOKKOS_FUNCTION S read(int v, const Coord& oc) const {
    const auto& n = ops_.template get<K>().node();
    if constexpr (Impl::is_cute_fragment_eval_v<op_eval_t<K>>)
      return static_cast<S>(n.frag_(v));
    else
      return static_cast<S>(cute::make_tensor(
          n.storage_.data(),
          Impl::select_seq(n.storage_.layout(), op_perm_t<K>{}))(oc));
  }

  template <typename Coord, std::size_t... Ks>
  KOKKOS_FUNCTION Kokkos::Array<S, NumOps> gather(
      int v, const Coord& oc, std::index_sequence<Ks...>) const {
    return {read<Ks>(v, oc)...};
  }

  template <typename Coord, std::size_t... Ds>
  KOKKOS_FUNCTION Kokkos::Array<int, Rank> global_index(
      const Coord& oc, std::index_sequence<Ds...>) const {
    return {origin_[Ds] + static_cast<int>(cute::get<Ds>(oc))...};
  }

  [[no_unique_address]] CombineFn fn_;
  DeviceTuple<OpEvals...>         ops_;
  Kokkos::Array<int, Rank>        origin_;
  bool                            active_;
  Coords                          coords_;
};

template <typename ES, typename Storage, int R, typename HookOp,
          typename Partitioner>
class Evaluator<
    CutePolicyTag<ES>,
    NodeHandle<IntermTag, Storage, std::integral_constant<int, R>, ES, HookOp>,
    CuteFragmentStoreTag<Partitioner>> {
  static_assert(cute::is_tensor<Storage>::value &&
                    cute::is_smem<Storage>::value,
                "CuTe fragment store: destination must be a shared-memory "
                "cute::Tensor");
  static_assert(cute::is_static<typename Storage::layout_type>::value,
                "CuTe fragment store: destination layout must be static");

 public:
  using node_type   = NodeHandle<IntermTag, Storage,
                                 std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = CuteFragmentStoreTag<Partitioner>;
  static constexpr int Rank = R;
  using storage_type        = Storage;
  using value_type          = typename node_type::value_type;
  using exec_space          = ES;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  template <typename FragEval>
  KOKKOS_FUNCTION auto operator=(const FragEval& src) const {
    static_assert(Impl::is_cute_fragment_eval_v<FragEval>,
                  "CuTe fragment store: the source must be a fragment "
                  "evaluator");
    static_assert(FragEval::Rank == R,
                  "CuTe fragment store: fragment and destination must have "
                  "equal rank");
    static_assert(
        std::is_same_v<decltype(cute::flatten(
                           cute::shape(std::declval<Storage>()))),
                       typename FragEval::tile_shape_type>,
        "CuTe fragment store: the destination must have the fragment's tile "
        "extents");
    static_assert(
        Impl::partitions_like_v<Partitioner, Storage,
                                typename FragEval::storage_type>,
        "CuTe fragment store: the partitioner must partition the destination "
        "exactly as the fragment's producer partitioned it");

    const auto dst = node_.storage_;
    if (tag_.part.active()) cute::copy(src.node().frag_, tag_.part(dst));
    return Impl::make_cute_value_evaluator<ES>(dst, src.node().hook_op);
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

template <typename ES, typename Frag, typename Coords, typename TileShape,
          int R, typename HookOp, typename Partitioner>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<FragmentTag, Frag, Coords, TileShape,
                           std::integral_constant<int, R>, ES, HookOp>,
                CuteFragmentStoreTag<Partitioner>> {
  static_assert(
      Impl::partitions_like_v<
          Partitioner, decltype(cute::make_identity_tensor(TileShape{})), Frag>,
      "CuTe fragment store: the partitioner must partition the tile exactly "
      "as the fragment's producer partitioned it");

 public:
  using node_type    = NodeHandle<FragmentTag, Frag, Coords, TileShape,
                                  std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag   = CutePolicyTag<ES>;
  using tiling_type  = CuteFragmentStoreTag<Partitioner>;
  using storage_type = Frag;
  using value_type   = typename node_type::value_type;
  using exec_space   = ES;
  static constexpr int Rank = R;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  template <typename Coord, typename T, typename ModesSeq, typename PermSeq>
  KOKKOS_FUNCTION void operator()(const Coord&                     coord,
                                  const TensorHandle<T, ModesSeq>& out,
                                  PermSeq                          perm) const {
    Impl::cute_store_global<R>(node_.frag_, tag_.part, TileShape{}, coord, out,
                               perm, node_.hook_op);
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

}  // namespace TensorOperations
