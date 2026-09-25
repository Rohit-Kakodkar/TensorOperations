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

template <typename ThrLayout>
struct CuteThreadTag {
  ThrLayout thr_layout;
  int       thr_idx;
};

template <typename ThrLayout>
struct CuteSmemLoadTag : CuteThreadTag<ThrLayout> {};

template <typename ThrLayout>
struct CuteStoreTag : CuteThreadTag<ThrLayout> {};

template <typename TileShape, typename ThrLayout>
struct CuteStagedTag : CuteThreadTag<ThrLayout> {};

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
          typename TileShape, typename ThrLayout>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>,
                CuteStagedTag<TileShape, ThrLayout>> {
 public:
  using node_type   = NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>;
  using policy_tag  = CutePolicyTag<ES>;
  using tiling_type = CuteStagedTag<TileShape, ThrLayout>;
  using value_type  = typename node_type::value_type;
  using exec_space  = ES;
  using modes_seq   = typename node_type::modes_seq;
  static constexpr int Rank = node_type::Rank;

  static_assert(cute::is_static<TileShape>::value &&
                    cute::rank_v<TileShape> == Rank,
                "CuTe staged: the tile shape must be static with one mode per "
                "operand mode");
  static_assert(cute::is_static<ThrLayout>::value &&
                    cute::rank_v<ThrLayout> == Rank,
                "CuTe staged: the thread layout must be static with one mode "
                "per operand mode");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  template <typename Coord>
  KOKKOS_FUNCTION auto operator()(const Coord& coord) const {
    const auto src =
        make_evaluator<CutePolicyTag<ES>>(node_.operand_, TileShape{})(coord);
    const CuteThreadPartitioner<ThrLayout> part{tag_.thr_layout, tag_.thr_idx};
    const auto coords = part(cute::make_identity_tensor(TileShape{}));
    auto       frag   = cute::make_tensor<value_type>(cute::shape(coords));
    if (part.active()) cute::copy(part(src.node().storage_), frag);
    return Impl::make_cute_fragment_value_evaluator<ES, Rank, TileShape>(
        frag, coords, src.node().hook_op);
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
KOKKOS_FUNCTION auto select_seq(const Layout& l,
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

template <typename ES, typename NA, typename NB, typename IntCRank, typename S,
          typename HookOp, typename CModesSeq, typename PermCSeq,
          typename AEval, typename BEval, typename TiledMma>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                           CModesSeq, PermCSeq>,
                CuteContractTag<AEval, BEval, TiledMma>> {
 public:
  using node_type  = NodeHandle<ContractionTag, NA, NB, IntCRank, S, ES, HookOp,
                                CModesSeq, PermCSeq>;
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
    cute::gemm(tag_.mma, thr.partition_A(sA), thr.partition_B(sB), frag);

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

template <typename... OpEvals>
constexpr std::size_t first_fragment_index() {
  const bool frag[] = {is_cute_fragment_eval_v<OpEvals>...};
  for (std::size_t k = 0; k < sizeof...(OpEvals); ++k)
    if (frag[k]) return k;
  return sizeof...(OpEvals);
}

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

template <typename... OpEvals>
struct CuteCombineTag {
  DeviceTuple<OpEvals...>                              ops;
  Kokkos::Array<int, Impl::combine_rank_v<OpEvals...>> origin{};
};

template <typename ThrLayout, typename... OpEvals>
struct CuteCombineThreadTag : CuteThreadTag<ThrLayout> {
  DeviceTuple<OpEvals...>                              ops;
  Kokkos::Array<int, Impl::combine_rank_v<OpEvals...>> origin{};
};

namespace Impl {

template <typename ES, typename Node, typename... OpEvals>
class CuteCombineEvaluator;

template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename... OpEvals>
class CuteCombineEvaluator<ES,
                           NodeHandle<CombineTag, CombineFn, IntCRank, S, ES,
                                      CModesSeq, IntNumOut, Ops...>,
                           OpEvals...> {
 public:
  using node_type  = NodeHandle<CombineTag, CombineFn, IntCRank, S, ES,
                                CModesSeq, IntNumOut, Ops...>;
  using policy_tag = CutePolicyTag<ES>;
  using value_type = S;
  using exec_space = ES;

  static constexpr int Rank   = node_type::Rank;
  static constexpr int NumOps = node_type::NumOps;
  static constexpr int NumOut = node_type::NumOut;

  static_assert(sizeof...(OpEvals) == NumOps,
                "CuTe combine: one operand evaluator per node operand");
  static_assert(NumOut >= 1,
                "CuTe combine: a sink combine (fn returning void) is not "
                "supported by the CuTe backend yet");
  static_assert(((OpEvals::Rank == Rank) && ...),
                "CuTe combine: every operand must have the output's rank");
  static_assert((cute_combine_operand_ok<OpEvals>() && ...),
                "CuTe combine: an operand must be a fragment node or a "
                "shared-memory cute::Tensor with a static layout");

  template <std::size_t K>
  using view_shape_t = decltype(cute::shape(select_seq(
      std::declval<typename std::tuple_element_t<
          K, std::tuple<OpEvals...>>::storage_type::layout_type>(),
      label_perm_seq_t<CModesSeq, typename std::tuple_element_t<
                                      K, std::tuple<Ops...>>::modes_seq>{})));

  template <std::size_t K>
  using flat_view_shape_t =
      decltype(cute::flatten(std::declval<view_shape_t<K>>()));

  KOKKOS_FUNCTION CuteCombineEvaluator(node_type n, DeviceTuple<OpEvals...> ops,
                                       Kokkos::Array<int, Rank> origin)
      : fn_(n.fn), ops_(ops), origin_(origin) {}

 protected:
  template <std::size_t K>
  using op_eval_t = std::tuple_element_t<K, std::tuple<OpEvals...>>;
  template <std::size_t K>
  using op_node_t = std::tuple_element_t<K, std::tuple<Ops...>>;
  template <std::size_t K>
  using op_perm_t =
      label_perm_seq_t<CModesSeq, typename op_node_t<K>::modes_seq>;

  template <std::size_t K>
  KOKKOS_FUNCTION auto view() const {
    const auto& s = ops_.template get<K>().node().storage_;
    return cute::make_tensor(s.data(), select_seq(s.layout(), op_perm_t<K>{}));
  }

  template <typename TileShape, typename Coords, typename Proto>
  KOKKOS_FUNCTION auto evaluate(const Coords& coords,
                                const Proto&  proto) const {
    using frag_t = decltype(cute::make_tensor<S>(cute::shape(proto)));
    Kokkos::Array<frag_t, NumOut> outs;
    for (int v = 0; v < static_cast<int>(cute::size(coords)); ++v) {
      const auto oc = cute::flatten(coords(v));
      const auto r  = as_output_array<S>(
          apply_combine(fn_, global_index(oc, std::make_index_sequence<Rank>{}),
                        gather(v, oc, std::make_index_sequence<NumOps>{})));
      for (int m = 0; m < NumOut; ++m) outs[m](v) = r[m];
    }
    return make_results<TileShape>(outs, coords,
                                   std::make_index_sequence<NumOut>{});
  }

  [[no_unique_address]] CombineFn fn_;
  DeviceTuple<OpEvals...>         ops_;
  Kokkos::Array<int, Rank>        origin_;

 private:
  template <std::size_t K, typename Coord>
  KOKKOS_FUNCTION S read(int v, const Coord& oc) const {
    if constexpr (is_cute_fragment_eval_v<op_eval_t<K>>)
      return static_cast<S>(ops_.template get<K>().node().frag_(v));
    else
      return static_cast<S>(view<K>()(oc));
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

  template <typename TileShape, typename Frags, typename Coords,
            std::size_t... Ms>
  KOKKOS_FUNCTION auto make_results(const Frags& outs, const Coords& coords,
                                    std::index_sequence<Ms...>) const {
    using result_t =
        decltype(make_cute_fragment_value_evaluator<ES, Rank, TileShape>(
            outs[0], coords, NoHook{}));
    return Kokkos::Array<result_t, NumOut>{
        make_cute_fragment_value_evaluator<ES, Rank, TileShape>(
            outs[Ms], coords, NoHook{})...};
  }
};

template <std::size_t D, typename CModesSeq, typename OpEvals, typename Ops,
          typename Is>
struct fragments_share_driver;

template <std::size_t D, typename CModesSeq, typename... OpEvals,
          typename... Ops, std::size_t... Ks>
struct fragments_share_driver<D, CModesSeq, std::tuple<OpEvals...>,
                              std::tuple<Ops...>, std::index_sequence<Ks...>> {
  using driver_t = std::tuple_element_t<D, std::tuple<OpEvals...>>;

  template <typename Eval, typename OpNode>
  static constexpr bool one() {
    if constexpr (!is_cute_fragment_eval_v<Eval>)
      return true;
    else
      return std::is_same_v<typename Eval::coords_type,
                            typename driver_t::coords_type> &&
             std::is_same_v<typename Eval::tile_shape_type,
                            typename driver_t::tile_shape_type> &&
             std::is_same_v<
                 decltype(cute::shape(
                     std::declval<typename Eval::storage_type>())),
                 decltype(cute::shape(
                     std::declval<typename driver_t::storage_type>()))> &&
             std::is_same_v<typename OpNode::modes_seq, CModesSeq>;
  }

  static constexpr bool value = (one<OpEvals, Ops>() && ...);
};

template <typename Base, typename TileShape, typename OpEvals, typename Is>
struct smem_operands_match_tile;

template <typename Base, typename TileShape, typename... OpEvals,
          std::size_t... Ks>
struct smem_operands_match_tile<Base, TileShape, std::tuple<OpEvals...>,
                                std::index_sequence<Ks...>> {
  template <typename Eval, std::size_t K>
  static constexpr bool one() {
    if constexpr (is_cute_fragment_eval_v<Eval>)
      return true;
    else
      return std::is_same_v<typename Base::template flat_view_shape_t<K>,
                            TileShape>;
  }

  static constexpr bool value = (one<OpEvals, Ks>() && ...);
};

template <typename Base, typename Is>
struct view_shapes_agree;

template <typename Base, std::size_t... Ks>
struct view_shapes_agree<Base, std::index_sequence<Ks...>> {
  static constexpr bool value =
      (std::is_same_v<typename Base::template view_shape_t<Ks>,
                      typename Base::template view_shape_t<0>> &&
       ...);
};

}  // namespace Impl

template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename... OpEvals>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                           IntNumOut, Ops...>,
                CuteCombineTag<OpEvals...>>
    : public Impl::CuteCombineEvaluator<
          ES,
          NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                     IntNumOut, Ops...>,
          OpEvals...> {
  using base =
      Impl::CuteCombineEvaluator<ES,
                                 NodeHandle<CombineTag, CombineFn, IntCRank, S,
                                            ES, CModesSeq, IntNumOut, Ops...>,
                                 OpEvals...>;
  static constexpr std::size_t D = Impl::first_fragment_index<OpEvals...>();

  static_assert(D < sizeof...(OpEvals),
                "CuTe combine: CuteCombineTag needs a fragment operand to "
                "decide which elements each thread owns; with only "
                "shared-memory operands use CuteCombineThreadTag");
  static_assert(
      Impl::fragments_share_driver<D, CModesSeq, std::tuple<OpEvals...>,
                                   std::tuple<Ops...>,
                                   std::index_sequence_for<OpEvals...>>::value,
      "CuTe combine: every fragment operand must share the driving "
      "fragment's partition and be labelled in the combine's output order");
  static_assert(
      Impl::smem_operands_match_tile<
          base,
          typename std::tuple_element_t<
              D, std::tuple<OpEvals...>>::tile_shape_type,
          std::tuple<OpEvals...>, std::index_sequence_for<OpEvals...>>::value,
      "CuTe combine: every shared-memory operand must present the driving "
      "fragment's tile extents in the output's mode order");

 public:
  using typename base::node_type;
  using tiling_type = CuteCombineTag<OpEvals...>;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : base(n, t.ops, t.origin) {}

  KOKKOS_FUNCTION auto operator()() const {
    const auto& d = this->ops_.template get<D>().node();
    return this->template evaluate<
        typename std::decay_t<decltype(d)>::tile_shape_type>(d.coords_,
                                                             d.frag_);
  }
};

template <typename ES, typename CombineFn, typename IntCRank, typename S,
          typename CModesSeq, typename IntNumOut, typename... Ops,
          typename ThrLayout, typename... OpEvals>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                           IntNumOut, Ops...>,
                CuteCombineThreadTag<ThrLayout, OpEvals...>>
    : public Impl::CuteCombineEvaluator<
          ES,
          NodeHandle<CombineTag, CombineFn, IntCRank, S, ES, CModesSeq,
                     IntNumOut, Ops...>,
          OpEvals...> {
  using base =
      Impl::CuteCombineEvaluator<ES,
                                 NodeHandle<CombineTag, CombineFn, IntCRank, S,
                                            ES, CModesSeq, IntNumOut, Ops...>,
                                 OpEvals...>;

  static_assert((!Impl::is_cute_fragment_eval_v<OpEvals> && ...),
                "CuTe combine: CuteCombineThreadTag takes shared-memory "
                "operands only; a fragment operand decides thread ownership "
                "itself, so use CuteCombineTag");
  static_assert(cute::is_static<ThrLayout>::value &&
                    cute::rank_v<ThrLayout> == base::Rank,
                "CuTe combine: the thread layout must be static with one mode "
                "per output mode");

  static_assert(
      Impl::view_shapes_agree<base, std::index_sequence_for<OpEvals...>>::value,
      "CuTe combine: every operand must present the same extents in the "
      "output's mode order");

 public:
  using typename base::node_type;
  using tiling_type = CuteCombineThreadTag<ThrLayout, OpEvals...>;

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type t)
      : base(n, t.ops, t.origin),
        thr_layout_(t.thr_layout),
        thr_idx_(t.thr_idx) {}

  KOKKOS_FUNCTION auto operator()() const {
    const auto cC = cute::local_partition(
        cute::make_identity_tensor(cute::shape(this->template view<0>())),
        thr_layout_, thr_idx_);
    return this
        ->template evaluate<typename base::template flat_view_shape_t<0>>(cC,
                                                                          cC);
  }

 private:
  ThrLayout thr_layout_;
  int       thr_idx_;
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
