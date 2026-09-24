#pragma once

#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/TensorHandle.hpp>

#include <cute/tensor.hpp>

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

template <typename Frag, typename Coords, typename IntRank, typename ExecSpace,
          typename HookOp>
struct NodeHandle<FragmentTag, Frag, Coords, IntRank, ExecSpace, HookOp> {
  static_assert(cute::is_tensor<Frag>::value && cute::is_rmem<Frag>::value,
                "fragment node: storage must be a register cute::Tensor");
  static_assert(cute::is_tensor<Coords>::value,
                "fragment node: coordinates must be a cute::Tensor");
  static_assert(std::is_same_v<decltype(cute::shape(std::declval<Frag>())),
                               decltype(cute::shape(std::declval<Coords>()))>,
                "fragment node: fragment and coordinates must share a shape");

  using node_tag            = FragmentTag;
  static constexpr int Rank = IntRank::value;
  using storage_type        = Frag;
  using coords_type         = Coords;
  using value_type          = typename Frag::value_type;
  using exec_space          = ExecSpace;
  using hook_type           = HookOp;

  Frag                         frag_;
  Coords                       coords_;
  [[no_unique_address]] HookOp hook_op;
};

template <typename ES, int R, typename Frag, typename Coords,
          typename HookOp = NoHook>
KOKKOS_FUNCTION auto make_cute_fragment_node(Frag frag, Coords coords,
                                             HookOp hook = {}) {
  return NodeHandle<FragmentTag, Frag, Coords, std::integral_constant<int, R>,
                    ES, HookOp>{std::move(frag), std::move(coords),
                                std::move(hook)};
}

template <typename ES, typename Frag, typename Coords, int R, typename HookOp>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<FragmentTag, Frag, Coords,
                           std::integral_constant<int, R>, ES, HookOp>,
                void> {
 public:
  using node_type    = NodeHandle<FragmentTag, Frag, Coords,
                                  std::integral_constant<int, R>, ES, HookOp>;
  using policy_tag   = CutePolicyTag<ES>;
  using tiling_type  = void;
  using storage_type = Frag;
  using coords_type  = Coords;
  using value_type   = typename node_type::value_type;
  using exec_space   = ES;
  static constexpr int Rank = R;

  KOKKOS_FUNCTION explicit Evaluator(node_type n) : node_(n) {}

  KOKKOS_FUNCTION const node_type& node() const { return node_; }

 private:
  node_type node_;
};

namespace Impl {

template <typename ES, int R, typename Frag, typename Coords, typename HookOp>
KOKKOS_FUNCTION auto make_cute_fragment_value_evaluator(Frag   frag,
                                                        Coords coords,
                                                        HookOp hook) {
  auto node = make_cute_fragment_node<ES, R>(std::move(frag), std::move(coords),
                                             std::move(hook));
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

template <typename Smem, typename ThrLayout>
struct CuteStagedTag : CuteThreadTag<ThrLayout> {
  Smem dst;
};

template <typename AEval, typename BEval, typename TiledMma>
struct CuteContractTag {
  AEval    a;
  BEval    b;
  TiledMma mma;
  int      thr_idx;
};

namespace Impl {

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
  template <typename Tensor>
  KOKKOS_FUNCTION auto partition(const Tensor& t) const {
    return cute::local_partition(t, tag_.thr_layout, tag_.thr_idx);
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
    cute::copy(this->partition(src.node().storage_), this->partition(dst));

    return Impl::make_cute_value_evaluator<ES>(dst, src.node().hook_op);
  }
};

template <typename ES, typename Operand, typename ModesSeq, typename NodeTile,
          typename Smem, typename ThrLayout>
class Evaluator<CutePolicyTag<ES>,
                NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>,
                CuteStagedTag<Smem, ThrLayout>> {
  static_assert(cute::is_tensor<Smem>::value,
                "CuTe staged: destination must be a cute::Tensor");
  static_assert(cute::is_static<typename Smem::layout_type>::value,
                "CuTe staged: destination layout must be static");

 public:
  using node_type    = NodeHandle<StagedTag, Operand, ModesSeq, NodeTile>;
  using policy_tag   = CutePolicyTag<ES>;
  using tiling_type  = CuteStagedTag<Smem, ThrLayout>;
  using value_type   = typename node_type::value_type;
  using exec_space   = ES;
  using modes_seq    = typename node_type::modes_seq;
  using storage_type = Smem;
  static constexpr int Rank = node_type::Rank;

  static_assert(Smem::rank == Rank,
                "CuTe staged: destination rank must equal the operand's rank");

  KOKKOS_FUNCTION Evaluator(node_type n, tiling_type tag)
      : node_(n), tag_(tag) {}

  template <typename Coord>
  KOKKOS_FUNCTION auto operator()(const Coord& coord) const {
    auto src = make_evaluator<CutePolicyTag<ES>>(node_.operand_,
                                                 cute::shape(tag_.dst))(coord);
    auto stager = make_evaluator<CutePolicyTag<ES>>(
        make_cute_interm_node<ES>(tag_.dst),
        CuteSmemLoadTag<ThrLayout>{{tag_.thr_layout, tag_.thr_idx}});
    return (stager = src);
  }

  KOKKOS_FUNCTION const storage_type& storage() const { return tag_.dst; }

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

  template <typename Coord, typename T, typename ModesSeq, int... Perm>
  KOKKOS_FUNCTION void operator()(const Coord&                     coord,
                                  const TensorHandle<T, ModesSeq>& out,
                                  std::integer_sequence<int, Perm...>) const {
    static_assert(sizeof...(Perm) == R,
                  "store permutation must have one entry per output mode");
    static_assert(TensorHandle<T, ModesSeq>::Rank == R,
                  "CuTe store: output rank must equal the tile's rank");

    const auto g = Impl::make_cute_handle(out).tensor;
    const auto gc =
        cute::make_tensor(g.data(), cute::select<Perm...>(g.layout()));
    const auto tiler = cute::shape(this->node_.storage_);
    const auto sp    = this->partition(this->node_.storage_);
    auto       gp    = this->partition(cute::local_tile(gc, tiler, coord));

    if constexpr (std::is_same_v<HookOp, NoHook>) {
      cute::copy(sp, gp);
    } else {
      const auto cp = this->partition(cute::local_tile(
          cute::make_identity_tensor(cute::shape(gc)), tiler, coord));
      for (int i = 0; i < static_cast<int>(cute::size(sp)); ++i) {
        auto v = sp(i);
        Impl::apply_cute_hook(this->node_.hook_op, cp(i), v,
                              std::make_index_sequence<R>{});
        gp(i) = v;
      }
    }
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

    return Impl::make_cute_fragment_value_evaluator<ES, RankC>(frag, cC,
                                                               node_.hook_op);
  }

 private:
  node_type   node_;
  tiling_type tag_;
};

}  // namespace TensorOperations
