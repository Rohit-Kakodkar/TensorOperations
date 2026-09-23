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

}  // namespace TensorOperations
