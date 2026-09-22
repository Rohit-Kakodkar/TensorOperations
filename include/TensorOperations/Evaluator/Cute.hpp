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

namespace Impl {

template <typename ES, typename Storage, typename HookOp>
KOKKOS_FUNCTION auto make_cute_value_evaluator(Storage tile, HookOp hook) {
  using node_t =
      NodeHandle<IntermTag, Storage, std::integral_constant<int, Storage::rank>,
                 ES, HookOp>;
  return Evaluator<CutePolicyTag<ES>, node_t, void>(
      node_t{std::move(tile), std::move(hook)});
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

}  // namespace TensorOperations
