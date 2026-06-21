#pragma once
#include <TensorOperations/TensorHandle.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// Tags for NodeHandle specializations
struct InputTag {};
struct IntermTag {};
struct ContractionTag {};

// Sentinel for "no hook"
struct NoHook {};

// Sentinel for "infer Scalar from NodeA::value_type" in make_contraction_node
struct InferScalar {};

// Primary template — undefined; must use a specialization
template <typename Tag, typename... Args>
struct NodeHandle;

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Safely extracts T::value_type when it exists; falls back to void.
template <typename T, typename = void>
struct value_type_of {
  using type = void;
};
template <typename T>
struct value_type_of<T, std::void_t<typename T::value_type>> {
  using type = typename T::value_type;
};

// Extracts ExecSpace from a Kokkos View-like T (has ::execution_space).
// Falls back to DefaultExecutionSpace for plain TensorLike types.
template <typename T, typename = void>
struct exec_space_of {
  using type = Kokkos::DefaultExecutionSpace;
};
template <typename T>
struct exec_space_of<T, std::void_t<typename T::execution_space>> {
  using type = typename T::execution_space;
};

// Map (Scalar, Rank, MemSpace) → the appropriate Kokkos::View type
template <typename S, int R, typename MS>
struct KokkosViewN;
template <typename S, typename MS>
struct KokkosViewN<S, 0, MS> {
  using type = Kokkos::View<S, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS>
struct KokkosViewN<S, 1, MS> {
  using type = Kokkos::View<S*, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS>
struct KokkosViewN<S, 2, MS> {
  using type = Kokkos::View<S**, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS>
struct KokkosViewN<S, 3, MS> {
  using type = Kokkos::View<S***, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS>
struct KokkosViewN<S, 4, MS> {
  using type = Kokkos::View<S****, Kokkos::LayoutRight, MS>;
};

// Construct a KokkosViewN from a shape array via index_sequence unpacking
template <typename ViewType, int R, std::size_t... Is>
ViewType make_view_impl(const Kokkos::Array<int, R>& shape,
                        std::index_sequence<Is...>) {
  return ViewType("intermediate", static_cast<std::size_t>(shape[Is])...);
}

template <typename S, int R, typename MS>
typename KokkosViewN<S, R, MS>::type make_kokkos_view(
    const Kokkos::Array<int, R>& shape) {
  using VT = typename KokkosViewN<S, R, MS>::type;
  return make_view_impl<VT, R>(shape, std::make_index_sequence<R>{});
}

}  // namespace Impl

// ---------------------------------------------------------------------------
// Input specialization — wraps an existing TensorHandle
// ---------------------------------------------------------------------------
template <TensorLike T, typename HookOp>
struct NodeHandle<InputTag, T, HookOp> {
  TensorHandle<T>              handle;
  [[no_unique_address]] HookOp hook_op;

  static constexpr int Rank = TensorHandle<T>::Rank;
  using value_type          = typename Impl::value_type_of<T>::type;
  using exec_space          = typename Impl::exec_space_of<T>::type;

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const {
    Kokkos::Array<int, Rank> s{};
    for (int i = 0; i < Rank; ++i) s[i] = static_cast<int>(handle.extent(i));
    return s;
  }
  KOKKOS_FUNCTION Kokkos::Array<int32_t, Rank> modes() const {
    return handle.modes;
  }
};

// ---------------------------------------------------------------------------
// Intermediate specialization — a staged/computed tile, parameterized by its
// storage. Storage is a Kokkos::View (scratch / team tier) exposing value_type,
// extent(i) and operator()(idx...), so the tile is read uniformly. Also reused
// for deferred full-tensor intermediates (empty View storage).
// ---------------------------------------------------------------------------
template <typename Storage, typename IntRank, typename ExecSpace,
          typename HookOp>
struct NodeHandle<IntermTag, Storage, IntRank, ExecSpace, HookOp> {
  static constexpr int Rank = IntRank::value;
  using storage_type        = Storage;
  using value_type          = typename Storage::value_type;
  using exec_space          = ExecSpace;

  uint32_t                     id{};
  Storage                      storage_;  // Kokkos::View (scratch tier)
  Kokkos::Array<int, Rank>     shape_;
  Kokkos::Array<int32_t, Rank> modes_;
  [[no_unique_address]] HookOp hook_op;

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
  KOKKOS_FUNCTION Kokkos::Array<int32_t, Rank> modes() const { return modes_; }
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

// Input node — wraps an existing TensorHandle
template <TensorLike T, typename HookOp = NoHook>
NodeHandle<InputTag, T, HookOp> make_input_node(TensorHandle<T> h,
                                                HookOp          hook = {}) {
  return {std::move(h), std::move(hook)};
}

// Intermediate node — deferred allocation, ExecSpace determines both memory
// spaces. Storage is the matching Kokkos::View (empty until allocated).
template <typename Scalar, int Rank,
          typename ExecSpace = Kokkos::DefaultExecutionSpace,
          typename HookOp    = NoHook>
auto make_interm_node(uint32_t id, std::array<int, Rank> shape,
                      std::array<int32_t, Rank> modes, HookOp hook = {}) {
  using mem_space = typename ExecSpace::memory_space;
  using view_type = typename Impl::KokkosViewN<Scalar, Rank, mem_space>::type;
  using NodeType =
      NodeHandle<IntermTag, view_type, std::integral_constant<int, Rank>,
                 ExecSpace, HookOp>;
  Kokkos::Array<int, Rank>     kshape{};
  Kokkos::Array<int32_t, Rank> kmodes{};
  for (int i = 0; i < Rank; ++i) {
    kshape[i] = shape[i];
    kmodes[i] = modes[i];
  }
  return NodeType{id, view_type{}, kshape, kmodes, std::move(hook)};
}

// ---------------------------------------------------------------------------
// Contraction specialization — C{free modes} = sum{contracted} A × B
// ---------------------------------------------------------------------------
template <typename NodeA, typename NodeB, typename IntCRank, typename Scalar,
          typename ExecSpace, typename HookOp>
struct NodeHandle<ContractionTag, NodeA, NodeB, IntCRank, Scalar, ExecSpace,
                  HookOp> {
  static constexpr int Rank          = IntCRank::value;
  static constexpr int NumContracted = (NodeA::Rank + NodeB::Rank - Rank) / 2;
  using hook_type                    = HookOp;
  using value_type                   = Scalar;
  using exec_space                   = ExecSpace;
  using node_a_type                  = NodeA;
  using node_b_type                  = NodeB;

  NodeA                        node_a;
  NodeB                        node_b;
  Kokkos::Array<int32_t, Rank> modes_;
  Kokkos::Array<int, Rank>     shape_;
  [[no_unique_address]] HookOp hook_op;

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
  KOKKOS_FUNCTION Kokkos::Array<int32_t, Rank> modes() const { return modes_; }
};

// ---------------------------------------------------------------------------
// Contraction node factory
// ---------------------------------------------------------------------------

template <typename Scalar    = InferScalar,
          typename ExecSpace = Kokkos::DefaultExecutionSpace, typename NodeA,
          typename NodeB, std::size_t CRank, typename HookOp = NoHook>
auto make_contraction_node(NodeA a, NodeB b,
                           std::array<int32_t, CRank> out_modes,
                           HookOp                     hook = {}) {
  using ActualScalar = std::conditional_t<std::is_same_v<Scalar, InferScalar>,
                                          typename NodeA::value_type, Scalar>;
  constexpr int Rank = static_cast<int>(CRank);
  static_assert((NodeA::Rank + NodeB::Rank - Rank) % 2 == 0,
                "Output rank is inconsistent with input ranks");

  const auto a_modes = a.modes();
  const auto b_modes = b.modes();
  const auto a_shape = a.shape();
  const auto b_shape = b.shape();

  auto in_array = [](auto val, const auto& arr, int n) {
    for (int i = 0; i < n; ++i)
      if (arr[i] == val) return true;
    return false;
  };

  // Each output mode must be a free mode (in exactly one input)
  for (int i = 0; i < Rank; ++i) {
    int32_t m    = out_modes[i];
    bool    in_a = in_array(m, a_modes, NodeA::Rank);
    bool    in_b = in_array(m, b_modes, NodeB::Rank);
    assert((in_a != in_b) &&
           "each output mode must appear in exactly one input tensor");
  }

  // Derive C's shape from input extents
  Kokkos::Array<int, Rank> c_shape{};
  for (int i = 0; i < Rank; ++i) {
    int32_t m = out_modes[i];
    if (in_array(m, a_modes, NodeA::Rank)) {
      for (int j = 0; j < NodeA::Rank; ++j)
        if (a_modes[j] == m) {
          c_shape[i] = a_shape[j];
          break;
        }
    } else {
      for (int j = 0; j < NodeB::Rank; ++j)
        if (b_modes[j] == m) {
          c_shape[i] = b_shape[j];
          break;
        }
    }
  }

  Kokkos::Array<int32_t, Rank> k_out_modes{};
  for (int i = 0; i < Rank; ++i) k_out_modes[i] = out_modes[i];

  return NodeHandle<ContractionTag, NodeA, NodeB,
                    std::integral_constant<int, Rank>, ActualScalar, ExecSpace,
                    HookOp>{std::move(a), std::move(b), k_out_modes, c_shape,
                            std::move(hook)};
}

}  // namespace TensorOperations
