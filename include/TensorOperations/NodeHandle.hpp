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

// Primary template — undefined; must use a specialization
template <typename Tag, typename... Args>
struct NodeHandle;

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Does Node instantiate NodeHandle with this tag? One trait covers all tags
// (InputTag, ContractionTag, ...).
template <typename Tag, typename Node>
struct has_node_tag : std::false_type {};
template <typename Tag, typename... Args>
struct has_node_tag<Tag, NodeHandle<Tag, Args...>> : std::true_type {};
template <typename Tag, typename Node>
inline constexpr bool has_node_tag_v = has_node_tag<Tag, Node>::value;

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

}  // namespace Impl

// ---------------------------------------------------------------------------
// Input specialization — wraps an existing TensorHandle
// ---------------------------------------------------------------------------
template <TensorLike T, typename ModesSeq, typename HookOp>
struct NodeHandle<InputTag, T, ModesSeq, HookOp> {
  TensorHandle<T, ModesSeq>    handle;
  [[no_unique_address]] HookOp hook_op;

  static constexpr int Rank = TensorHandle<T, ModesSeq>::Rank;
  using value_type          = typename Impl::value_type_of<T>::type;
  using exec_space          = typename Impl::exec_space_of<T>::type;
  using modes_seq           = ModesSeq;

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const {
    Kokkos::Array<int, Rank> s{};
    for (int i = 0; i < Rank; ++i) s[i] = static_cast<int>(handle.extent(i));
    return s;
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

  Storage                      storage_;  // Kokkos::View (scratch tier)
  [[no_unique_address]] HookOp hook_op;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

// Input node — wraps an existing TensorHandle
template <TensorLike T, typename ModesSeq, typename HookOp = NoHook>
KOKKOS_FUNCTION NodeHandle<InputTag, T, ModesSeq, HookOp> make_input_node(
    TensorHandle<T, ModesSeq> h, HookOp hook = {}) {
  return {std::move(h), std::move(hook)};
}

// ---------------------------------------------------------------------------
// Contraction specialization — C{free modes} = sum{contracted} A × B
// ---------------------------------------------------------------------------
// ModesSeq holds the output labels in *canonical* (freeA ++ freeB) order; the
// stored shape_ is in that order too. PermCSeq maps canonical output axes back
// to the user-requested output order (canonical -> user), so the driver can
// present the user's output view/tile canonically.
template <typename NodeA, typename NodeB, typename IntCRank, typename Scalar,
          typename ExecSpace, typename HookOp, typename ModesSeq,
          typename PermCSeq>
struct NodeHandle<ContractionTag, NodeA, NodeB, IntCRank, Scalar, ExecSpace,
                  HookOp, ModesSeq, PermCSeq> {
  static constexpr int Rank          = IntCRank::value;
  static constexpr int NumContracted = (NodeA::Rank + NodeB::Rank - Rank) / 2;
  using hook_type                    = HookOp;
  using value_type                   = Scalar;
  using exec_space                   = ExecSpace;
  using node_a_type                  = NodeA;
  using node_b_type                  = NodeB;
  using modes_seq                    = ModesSeq;  // canonical output labels
  using permC_seq                    = PermCSeq;  // canonical -> user output

  NodeA                        node_a;
  NodeB                        node_b;
  Kokkos::Array<int, Rank>     shape_;  // canonical output extents
  [[no_unique_address]] HookOp hook_op;

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
};

// ---------------------------------------------------------------------------
// Contraction node factory
//
// C{out modes} = sum{contracted} A x B, with the output modes given as
// compile-time labels. Contracted modes are inferred as the labels shared by A
// and B; free modes are those in exactly one input. The output is stored in
// canonical (freeA ++ freeB) order; the user's requested order is recorded as a
// canonical->user permutation so the driver can write the result back
// correctly.
// ---------------------------------------------------------------------------

namespace Impl {

template <typename ActualScalar, typename ExecSpace, int32_t... OutModes,
          typename NodeA, typename NodeB, typename HookOp>
auto make_contraction_node_impl(NodeA a, NodeB b, HookOp hook) {
  constexpr int Rank = static_cast<int>(sizeof...(OutModes));
  static_assert((NodeA::Rank + NodeB::Rank - Rank) % 2 == 0,
                "Output rank is inconsistent with input ranks");

  using AModes = typename NodeA::modes_seq;
  using BModes = typename NodeB::modes_seq;
  using OutSeq = std::integer_sequence<int32_t, OutModes...>;
  static_assert(
      Impl::valid_contraction<Rank, AModes, BModes, OutSeq>(),
      "each output mode must appear in exactly one input tensor, output modes "
      "must be pairwise distinct, and the output rank must equal the number "
      "of free modes");

  // Canonical output labels (freeA ++ freeB) and the canonical->user map.
  using CanonModes = Impl::canonC_modes_seq_t<Rank, AModes, BModes>;
  using PermC      = Impl::permC_seq_t<Rank, AModes, BModes, OutSeq>;

  // Canonical C is freeA ++ freeB, and permA/permB already locate those axes
  // in each operand: canonical output axis i draws its extent from A axis
  // pA[i] (i < FreeA) or from B axis pB[NC + (i - FreeA)].
  constexpr auto pA    = Impl::permA_v<AModes, BModes>;
  constexpr auto pB    = Impl::permB_v<AModes, BModes>;
  constexpr int  NC    = Impl::num_contracted<AModes, BModes>();
  constexpr int  FreeA = NodeA::Rank - NC;

  const auto a_shape = a.shape();
  const auto b_shape = b.shape();

  // Contracted extents must agree between A and B (A axis pA[FreeA + i] pairs
  // with B axis pB[i] on the same label).
  for (int i = 0; i < NC; ++i)
    assert(a_shape[pA[FreeA + i]] == b_shape[pB[i]] &&
           "contracted mode extents must match between A and B");

  Kokkos::Array<int, Rank> c_shape{};
  for (int i = 0; i < Rank; ++i)
    c_shape[i] = i < FreeA ? a_shape[pA[i]] : b_shape[pB[NC + (i - FreeA)]];

  return NodeHandle<ContractionTag, NodeA, NodeB,
                    std::integral_constant<int, Rank>, ActualScalar, ExecSpace,
                    HookOp, CanonModes, PermC>{std::move(a), std::move(b),
                                               c_shape, std::move(hook)};
}

}  // namespace Impl

// Primary form: infer the output scalar from NodeA, default execution space.
//   make_contraction_node<'l','i'>(a, b)
template <int32_t... OutModes, typename NodeA, typename NodeB,
          typename HookOp = NoHook>
auto make_contraction_node(NodeA a, NodeB b, HookOp hook = {}) {
  return Impl::make_contraction_node_impl<
      typename NodeA::value_type, Kokkos::DefaultExecutionSpace, OutModes...>(
      std::move(a), std::move(b), std::move(hook));
}

// Explicit scalar (and execution-space) override:
//   make_contraction_node<double, Kokkos::DefaultExecutionSpace, 'i','k'>(a, b)
template <typename Scalar, typename ExecSpace = Kokkos::DefaultExecutionSpace,
          int32_t... OutModes, typename NodeA, typename NodeB,
          typename HookOp = NoHook>
auto make_contraction_node(NodeA a, NodeB b, HookOp hook = {}) {
  return Impl::make_contraction_node_impl<Scalar, ExecSpace, OutModes...>(
      std::move(a), std::move(b), std::move(hook));
}

// ---------------------------------------------------------------------------
// canonicalize_input — present an operand in the GEMM's canonical axis order.
//
// Identity permutation returns the node unchanged (native layout, fast staging
// path); otherwise the operand's data is wrapped in a zero-copy PermutedView
// carrying canonical labels. Only input operands may be permuted (permuted
// nested/intermediate operands are a follow-up).
// ---------------------------------------------------------------------------
namespace Impl {

template <typename Node, int... Perm>
KOKKOS_FUNCTION auto canonicalize_input(
    const Node& node, std::integer_sequence<int, Perm...> perm) {
  if constexpr (is_identity_seq(perm)) {
    return node;
  } else {
    static_assert(has_node_tag_v<InputTag, Node>,
                  "permuted non-input operands are not supported yet");
    using PermSeq    = std::integer_sequence<int, Perm...>;
    using CanonModes = gather_modes_seq_t<typename Node::modes_seq, PermSeq>;
    auto pv          = permuted_alias(node.handle, perm);
    return make_input_node(make_handle_seq(pv, CanonModes{}), node.hook_op);
  }
}

}  // namespace Impl

}  // namespace TensorOperations
