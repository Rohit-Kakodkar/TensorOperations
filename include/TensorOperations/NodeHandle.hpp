#pragma once
#include <TensorOperations/DeviceTuple.hpp>
#include <TensorOperations/TensorHandle.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// Tags for NodeHandle specializations
struct InputTag {};
struct IntermTag {};
struct ContractionTag {};
struct CombineTag {};
struct SlotTag {};
struct StagedTag {};

// Sentinel for "no hook"
struct NoHook {};

// Forward declaration — TiledLayout.hpp defines it; exec_space_of below has to
// look through it, and including it here would be a cycle.
template <typename ViewType, typename Layout>
struct View;

// Primary template — undefined; must use a specialization
template <typename Tag, typename... Args>
struct NodeHandle;

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Does Node instantiate NodeHandle with this tag? One trait covers all tags
// (InputTag, ContractionTag, ...).
//
// The _v alias strips cv-ref before matching, and that is a safety property
// rather than a convenience. A partial specialization on NodeHandle<Tag,...>
// does not match `const NodeHandle<Tag,...>`, which is exactly what
// `decltype(some_const_node)` yields -- so without the decay the trait answers
// FALSE for a node it should recognize. Answering false is not inert: the
// staging predicates in Evaluator.hpp are written as if-constexpr chains whose
// fallthrough branch is the permissive one ("copied into its own buffer,
// therefore shape-unconstrained"), so a cv-qualified node silently acquires
// permissions its bare type is denied. Decay here, once, rather than at each of
// the call sites that would have to remember.
template <typename Tag, typename Node>
struct has_node_tag : std::false_type {};
template <typename Tag, typename... Args>
struct has_node_tag<Tag, NodeHandle<Tag, Args...>> : std::true_type {};
template <typename Tag, typename Node>
inline constexpr bool has_node_tag_v =
    has_node_tag<Tag, std::remove_cv_t<std::remove_reference_t<Node>>>::value;

// Is T any NodeHandle instantiation? Guards factory operands against
// structurally node-like non-node types: a CombineOutputHandle carries
// modes_seq/Rank/shape() (so it passes every label check) but is a terminal
// output, and would otherwise only fail deep inside evaluator instantiation.
template <typename T>
struct is_node_handle : std::false_type {};
template <typename Tag, typename... Args>
struct is_node_handle<NodeHandle<Tag, Args...>> : std::true_type {};
// Decayed for the same reason as has_node_tag_v above.
template <typename T>
inline constexpr bool is_node_handle_v =
    is_node_handle<std::remove_cv_t<std::remove_reference_t<T>>>::value;

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
// A TensorOperations::View is a layout reinterpretation of a backing view and
// does not republish its typedefs, so look through to the backing. Without
// this, storage carved from a non-default space -- host scratch in a CUDA
// build, say -- would deduce Kokkos::DefaultExecutionSpace and disagree with
// the policy tag it is evaluated under, which shows up as an Evaluator with no
// matching specialization rather than as a diagnosable error.
template <typename ViewType, typename Layout>
struct exec_space_of<TensorOperations::View<ViewType, Layout>, void> {
  using type = typename exec_space_of<ViewType>::type;
};

}  // namespace Impl

// ---------------------------------------------------------------------------
// Input specialization — wraps an existing TensorHandle
// ---------------------------------------------------------------------------
template <TensorLike T, typename ModesSeq, typename HookOp>
struct NodeHandle<InputTag, T, ModesSeq, HookOp> {
  TensorHandle<T, ModesSeq>    handle;
  [[no_unique_address]] HookOp hook_op;

  using node_tag            = InputTag;
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
  using node_tag            = IntermTag;
  static constexpr int Rank = IntRank::value;
  using storage_type        = Storage;
  using value_type          = typename Storage::value_type;
  using exec_space          = ExecSpace;

  Storage                      storage_;  // Kokkos::View (scratch tier)
  [[no_unique_address]] HookOp hook_op;
};

// Tile is `void` until the graph resolves it.
//
// Every other member kind DERIVES its output tile from its operands, so
// MemberOutTile can compute one from the node alone. A stage has no operands to
// derive from -- its tile is a property of the graph's label map, which the
// node does not know. So `make_stage_node(input)` leaves it unresolved and
// LevelGraph::add fills it in: `add` is the first place where both the node and
// the map are in scope. Downstream, `tile_type` is simply what MemberOutTile
// reports.
template <typename Operand, typename ModesSeq, typename Tile>
struct NodeHandle<StagedTag, Operand, ModesSeq, Tile> {
  Operand operand_;

  using node_tag            = StagedTag;
  using operand_type        = Operand;
  using tile_type           = Tile;
  static constexpr int Rank = Operand::Rank;
  using value_type          = typename Operand::value_type;
  using exec_space          = typename Operand::exec_space;
  using modes_seq           = ModesSeq;

  static_assert(static_cast<int>(ModesSeq::size()) == Rank,
                "staged node: one label per axis");
  static_assert(Impl::labels_distinct_v<ModesSeq>,
                "staged node: labels must be distinct");

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const {
    return operand_.shape();
  }

  // Relabelling keeps the tile UNRESOLVED even if it was resolved, because the
  // tile follows the labels: a different label order is a different tile, and
  // re-deriving it is the graph's job, not this node's.
  template <int32_t... Modes>
  KOKKOS_FUNCTION auto as() const {
    return NodeHandle<StagedTag, Operand,
                      std::integer_sequence<int32_t, Modes...>, void>{operand_};
  }
};

// ---------------------------------------------------------------------------
// Slot specialization — a LABELLED, read-only view of a buffer some OTHER node
// owns.
//
// This is the node kind that lets a consumer NAME its operand rather than nest
// that operand's whole subtree by value. Nesting is what makes a shared subtree
// evaluate once PER CONSUMER (each ScratchAllocator holds its own inner
// Evaluator, ScratchAllocator.hpp); naming it evaluates it once, full stop.
//
// Structurally this is an IntermTag node plus the two things an OPERAND needs
// and IntermTag does not carry: `modes_seq`, from which the contraction
// evaluator derives permA/permB, and `shape()`, which its k-tile counts index.
// A distinct tag rather than widening IntermTag, which is used pervasively for
// internal handoff between evaluators and whose Specializations 6/7/8 would all
// have to absorb the extra parameters.
//
// DELIBERATELY NO HOOK, and that is load-bearing rather than an omission. A
// hook is applied by writing back into the buffer it rides on (Evaluator/
// Team.hpp Specializations 7 and 8), which is sound only while a buffer has
// exactly one consumer. A slot exists in order to be shared, so the mutating
// paths must be unreachable from it — carrying no hook member makes that
// structural instead of a rule someone has to remember. The complementary
// restriction, on in-place REORDERING, cannot be expressed structurally and is
// stated in Impl::operand_stageable_v.
//
// Storage is the producing node's own output scratch view. Its type depends
// only on (value_type, exec space, output tile) — never on the producing node —
// so a slot is typeable from the tile alone, with no recursion into whatever
// computed it.
// ---------------------------------------------------------------------------
// WHICH buffer it names is a TEMPLATE parameter, not a member. A driver reads
// the store with `store.get<SlotIdx>()`, and the store is a heterogeneous tuple
// -- its element types differ per slot -- so the index has to be available at
// compile time. Carrying it in the node rather than in a table beside the node
// is what keeps the two from desynchronising: there is no second structure to
// get wrong.
//
// Two slot nodes may share an index and differ in their labels. That is the
// relabel mechanism (see make_slot_node), and it is why the index alone does
// not determine the type.
template <typename IntSlotIdx, typename Storage, typename IntRank,
          typename ExecSpace, typename ModesSeq>
struct NodeHandle<SlotTag, IntSlotIdx, Storage, IntRank, ExecSpace, ModesSeq> {
  using node_tag            = SlotTag;
  static constexpr int Rank = IntRank::value;
  // Index of the buffer this node names, within the driver's slot store.
  static constexpr std::size_t SlotIdx = IntSlotIdx::value;
  using storage_type                   = Storage;
  using value_type                     = typename Storage::value_type;
  using exec_space                     = ExecSpace;
  using modes_seq                      = ModesSeq;

  Storage                  storage_;
  Kokkos::Array<int, Rank> shape_;  // the PRODUCER's full output extents

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }

  template <int32_t... Modes>
  KOKKOS_FUNCTION auto as() const {
    static_assert(static_cast<int>(sizeof...(Modes)) == Rank,
                  "slot node as(): one label per axis");
    return NodeHandle<SlotTag, IntSlotIdx, Storage, IntRank, ExecSpace,
                      std::integer_sequence<int32_t, Modes...>>{storage_,
                                                                shape_};
  }
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

// Input node — wraps an existing TensorHandle
template <TensorLike T, typename ModesSeq, typename HookOp = NoHook>
KOKKOS_FUNCTION NodeHandle<InputTag, T, ModesSeq, HookOp> make_input_node(
    TensorHandle<T, ModesSeq> h, HookOp hook = {}) {
  static_assert(
      std::same_as<HookOp, NoHook> ||
          HookLike<HookOp, TensorHandle<T, ModesSeq>::Rank,
                   typename Impl::value_type_of<T>::type>,
      "input hook must be callable as op(i_0, ..., i_{Rank-1}, value_type&)");
  return {std::move(h), std::move(hook)};
}

// Intermediate node — wraps an existing storage View (a scratch/team tile, or
// an empty View for a deferred full-tensor intermediate).
template <typename Storage, typename HookOp = NoHook>
KOKKOS_FUNCTION auto make_interm_node(Storage storage, HookOp hook = {}) {
  constexpr int Rank = static_cast<int>(Storage::rank);
  using ExecSpace    = typename Impl::exec_space_of<Storage>::type;
  static_assert(
      std::same_as<HookOp, NoHook> ||
          HookLike<HookOp, Rank, typename Storage::value_type>,
      "interm hook must be callable as op(i_0, ..., i_{Rank-1}, value_type&)");
  return NodeHandle<IntermTag, Storage, std::integral_constant<int, Rank>,
                    ExecSpace, HookOp>{std::move(storage), std::move(hook)};
}

template <typename Operand>
KOKKOS_FUNCTION auto make_stage_node(Operand op) {
  return NodeHandle<StagedTag, Operand, typename Operand::modes_seq, void>{
      std::move(op)};
}

// Slot node — label an existing buffer (a producing node's output scratch) so
// it can be read as an operand.
//
// `shape` is the PRODUCER's full per-mode extents in the order the labels are
// given, NOT the tile's: a consumer computes its k-tile counts against the
// operand's global shape (Evaluator/Team.hpp, accumulate_block), exactly as it
// does for an input operand's tensor extents.
//
// Two slot nodes MAY alias one buffer with different labels, and that is the
// relabel mechanism. When a shared result has to be spelled in two consumers'
// differing canonical orders, declare it twice rather than moving data; the
// consumer whose labels do not match the storage order resolves the difference
// through its own gather permutation, zero-copy (Impl::operand_relabelable_v).
// `SlotIdx` names the buffer within the driver's slot store. Standalone uses --
// binding a buffer by hand, outside any driver -- can pass 0 and ignore it; it
// only means something to a DagGraph.
template <std::size_t SlotIdx, typename ModesSeq, typename Storage>
KOKKOS_FUNCTION auto make_slot_node_seq(
    Storage storage, Kokkos::Array<int, ModesSeq::size()> shape) {
  constexpr int Rank = static_cast<int>(ModesSeq::size());
  using ExecSpace    = typename Impl::exec_space_of<Storage>::type;
  static_assert(static_cast<int>(Storage::rank) == Rank,
                "slot node: one label per storage mode");
  return NodeHandle<SlotTag, std::integral_constant<std::size_t, SlotIdx>,
                    Storage, std::integral_constant<int, Rank>, ExecSpace,
                    ModesSeq>{std::move(storage), shape};
}

template <std::size_t SlotIdx, int32_t... Modes, typename Storage>
KOKKOS_FUNCTION auto make_slot_node(
    Storage storage, Kokkos::Array<int, sizeof...(Modes)> shape) {
  return make_slot_node_seq<SlotIdx, std::integer_sequence<int32_t, Modes...>>(
      std::move(storage), shape);
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
  using node_tag                     = ContractionTag;
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
  static_assert(Impl::is_node_handle_v<NodeA> && Impl::is_node_handle_v<NodeB>,
                "contraction operands must be node handles; a multi-output "
                "slice (CombineOutputHandle) is a terminal output, not an "
                "operand");
  constexpr int Rank = static_cast<int>(sizeof...(OutModes));
  static_assert((NodeA::Rank + NodeB::Rank - Rank) % 2 == 0,
                "Output rank is inconsistent with input ranks");
  static_assert(
      std::same_as<HookOp, NoHook> || HookLike<HookOp, Rank, ActualScalar>,
      "contraction hook must be callable as "
      "op(i_0, ..., i_{Rank-1}, value_type&)");

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
// Combine specialization — P{modes} = fn(A{modes}, B{modes}, ...) (pointwise)
// ---------------------------------------------------------------------------
// A pure per-coordinate combine over N operands: every operand shares the same
// index set (no mode is reduced). ModesSeq holds the output labels, which also
// serve as the canonical order; operands may be presented in any axis order and
// are gathered into this order at the evaluator boundary. fn is applied per
// element with the global output coordinate (hook-style):
// fn(i_0, ..., i_{Rank-1}, v_0, ..., v_{N-1}). There is no separate store hook
// — fn already sees the coordinates and every operand value, so it subsumes any
// per-coordinate transform. The N operand nodes (heterogeneous types) are held
// in a DeviceTuple so the pack survives into device code. fn may return either
// a scalar (NumOut == 1) or a Kokkos::Array<Scalar, M> (NumOut == M): a
// multi-output combine emits M tensors that share these modes in one pass.
template <typename CombineFn, typename IntRank, typename Scalar,
          typename ExecSpace, typename ModesSeq, typename IntNumOut,
          typename... Ops>
struct NodeHandle<CombineTag, CombineFn, IntRank, Scalar, ExecSpace, ModesSeq,
                  IntNumOut, Ops...> {
  using node_tag              = CombineTag;
  static constexpr int Rank   = IntRank::value;
  static constexpr int NumOps = static_cast<int>(sizeof...(Ops));
  static constexpr int NumOut = IntNumOut::value;  // outputs emitted by fn
  using value_type            = Scalar;
  using exec_space            = ExecSpace;
  using combine_type          = CombineFn;
  using modes_seq             = ModesSeq;  // output labels (== canonical order)
  using ops_tuple_t           = DeviceTuple<Ops...>;

  [[no_unique_address]] CombineFn fn;
  ops_tuple_t                     operands;
  Kokkos::Array<int, Rank>        shape_;  // output extents (canonical order)

  KOKKOS_FUNCTION Kokkos::Array<int, Rank> shape() const { return shape_; }
};

// ---------------------------------------------------------------------------
// Combine node factory
//
// P{out modes} = fn(A, B, ...), pointwise, with the output modes given as
// compile-time labels. Every operand must carry exactly the output label set
// (in any order); no mode is reduced. The output is stored in the requested
// output order, which is treated as canonical (operands are gathered into it).
// Public signature is operands-first, combine fn last:
//   make_combine_node<'i','j'>(a, b, ..., fn)
// ---------------------------------------------------------------------------
namespace Impl {

// --- combine fn return-type introspection ----------------------------------
// Classify combine_ret_t (Concept.hpp: the type fn returns for the combine call
// shape) as a scalar (NumOut == 1) or a Kokkos::Array<U, M> (NumOut == M).
template <typename Ret>
struct combine_out {  // scalar result
  static constexpr int num = 1;
  using elem               = Ret;
};
template <typename U, std::size_t M>
struct combine_out<Kokkos::Array<U, M>> {  // Kokkos::Array<U, M> result
  static constexpr int num = static_cast<int>(M);
  using elem               = U;
};

// An operand's shape() gathered into the TargetSeq (output) axis order, so
// operand extents can be compared mode-for-mode whatever each operand's own
// axis order is.
template <typename TargetSeq, typename Op>
Kokkos::Array<int, TargetSeq::size()> gathered_shape(const Op& op) {
  constexpr int  Rank = static_cast<int>(TargetSeq::size());
  constexpr auto perm = compute_label_perm<TargetSeq, typename Op::modes_seq>();
  const auto     s    = op.shape();
  Kokkos::Array<int, TargetSeq::size()> out{};
  for (int i = 0; i < Rank; ++i) out[i] = s[perm[i]];
  return out;
}

template <typename ActualScalar, typename ExecSpace, int32_t... OutModes,
          typename CombineFn, typename... Ops>
auto make_combine_node_impl(CombineFn fn, Ops... ops) {
  constexpr int Rank = static_cast<int>(sizeof...(OutModes));
  constexpr int N    = static_cast<int>(sizeof...(Ops));
  static_assert(N >= 1, "combine node needs at least one operand");
  static_assert((Impl::is_node_handle_v<Ops> && ...),
                "combine operands must be node handles; a multi-output slice "
                "(CombineOutputHandle) is a terminal output, not an operand");
  static_assert(((static_cast<int>(Ops::Rank) == Rank) && ...),
                "combine node: every operand and the output must have equal "
                "rank");

  using OutSeq = std::integer_sequence<int32_t, OutModes...>;
  static_assert(Impl::all_distinct(Impl::seq_to_array(OutSeq{})),
                "combine node: output labels must be pairwise distinct");
  static_assert(
      (Impl::same_label_set<typename Ops::modes_seq, OutSeq>() && ...),
      "combine node: each operand's label set must equal the output label set "
      "(a pointwise op reduces no mode)");
  static_assert(CombineLike<CombineFn, Rank, N, ActualScalar>,
                "combine fn must be const-callable as "
                "fn(i_0, ..., i_{Rank-1}, v_0, ..., v_{N-1}) -> value_type "
                "(the evaluator invokes it through a const kernel capture)");

  // Number of outputs, deduced from fn's return type: a scalar gives one
  // output, a Kokkos::Array<U, M> gives M. The element type must be the operand
  // scalar (homogeneous outputs sharing these modes).
  using Ret            = combine_ret_t<CombineFn, Rank, N, ActualScalar>;
  using OutInfo        = combine_out<Ret>;
  constexpr int NumOut = OutInfo::num;
  static_assert(std::is_convertible_v<typename OutInfo::elem, ActualScalar>,
                "combine fn output element type must be convertible to the "
                "operand scalar (multi-output combine is homogeneous)");

  // Operand 0 fixes the output extents (its axes gathered into output order);
  // every operand must agree on every mode extent.
  const std::array<Kokkos::Array<int, Rank>, N> op_shapes{
      Impl::gathered_shape<OutSeq>(ops)...};
  const Kokkos::Array<int, Rank> shape = op_shapes[0];
  for (int k = 1; k < N; ++k)
    for (int i = 0; i < Rank; ++i)
      assert(op_shapes[k][i] == shape[i] &&
             "combine node: operand extents must match on every mode");

  return NodeHandle<CombineTag, CombineFn, std::integral_constant<int, Rank>,
                    ActualScalar, ExecSpace, OutSeq,
                    std::integral_constant<int, NumOut>, Ops...>{
      std::move(fn), DeviceTuple<Ops...>(ops...), shape};
}

// Split the operands-first / fn-last argument list: the last argument is the
// combine fn, the leading ones are the operands. Runs on the host, so std::get
// is fine here.
template <typename ActualScalar, typename ExecSpace, int32_t... OutModes,
          typename Tuple, std::size_t... Is>
auto combine_from_args(Tuple args, std::index_sequence<Is...>) {
  return make_combine_node_impl<ActualScalar, ExecSpace, OutModes...>(
      std::get<sizeof...(Is)>(std::move(args)),  // fn (last)
      std::get<Is>(std::move(args))...);         // operands (leading)
}

}  // namespace Impl

// Primary form: infer the output scalar from the first operand, default exec
// space.  make_combine_node<'i','j'>(a, b, ..., fn)   // operands first, fn
// last
template <int32_t... OutModes, typename... Args>
auto make_combine_node(Args... args) {
  constexpr std::size_t N = sizeof...(Args);
  static_assert(
      N >= 2, "make_combine_node needs at least one operand and a combine fn");
  using FirstOp = std::tuple_element_t<0, std::tuple<Args...>>;
  return Impl::combine_from_args<typename FirstOp::value_type,
                                 Kokkos::DefaultExecutionSpace, OutModes...>(
      std::tuple<Args...>(std::move(args)...),
      std::make_index_sequence<N - 1>{});
}

// Explicit scalar (and execution-space) override:
//   make_combine_node<double, Kokkos::DefaultExecutionSpace, 'i','j'>(a, b, fn)
template <typename Scalar, typename ExecSpace = Kokkos::DefaultExecutionSpace,
          int32_t... OutModes, typename... Args>
auto make_combine_node(Args... args) {
  constexpr std::size_t N = sizeof...(Args);
  static_assert(
      N >= 2, "make_combine_node needs at least one operand and a combine fn");
  return Impl::combine_from_args<Scalar, ExecSpace, OutModes...>(
      std::tuple<Args...>(std::move(args)...),
      std::make_index_sequence<N - 1>{});
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
