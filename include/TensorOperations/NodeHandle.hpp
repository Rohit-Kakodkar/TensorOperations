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
template <typename Tag, typename... Args> struct NodeHandle;

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------
namespace Impl {

// Map (Scalar, Rank, MemSpace) → the appropriate Kokkos::View type
template <typename S, int R, typename MS> struct KokkosViewN;
template <typename S, typename MS> struct KokkosViewN<S, 0, MS> {
  using type = Kokkos::View<S, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS> struct KokkosViewN<S, 1, MS> {
  using type = Kokkos::View<S *, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS> struct KokkosViewN<S, 2, MS> {
  using type = Kokkos::View<S **, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS> struct KokkosViewN<S, 3, MS> {
  using type = Kokkos::View<S ***, Kokkos::LayoutRight, MS>;
};
template <typename S, typename MS> struct KokkosViewN<S, 4, MS> {
  using type = Kokkos::View<S ****, Kokkos::LayoutRight, MS>;
};

// Construct a KokkosViewN from a shape array via index_sequence unpacking
template <typename ViewType, int R, std::size_t... Is>
ViewType make_view_impl(const std::array<int, R> &shape,
                        std::index_sequence<Is...>) {
  return ViewType("intermediate", static_cast<std::size_t>(shape[Is])...);
}

template <typename S, int R, typename MS>
typename KokkosViewN<S, R, MS>::type
make_kokkos_view(const std::array<int, R> &shape) {
  using VT = typename KokkosViewN<S, R, MS>::type;
  return make_view_impl<VT, R>(shape, std::make_index_sequence<R>{});
}

} // namespace Impl

// ---------------------------------------------------------------------------
// Input specialization — wraps an existing TensorHandle
// ---------------------------------------------------------------------------
template <TensorLike T, typename HookOp>
struct NodeHandle<InputTag, T, HookOp> {
  TensorHandle<T> handle;
  [[no_unique_address]] HookOp hook_op;

  static constexpr int Rank = TensorHandle<T>::Rank;

  std::array<int, Rank> shape() const {
    std::array<int, Rank> s;
    for (int i = 0; i < Rank; ++i)
      s[i] = handle.extent(i);
    return s;
  }
  std::array<int32_t, Rank> modes() const { return handle.modes; }

  const TensorHandle<T> &concretize() const { return handle; }

  template <typename TeamMember>
  auto concretize(const TeamMember &team,
                  std::array<int, Rank> tile_shape) const {
    // Deep copy the relevant subtile into scratch memory
  }
};

// ---------------------------------------------------------------------------
// Intermediate specialization — deferred allocation
// ---------------------------------------------------------------------------
template <typename Scalar, typename IntRank, typename ExecSpace,
          typename HookOp>
struct NodeHandle<IntermTag, Scalar, IntRank, ExecSpace, HookOp> {
  static constexpr int Rank = IntRank::value;

  uint32_t id;
  std::array<int, Rank> shape_;
  std::array<int32_t, Rank> modes_;
  [[no_unique_address]] HookOp hook_op;

  std::array<int, Rank> shape() const { return shape_; }
  std::array<int32_t, Rank> modes() const { return modes_; }

  // Global memory — allocates in ExecSpace::memory_space
  auto concretize() const {
    using MS = typename ExecSpace::memory_space;
    auto view = Impl::make_kokkos_view<Scalar, Rank, MS>(shape_);
    Kokkos::deep_copy(view, Scalar(0));
    return make_handle(std::move(view), modes_);
  }

  // Scratch memory — allocates from team scratch in
  // ExecSpace::scratch_memory_space
  template <typename TeamMember>
  KOKKOS_INLINE_FUNCTION auto
  concretize(TeamMember team, std::array<int, Rank> tile_shape) const {
    using ScratchSpace = typename ExecSpace::scratch_memory_space;
    using VT = typename Impl::KokkosViewN<Scalar, Rank, ScratchSpace>::type;
    auto view = Impl::make_view_impl<VT, Rank>(
        tile_shape, std::make_index_sequence<Rank>{});
    return make_handle(std::move(view), modes_);
  }
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

// Input node — wraps an existing TensorHandle
template <TensorLike T, typename HookOp = NoHook>
NodeHandle<InputTag, T, HookOp> make_input_node(TensorHandle<T> h,
                                                HookOp hook = {}) {
  return {std::move(h), std::move(hook)};
}

// Intermediate node — deferred allocation, ExecSpace determines both memory
// spaces
template <typename Scalar, int Rank,
          typename ExecSpace = Kokkos::DefaultExecutionSpace,
          typename HookOp = NoHook>
NodeHandle<IntermTag, Scalar, std::integral_constant<int, Rank>, ExecSpace,
           HookOp>
make_interm_node(uint32_t id, std::array<int, Rank> shape,
                 std::array<int32_t, Rank> modes, HookOp hook = {}) {
  return {id, shape, modes, std::move(hook)};
}

// ---------------------------------------------------------------------------
// Contraction specialization — C{free modes} = sum{contracted} A × B
// ---------------------------------------------------------------------------
template <typename NodeA, typename NodeB, typename IntCRank, typename Scalar,
          typename ExecSpace, typename HookOp>
struct NodeHandle<ContractionTag, NodeA, NodeB, IntCRank, Scalar, ExecSpace,
                  HookOp> {
  static constexpr int Rank = IntCRank::value;
  static constexpr int NumContracted = (NodeA::Rank + NodeB::Rank - Rank) / 2;
  using hook_type = HookOp;

  NodeA node_a;
  NodeB node_b;
  std::array<int32_t, Rank> modes_;
  std::array<int, Rank> shape_;
  [[no_unique_address]] HookOp hook_op;

  std::array<int, Rank> shape() const { return shape_; }
  std::array<int32_t, Rank> modes() const { return modes_; }
};

// ---------------------------------------------------------------------------
// Contraction node factory
// ---------------------------------------------------------------------------

template <typename Scalar, typename ExecSpace = Kokkos::DefaultExecutionSpace,
          typename NodeA, typename NodeB, std::size_t CRank,
          typename HookOp = NoHook>
auto make_contraction_node(NodeA a, NodeB b,
                           std::array<int32_t, CRank> out_modes,
                           HookOp hook = {}) {
  constexpr int Rank = static_cast<int>(CRank);
  static_assert((NodeA::Rank + NodeB::Rank - Rank) % 2 == 0,
                "Output rank is inconsistent with input ranks");

  const auto a_modes = a.modes();
  const auto b_modes = b.modes();
  const auto a_shape = a.shape();
  const auto b_shape = b.shape();

  auto in_array = [](auto val, const auto &arr, int n) {
    for (int i = 0; i < n; ++i)
      if (arr[i] == val)
        return true;
    return false;
  };

  // Each output mode must be a free mode (in exactly one input)
  for (int i = 0; i < Rank; ++i) {
    int32_t m = out_modes[i];
    bool in_a = in_array(m, a_modes, NodeA::Rank);
    bool in_b = in_array(m, b_modes, NodeB::Rank);
    assert((in_a != in_b) &&
           "each output mode must appear in exactly one input tensor");
  }

  // Derive C's shape from input extents
  std::array<int, Rank> c_shape{};
  for (int i = 0; i < Rank; ++i) {
    int32_t m = out_modes[i];
    if (in_array(m, a_modes, NodeA::Rank)) {
      for (int j = 0; j < NodeA::Rank; ++j)
        if (a_modes[j] == m) { c_shape[i] = a_shape[j]; break; }
    } else {
      for (int j = 0; j < NodeB::Rank; ++j)
        if (b_modes[j] == m) { c_shape[i] = b_shape[j]; break; }
    }
  }

  return NodeHandle<ContractionTag, NodeA, NodeB,
                    std::integral_constant<int, Rank>, Scalar, ExecSpace,
                    HookOp>{std::move(a), std::move(b), out_modes, c_shape,
                            std::move(hook)};
}

} // namespace TensorOperations
