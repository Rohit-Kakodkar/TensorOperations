#pragma once
#include <TensorOperations/TensorHandle.hpp>
#include <array>
#include <cstdint>
#include <type_traits>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

// Tags for NodeHandle specializations
struct InputTag {};
struct IntermTag {};

// Primary template — undefined; must use a specialization
template <typename Tag, typename... Args> struct NodeHandle;

// ---------------------------------------------------------------------------
// Input specialization — wraps an existing TensorHandle
// ---------------------------------------------------------------------------
template <TensorLike T, typename HookOp = void> struct NodeHandle<InputTag, T> {
  TensorHandle<T> handle;
  HookOp hook_op;

  // Check that HookOp is invocable with indices with the same arity as the
  // tensor rank
  static_assert(
      (std::invocable<HookOp::template operator(),
                      std::make_index_sequence<TensorHandle<T>::Rank>>),
      "HookOp must be invocable with indices matching the tensor rank");

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
// Impl helpers for the intermediate specialization
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
// Intermediate specialization — deferred allocation
// ---------------------------------------------------------------------------
template <typename Scalar, typename IntRank, typename ExecSpace,
          typename HookOp = void>
struct NodeHandle<IntermTag, Scalar, IntRank, ExecSpace> {
  static constexpr int Rank = IntRank::value;

  uint32_t id;
  std::array<int, Rank> shape_;
  std::array<int32_t, Rank> modes_;
  HookOp hook_op;

  static_assert(
      (std::invocable<HookOp::template operator(),
                      std::make_index_sequence<Rank>>),
      "HookOp must be invocable with indices matching the tensor rank");

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
// Factory functions — both overloads of make_node
// ---------------------------------------------------------------------------

// Input node — wraps an existing TensorHandle
template <TensorLike T> NodeHandle<InputTag, T> make_node(TensorHandle<T> h) {
  return {std::move(h)};
}

// Intermediate node — deferred allocation, ExecSpace determines both memory
// spaces
template <typename Scalar, int Rank,
          typename ExecSpace = Kokkos::DefaultExecutionSpace>
NodeHandle<IntermTag, Scalar, std::integral_constant<int, Rank>, ExecSpace>
make_node(uint32_t id, std::array<int, Rank> shape,
          std::array<int32_t, Rank> modes) {
  return {id, shape, modes};
}

} // namespace TensorOperations
