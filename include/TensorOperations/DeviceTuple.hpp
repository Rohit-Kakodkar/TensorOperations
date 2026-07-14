#pragma once
/**
 * @file DeviceTuple.hpp
 * @brief A minimal heterogeneous tuple usable in Kokkos device code.
 *
 * `std::tuple` / `std::get` are unreliable inside CUDA device code, so this
 * header provides a small, device-copyable, device-indexable tuple. Leaves are
 * keyed by *index* (not by type) so that a tuple may hold several elements of
 * the same type without `get<I>` becoming ambiguous.
 *
 * Public, std-like API (in namespace `TensorOperations`):
 *   - `DeviceTuple<Ts...>`           — the tuple type
 *   - `get<I>(tuple)`                — free function, `KOKKOS_FUNCTION`
 *   - `tuple.template get<I>()`      — member, `KOKKOS_FUNCTION`
 *   - `tuple_element_t<I, Tuple>`    — the type at index I
 *   - `tuple_size<Tuple>` / `tuple_size_v<Tuple>` — element count
 *   - `DeviceTuple<Ts...>::size`     — element count (member constant)
 *
 * Nothing is injected into `namespace std`.
 */
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace TensorOperations {

namespace Impl {
// One tuple element, tagged by its index I so equal types at different indices
// remain distinct base classes (no ambiguous upcast in get<I>).
template <std::size_t I, typename T>
struct TupleLeaf {
  T                         value_;
  KOKKOS_DEFAULTED_FUNCTION TupleLeaf() = default;
  KOKKOS_FUNCTION explicit TupleLeaf(const T& v) : value_(v) {}
};

// Type at index I: unevaluated overload resolution against the leaf bases.
template <std::size_t I, typename T>
T tuple_leaf_type(TupleLeaf<I, T>);

template <typename Seq, typename... Ts>
struct TupleImpl;
template <std::size_t... Is, typename... Ts>
struct TupleImpl<std::index_sequence<Is...>, Ts...> : TupleLeaf<Is, Ts>... {
  KOKKOS_DEFAULTED_FUNCTION TupleImpl() = default;
  KOKKOS_FUNCTION explicit TupleImpl(const Ts&... ts)
      : TupleLeaf<Is, Ts>(ts)... {}
};
}  // namespace Impl

// --- public API ------------------------------------------------------------

// Free get<I>: an implicit upcast to the unique TupleLeaf<I,T> base deduces T.
template <std::size_t I, typename T>
KOKKOS_FUNCTION T& get(Impl::TupleLeaf<I, T>& leaf) noexcept {
  return leaf.value_;
}
template <std::size_t I, typename T>
KOKKOS_FUNCTION const T& get(const Impl::TupleLeaf<I, T>& leaf) noexcept {
  return leaf.value_;
}

// The type stored at index I of a tuple.
template <std::size_t I, typename Tuple>
using tuple_element_t =
    decltype(Impl::tuple_leaf_type<I>(std::declval<Tuple>()));

// A heterogeneous, device-copyable tuple.
template <typename... Ts>
struct DeviceTuple : Impl::TupleImpl<std::index_sequence_for<Ts...>, Ts...> {
  using base_t = Impl::TupleImpl<std::index_sequence_for<Ts...>, Ts...>;
  static constexpr std::size_t size = sizeof...(Ts);

  KOKKOS_DEFAULTED_FUNCTION DeviceTuple() = default;
  KOKKOS_FUNCTION explicit DeviceTuple(const Ts&... ts) : base_t(ts...) {}

  template <std::size_t I>
  KOKKOS_FUNCTION auto& get() {
    return TensorOperations::get<I>(*this);
  }
  template <std::size_t I>
  KOKKOS_FUNCTION const auto& get() const {
    return TensorOperations::get<I>(*this);
  }
};

// std-like element-count trait.
template <typename Tuple>
struct tuple_size;
template <typename... Ts>
struct tuple_size<DeviceTuple<Ts...>>
    : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template <typename Tuple>
inline constexpr std::size_t tuple_size_v = tuple_size<Tuple>::value;

}  // namespace TensorOperations
