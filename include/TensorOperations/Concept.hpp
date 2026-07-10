#pragma once
#include <concepts>
#include <cstdint>
#include <utility>

namespace TensorOperations {

namespace Impl {

template <typename T, std::size_t... Is>
concept CallableWithNInts = requires(T t) {
  { t((void(Is), 0)...) };
};

// Helper traits expand the rank into an index sequence without using an
// immediately-invoked lambda inside the concept. Lambdas embedded in concept
// definitions are evaluated inconsistently by some compilers across
// instantiation contexts, which made the surrounding concepts spuriously fail.
template <typename T, typename Seq>
struct CallableWithRankImpl : std::false_type {};

template <typename T, std::size_t... Is>
struct CallableWithRankImpl<T, std::index_sequence<Is...>>
    : std::bool_constant<CallableWithNInts<T, Is...>> {};

template <typename T>
concept CallableWithRank = CallableWithRankImpl<
    T, std::make_index_sequence<static_cast<std::size_t>(T::rank)>>::value;

}  // namespace Impl

template <typename T>
concept TensorLike = requires(T t) {
  requires(static_cast<int>(T::rank) >= 0);
  { t.extent(0) } -> std::convertible_to<int>;
  t.data();
  { t.stride(0) } -> std::convertible_to<std::ptrdiff_t>;
} && Impl::CallableWithRank<T>;

namespace Impl {

// Is Op callable with Rank int indices followed by a mutable V&, i.e. the
// library-wide hook signature op(i, j, k, ..., v)? Mirrors CallableWithRank
// above, but Rank/V are supplied explicitly rather than read off T::rank,
// since a hook has no rank of its own — it borrows the node's.
template <typename Op, typename V, std::size_t... Is>
concept HookCallableWithNInts = requires(Op op, V& v) {
  { op((void(Is), 0)..., v) };
};

template <typename Op, int Rank, typename V, typename Seq>
struct HookLikeImpl : std::false_type {};

template <typename Op, int Rank, typename V, std::size_t... Is>
struct HookLikeImpl<Op, Rank, V, std::index_sequence<Is...>>
    : std::bool_constant<HookCallableWithNInts<Op, V, Is...>> {};

}  // namespace Impl

// Is Op callable as a hook for a Rank-dimensional operand of element type V,
// i.e. op(i_0, ..., i_{Rank-1}, v) with v a mutable V&? Does not special-case
// NoHook (defined later, in NodeHandle.hpp) — callers combine this with a
// same_as<Op, NoHook> check.
template <typename Op, int Rank, typename V>
concept HookLike = Impl::HookLikeImpl<
    Op, Rank, V,
    std::make_index_sequence<static_cast<std::size_t>(Rank)>>::value;

}  // namespace TensorOperations
