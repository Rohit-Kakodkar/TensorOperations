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

namespace Impl {

// The combine-op call shape, spelled ONCE: invoke a const Fn with one int index
// per rank followed by one V per operand, fn(i_0, ..., i_{Rank-1}, v_0, ...,
// v_{N-1}). Declaration-only, used in unevaluated contexts as both the SFINAE
// probe behind CombineLike and the return-type introspection behind the combine
// factory's output-count deduction. Fn is invoked const because the evaluator
// calls fn through a const by-value kernel capture. The two index sequences
// (Rank indices, N values) are expanded via the comma-operator idiom rather
// than two packs inside one requires-clause (which is ambiguous) or an
// alias-template pack replacement (which nvcc's device frontend mishandles).
template <typename Fn, typename V, std::size_t... Is, std::size_t... Ks>
auto combine_ret(std::index_sequence<Is...>, std::index_sequence<Ks...>)
    -> decltype(std::declval<const Fn&>()((void(Is), 0)...,
                                          (void(Ks), std::declval<V>())...));

// The type fn returns for that call shape: a scalar, or a Kokkos::Array<U, M>
// for a multi-output combine (classified by the factory in NodeHandle.hpp).
template <typename Fn, int Rank, int N, typename V>
using combine_ret_t = decltype(combine_ret<Fn, V>(
    std::make_index_sequence<static_cast<std::size_t>(Rank)>{},
    std::make_index_sequence<static_cast<std::size_t>(N)>{}));

// Is Fn callable in the combine shape? Probe-with-ellipsis-fallback over
// combine_ret, so validity and return-type introspection cannot diverge.
template <typename Fn, int Rank, int N, typename V>
struct CombineLikeImpl {
 private:
  template <typename SeqI, typename SeqK>
  static auto probe(SeqI is, SeqK ks)
      -> decltype((void)combine_ret<Fn, V>(is, ks), std::true_type{});
  static std::false_type probe(...);

 public:
  static constexpr bool value = decltype(probe(
      std::make_index_sequence<static_cast<std::size_t>(Rank)>{},
      std::make_index_sequence<static_cast<std::size_t>(N)>{}))::value;
};

}  // namespace Impl

// Is Fn callable (const) as a pointwise combine for a Rank-dimensional node of
// N operands with element type V, i.e. fn(i_0, ..., i_{Rank-1}, v_0, ...,
// v_{N-1}) — hook-style coordinate indices followed by the N operand values?
// The return type is deliberately NOT constrained here: the combine factory
// deduces the output count from it (scalar vs Kokkos::Array) and checks element
// convertibility separately.
template <typename Fn, int Rank, int N, typename V>
concept CombineLike = Impl::CombineLikeImpl<Fn, Rank, N, V>::value;

}  // namespace TensorOperations
