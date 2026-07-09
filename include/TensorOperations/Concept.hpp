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

}  // namespace TensorOperations
