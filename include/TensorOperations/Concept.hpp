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

} // namespace Impl

template <typename T>
concept TensorLike = requires(T t) {
  { T::rank() } -> std::convertible_to<int>;
  { t.extent(0) } -> std::convertible_to<int>;
} && []<std::size_t... Is>(std::index_sequence<Is...>) {
  return Impl::CallableWithNInts<T, Is...>;
}(std::make_index_sequence<static_cast<std::size_t>(T::rank())>{});

struct NodeHandle {
  uint32_t id;
};

} // namespace TensorOperations
