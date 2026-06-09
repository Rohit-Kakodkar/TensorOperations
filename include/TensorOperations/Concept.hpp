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

template <typename T, std::size_t... Is>
concept WritableWithNInts = requires(T t) {
    t((void(Is), 0)...) = typename T::value_type{};
};

} // namespace Impl

template <typename T>
concept TensorLike =
    requires(T t) {
        requires (static_cast<int>(T::rank) >= 0);
        { t.extent(0) } -> std::convertible_to<int>;
    }
    && []<std::size_t... Is>(std::index_sequence<Is...>) {
           return Impl::CallableWithNInts<T, Is...>;
       }(std::make_index_sequence<static_cast<std::size_t>(T::rank)>{});

template <typename T>
concept WritableTensorLike =
    TensorLike<T> &&
    requires { typename T::value_type; } &&
    []<std::size_t... Is>(std::index_sequence<Is...>) {
        return Impl::WritableWithNInts<T, Is...>;
    }(std::make_index_sequence<static_cast<std::size_t>(T::rank)>{});

} // namespace TensorOperations
