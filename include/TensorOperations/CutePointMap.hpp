#pragma once

#include <cute/tensor.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace TensorOperations {
namespace Impl {

template <typename Seq>
struct cute_label_values;
template <typename T, T... Ls>
struct cute_label_values<std::integer_sequence<T, Ls...>> {
  static constexpr std::array<long, sizeof...(Ls)> value{
      static_cast<long>(Ls)...};
};

template <typename Shape, std::size_t... Is>
constexpr std::array<long, sizeof...(Is)> cute_shape_values(
    std::index_sequence<Is...>) {
  return {static_cast<long>(decltype(cute::get<Is>(Shape{}))::value)...};
}

template <typename CModes, typename CanonModes, typename CombineShape>
struct cute_relabel_info {
  static constexpr std::size_t R  = CModes::size();
  static constexpr auto        cm = cute_label_values<CModes>::value;
  static constexpr auto        cn = cute_label_values<CanonModes>::value;
  static constexpr auto        ex =
      cute_shape_values<CombineShape>(std::make_index_sequence<R>{});

  static constexpr std::size_t position(std::size_t c) {
    std::size_t q = 0;
    while (q < R && cm[q] != cn[c]) ++q;
    return q;
  }
  static constexpr long extent(std::size_t c) { return ex[position(c)]; }
  static constexpr long stride(std::size_t c) {
    long s = 1;
    for (std::size_t p = 0; p < position(c); ++p) s *= ex[p];
    return s;
  }
};

template <typename Info, typename Seq>
struct cute_relabel_layout;
template <typename Info, std::size_t... Cs>
struct cute_relabel_layout<Info, std::index_sequence<Cs...>> {
  using type = cute::Layout<
      cute::Shape<cute::Int<static_cast<int>(Info::extent(Cs))>...>,
      cute::Stride<cute::Int<static_cast<int>(Info::stride(Cs))>...>>;
};

template <typename CModes, typename CanonModes, typename CombineShape>
using cute_relabel_t = typename cute_relabel_layout<
    cute_relabel_info<CModes, CanonModes, CombineShape>,
    std::make_index_sequence<CModes::size()>>::type;

template <typename Mma, typename CModes, typename CanonModes,
          typename CombineShape>
using cute_point_tv_t = decltype(cute::composition(
    cute_relabel_t<CModes, CanonModes, CombineShape>{},
    std::declval<const Mma&>().get_layoutC_TV()));

template <typename TVK, typename TVD>
constexpr bool cute_points_aligned() {
  using TK = decltype(cute::coalesce(cute::layout<0>(TVK{})));
  using TD = decltype(cute::coalesce(cute::layout<0>(TVD{})));
  using VK = decltype(cute::layout<1>(TVK{}));
  using VD = decltype(cute::layout<1>(TVD{}));
  if constexpr (!std::is_same_v<TK, TD>)
    return false;
  else if constexpr (decltype(cute::size(VK{}))::value !=
                     decltype(cute::size(VD{}))::value)
    return false;
  else
    return std::is_same_v<decltype(cute::coalesce(cute::composition(
                              VK{}, cute::composition(cute::right_inverse(VK{}),
                                                      VD{})))),
                          decltype(cute::coalesce(VD{}))>;
}

template <typename TVK, typename TVD>
inline constexpr bool cute_points_aligned_v = cute_points_aligned<TVK, TVD>();

template <typename TVK, typename TVD>
using cute_value_remap_t = decltype(cute::composition(
    cute::right_inverse(cute::layout<1>(TVK{})), cute::layout<1>(TVD{})));

}  // namespace Impl
}  // namespace TensorOperations
