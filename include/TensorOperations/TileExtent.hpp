#pragma once

#include <TensorOperations/Liveness.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <utility>

namespace TensorOperations {
namespace Impl {

struct WholeTile {
  template <typename Point>
  KOKKOS_FUNCTION constexpr bool contains(const Point&) const {
    return true;
  }

  template <int NodeRank, int... G>
  KOKKOS_FUNCTION WholeTile project(std::integer_sequence<int, G...>) const {
    return {};
  }
};

template <std::size_t Rank>
struct ClippedTile {
  Kokkos::Array<int, Rank> extent;

  template <typename Point>
  KOKKOS_FUNCTION bool contains(const Point& point) const {
    return inside(point, std::make_index_sequence<Rank>{});
  }

  template <typename Tile>
  KOKKOS_FUNCTION bool is_whole() const {
    bool whole = true;
    for (std::size_t d = 0; d < Rank; ++d)
      whole = whole && extent[d] >= Tile::extent(static_cast<int>(d));
    return whole;
  }

  template <int NodeRank, int... G>
  KOKKOS_FUNCTION auto project(std::integer_sequence<int, G...>) const {
    const int                                       grid[NodeRank] = {G...};
    ClippedTile<static_cast<std::size_t>(NodeRank)> out{};
    for (int d = 0; d < NodeRank; ++d)
      out.extent[d] = grid[d] >= 0 ? extent[static_cast<std::size_t>(grid[d])]
                                   : Kokkos::Experimental::finite_max_v<int>;
    return out;
  }

 private:
  template <typename Point, std::size_t... Ds>
  KOKKOS_FUNCTION bool inside(const Point& point,
                              std::index_sequence<Ds...>) const {
    return ((static_cast<int>(get<Ds>(point)) < extent[Ds]) && ...);
  }
};

template <typename Tile, std::size_t Rank>
KOKKOS_FUNCTION ClippedTile<Rank> clip_tile(
    const Kokkos::Array<int, Rank>& shape,
    const Kokkos::Array<int, Rank>& index) {
  ClippedTile<Rank> out{};
  for (std::size_t d = 0; d < Rank; ++d)
    out.extent[d] = shape[d] - index[d] * Tile::extent(static_cast<int>(d));
  return out;
}

template <std::size_t RootR, typename Extent>
struct PlacedTile {
  Kokkos::Array<int, RootR> index;
  Extent                    extent;

  template <int NodeRank, int... G>
  KOKKOS_FUNCTION auto node_index(std::integer_sequence<int, G...> grid) const {
    return Impl::node_index<NodeRank, RootR>(index, grid);
  }

  template <int NodeRank, int... G>
  KOKKOS_FUNCTION auto node_extent(
      std::integer_sequence<int, G...> grid) const {
    return extent.template project<NodeRank>(grid);
  }
};

}  // namespace Impl
}  // namespace TensorOperations
