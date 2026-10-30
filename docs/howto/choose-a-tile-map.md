# Choose a tile map

The tile map is the first argument to `make_level_graph`. It lists every label
your nodes use and gives each one an extent. This page is for when you move from
a toy size to your own: which labels to block, how big a tile to use, and what
to do when your element count is not a multiple of the tile. It covers the Team
policy only. The basic graph is built in {doc}`../tutorials/first-level-graph`;
the example here is the one from {doc}`indirect-access`.

## What the two entries mean

```cpp
using TileMap = TOps::LabelTiles<TOps::LabelTile<'e', 4>,
                                 TOps::LabelWhole<'p', 3>>;
```

- `LabelTile<'e', 4>` cuts the `'e'` axis into tiles of 4. Each tile is one
  team, so `'e'` is part of the launch grid.
- `LabelWhole<'p', 3>` says the `'p'` axis is exactly 3 long and every team
  sees all of it. It stays out of the grid.

## Rules

| Rule | What you see when you break it |
| --- | --- |
| Every label a node carries appears in the map, once. | Compile error: `every label a node carries must appear in the graph's tile map`, or `a label may appear at most once in the map`. |
| A label that is summed over (contracted) is `LabelWhole`. Blocking it would split the sum across teams, and nothing adds the partial sums. | Compile error: `a contracted label must be LabelWhole`. |
| At least one label is `LabelTile` and carried by some node; a `LabelTile` entry for an unused label does not count. | Compile error at `execute`: `at least one label must be blocked (LabelTile)`. |
| A `LabelWhole` extent is not smaller than the axis. | Runtime abort, in every build: `a label declared LabelWhole is longer than its tile ... declare it LabelTile instead`. |
| Every blocked extent is a multiple of its tile (next section). | Nothing checks a staged input (a tensor the graph copies into team scratch, such as `local_to_global`) in any build. Behavior is undefined. |

The compile-time cases are exercised in
[`labeltiles_guard_negative.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/negative/labeltiles_guard_negative.cpp).
Other checks are listed in {doc}`../reference/checks-and-guards`.

## Which labels to block

Start with one `LabelTile`: the axis with the most independent work, usually
the element axis. Make everything else `LabelWhole`. Each blocked label costs
every team a register for its tile index and a division to decode it, whether
or not it ever varies. Block a second label only when one axis alone cannot
fill the device. The map accepts several `LabelTile` entries
(`tests/test_label_tiles.cpp`).

## Choose the tile extent

The extent is a compile-time constant, so each value is a separate
instantiation of the graph. It is a trade-off:

- A larger tile spreads per-team staging (for example a shared operator matrix)
  over more elements and gives each team more work.
- Scratch grows with it. In the SEM stiffness benchmark it scales linearly with
  the tile (`sfpp_min/bench_te.cpp`). More scratch per team means fewer teams
  resident at once, and past the backend's limit the launch aborts.

There is no universal best value. Measure on your hardware, and print the
scratch request first.

### Check the scratch request before launching

`graph.outputs()` knows the request without running anything:

```cpp
auto launch = graph.outputs();
std::printf("scratch: %zu bytes at level %d\n", launch.scratch_bytes(),
            launch.scratch_level());
```

`scratch_bytes()` is the request the launch makes. `scratch_level()` is the
Kokkos scratch level that holds it: 0 if it fits there, else 1, else -1.
At -1, `execute` aborts with `the slot store exceeds this backend's team scratch
at every level; use a smaller tile`. Reduce the tile extent and recheck.
`slot_bytes()` is the same total without reusing storage between levels.

The limit comes from the backend, so the same tile can fit on Serial and fail on
a GPU. Check on the device you will run on.

## When the extent is not a multiple of the tile

The Team path has no remainder tile. Round the blocked axis up to a multiple of
the tile, and make the padded entries harmless:

1. Allocate the connectivity table with the padded element count.
2. Point every padded entry at one spare node past the real ones.
3. Size the global fields with one extra node, and ignore it afterwards.

The spare node absorbs the padded elements' scatters, so real nodes are
unchanged. The gather from it reads a valid value. This is the recipe
`sfpp_min/bench_te.cpp` uses for its element tile.

The program below runs 10 elements with a tile of 4 (padded to 12) and compares
the real nodes with a host reference over the 10 real elements.

```cpp
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <vector>

struct DoublePlusOne {
  KOKKOS_FUNCTION float operator()(int, int, float value) const {
    return 2.0f * value + 1.0f;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    namespace TOps = TensorOperations;
    using Space  = Kokkos::DefaultExecutionSpace;

    constexpr int tile_elements      = 4;   // elements per team
    constexpr int points_per_element = 3;
    constexpr int num_nodes          = 7;
    constexpr int real_elements      = 10;  // not a multiple of the tile

    // Round up to a whole number of tiles.
    constexpr int padded_elements =
        (real_elements + tile_elements - 1) / tile_elements * tile_elements;
    // Padded entries all point at one spare node past the real ones.
    constexpr int spare_node = num_nodes;

    using TileMap =
        TOps::LabelTiles<TOps::LabelTile<'e', tile_elements>,
                         TOps::LabelWhole<'p', points_per_element>>;

    Kokkos::View<int**, Kokkos::LayoutRight, Space> local_to_global(
        "local_to_global", padded_elements, points_per_element);
    Kokkos::View<float*, Space> node_values("node_values", num_nodes + 1);
    Kokkos::View<float*, Space> node_sums("node_sums", num_nodes + 1);

    auto host_map = Kokkos::create_mirror_view(local_to_global);
    for (int e = 0; e < padded_elements; ++e)
      for (int p = 0; p < points_per_element; ++p)
        host_map(e, p) =
            e < real_elements ? (2 * e + p) % num_nodes : spare_node;
    Kokkos::deep_copy(local_to_global, host_map);

    auto host_values = Kokkos::create_mirror_view(node_values);
    for (int n = 0; n <= num_nodes; ++n)
      host_values(n) = static_cast<float>(n + 1);
    Kokkos::deep_copy(node_values, host_values);
    Kokkos::deep_copy(node_sums, 0.0f);

    auto empty = TOps::make_level_graph<float, Space>(TileMap{});
    auto [with_map, map_slot] =
        empty.add(TOps::make_index_node<'e', 'p'>(local_to_global));
    auto [with_gather, gathered] =
        with_map.add(TOps::make_gather_node<'e', 'p'>(map_slot, node_values));
    auto [with_math, doubled] = with_gather.add(
        TOps::make_combine_node<'e', 'p'>(gathered, DoublePlusOne{}));
    auto graph = with_math.add(
        TOps::make_scatter_add_node<'e', 'p'>(map_slot, node_sums, doubled));

    auto launch = graph.outputs();
    std::printf("scratch: %zu bytes at level %d\n", launch.scratch_bytes(),
                launch.scratch_level());
    launch.execute(TOps::TeamPolicyTag<Space>{});
    Kokkos::fence();

    auto host_sums =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, node_sums);
    std::vector<float> expected(num_nodes, 0.0f);
    for (int e = 0; e < real_elements; ++e)
      for (int p = 0; p < points_per_element; ++p) {
        const int n = (2 * e + p) % num_nodes;
        expected[n] += 2.0f * host_values(n) + 1.0f;
      }
    for (int n = 0; n < num_nodes; ++n)
      if (host_sums(n) != expected[n]) exit_code = 1;
    std::printf("%s (spare node holds %g, ignored)\n",
                exit_code == 0 ? "ok" : "mismatch", host_sums(spare_node));
  }
  Kokkos::finalize();
  return exit_code;
}
```

On Serial it prints the scratch line (152 bytes at level 0 for this graph),
then `ok (spare node holds 102, ignored)`. The spare node's sum is whatever the
padded entries add; read only real nodes.

If you skip the padding, nothing tells you, in any build: a staged input whose
blocked extent is not a multiple of the tile is not checked on the Team path.
Padding is the only protection. The last tile reads past the end of the
connectivity table, which can crash or silently produce wrong sums; do not rely
on a particular failure.

Two checks that do exist, both `assert` only (removed by `NDEBUG`):

- If you pass an output view to `execute`, its blocked extents must also be
  multiples of the tile. Size those
  views to the padded count; the rows past the real elements are not meaningful.
- Staged tensors that disagree on a blocked label's extent.

## See also

- {doc}`indirect-access` for the gather and scatter nodes used above.
- {doc}`../reference/checks-and-guards` for other checks and whether they survive
  a Release build.
