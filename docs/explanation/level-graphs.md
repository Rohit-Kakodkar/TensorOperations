# What one execute runs

Read this before you put a whole step, such as assemble a global field and then
use the assembled values, into one graph. A later level does not see other
teams' writes to global memory, and nothing reports an error when it reads
them. You get a wrong answer.

## What one `execute` launches

On the Team path, `outputs(...).execute(TeamPolicyTag<Space>{})` is one
`Kokkos::parallel_for` over a `Kokkos::TeamPolicy`.

- The league has one team per tile. The tile map decides how many; see
  {doc}`../howto/choose-a-tile-map`. `execute` returns that count.
- Each team runs every level, in order, for its own tile.
- A level ends with `team_barrier()`. That barrier is inside one team. Nothing
  synchronizes teams between levels.
- When a team has run its last level, it stores the designated outputs for its
  tile.

So "a later level runs after an earlier one" holds within a tile. It does not
mean "after every tile has finished the earlier level".

## What a later level can see

A level sees what its own team produced: slots in team scratch, and its own
tile of any global view. It does not reliably see what other teams wrote to
global memory during the same `execute`. That covers scatters, sink combines
(a combine whose function returns `void` and writes a view itself) and any
other view the graph writes.

Take the gather, combine and scatter example from {doc}`../howto/indirect-access`.
Add a gather of the assembled field after the scatter, and store the result per
element:

```text
index -> gather node_values -> combine (returns 1) -> scatter-add into assembled
      -> gather assembled -> sink combine storing into per_element
```

Every node ends up with two contributions, so every local point should read 2.
On Serial the single graph prints:

```text
1 1 1
2 2 2
```

Element 0's team ran first, so it read `assembled` before element 1 had added
to it. On a parallel backend the result depends on how the teams are scheduled.

## Split the graph at the dependency

When a level needs a value that other tiles produce through global memory, end
the first graph there and start a second one. Call `execute` on each.

```cpp
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>

namespace TOps = TensorOperations;
using Space    = Kokkos::DefaultExecutionSpace;

constexpr int num_elements       = 2;
constexpr int points_per_element = 3;
constexpr int num_nodes          = 3;

using TileMap = TOps::LabelTiles<TOps::LabelTile<'e', 1>,
                                 TOps::LabelWhole<'p', points_per_element>>;
using Map     = Kokkos::View<int**, Kokkos::LayoutRight, Space>;
using Field   = Kokkos::View<float*, Space>;
using Local   = Kokkos::View<float**, Kokkos::LayoutRight, Space>;

// Every local point contributes 1 to its node.
struct One {
  KOKKOS_FUNCTION float operator()(int, int, float) const { return 1.0f; }
};

// A sink: returns nothing and writes the value itself.
struct Store {
  Local out;
  KOKKOS_FUNCTION void operator()(int element, int point, float v) const {
    out(element, point) = v;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    Map   local_to_global("local_to_global", num_elements, points_per_element);
    Field node_values("node_values", num_nodes);
    Field assembled("assembled", num_nodes);
    Local per_element("per_element", num_elements, points_per_element);

    auto host_map = Kokkos::create_mirror_view(local_to_global);
    for (int e = 0; e < num_elements; ++e)
      for (int p = 0; p < points_per_element; ++p)
        host_map(e, p) = (e + p) % num_nodes;
    Kokkos::deep_copy(local_to_global, host_map);
    Kokkos::deep_copy(node_values, 1.0f);
    Kokkos::deep_copy(assembled, 0.0f);

    // Graph 1: add 1 to `assembled` at every (element, point).
    {
      auto g0 = TOps::make_level_graph<float, Space>(TileMap{});
      auto [g1, map] =
          g0.add(TOps::make_index_node<'e', 'p'>(local_to_global));
      auto [g2, v] = g1.add(TOps::make_gather_node<'e', 'p'>(map, node_values));
      auto [g3, one] = g2.add(TOps::make_combine_node<'e', 'p'>(v, One{}));
      auto g4 =
          g3.add(TOps::make_scatter_add_node<'e', 'p'>(map, assembled, one));
      g4.outputs().execute(TOps::TeamPolicyTag<Space>{});
    }

    // Graph 2: read `assembled` back per element. No fence between the two.
    {
      auto g0 = TOps::make_level_graph<float, Space>(TileMap{});
      auto [g1, map] =
          g0.add(TOps::make_index_node<'e', 'p'>(local_to_global));
      auto [g2, v] = g1.add(TOps::make_gather_node<'e', 'p'>(map, assembled));
      auto g3 =
          g2.add(TOps::make_combine_node<'e', 'p'>(v, Store{per_element}));
      g3.outputs().execute(TOps::TeamPolicyTag<Space>{});
    }
    Kokkos::fence();  // before the host reads per_element

    auto host_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        per_element);
    for (int e = 0; e < num_elements; ++e) {
      for (int p = 0; p < points_per_element; ++p) {
        std::printf("%g ", host_out(e, p));
        if (host_out(e, p) != 2.0f) exit_code = 1;
      }
      std::printf("\n");
    }
  }
  Kokkos::finalize();
  return exit_code;
}
```

It prints `2 2 2` on both lines. Built against the `tensor` target and run with
the Serial backend (clang, macOS); other backends are untried.

Use one graph when every dependency stays inside a tile: gathers, contractions
and combines that read slots produced by earlier levels of the same graph.
Use a second graph when a level reads a global view that an earlier level
wrote through a scatter or a sink.

## Ordering between `execute` calls

`execute` returns without waiting for the kernel. Two calls on the same
execution space need no fence between them, because the launch uses the
default instance of that space (the policy is built from `Space` alone) and
Kokkos runs kernels on one instance in submission order. The example relies on
this.

Fence before the host reads a result, as the example does. Graphs on different
execution spaces, or on a non-default instance, are not ordered by this
rule.

## Launch settings

| Setting | What it does |
| --- | --- |
| `outputs(...).team_size(n)` | Fixes the team size. Without it the policy passes `Kokkos::AUTO`. It returns a new `LevelOutputs`; chain it before `execute`. |
| `scratch_bytes()` | Bytes of team scratch the launch asks for, after slots with disjoint lifetimes share space. |
| `slot_bytes()` | The same total without that sharing. |
| `scratch_level()` | The Kokkos scratch level the launch uses: 0 if the request fits there, otherwise 1, or -1 if it fits neither. |

If the request fits neither level, `execute` calls `Kokkos::abort` with
"the slot store exceeds this backend's team scratch at every level; use a
smaller tile". Check `scratch_level()` before you launch, and shrink the tile
in the tile map if it is -1. `Kokkos::TeamPolicy<Space>::scratch_size_max(level)`
gives the limit for each level.

## CuTe

This page describes the Team path only. It does not say whether the CuTe launch
shares these visibility rules.
