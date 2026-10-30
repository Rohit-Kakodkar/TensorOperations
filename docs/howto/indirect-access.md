# Gather, compute and scatter through connectivity

Read a global field through an integer connectivity table, compute on the local
values, and add the results back into a global field. This page covers the
Team policy with default staging; it does not establish CuTe or register-staging
behavior. Signatures are in the
{doc}`API reference <../reference/api/graph-and-nodes>`.

## What we are building

A mesh has 2 **elements**, each with 3 local **points**. Points are shared:
several local points land on the same **global node**. A connectivity table says
which one. The program reads a value per global node, doubles it and adds 1 at
every local point, then adds the results back per node.

```text
local_to_global(e, p)     node_values          after gather      after 2u+1
 e=0:  0  1  2             node 0:  1           e=0:  1  2  3     e=0:  3  5  7
 e=1:  1  2  0             node 1:  2           e=1:  2  3  1     e=1:  5  7  3
                           node 2:  3

node_sums (result): node 0 = 3 + 3 = 6,  node 1 = 5 + 5 = 10,  node 2 = 7 + 7 = 14
```

Labels `'e'` (element) and `'p'` (point) name the two axes of
`local_to_global`, in that order.

## The code

The program needs the `tensor` CMake target. Each numbered comment is one step.

```cpp
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>

// Local computation: takes the element and point coordinates, then the value.
struct DoublePlusOne {
  KOKKOS_FUNCTION float operator()(int /*element*/, int /*point*/,
                                   float value) const {
    return 2.0f * value + 1.0f;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    namespace TOps = TensorOperations;
    using Space  = Kokkos::DefaultExecutionSpace;

    constexpr int num_elements       = 2;
    constexpr int points_per_element = 3;
    constexpr int num_nodes          = 3;

    // One team per element; the point axis is kept whole inside each team.
    using TileMap = TOps::LabelTiles<TOps::LabelTile<'e', 1>,
                                     TOps::LabelWhole<'p', points_per_element>>;

    // 1. Data. Everything lives in the graph's execution space.
    Kokkos::View<int**, Kokkos::LayoutRight, Space> local_to_global(
        "local_to_global", num_elements, points_per_element);
    Kokkos::View<float*, Space> node_values("node_values", num_nodes);
    Kokkos::View<float*, Space> node_sums("node_sums", num_nodes);

    auto host_map = Kokkos::create_mirror_view(local_to_global);
    for (int e = 0; e < num_elements; ++e)
      for (int p = 0; p < points_per_element; ++p)
        host_map(e, p) = (e + p) % num_nodes;
    Kokkos::deep_copy(local_to_global, host_map);

    auto host_values = Kokkos::create_mirror_view(node_values);
    for (int n = 0; n < num_nodes; ++n)
      host_values(n) = static_cast<float>(n + 1);
    Kokkos::deep_copy(node_values, host_values);

    // Scatter adds into node_sums, so it must start at zero.
    Kokkos::deep_copy(node_sums, 0.0f);

    // 2. Graph. Each add() is one level; a level may only read earlier ones.
    auto empty = TOps::make_level_graph<float, Space>(TileMap{});

    // Level 1: stage the connectivity table.
    auto [with_map, map_slot] =
        empty.add(TOps::make_index_node<'e', 'p'>(local_to_global));

    // Level 2: gathered(e, p) = node_values(local_to_global(e, p)).
    auto [with_gather, gathered] =
        with_map.add(TOps::make_gather_node<'e', 'p'>(map_slot, node_values));

    // Level 3: doubled(e, p) = 2 * gathered(e, p) + 1.
    auto [with_math, doubled] = with_gather.add(
        TOps::make_combine_node<'e', 'p'>(gathered, DoublePlusOne{}));

    // Level 4: node_sums(local_to_global(e, p)) += doubled(e, p).
    // A level of only scatters returns just the graph (no value handle).
    auto graph = with_math.add(
        TOps::make_scatter_add_node<'e', 'p'>(map_slot, node_sums, doubled));

    // 3. Run. No output views: the result is the mutation of node_sums.
    graph.outputs().execute(TOps::TeamPolicyTag<Space>{});
    Kokkos::fence();  // execute() does not wait; fence before reading.

    // 4. Check on the host. Expected node_sums: 6, 10, 14.
    auto host_sums =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, node_sums);
    for (int n = 0; n < num_nodes; ++n) {
      const float expected = 2.0f * (2.0f * host_values(n) + 1.0f);
      if (host_sums(n) != expected) exit_code = 1;
    }
    std::printf("%s\n", exit_code == 0 ? "ok" : "mismatch");
  }
  Kokkos::finalize();
  return exit_code;
}
```

Every node receives exactly two contributions, which are equal, so the exact
float comparison holds. Do not rely on exact comparisons in general: atomic
updates do not fix the summation order.

## Reading the graph

| Level | Call | Result handle | Meaning |
| --- | --- | --- | --- |
| 1 | `make_index_node` | `map_slot` | Connectivity staged in graph storage |
| 2 | `make_gather_node` | `gathered` | `node_values` read through `map_slot` |
| 3 | `make_combine_node` | `doubled` | `DoublePlusOne` applied at each `(e, p)` |
| 4 | `make_scatter_add_node` | none | Atomic add into `node_sums` |

To write a combine with several operands or results, see
{doc}`write-device-callables`.

## Rules to remember

- **Labels name existing axes.** For `local_to_global(e, p)`, use `<'e', 'p'>`. Labels never
  transpose storage.
- **One level per dependency.** Put the index and each producer in an earlier
  `add(...)` than its consumers; members of one level run concurrently.
- **Scatter adds, it never clears.** Zero the destination first for a fresh sum.
- **Nothing checks your indices.** Every connectivity entry must be a valid,
  `int`-representable index for both source and destination.
- **Fence before you read.** Execution returns without waiting for completion.

The full list of obligations and what the library does check is in
{doc}`../reference/checks-and-guards`.

## Direct view or index slot?

| | Direct view | Index slot |
| --- | --- | --- |
| Use | Pass `local_to_global` to each gather and scatter | `add(make_index_node<'e', 'p'>(local_to_global))`, pass the handle on |
| Index read | In the kernel, per operation | Staged into graph storage once |
| Labels | One distinct label per axis | Must match the handle's label sequence exactly |
| Layout argument | Allowed on the gather | Rejected on the gather; give it to `make_index_node` |

Neither form is established as faster; the slot makes sharing one index
explicit.

## Sources and destinations

| | Accepted | Not accepted |
| --- | --- | --- |
| Source | Rank-one Kokkos view (including a column `subview`), or an object with a `const` `source(int)` | A rank-two field passed directly |
| Destination | Writable rank-one view, or a `const` `dst(int)` returning `T&` that supports `Kokkos::atomic_add` | A view of `const` elements; an accessor returning a value |

For a column of a rank-two field, use
`Kokkos::subview(field, Kokkos::ALL, component)`. Custom call operators run in
the kernel: make them `KOKKOS_FUNCTION` and `const`.

## Common mistakes

| Compiler message or symptom | Cause |
| --- | --- |
| `an index handle must be read with the gather's labels, in the same order` | Consumer's labels differ from the index handle's; `as<...>()` does not fix this, see {doc}`reuse-a-tensor-under-new-labels` (scatter says "scatter's labels") |
| `an index slot is read only by gather` | An integer slot was used as a numerical operand; gather from a floating-point source instead |
| `may not read a slot ITS OWN LEVEL` | A slot producer is in the same `add(...)` level as its consumer |
| Destination holds old values plus new sums | Destination was not zeroed; running again adds again |
| Wrong or stale values on the host | Missing `Kokkos::fence()` before reading |
| Crash or garbage with no diagnostic | An index outside the source or destination extent |

## Variations

**Direct-view form.** Drop level 1. Add the gather to `empty`, passing
`local_to_global` where the example passes `map_slot`, and pass
`local_to_global` to the scatter as well. The combine and the zero-output
execution stay the same.

**Accumulate onto an existing field.** Instead of zeroing `node_sums`, fill it
with the current values and do not clear it. Make sure all earlier updates have
completed, or are correctly ordered, before submission.

## See also

- {doc}`../reference/checks-and-guards` for the exact contract.
- Working tests: [`test_gather_node.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_gather_node.cpp),
  [`test_scatter_add_node.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_scatter_add_node.cpp),
  [`test_index_node.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_index_node.cpp).
