# Contract two tensors with a level graph

You have two Kokkos views and want their product in a third, computed per tile
in team scratch. This page covers the Team policy and a binary contraction:
`C[j,i] = sum over k of A[i,k] * B[k,j]`. For gather and scatter through
connectivity see {doc}`indirect-access`. For the program to build against the
`tensor` target see {doc}`use-in-your-project`. Signatures are in the
{doc}`API reference <../reference/api/graph-and-nodes>`.

## The code

Each numbered comment is one step.

```cpp
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>

namespace TOps = TensorOperations;
using Space    = Kokkos::DefaultExecutionSpace;

constexpr int rows      = 6;  // i: cut into tiles of 2, so 3 teams
constexpr int tile_rows = 2;
constexpr int inner     = 3;  // k: summed
constexpr int cols      = 4;  // j

// i is blocked; k and j stay whole inside every team.
using TileMap =
    TOps::LabelTiles<TOps::LabelTile<'i', tile_rows>,
                     TOps::LabelWhole<'k', inner>, TOps::LabelWhole<'j', cols>>;
using Matrix = Kokkos::View<float**, Kokkos::LayoutRight, Space>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    // 1. Data, all in the graph's execution space.
    Matrix a("a", rows, inner);  // labels (i, k)
    Matrix b("b", inner, cols);  // labels (k, j)
    Matrix c("c", cols, rows);   // output, labels (j, i)

    auto host_a = Kokkos::create_mirror_view(a);
    auto host_b = Kokkos::create_mirror_view(b);
    for (int i = 0; i < rows; ++i)
      for (int k = 0; k < inner; ++k) host_a(i, k) = 1.0f + i + 0.5f * k;
    for (int k = 0; k < inner; ++k)
      for (int j = 0; j < cols; ++j) host_b(k, j) = 1.0f - k + 0.25f * j;
    Kokkos::deep_copy(a, host_a);
    Kokkos::deep_copy(b, host_b);

    // 2. Graph. Each add() returns a new graph and the handles it produced.
    auto g0           = TOps::make_level_graph<float, Space>(TileMap{});
    auto [g1, a_tile] = g0.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'i', 'k'>(a))));
    auto [g2, b_tile] = g1.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'k', 'j'>(b))));
    auto [g3, product] =
        g2.add(TOps::make_contraction_node<'j', 'i'>(a_tile, b_tile));

    // 3. Run. One output slot, so one view.
    const int teams =
        g3.outputs(product).execute(TOps::TeamPolicyTag<Space>{}, c);
    Kokkos::fence();  // execute() does not wait; fence before the host reads.

    // 4. Check against a host loop.
    auto   host_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, c);
    double max_err = 0.0;
    for (int j = 0; j < cols; ++j)
      for (int i = 0; i < rows; ++i) {
        double ref = 0.0;
        for (int k = 0; k < inner; ++k) ref += host_a(i, k) * host_b(k, j);
        max_err = std::fmax(max_err, std::fabs(ref - host_c(j, i)));
      }
    std::printf("teams=%d max_err=%g\n", teams, max_err);
    if (teams != rows / tile_rows || max_err > 1e-5) exit_code = 1;
  }
  Kokkos::finalize();
  return exit_code;
}
```

On Serial it prints `teams=3 max_err=0`. Verified on macOS with clang, in Debug
and Release; other backends are untried.

## What each step does

**Labels.** `make_handle<'i', 'k'>(a)` names the axes of `a`, in order. It does
not move data. A label that appears on both operands, here `'k'`, is summed.
The labels you give `make_contraction_node<'j', 'i'>` are the output's axes, in
that order, so `c` is `(cols, rows)`, not the operands' order. Give `c` the
shape in that order, with full extents.

**Tile map.** `LabelTile<'i', 2>` cuts `i` into tiles of 2, and each tile is
one team. `LabelWhole` labels stay whole inside a team. A summed label must be
`LabelWhole`; `LabelTile<'k', ...>` fails to compile with `a contracted label
must be LabelWhole`. How to pick the tiles is in {doc}`choose-a-tile-map`.

**Staging and levels.** `add` never changes the graph you call it on. It
returns a new graph and a handle for each value the nodes produce, so you keep
the new graph and pass the handle on. The contraction reads the handles
`a_tile` and `b_tile`, not the views. A handle exists only after the `add` that
makes it, so a node can only read nodes from earlier `add` calls.

Put nodes in the same `add` only when they are the same kind (all stages, all
contractions or all combines) and have the same tile shape. Here the two
stages have different tile shapes, so they get a level each. `add` does not
enforce this. Two stages of different shapes in one `add` compile and give
wrong values; a stage and a contraction in one `add` fail with an error from
deep in the Team code, for example `no member named 'NumOps'`. To have the
rules checked, instantiate the plan for the graph (include
`<TensorOperations/LevelPlan.hpp>`), anywhere after the last `add`:

```cpp
static_assert(sizeof(TOps::LevelPlan<decltype(g3)::levels_type>) > 0);
```

A violation then stops compilation with a message such as `every member of a
level must share ONE iteration space`.

**Outputs.** `outputs(product)` picks the slots that become views. `execute`
takes one view per slot, in that order, all of the same type. Pass two
roots with views of different types and you get `output views must share one
type`.

**Launch.** The tag's execution space must be the one the graph was made with
(`Space`); a different one fails with `policy tag must match the graph's
execution space`. `execute` returns the number of teams, here 3. Without
`team_size(n)`, or with `n <= 0`, Kokkos chooses the team size
(`Kokkos::AUTO`). A positive `n` is used as given:

```cpp
g3.outputs(product).team_size(threads_per_team).execute(TOps::TeamPolicyTag<Space>{}, c);
```

`execute` does not wait for the kernel. Call `Kokkos::fence()` before the host
reads `c`, as step 3 does. The graph holds copies of the view handles, not of
the elements: keep `a`, `b` and `c` alive and accessible from `Space` until the
fence.

## Sizes the Team path requires

The Team path has no remainder tile. Every `LabelTile` extent must be a
multiple of its tile, and every `LabelWhole` extent must equal its tile. Here
`rows = 6` with tiles of 2 is fine; `rows = 5` is not.

Breaking this is a caller error, mostly unchecked:

| Mistake | What happens |
| --- | --- |
| A `LabelWhole` axis longer than its tile | Aborts in every build: `a label declared LabelWhole is longer than its tile`. |
| A `LabelTile` extent that is not a multiple | No diagnostic in any build. The last tile reads past the data and the results are wrong. |
| A `LabelWhole` axis shorter than its tile | No diagnostic in a Release build. A Debug build asserts (`the tile overhangs the tensor`). |
| An output view shaped in the operands' order instead of the declared order | No diagnostic in any build. With `c` as `(rows, cols)` the program runs and the results are wrong. |

To use a size that is not a multiple, pad the blocked axis up to the next
multiple with harmless values (zeros in `a` here). Size `c` as
`(cols, padded_rows)` and read only `c(j, 0..rows-1)`. {doc}`choose-a-tile-map`
shows the same padding idea for a different recipe.

## Not covered

This page describes the Team path. The CuTe policy has its own launch rules
(for instance, it rejects `team_size`) and this page does not claim these sizes
or this example hold there. What a later level can see of other teams' writes is
in {doc}`../explanation/level-graphs`. The full list of checks is in
{doc}`../reference/checks-and-guards`.
