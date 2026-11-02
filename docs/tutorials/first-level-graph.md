# Your first level graph

You will write one program that applies a small matrix to every element's data,
then scales each result by a weight:

```text
out(e, i) = w(i) * sum over j of D(i, j) * u(e, j)
```

`u` holds one row of 5 values per element, `D` is a 3 by 5 matrix shared by all
elements, and `w` has one weight per row of `D`. It is the kind of team kernel
you would otherwise write by hand with `Kokkos::TeamPolicy`. Here you describe
the work as a graph and the library runs it. You need a build that links the
`tensor` target; see {doc}`../howto/use-in-your-project`.

## The program

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

constexpr int elements   = 8;  // e: how many elements
constexpr int tile_elems = 4;  // e is cut into tiles of 4, so 2 teams
constexpr int rows       = 3;  // i: rows of D
constexpr int cols       = 5;  // j: summed

using TileMap =
    TOps::LabelTiles<TOps::LabelTile<'e', tile_elems>,
                     TOps::LabelWhole<'i', rows>, TOps::LabelWhole<'j', cols>>;
using Matrix = Kokkos::View<float**, Kokkos::LayoutRight, Space>;

// Multiplies each product by the weight of its row.
struct Weight {
  Kokkos::View<float*, Space> w;
  KOKKOS_FUNCTION float operator()(int /*e*/, int i, float product) const {
    return w(i) * product;
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    Matrix                      d("d", rows, cols);      // labels (i, j)
    Matrix                      u("u", elements, cols);  // labels (e, j)
    Kokkos::View<float*, Space> w("w", rows);
    Matrix                      out("out", elements, rows);  // labels (e, i)

    auto h_d = Kokkos::create_mirror_view(d);
    auto h_u = Kokkos::create_mirror_view(u);
    auto h_w = Kokkos::create_mirror_view(w);
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) h_d(i, j) = 1.0f + i - 0.5f * j;
    for (int e = 0; e < elements; ++e)
      for (int j = 0; j < cols; ++j) h_u(e, j) = 0.25f * e + j;
    for (int i = 0; i < rows; ++i) h_w(i) = 2.0f + i;
    Kokkos::deep_copy(d, h_d);
    Kokkos::deep_copy(u, h_u);
    Kokkos::deep_copy(w, h_w);

    auto g0            = TOps::make_level_graph<float, Space>(TileMap{});
    auto [g1, d_tile]  = g0.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'i', 'j'>(d))));
    auto [g2, u_tile]  = g1.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'e', 'j'>(u))));
    auto [g3, product] =
        g2.add(TOps::make_contraction_node<'e', 'i'>(d_tile, u_tile));
    auto [g4, weighted] =
        g3.add(TOps::make_combine_node<'e', 'i'>(product, Weight{w}));

    const int teams =
        g4.outputs(weighted).execute(TOps::TeamPolicyTag<Space>{}, out);
    Kokkos::fence();

    auto h_out =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
    double max_err = 0.0;
    for (int e = 0; e < elements; ++e)
      for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) sum += h_d(i, j) * h_u(e, j);
        max_err = std::fmax(max_err, std::fabs(h_w(i) * sum - h_out(e, i)));
      }
    std::printf("teams=%d max_err=%g\n", teams, max_err);
    if (teams != elements / tile_elems || max_err > 1e-4) exit_code = 1;
  }
  Kokkos::finalize();
  return exit_code;
}
```

On Serial it prints `teams=2 max_err=0`. Verified on macOS with clang and the
Serial backend; other backends are untried.

## Walk-through

**Labels.** A label is a one-character name for an axis, written as a template
argument: `make_handle<'i', 'j'>(d)` says `d`'s first axis is `i` and its
second is `j`. Labels do not move data. They decide how operands line up: a
label on both operands of a contraction (`j`, here) is summed. This example uses
extents 8, 3 and 5, all different, so a swapped axis shows up instead of passing
unnoticed.

**Tile map.** The first argument to `make_level_graph` lists every label and
how it is split. `LabelTile<'e', 4>` cuts `e` into tiles of 4, and each tile
becomes one team. `LabelWhole` labels are not split, so every team sees all of
`i` and `j`. A summed label must be `LabelWhole`. For how to pick tile sizes for
real data, see {doc}`../howto/choose-a-tile-map`. `elements` must be a multiple
of 4. There is no remainder tile: with a partial tile the last team still runs,
reads past the inputs and writes past the end of `out`. No build reports it and
the in-range results can still look right, so a passing check does not prove
the size is safe. For 10 elements, pad `u` with zero rows to 12, size `out` as
12 by 3 and read only rows 0 to 9; see "Sizes the Team path requires" in
{doc}`../howto/contract-with-a-level-graph`.

**Stage.** `make_stage_node(make_input_node(...))` copies each team's tile of
an operand from its `Kokkos::View` into the team's scratch memory. `d` is
staged as well as `u` because a contraction reads staged operands, not views.

**Level, slot handle.** Each `add` call appends one level and returns a new
graph plus a **slot handle** for every value the nodes in it produce. A slot
handle stands for a value inside the graph, not for data you can read. You pass
it to later nodes (`d_tile`, `u_tile`, `product`) instead of the view.
`add` never changes the graph you call it on, so keep the new one: `g1`, `g2`,
and so on. A node can only read handles from earlier `add` calls, so the order
of the calls is the order the work runs. Here the two stages sit in separate
levels because their tile shapes differ (`d` has no `e`), and the combine sits
in a level after the contraction it reads. What may share a level is in
{doc}`../howto/contract-with-a-level-graph`.

**Contraction.** `make_contraction_node<'e', 'i'>(d_tile, u_tile)` sums over
`j`, the one label both operands share. The labels in the angle brackets are
the axes of the result, in order. {doc}`../howto/contract-with-a-level-graph`
covers the rules and the mistakes that compile but give wrong values.

**Combine.** A combine applies your function to each point of its labels. The
function gets the global `e` and `i`, then one value for each operand you
passed, and returns the new value. `Weight` reads `w(i)` from a view it holds
by value, because a combine's operands must all have the combine's own labels,
and `w` has only `i`. Write it as a functor with `KOKKOS_FUNCTION` and a
`const` call operator. More on arguments, several results and sinks is in
{doc}`../howto/write-device-callables`.

**Output.** `outputs(weighted)` picks the slots that come back as views, and
`execute` takes one view per slot, in that order. `out` must be shaped in the
label order of the node that made the slot: `(e, i)`.

**Launch and fence.** `execute` returns the number of teams (2 here) but does
not wait for the kernel. Call `Kokkos::fence()` before the host touches the
results, as the program does. `create_mirror_view_and_copy` also waits, so the
fence only matters if you read `out` directly on a host backend, or overwrite
`d`, `u` or `w`, while the kernel may still be running; on a host backend that
is a data race. Keep `d`, `u`, `w` and `out` alive until the fence: the graph
holds copies of the view handles, not of the elements. What one `execute`
runs, and which writes a later level can see, is in
{doc}`../explanation/level-graphs`.

## Where to go next

- Sizes that do not divide the tile, and the failures they cause:
  {doc}`../howto/contract-with-a-level-graph` and
  {doc}`../howto/choose-a-tile-map`.
- Look up elements through connectivity: {doc}`../howto/indirect-access`.
- Combines with several operands or results:
  {doc}`../howto/write-device-callables`.
- Every compile-time and runtime check: {doc}`../reference/checks-and-guards`.
