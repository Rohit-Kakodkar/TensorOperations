# Reuse a tensor under new labels

A contraction sums over the labels two operands share. To contract one matrix
along a different axis of a field, or to feed one result into the next
contraction, give the handle new label names with `as<...>()`. The tensor is
staged once and the new handle reads the same buffer.

This page covers the Team policy on Kokkos Serial. CuTe and register staging
are not covered here.

## Contract one matrix along two axes

A derivative matrix `D(q, k)` (3 x 5) acts on one axis of a field
`u(e, a, b, c)` (6 x 5 x 5 x 4). You want both

```text
Ga(q, e, b, c) = sum_a D(q, a) u(e, a, b, c)
Gb(q, e, a, c) = sum_b D(q, b) u(e, a, b, c)
```

`D` is staged as `(q, a)`, so the first contraction finds the shared label `a`
on its own. For the second, `d.as<'q', 'b'>()` calls the same matrix `(q, b)`.

```cpp
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>

namespace to = TensorOperations;
using Space  = Kokkos::DefaultExecutionSpace;
using Mat    = Kokkos::View<float**, Kokkos::LayoutRight, Space>;
using Field  = Kokkos::View<float****, Kokkos::LayoutRight, Space>;

constexpr int Q = 3, N = 5, C = 4, E = 6, TE = 2;

// One extent per label. 'e' is blocked into tiles of TE, so the graph runs one
// team per tile; every other label is taken whole. The contracted labels
// 'a' and 'b' must be LabelWhole.
using Map = to::LabelTiles<to::LabelWhole<'q', Q>, to::LabelWhole<'a', N>,
                           to::LabelWhole<'b', N>, to::LabelWhole<'c', C>,
                           to::LabelTile<'e', TE>>;

float dval(int q, int k) { return 0.1f + 0.3f * q - 0.2f * k + 0.05f * q * k; }
float uval(int e, int a, int b, int c) {
  return 0.2f + 0.11f * e - 0.07f * a + 0.13f * b - 0.03f * c +
         0.01f * (e + a) * (b + 1);
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    Mat   D("D", Q, N);
    Field U("U", E, N, N, C);
    Field Ga("Ga", Q, E, N, C);  // labels (q, e, b, c)
    Field Gb("Gb", Q, E, N, C);  // labels (q, e, a, c)

    auto Dh = Kokkos::create_mirror_view(D);
    auto Uh = Kokkos::create_mirror_view(U);
    for (int q = 0; q < Q; ++q)
      for (int k = 0; k < N; ++k) Dh(q, k) = dval(q, k);
    for (int e = 0; e < E; ++e)
      for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
          for (int c = 0; c < C; ++c) Uh(e, a, b, c) = uval(e, a, b, c);
    Kokkos::deep_copy(D, Dh);
    Kokkos::deep_copy(U, Uh);

    auto g0      = to::make_level_graph<float, Space>(Map{});
    auto [g1, d] = g0.add(
        to::make_stage_node(to::make_input_node(to::make_handle<'q', 'a'>(D))));
    auto [g2, u] = g1.add(to::make_stage_node(
        to::make_input_node(to::make_handle<'e', 'a', 'b', 'c'>(U))));

    auto along_a = to::make_contraction_node<'q', 'e', 'b', 'c'>(d, u);
    auto along_b =
        to::make_contraction_node<'q', 'e', 'a', 'c'>(d.as<'q', 'b'>(), u);
    auto [g3, ra, rb] = g2.add(along_a, along_b);

    g3.outputs(ra, rb).execute(to::TeamPolicyTag<Space>{}, Ga, Gb);
    Kokkos::fence();

    auto Gah = Kokkos::create_mirror_view(Ga);
    auto Gbh = Kokkos::create_mirror_view(Gb);
    Kokkos::deep_copy(Gah, Ga);
    Kokkos::deep_copy(Gbh, Gb);

    double max_err = 0.0;
    for (int q = 0; q < Q; ++q)
      for (int e = 0; e < E; ++e)
        for (int m = 0; m < N; ++m)
          for (int c = 0; c < C; ++c) {
            double ref_a = 0.0, ref_b = 0.0;
            for (int k = 0; k < N; ++k) {
              ref_a += dval(q, k) * uval(e, k, m, c);
              ref_b += dval(q, k) * uval(e, m, k, c);
            }
            max_err = std::fmax(max_err, std::fabs(ref_a - Gah(q, e, m, c)));
            max_err = std::fmax(max_err, std::fabs(ref_b - Gbh(q, e, m, c)));
          }
    std::printf("max error %g\n", max_err);
    if (max_err > 1e-4) exit_code = 1;
  }
  Kokkos::finalize();
  return exit_code;
}
```

It prints an error near `1e-7`. The extents `q`=3, `a`=`b`=5, `c`=4, `e`=6 are
chosen so that a wrong axis pairing changes the answer.

## What `as<...>()` does

- **It renames by position.** The labels you pass go to the handle's axes in
  order. `d` has axes `(q, a)`, so `d.as<'q', 'b'>()` calls axis 0 `q` and
  axis 1 `b`. You must pass one label per axis (`slot node as(): one label per
  axis`).
- **It copies nothing.** The new handle reads the same staged buffer.
- **The handle's labels are the declared ones.** For a contraction result,
  that is the output order you gave `make_contraction_node`, not a canonical
  order. In the example, `ra` carries `(q, e, b, c)`, so renaming it is
  positional on those four.
- **Swapping two names transposes.** Axis order is fixed by the storage; only
  the names move. With `m` staged as `(x, y)` and `n` as `(x, z)`:

  ```cpp
  // x, y, z mapped to 4, 4 and 3 (z is LabelTile<'z', 3>); M is 4 x 4, N is 4 x 3.
  auto plain   = to::make_contraction_node<'y', 'z'>(m, n);
  auto swapped = to::make_contraction_node<'y', 'z'>(m.as<'y', 'x'>(), n);
  ```

  `plain` sums `m`'s first axis and gives `M^T N`. In `swapped`, `m`'s first
  axis is now called `y`, so the sum runs over its second axis and gives
  `M N`.

## Rename, or let the name matching do it?

Consumers match labels by name, in any order. A contraction or combine that
reads a slot declares its own labels and its own order, and the library
permutes the read. You do not rename to reorder. For example, a combine
declared `<'q', 'c', 'e', 'b'>` can read the `(e, b, q, c)` result of a
contraction directly (`CombineReadsDeclaredSlotInItsOwnOrder` in
`test_level_graph.cpp`).

Rename when you need a different label name, which is one of:

- to sum over a different axis (the example above),
- to match the label the next operand uses (below),
- to transpose, as `swapped` does.

## Feed one result into the next contraction

`ab = A B` has labels `(i, l)`. The next operand `F` calls its first axis `j`,
so rename `ab` to `(i, j)`. A slot cannot be read in the level that produces
it, so the second contraction goes in a later `add`.

```cpp
// i=4 (tile 2), k=4, l=5, j=5, m=6: A is 4x4, B is 4x5, F is 5x6.
using Map = to::LabelTiles<to::LabelTile<'i', 2>, to::LabelWhole<'k', 4>,
                           to::LabelWhole<'l', 5>, to::LabelWhole<'j', 5>,
                           to::LabelWhole<'m', 6>>;

auto g0    = to::make_level_graph<float, Space>(Map{});
auto stage = [](auto handle) {
  return to::make_stage_node(to::make_input_node(handle));
};
auto [g1, a]  = g0.add(stage(to::make_handle<'i', 'k'>(A)));
auto [g2, b]  = g1.add(stage(to::make_handle<'k', 'l'>(B)));
auto [g3, f]  = g2.add(stage(to::make_handle<'j', 'm'>(F)));
auto [g4, ab] = g3.add(to::make_contraction_node<'i', 'l'>(a, b));
auto [g5, r]  = g4.add(to::make_contraction_node<'i', 'm'>(ab.as<'i', 'j'>(), f));
```

## Which labels the tile map must list

List every label a tensor is staged under, and every label you rename to. A
staged label missing from the map stops compilation:

```text
error: static assertion failed ... LabelTiles: every label a node carries must
appear in the graph's tile map -- an absent label has no tile extent to give,
and defaulting one would silently change the iteration space
```

A renamed label that is summed needs its extent from the map unless the call
is a plain two-operand product: two operands, each output label on exactly one
operand, and every summed label on both. Every other call (more than two operands, structured operands such
as `make_outer_product_node`, or a label shared by both operands and the
output) needs every summed label in the map, and reports
`make_contraction_node: a summed label is not in the graph's label map, so its
extent is unknown`. For example, summing over `w` here fails unless `w` is in
the map, because `q` and `e` are shared by both operands and the output:

```cpp
auto z = to::make_contraction_node<'q', 'e', 'c'>(ra.as<'q', 'e', 'w', 'c'>(),
                                                  rb.as<'q', 'e', 'w', 'c'>());
```

For a call like this, add `to::LabelWhole<'w', N>`.

Be careful with a summed label that is on one operand only. The library sums
it by itself, so a rename that fails to match the other operand does not
stop compilation as a mismatch. Without the label in the map you get the
"summed label is not in the graph's label map" error; with it in the map the
call compiles and silently returns the wrong numbers. Fix the rename, not the
map.

Map extents must match the tensors. When you rename an axis, give the new label
the same extent as the label it replaces:

| Situation | What you see |
| --- | --- |
| The new label's extent differs from the other operand's extent for the label it meets (here `d.as<'q', 'c'>()` with `c` mapped to 4, while `D`'s axis is 5 long) | Compile error: `staged A's and B's contracted extents must match`. The same happens if you swap two names whose map extents differ. |
| A `LabelWhole` extent is smaller than the staged tensor's real axis | Runtime abort at `execute`: `a label declared LabelWhole is longer than its tile, so it needs to be gridded -- declare it LabelTile instead` |
| A label's extent is larger than the staged tensor's real axis (for example `c` mapped to 5 on a 4-long axis) | In a Debug build, an assertion: `subview_tile: the tile overhangs the tensor along some mode`. With `NDEBUG` there is no diagnostic and the results are wrong. |
| No label is a `LabelTile` | Compile error: `at least one label must be blocked (LabelTile)` |
| A contracted label is a `LabelTile` | Compile error: `a contracted label must be LabelWhole` |

## Generic code needs `.template`

Where the handle's type is a template parameter, the compiler cannot tell that
`as` is a member template, so write `.template`:

```cpp
template <typename Slot>
auto read_as_j(const Slot& s) {
  return s.template as<'i', 'j'>();  // `s.as<'i', 'j'>()` does not compile
}
```

Without it, Clang reports `missing 'template' keyword prior to dependent
template name 'as'`. This applies in generic lambdas too. In a non-template
function, where `d`'s type is known, plain `d.as<...>()` is fine.

## Renaming does not fix an index slot

An index slot must be read by a gather or scatter with the labels it was
created with, in the same order. `ig.as<...>()` on an index handle does not
get around that; see {doc}`indirect-access` for the rule and the diagnostic.

## See also

- Tests that pin this behavior:
  [`test_level_graph.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_level_graph.cpp)
  (`ThreeMemberFusedGradientLevelEqualsReference`,
  `CombineReadsDeclaredSlotInItsOwnOrder`),
  [`test_level_liveness.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_level_liveness.cpp)
  (chained contractions).
