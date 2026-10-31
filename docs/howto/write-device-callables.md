# Write a combine function

Apply your own per-point function in a graph: read one value from each of
several operands at the same point, and return one result, several, or none.
This page assumes you have read {doc}`indirect-access`, which shows a
single-operand combine. It covers the Team policy; the CuTe backend is not
covered. The factory is `make_combine_node`; see the
{doc}`API reference <../reference/api/graph-and-nodes>`.

## What the function receives

```text
make_combine_node<'e', 'p'>(lambda, strain, fn)

fn(e, p, lambda_value, strain_value)
   |  |  `--------- one value per operand, in the order you passed them
   `--`------------ global coordinates, in the order of the labels <'e', 'p'>
```

- The coordinates are **global**, not coordinates inside a tile. They follow
  the combine's labels, not any operand's.
- Operands may store their labels in any order. In the example below `lambda`
  is `(e, p)` and `strain` is `(p, e)`; the function still sees `(e, p, ...)`.
- Every operand must carry exactly the combine's label set. A combine neither
  reduces nor broadcasts: it cannot read `offset(e)` as a `(e, p)` operand. Use
  a contraction to reduce.

## Example

`stress(e, p) = lambda(e, p) * strain(p, e)`, written twice: once returning two
results, once as a sink that adds each element's stresses into a total. The
second result adds `100 * e + p` to the stress so that swapped coordinates show
up in the output. Extents differ (4 elements, 3 points) and the values depend on position, so
swapped coordinates give wrong numbers.

```cpp
#include <TensorOperations/Evaluator.hpp>
#include <TensorOperations/LevelGraph.hpp>
#include <TensorOperations/NodeHandle.hpp>
#include <TensorOperations/Tiling.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>

// Two operands, two results. Arguments: coordinates (e, p), then the values of
// lambda and strain in the order they were passed to make_combine_node.
struct Stress {
  KOKKOS_FUNCTION Kokkos::Array<float, 2> operator()(int e, int p,
                                                     float lambda,
                                                     float strain) const {
    const float stress = lambda * strain;
    return {stress, stress + 100.0f * e + p};
  }
};

// Same inputs, no results: the callable stores the value itself.
struct SumPerElement {
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace> total;
  KOKKOS_FUNCTION void operator()(int e, int /*p*/, float lambda,
                                  float strain) const {
    Kokkos::atomic_add(&total(e), lambda * strain);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    namespace TOps = TensorOperations;
    using Space  = Kokkos::DefaultExecutionSpace;

    constexpr int E = 4;  // elements
    constexpr int P = 3;  // points per element

    using TileMap =
        TOps::LabelTiles<TOps::LabelTile<'e', 2>, TOps::LabelWhole<'p', P>>;

    // lambda is stored (e, p); strain is stored (p, e).
    Kokkos::View<float**, Kokkos::LayoutRight, Space> lambda("lambda", E, P);
    Kokkos::View<float**, Kokkos::LayoutRight, Space> strain("strain", P, E);
    auto h_lambda = Kokkos::create_mirror_view(lambda);
    auto h_strain = Kokkos::create_mirror_view(strain);
    for (int e = 0; e < E; ++e)
      for (int p = 0; p < P; ++p) {
        h_lambda(e, p) = 1.0f + e;
        h_strain(p, e) = 0.5f * (p + 1);
      }
    Kokkos::deep_copy(lambda, h_lambda);
    Kokkos::deep_copy(strain, h_strain);

    // Output views are indexed in the combine's label order (e, p).
    Kokkos::View<float**, Kokkos::LayoutRight, Space> stress("stress", E, P);
    Kokkos::View<float**, Kokkos::LayoutRight, Space> tagged("tagged", E, P);
    Kokkos::View<float*, Space>                       total("total", E);
    Kokkos::deep_copy(total, 0.0f);  // the sink adds; it never clears

    // Stage both inputs, then combine.
    auto g0 = TOps::make_level_graph<float, Space>(TileMap{});
    auto [g1, l] = g0.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'e', 'p'>(lambda))));
    auto [g2, s] = g1.add(TOps::make_stage_node(
        TOps::make_input_node(TOps::make_handle<'p', 'e'>(strain))));

    // Two results: add() returns the graph, then one handle per result.
    auto [g3, out_stress, out_tagged] =
        g2.add(TOps::make_combine_node<'e', 'p'>(l, s, Stress{}));
    g3.outputs(out_stress, out_tagged)
        .execute(TOps::TeamPolicyTag<Space>{}, stress, tagged);

    // No results: add() returns only the graph, and outputs() takes no views.
    auto g4 =
        g2.add(TOps::make_combine_node<'e', 'p'>(l, s, SumPerElement{total}));
    g4.outputs().execute(TOps::TeamPolicyTag<Space>{});
    Kokkos::fence();  // execute() does not wait; fence before reading.

    auto h_stress =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, stress);
    auto h_tagged =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tagged);
    auto h_total =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, total);
    for (int e = 0; e < E; ++e) {
      float sum = 0.0f;
      for (int p = 0; p < P; ++p) {
        const float want = (1.0f + e) * 0.5f * (p + 1);
        sum += want;
        if (h_stress(e, p) != want) exit_code = 1;
        if (h_tagged(e, p) != want + 100.0f * e + p) exit_code = 1;
      }
      if (h_total(e) != sum) exit_code = 1;
    }
    std::printf("%s\n", exit_code == 0 ? "ok" : "mismatch");
  }
  Kokkos::finalize();
  return exit_code;
}
```

The exact float comparisons hold here because every product and every partial
sum is a small multiple of 0.5, which float represents exactly in any order.
Atomic updates do not fix the summation order, so do not rely on exact equality
for general atomic accumulation.

## What the function may return

| Return type | Results | What `add(...)` gives you |
| --- | --- | --- |
| A scalar | 1 | The graph and one handle |
| `Kokkos::Array<Scalar, M>` | M | The graph and M handles, in array order |
| `void` (a **sink**) | 0 | Only the graph, if every member of the level is a sink |

Pass the handles to `outputs(...)`, and the output views to `execute(...)` in
the same order. A sink writes wherever it likes, so the library does not know
about its stores: it is on you to fence, to make colliding writes safe (for
example with `Kokkos::atomic_add`, as above) and to keep indices in range. A rooted combine
and a sink may share a level.

The scalar is the first operand's `value_type`. Each returned element must be
convertible to it. To choose another, use the explicit form
`make_combine_node<Scalar, ExecSpace, labels...>(...)`.

## What the function must be

- Callable on a `const` object: `operator() const`. The evaluator calls it
  through a const kernel capture.
- Device-callable on your backend: `KOKKOS_FUNCTION` (or
  `KOKKOS_INLINE_FUNCTION`), with state captured by value. A functor holding a
  `Kokkos::View` as in `SumPerElement` is the usual form.
- Taking one `int` per label, then one parameter per operand that accepts the
  scalar.

## What the operands may be

| Operand | Allowed |
| --- | --- |
| Staged input (`make_stage_node(make_input_node(...))`) | Yes |
| Result of a contraction, a gather, or a register node | Yes |
| Functional input | No: wrap it in `make_stage_node` first |
| Index slot (an integer value) | No: only gather and scatter read it |
| A different label set from the output | No |

A level holds either contractions or combines, never both. Put them in
separate `add(...)` calls; the combine goes in a later level than anything it
reads.

## Common mistakes

| Compiler message (abridged) | Cause |
| --- | --- |
| `combine node: each operand's label set must equal the output label set (a pointwise op reduces no mode)` | An operand has labels the output lacks, or lacks some. Contract to reduce. |
| `combine fn must be const-callable as fn(i_0, ..., i_{Rank-1}, v_0, ..., v_{N-1}) -> value_type` | Wrong number of parameters (forgetting the coordinates is the usual cause), parameters that cannot take the scalar, or a non-`const` `operator()`. |
| `a functional input has no address, and a combine reads its operands as strided memory. Wrap it in make_stage_node first` | A functional input passed straight to the combine. |
| `combine node: an index slot is read only by gather and scatter nodes, not as a combine operand` | An integer slot used as a value. |
| `combine node: every operand and the output must have equal rank` | The label count differs from an operand's rank. |
| `a level must be NON-EMPTY and HOMOGENEOUS` | A contraction and a combine in one `add(...)`. |
| `Structured binding ... binds to 3 elements, but only 2 names were provided` (wording varies by compiler) | The destructuring does not match the number of results. |

Operand extents must agree on every label. That check is an `assert`, so it
disappears under `NDEBUG`; all of the above are compile-time checks.

## See also

- {doc}`indirect-access` for gathers and scatters around a combine.
- {doc}`../reference/checks-and-guards`.
- Working tests: [`test_level_graph.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_level_graph.cpp)
  (multi-output, global coordinates),
  [`test_sink_combine.cpp`](https://github.com/Rohit-Kakodkar/TensorOperations/blob/main/tests/test_sink_combine.cpp).
