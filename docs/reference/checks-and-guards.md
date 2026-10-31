# Checks and guards

Contracts the caller must uphold, and what the library does and does not
verify. Task-oriented usage is in the how-to guides.

## Indirect gather and scatter: obligations

A source is either a rank-one Kokkos view (including a strided field-column
subview) or an object callable as `source(int)` on a **const** object with a
non-void result. A destination is a writable rank-one Kokkos view or a
const-callable `dst(int)` accessor returning a non-const lvalue reference
`T&`, whose element type supports `Kokkos::atomic_add`. A rank-two field cannot
be passed directly to either factory: select a column with
`Kokkos::subview(field, Kokkos::ALL, component)` or wrap it in an accessor.
A const view handle can still write mutable elements; a view of const elements
cannot be a scatter destination.

Custom call operators execute in the kernel. Make them device-callable for the
chosen backend, for example with `KOKKOS_FUNCTION`, and const-qualified. A
destination accessor must return a reference to the actual destination element,
not a temporary or a value. The factories inspect callability and reference
types; they do not establish that arbitrary captured pointers are valid there.

Before execution, the caller must establish the following:

- Every connectivity entry denotes a valid source/destination element, or a
  valid argument in the accessor's domain. Gather passes the stored integer to
  the source; an accessor taking `int` can narrow it. Scatter explicitly converts
  the stored integer to `int` in both forms. For a workflow sharing connectivity,
  use values representable as `int` and valid for both accesses. For a view,
  this means a nonnegative index below its extent. Accepting an integral type
  does not make all values of that type safe.
- Connectivity and numerical values have equal extents for corresponding
  labels. Direct scatter iterates over the value's shape, not the connectivity
  view's shape. Slot scatter also derives its shape from the value and does not
  compare it with the index slot's shape. Every label's extent must also be
  consistent with the graph's tile map (`LabelTile` or `LabelWhole`) for each
  staged tensor that carries it.
- Connectivity, source, destination, and any accessor captures are accessible
  from the graph's execution space. The example uses `Kokkos::DefaultExecutionSpace`
  consistently. The factories do not infer or validate all memory-space
  relationships through arbitrary accessors.
- Referenced storage stays valid through kernel completion. Factories retain
  views and accessors by value; managed Kokkos views share ownership, while raw
  pointers, unmanaged views, and borrowed accessor state remain borrowed.
  Order input initialization before execution, and wait for completion before
  host observation, clearing destinations, or releasing borrowed storage.

### What the library checks

| Check | Mechanism | Present in release builds |
| --- | --- | --- |
| Integral indices, label count and order, value label set, supported source and destination forms, earlier-level slot producers, in-memory Team index | `static_assert` / level-plan check | Compile time only |
| Staged inputs disagreeing on a blocked label's extent | `assert` | No: removed by `NDEBUG` |
| A `LabelWhole` label longer than its tile | `Kokkos::abort` | Yes |
| Slot store larger than the backend's team scratch | `Kokkos::abort` | Yes |
| Connectivity values in range, index/value extent agreement | None | Not checked |

None of these validates connectivity contents or provides general shape
validation, and the indirect accesses add no range check. Backend or view bounds
checking, when enabled, is not a substitute for the caller obligations above.

### Local computation between gather and scatter

A combine function receives the global coordinates first, one `int` per label in
the node's label order, followed by the values of its operands. `DoublePlusOne`
in the how-to ignores the coordinates `(e, p)` and uses only the gathered value.
See {doc}`../howto/write-device-callables` for several operands and results.

### Accumulate and complete

Scatter performs `Kokkos::atomic_add(&dst(index), converted_value)` for each
local contribution. It converts the contribution to the destination element
type before adding. Collisions add into the same element, including across
teams. Scatter does not clear the destination: a prefilled destination keeps
its initial value plus contributions, and another execution adds again.
Explicitly initialize to zero for a fresh sum. Atomic updates do not fix the
floating-point summation order, so do not infer reproducible results from
exact comparisons in small tests.

Scatter is a zero-output sink. A level containing only scatter sinks returns
the new graph directly from `add(...)`, without a value handle to destructure.
`graph.outputs().execute(TeamPolicyTag<ES>{})` runs such a graph with no selected
roots and no output-view arguments. Destination mutation is the observable
result. Submission does not include a completion fence; use `Kokkos::fence()`
before reading the result on the host, as below.

The scatter tests cover collisions, both connectivity forms and layouts, column
subviews, and a writable-reference accessor. Prefilled destinations and repeated
execution follow from the atomic-add implementation; they are not covered by
dedicated tests.

Worked example: {doc}`../howto/indirect-access`.

## Staged contraction and output views

Worked example: {doc}`../howto/contract-with-a-level-graph`. Team path only.

| Check | Mechanism | Present in release builds |
| --- | --- | --- |
| A contracted label declared `LabelTile` | `static_assert` in `add` | Compile time only |
| A label missing from the tile map, or listed twice | `static_assert` | Compile time only |
| View count differs from output count; output views of different types; policy tag for another execution space; no blocked label | `static_assert` in `execute` | Compile time only |
| Level rules (one kind and one iteration space per level) | Only when you instantiate `LevelPlan` for the graph; `add` does not check them | Compile time, on request |
| A `LabelWhole` axis longer than its tile | `Kokkos::abort` | Yes |
| Contracted extents differ between operands | `assert` | No: removed by `NDEBUG` |
| A partial tile in a staged input | None | Not checked: wrong values |
| A `LabelWhole` axis shorter than its tile | `assert` when a tile is staged or stored | No: removed by `NDEBUG` |
| Output view axis order and extents matching the declared labels | None | Not checked: wrong values |
