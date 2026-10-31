# Graphs and nodes

For a worked contraction, see {doc}`../../howto/contract-with-a-level-graph`.
For a worked gather and scatter, with the caller obligations, see
{doc}`../../howto/indirect-access`. For when one `execute` is not enough, see
{doc}`../../explanation/level-graphs`.

## Building and launching a graph

```{doxygenfunction} TensorOperations::make_level_graph
```

```{doxygenstruct} TensorOperations::LevelGraph
:members: add, outputs
```

```{doxygenstruct} TensorOperations::LevelOutputs
:members: team_size, execute
```

Staging and contraction nodes are in {doc}`contraction`.

## Indirect access

```{doxygenfunction} TensorOperations::make_index_node
```

```{doxygenfunction} TensorOperations::make_gather_node
```

```{doxygenfunction} TensorOperations::make_scatter_add_node
```

## Pointwise combine

`make_combine_node<'e', 'p'>(a, b, fn)` takes the scalar from the first operand.
`make_combine_node<double, Space, 'e', 'p'>(a, b, fn)` sets the scalar and
execution space. The Doxygen comment is on the first form in `NodeHandle.hpp`.
To write the function it calls, see
{doc}`../../howto/write-device-callables`.
