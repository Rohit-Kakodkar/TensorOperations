# Graphs and nodes

For a worked example and the caller obligations, see
{doc}`../../howto/indirect-access`.

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
