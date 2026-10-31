# Inputs, staging and contraction

How to use these together is in
{doc}`../../howto/contract-with-a-level-graph`.

```{doxygenfunction} TensorOperations::make_input_node
```

```{doxygenfunction} TensorOperations::make_stage_node
```

## Contraction

`make_contraction_node<'j', 'i'>(a_tile, b_tile)` contracts the labels the two
staged operands share and gives the output axes in the order you list them.
`make_contraction_node<double, Space, 'j', 'i'>(a_tile, b_tile)` sets the output
scalar and execution space. Both are `make_contraction_node` in
`NodeHandle.hpp`.

The rules, in short:

- Every output label is on exactly one operand, and every label on both
  operands is summed. Output labels are distinct.
- The output view you pass to `execute` has its axes in the order you list the
  labels, not the operands' order.
- A summed label must be `LabelWhole` in the tile map.
- A call that does not fit this shape (more operands, a structured operand, a
  label shared with the output) selects the general contraction, whose
  compile errors differ.
