# Build options

Variables read by the repository's top-level `CMakeLists.txt`. To bring the
library into your own project, see {doc}`../howto/use-in-your-project`.

| Variable | Default | Effect |
| --- | --- | --- |
| `TENSOR_OPS_ENABLE_TIMING` | `OFF` | Adds the `TENSOR_OPS_ENABLE_TIMING` definition to `tensor` |
| `TENSOR_OPS_VERIFY_RESHAPE` | `OFF` | Adds `TENSOR_OPS_VERIFY_RESHAPE=1` to `tensor`; see `TileLayout.hpp` for what it checks (debug builds enable the check without it) |
| `TENSOR_OPS_BUILD_BENCHMARKS` | `OFF` | Builds the extra benchmarks in `CMakeLists.txt`. The tests and the `sfpp_min` executables are built regardless |
| `TENSOR_OPS_CUTLASS_DIR` | empty | CUTLASS checkout. With `Kokkos_ENABLE_CUDA` also set, adds its `include/` and `TENSOR_OPS_ENABLE_CUTE` to `tensor` |

Kokkos options such as `Kokkos_ENABLE_SERIAL` are passed through to the Kokkos
the library fetches.

```{admonition} Not yet written
:class: note
Registered test names.
```
