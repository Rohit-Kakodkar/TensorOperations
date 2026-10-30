# Backends

| Backend | Build requirement | Validated while writing this documentation |
| --- | --- | --- |
| Serial | `-DKokkos_ENABLE_SERIAL=ON` | Yes: the indirect-access example, with clang on macOS |
| OpenMP | `-DKokkos_ENABLE_OPENMP=ON` | No |
| CUDA (Kokkos) | `-DKokkos_ENABLE_CUDA=ON` and a CUDA toolchain | No |
| CUDA + CuTe | CUDA plus `TENSOR_OPS_CUTLASS_DIR` | No |

The library is built and run on all four backends by its maintainer, but the
last column records only what the documentation's own examples were checked
against. A pass on one backend does not establish another.
