# `bench_matrix_multiply`

Benchmarks the team-policy contraction path
(`Graph::execute(TeamPolicyTag{}, Tile<...>, C)`) against a hand-written Kokkos
baseline for plain dense matrix-matrix multiplication

```
C[i,j] = sum_{k} A[i,k] * B[k,j]      # a dense GEMM  I x K x J
```

## What it compares

| impl      | description                                                                  |
|-----------|------------------------------------------------------------------------------|
| `library` | `Graph::execute` with `TeamPolicyTag`. Generic over rank/pattern; stages tiles into team scratch with a **coalesced (memory-order) access pattern for any input layout**. |
| `naive`   | Plain Kokkos team kernel reading `A`/`B` straight from global memory in the inner loop — no scratch, no reuse. The obvious first kernel a user writes. |

## The stories

1. **Throughput vs `naive` (GFLOP/s).** The library beats the naive
   global-memory kernel by a wide margin (scratch staging raises arithmetic
   intensity / cuts global traffic), and the margin *grows* on `LayoutLeft`.
2. **Layout robustness / coalescing.** Each impl is run with the inputs `A`,`B`
   in both `Kokkos::LayoutRight` and `Kokkos::LayoutLeft`. The library stages
   with a memory-order access pattern, so its GFLOP/s is ~flat across layouts;
   the naive baseline's global reads become strided/uncoalesced on `LayoutLeft`
   and throughput drops. The "Layout robustness" summary reports the Left/Right
   ratio per impl (closer to `1.0` = more robust).

The benchmark also prints a correctness check (`library` vs `naive` within a
relative tolerance; `PASS`/`FAIL`) and a lines-of-code / generality table.

## Build & run

CPU (Serial/OpenMP) build:

```bash
cmake -S . -B build -DTENSOR_OPS_BUILD_BENCHMARKS=ON
cmake --build build --target bench_matrix_multiply
./build/bench_matrix_multiply
```

GPU (CUDA) build: configure Kokkos for your backend/arch as usual (e.g.
`-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON`) and add
`-DTENSOR_OPS_BUILD_BENCHMARKS=ON`. Run on a node with the GPU visible.

Sizes auto-select a CPU- or GPU-appropriate preset from the default execution
space. Override the sweep:

```bash
./build/bench_matrix_multiply [N [reps [warmup]]]
```

`N` must be divisible by the output tile (`TI`/`TJ`, both 32). `K` and the tile
extents are compile-time (see the `cfg` namespace in
`bench_matrix_multiply.cpp`).
