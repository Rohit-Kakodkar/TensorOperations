# Benchmarks

Performance benchmarks for the library, opt-in via CMake.

## `bench_team_contraction`

Benchmarks the team-policy contraction path
(`Graph::execute(TeamPolicyTag{}, Tile<...>, C)`) against two hand-written Kokkos
baselines for the representative contraction

```
C[i,l] = sum_{j,k} A[i,j,k] * B[j,k,l]      # a GEMM  I x (J*K) x L
```

### What it compares

| impl      | description                                                                  |
|-----------|------------------------------------------------------------------------------|
| `library` | `Graph::execute` with `TeamPolicyTag`. Generic over rank/pattern; stages tiles into team scratch with a **coalesced (memory-order) access pattern for any input layout**. |
| `naive`   | Plain Kokkos team kernel reading `A`/`B` straight from global memory in the inner loop — no scratch, no reuse. The obvious first kernel a user writes. |
| `tiled`   | Competent hand-written team+scratch GEMM, **hardcoded for this one rank-3 × rank-3 contraction**. Stages tiles → block GEMM → store, mirroring the library, but its staging uses a fixed row-major (LayoutRight-tuned) thread→index mapping. |

### The stories

1. **Throughput vs `naive` (GFLOP/s).** The library beats the naive global-memory
   kernel by a wide margin (scratch staging raises arithmetic intensity / cuts
   global traffic), and the margin *grows* on `LayoutLeft` (see below).
2. **Layout robustness / coalescing.** Each impl is run with the inputs `A`,`B` in
   both `Kokkos::LayoutRight` and `Kokkos::LayoutLeft`. The library stages with a
   memory-order access pattern (see `SubviewLayout::operator[]` /
   `tile_view` reading `view.stride(d)` in `TiledLayout.hpp`), so its GFLOP/s is
   ~flat across layouts. The baselines hardcode a LayoutRight-tuned index order, so
   their global reads become strided/uncoalesced on `LayoutLeft` and throughput
   drops. The "Layout robustness" summary table reports the Left/Right ratio per
   impl (closer to `1.0` = more robust); on an H100 the library is the most robust
   (≈0.95), the hand-`tiled` baseline degrades (≈0.79), and `naive` collapses
   (≈0.38).
3. **Generality vs the hand-`tiled` baseline.** The hand-tuned, register-blocked
   `tiled` GEMM reaches higher *peak* GFLOP/s than the library — that is expected,
   and the gap is a useful regression signal for the team-policy path. The
   library's advantage over it is not peak throughput but that it delivers
   competitive performance plus better layout-robustness from ~15 generic lines
   that work for *any* rank and contraction pattern, versus ~75 lines hardcoded to
   this one rank-3 × rank-3 contraction.

The benchmark also prints a correctness check (all three impls compared within a
relative tolerance; `PASS`/`FAIL`) and a lines-of-code / generality table: the
library call is a handful of generic lines that handle arbitrary rank and
contraction pattern from einsum-style mode labels, while the baselines are
hardcoded to this one contraction.

### Build & run

CPU (Serial/OpenMP) build:

```bash
cmake -S . -B build -DTENSOR_OPS_BUILD_BENCHMARKS=ON
cmake --build build --target bench_team_contraction
./build/bench_team_contraction
```

GPU (CUDA) build: configure Kokkos for your backend/arch as usual (e.g.
`-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON`) and add
`-DTENSOR_OPS_BUILD_BENCHMARKS=ON`. Run on a node with the GPU visible.

Sizes auto-select a CPU- or GPU-appropriate preset from the default execution
space. Override the sweep:

```bash
./build/bench_team_contraction [N [reps [warmup]]]
```

`N` must be divisible by the output tile (`TI`/`TL`, both 32). `J`, `K` and the
tile extents are compile-time (see the `cfg` namespace in
`bench_team_contraction.cpp`).
