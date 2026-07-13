# `bench_multilevel_contraction`

Benchmarks the library's **fused multi-level (chained) contraction** path
(`Graph::execute(TeamPolicyTag{}, flat-tile-list, E)`) against two hand-written Kokkos
baselines for the chained contraction

```
M[i,j] = sum_k A[i,k] B[k,j]        # GEMM  I x K x J
E[i,l] = sum_j M[i,j] D[j,l]        # GEMM  I x J x L
=>  E = (A·B)·D                     # matrix-chain / two-level GEMM
```

This is the matricized form of the operator/transfer chains ubiquitous in scientific
computing — triple products `Xᵀ H X` / `Bᵀ D B`, change-of-basis, and the reduced form
of sum-factorization (tensor-product operator application) in high-order FEM/spectral
methods.

## What it compares

| impl      | description |
|-----------|-------------|
| `library` | `Graph::execute` with a flat per-contraction tile list. Generic over rank/chain; **fused** — the intermediate `M` never touches global memory (produced tile-by-tile in team scratch, recomputed per output tile) — and stages with a coalesced (memory-order) access pattern for any input layout. |
| `naive`   | Two passes that **materialize `M` to global memory**: kernel 1 `M = A·B`, kernel 2 `E = M·D`, each a plain team kernel reading operands straight from global memory. The obvious first approach. |
| `tiled`   | A **straightforward** hand-written **fused** team+scratch kernel (what a practitioner realistically writes — *not* register-blocked), hardcoded for this chain. One team per `E` tile, looping over `j`-tiles: stage → scratch GEMM for `M` → scratch GEMM for `E`; no global `M`. |

Each impl runs with `A`,`B`,`D` in both `Kokkos::LayoutRight` and `Kokkos::LayoutLeft`,
and in two intermediate regimes: **single-tile** (`J = TJ`) and **multi-tiled**
(`J = 4·TJ`, which makes the fused paths recompute `M` across several `j`-tiles).

## The stories

1. **Coalescing / layout robustness — the headline.** `naive` reads its operands with a
   fixed (LayoutRight-tuned) index order, so on `LayoutLeft` its global reads become
   strided/uncoalesced and throughput **collapses** (Left/Right ≈ 0.11 on an A100). The
   library stages with a memory-order access pattern (`SubviewLayout::operator[]` /
   `tile_view` reading `view.stride(d)` in `TiledLayout.hpp`), so it stays usable across
   layouts (≈ 0.74) and beats `naive` by **6–9×** on `LayoutLeft`. On `LayoutRight` the
   two are close (both bandwidth-bound; the intermediate is small).
2. **Fuse vs. materialize.** `library` and `tiled` fuse — they trade the global `M`
   write + repeated `M` reads for recomputing `M` in scratch. `naive` materializes `M`.
   In the multi-tiled regime the library's advantage over `naive` grows as the
   intermediate does.
3. **Fast fused kernels are hard to hand-write.** The `tiled` baseline is a *correct*
   fused kernel but not register-blocked, and runs far below the library (which applies
   the same register-blocking/tiling to **both** GEMMs of the chain). Note its Left/Right
   ratio ≈ 1.0 is **not** coalescing quality — it is throughput-limited by team barriers
   and lack of register reuse, so memory layout barely matters. The library delivers
   register-blocked throughput for the whole chain from ~18 generic lines (any rank /
   chain via einsum modes + `StaticTile`), versus a hardcoded, un-tuned hand kernel.

The benchmark also prints a correctness check (all three impls within a relative
tolerance; `PASS`/`FAIL`) and a lines-of-code / generality table.

## Build & run

CPU (Serial/OpenMP):

```bash
cmake -S . -B build -DTENSOR_OPS_BUILD_BENCHMARKS=ON
cmake --build build --target bench_multilevel_contraction
./build/bench_multilevel_contraction
```

GPU (CUDA): configure Kokkos for your backend/arch (e.g. `-DKokkos_ENABLE_CUDA=ON
-DKokkos_ARCH_AMPERE80=ON`) plus `-DTENSOR_OPS_BUILD_BENCHMARKS=ON`, and run on a node
with the GPU visible.

Sizes auto-select a CPU- or GPU-appropriate preset from the default execution space.
Override the sweep:

```bash
./build/bench_multilevel_contraction [N [reps [warmup]]]
```

`N` must be divisible by the output tile (`TI`/`TL`, both 32). `K` and the tile extents
are compile-time; the two `J` regimes (`TJ` and `4·TJ`) are in the `cfg` namespace in
`bench_multilevel_contraction.cpp`.
