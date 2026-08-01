# SEM stiffness: where the library's 4.26x gap actually comes from

**Status:** measured 2026-08-01 on `feat/fused-combine-operands`, H100 NVL (sm_80
build, PTX JIT) and Serial/AVX-512. All numbers below are `min` of 3 reps after
1 warmup, `E = 2500000` (GPU) / `E = 16384` (CPU).

This document answers one question — *is redundant gradient recompute the
bottleneck?* — and turns the answer into an ordered work plan.

**Short answer: no, not on its own.** Gradient recompute is the largest single
line item (~50% of library runtime, worth 1.62x), but it is one of three
comparable factors that multiply together. Removing it entirely leaves the
library ~2.6x slower than hand-written code.

---

## Table of contents

- [Results](#results)
- [Method: the CONTROL kernel](#method-the-control-kernel)
- [The decomposition](#the-decomposition)
- [Factor 1 — redundancy (2.90x)](#factor-1--redundancy-290x)
- [Factor 2 — library overhead (1.48x)](#factor-2--library-overhead-148x)
- [Cross-backend comparison](#cross-backend-comparison)
- [Corrections to the benchmark header](#corrections-to-the-benchmark-header)
- [Statement of work](#statement-of-work)
- [Caveats](#caveats)

---

## Results

### GPU (H100 NVL, E = 2500000)

```
impl                             time(ms)    useful GF/s    executed GF/s
baseline (hand, 1 kernel)           9.737         2727.8                -
library (fused, 2 kernels)         41.487          640.2           1612.1
CONTROL   (hand, lib's redundancy) 28.190          942.2           2372.5
CONTROL-B (grads deduped)          17.401         1526.4                -
  [diag] 1 gradient only            1.984         1290.2
  [diag] 4 gradients, no aux        5.551         1844.6
  [diag] 1 F node (4grad+stress)    6.335         1616.5
  [diag] library, 1 component      20.969
  [diag] 1 divergence (mat. B)      2.003         1278.1
```

`speedup (base/lib) = 0.2347x`, i.e. the library is **4.26x slower**.

### CPU (Serial / AVX-512, E = 16384)

```
baseline (hand, 1 kernel)          41.861            4.2                -
library (fused, 2 kernels)        154.539            1.1              2.8
CONTROL   (hand, lib's redundancy) 134.453           1.3              3.3
CONTROL-B (grads deduped)          91.814            1.9                -
```

**3.69x slower**, factoring as **3.21x redundancy x 1.11x residual**. The
2.52x executed-FLOP ratio is a *floor* on the redundancy term, not an estimate
of it: the measured 3.21x exceeds it because redundancy also carries ~4x the
auxiliary global traffic. See [Cross-backend](#cross-backend-comparison) — an
earlier revision of this document used 2.52x here as if it were the measurement,
and drew a wrong conclusion from it.

---

## Method: the CONTROL kernel

The two headline rows aren't comparable on their own — the library does 2.52x
the FLOPs and ~4x the auxiliary global traffic, so "4.26x slower" conflates
*doing more work* with *doing work less efficiently*.

`CONTROL` separates them. It is a hand-written Kokkos kernel that reproduces the
library's **exact work profile** at hand-written efficiency:

| | launches | gradient sums | stress evals | divergence sums | aux reads |
|---|---|---|---|---|---|
| baseline | 1 | 4 | 1 | 4 | 1x |
| **CONTROL** | 2 | **16** | **4** | 4 | **4x** |
| **CONTROL-B** | 2 | **8** | **4** | 4 | 4x |
| library | 2 | 16 | 4 | 4 | 4x |

`CONTROL-B` is the identical kernel with one `constexpr bool` flipped, hoisting
the gradient stage out of the F-node loop so it runs once per launch instead of
twice. That single difference isolates the intra-kernel gradient recompute.

Both controls validate against the host reference (`max rel diff 0.00e+00`), so
they are doing the real physics, not a stripped-down imitation.

- `baseline -> CONTROL` = the price of the redundancy itself.
- `CONTROL -> library` = everything else the library costs.

---

## The decomposition

The 4.26x gap factors into three multiplicative terms:

| factor | cost | what it is |
|---|---|---|
| 2 launches + 4x stress + cross-kernel gradient recompute | **1.79x** | `baseline -> CONTROL-B` |
| intra-kernel gradient recompute | **1.62x** | `CONTROL-B -> CONTROL` |
| library vs equivalent hand-written code | **1.48x** | `CONTROL -> library` |

`1.79 x 1.62 x 1.48 = 4.26` ✓

Only the third term is GPU-specific in any part: on Serial the same `CONTROL`
construction gives a residual of just 1.11x. See
[Cross-backend](#cross-backend-comparison).

Gradient recompute is split across the first two terms: the library computes
each gradient **16 times where the baseline computes it 4 times** — 2x across
the component kernels, 2x across the two F nodes within each kernel.

---

## Factor 1 — redundancy (2.90x)

### It is dominated by the gradients, not the stress

From the diagnostics, the marginal cost of one gradient sweep inside a combine
is obtained by fitting `t = a + b*n` to the two measured points
(`n=1: 1.984 ms`, `n=4: 5.551 ms`):

```
b = (5.551 - 1.984) / 3 = 1.19 ms per gradient sweep
a = 0.79 ms fixed per kernel
```

The library runs 16 of them → **~20-22 ms of the 41.49 ms, roughly half the
runtime**. The baseline runs 4.

The stress/chain-rule redundancy is small by comparison: the aux-read cost of a
whole F node is `6.335 - 5.551 = 0.78 ms`, so all four F nodes' auxiliary
traffic is ~3.1 ms, and cutting it 4x -> 1x saves ~2.3 ms.

### Why it happens

The hand kernel derives all four integrand slots (`Fxi0, Fgm0, Fxi1, Fgm1`)
from **one** stress tensor and **one** set of gradients. The library cannot: a
combine node is `NumOut == 1` when used as a fused operand, so each of the four
slots is a separate node that drags its own copy of all four gradients and its
own stress evaluation underneath it.

Multi-output combine already works at the **graph root** (`fn` returning
`Kokkos::Array<V,M>`, `ops()` expanding to `[g, p0, p1]`). What is missing is
multi-output combine as a fused **operand** — the phase-2 work gated by
`Impl::single_result`.

### The mode-order constraint

A single 4-output combine will not work. The slots do not share modes: `Fxi*` is
`{q,e,j}` and `Fgm*` is `{q,e,i}`, because each node must be declared in its
consumer's canonical order (the "LOAD-BEARING RULE" in the benchmark header).
Emitting all four from one node would force a reorder on the **B slot** of the
divergence contraction, which `FUSED_OPERAND_STAGING_ANALYSIS.md` Part 3 prices
at 2-8x — catastrophic.

**Two 2-output combines, one per reference direction,** avoids this entirely:
same modes within each node, identity permutation, zero-copy passthrough
preserved.

---

## Factor 2 — library overhead (1.48x)

`CONTROL` and the library execute the same FLOPs and move the same bytes, yet
the library takes 28.19 ms -> 41.49 ms. **This term is not yet attributed.**

What is known:

- It is **not** the GEMM kernel. A gradient contraction runs at 1290 GF/s
  standalone, and the library's marginal gradient sweep (1.19 ms) is *cheaper*
  than the hand kernel's (`(28.190-17.401)/8 = 1.35 ms`).
- It is **not** the fused-operand passthrough. A divergence GEMM with a
  materialized `View` operand (2.003 ms) is within noise of a gradient GEMM
  (1.984 ms) of the same shape.
- Budgeting the library's 41.49 ms as `16 x 1.19` (gradients) `+ 4 x ~1.2`
  (divergences) `+ 4 x 0.78` (aux/stress) `≈ 30 ms` leaves **~11 ms
  unexplained**, which looks fixed-per-tree rather than proportional to work.

Candidates, untested: per-thread state (the evaluator is 4600 B, ~1150 registers
if resident), barrier count across ~20 serialized stages, and scratch staging
traffic that the hand kernel does not perform.

---

## Cross-backend comparison

> **CORRECTED 2026-08-01, after Task 1 landed.** An earlier revision of this
> section claimed the CPU and GPU residuals agreed almost exactly (1.46x vs
> 1.47x) and concluded the remaining gap was entirely algorithmic and
> backend-independent. **That was wrong, and the error was methodological:** the
> CPU redundancy figure was not measured, it was *assumed* to equal the 2.52x
> executed-FLOP ratio, on the reasoning that Serial has no team parallelism to
> waste so time should track FLOPs. Landing `CONTROL` on Serial (Task 1) made it
> measurable, and it measures **3.21x**, not 2.52x — redundancy costs more than
> its FLOPs alone, because it also carries ~4x the auxiliary global traffic.

| backend | measured gap | redundancy (**measured**) | **residual** |
|---|---|---|---|
| GPU (H100) | 4.26x | 2.89x | **1.48x** |
| CPU (Serial) | 3.69x | 3.21x | **1.11x** |

Both columns are now `CONTROL`-measured on their own backend. Stability: the
GPU residual reproduces at 1.48x across runs; the CPU residual sits in
1.09–1.15x.

**The residuals do not agree.** The honest reading is a product:

```
residual_GPU  =  1.11x  (backend-independent library cost)
              x  1.33x  (appears only on the GPU)
```

So there really is a backend-independent term — it is just much smaller than
claimed, and roughly a third of the GPU residual is something the Serial
backend does not pay. **Team utilisation is back on the table.** It is not
proven to be the cause, but the evidence that previously ruled it out has
evaporated, and Task 2 must now check it rather than assume it away.

Note the redundancy term is *larger* on CPU (3.21x vs 2.89x) — the CPU is
relatively more sensitive to the extra work and less sensitive to whatever the
GPU-specific residual is, which is consistent with the residual being a
latency/occupancy effect rather than an instruction-count one.

---

## Corrections to the benchmark header

`bench_sem_stiffness.cpp`'s comment block predates the per-thread-state work on
this branch (`Team.hpp` E1/E2, static View extents) and is now wrong in three
places:

1. **"~91x slower on H100"** → now **4.26x**.
2. **"~95% of the runtime is neither GEMM work nor recompute"** → now false.
   ~30 of 41.5 ms is GEMM sweeps running at the standalone contraction rate.
   The large unexplained residual it describes has largely been paid off.
3. **"the register kernel's work-item count is 32 while the team is 256 wide"**
   as the leading explanation → *demoted, but NOT ruled out.* This document
   originally superseded it outright on the strength of the CPU/GPU residual
   agreement; that agreement was an artefact of estimating the CPU redundancy
   (see [Cross-backend](#cross-backend-comparison)). With both backends now
   measured, ~1.33x of the GPU residual is GPU-specific, and this is a live
   candidate for it again.

**Status: applied.** The header was rewritten when Task 1 landed and now states
the corrected decomposition, including the non-agreement of the residuals.

---

## Statement of work

Ordered by expected value per unit effort. Each task states what it is worth in
measured terms, so the ordering can be re-derived if a number moves.

### Task 1 — Land the controls and correct the header  ✅ **DONE 2026-08-01**

`CONTROL`, `CONTROL-B` and the four diagnostic rows are now in
`bench_sem_stiffness.cpp`; it prints the full decomposition on both backends and
the header has been rewritten. All rows validate (`max rel diff 0.00e+00`).

**It immediately paid for itself.** Running `CONTROL` on Serial — which was not
possible while it lived in a scratchpad — is what exposed that this document's
"cross-backend confirmation" was wrong: the CPU redundancy had been *assumed*
equal to the 2.52x FLOP ratio, and measures 3.21x. The residuals do not agree
(1.48x GPU vs 1.11x CPU), which reopens a hypothesis this document had closed.

That is exactly the failure mode this task was meant to prevent, and it was
already latent when the document was written. Numbers that live only in a
scratchpad do not get re-run, and conclusions drawn from estimates treated as if
they were measurements survive unchallenged.

---

### Task 2 — Attribute the 1.48x residual  *(spike, before committing to Task 3)*

**Priority: HIGH — cheap, and it can reorder everything after it.**

Profile the fused kernel (`ncu`, via `srun --account=rse --gres=gpu:1`) and
attribute the ~11 ms that the GEMM + aux budget does not explain. Check, in
order: stall reasons on the staging loops, barrier count per team, register
spills against the 4600 B evaluator, and scratch-write traffic that the hand
kernel does not perform.

- **Effort:** 1-2 days, measurement only, no library change.
- **Depends on:** nothing (Task 1 in parallel).
- **Risk:** low. Worst case it confirms the residual is diffuse.
- **Done when:** the ~11 ms is attributed to named mechanisms with a measured
  share each.

**Why this early, ahead of the big feature:** the residual is 1.48x and it
applies *multiplicatively to every future improvement* — it does not shrink when
the redundancy does. If it turns out to be one fixable staging or spill problem,
it is worth more per unit effort than Task 3 and the order should swap. Spending
two days to find out before spending weeks is the cheap de-risking move.

---

### Task 3 — Multi-output combine as a fused operand (phase 2)

**Priority: HIGH — the largest single algorithmic win.**

Widen the `Impl::single_result` gate so a combine node with `NumOut > 1` can be
selected as a fused operand, and express the SEM tree as **two 2-output
combines**, one per reference direction:

- `Cxi{q,e,j}` emits `[Fxi_0, Fxi_1]` from one gradient set and one stress eval.
- `Cgm{q,e,i}` emits `[Fgm_0, Fgm_1]` likewise.
- The root combine already supports multi-output, so both force components come
  out of **one launch**.

This collapses 16 gradient sums -> 8, 4 stress evals -> 2, and 2 launches -> 1.

- **Projected payoff:** the measured 1-component run (20.969 ms) is the closest
  available proxy — one launch, 8 gradient sums, 2 stress evals, 2 divergences.
  Adding back 2 divergence sweeps (~1.2 ms each) and a second force write
  (~0.4 ms) projects **~24 ms, a 1.75x speedup, gap 4.26x -> ~2.45x.**
- **Effort:** the substantial one. Multi-output selection, scratch layout for M
  result tiles, and operand plumbing.
- **Depends on:** Task 2's verdict (see above). Task 1 for measurement.
- **Risk:** medium. The mode-order constraint is a real trap — a single
  4-output node instead of two 2-output nodes would force a B-slot reorder and
  could *lose* 2-8x. Guard this with a test that asserts the identity
  permutation on both fused operands.
- **Done when:** the SEM graph runs in one launch with 8 gradient sums, still
  `PASS` against the host reference, and the benchmark shows <=2.6x.

---

### Task 4 — Fan-out deduplication (CSE) of shared subtrees

**Priority: MEDIUM — real, but smaller and much harder than Task 3.**

After Task 3 the two direction combines still each compute all four gradients,
because the chain rule mixes them — 8 sums where 4 suffice. Deduplicating
requires recognising that `Cxi`'s `d(u0)/dxi` (modes `{q,e,j}`) and `Cgm`'s
(modes `{i,e,q}`) are the same tensor under alpha-renaming: both are
`(x-role, e, z-role)`.

- **Projected payoff:** removes 4 gradient sweeps at 1.19 ms → **~4.8 ms, gap
  ~2.45x -> ~1.95x.**
- **Effort:** large. Needs node identity up to label renaming, a DAG rather than
  a tree evaluator, and shared (non-additive) scratch.
- **Depends on:** Task 3. Doing it first buys only 8 of the 12 redundant sweeps
  and leaves the 2-launch split in place.
- **Risk:** high — this is the architectural change, and scratch is currently
  additive down the whole tree with no recycling.
- **Done when:** the SEM graph runs 4 gradient sums, matching the hand kernel.

**Why after Task 3:** Task 3 subsumes part of this (the cross-kernel half of the
gradient redundancy) at a fraction of the cost, and it changes the shape of what
Task 4 has to deduplicate. Doing Task 4 first means solving the harder problem
against a tree that Task 3 is about to restructure.

---

### Task 5 — Act on Task 2's findings

**Priority: DEFERRED — scope unknown until Task 2 reports.**

Whatever the residual turns out to be. Sized after Task 2. Note that at 1.48x
this is, after Task 3 and Task 4 land, the *largest remaining term*: a library
at ~19 ms against a 9.7 ms baseline is ~1.95x off, and essentially all of that
is this factor.

---

### Not recommended

- ~~**Chasing team occupancy / work-item count.**~~ **Retracted.** This entry
  rested on the CPU and GPU residuals agreeing to within 1%. They do not — that
  agreement came from estimating the CPU redundancy rather than measuring it.
  With `CONTROL` now running on both backends, ~1.33x of the GPU residual is
  GPU-specific, and occupancy is a legitimate candidate. Task 2 checks it.
- **Reducing the recompute by shrinking `TE`.** Scratch is additive and per-team
  cost is flat; this trades one problem for another.
- **A single 4-output combine.** See the mode-order constraint above.

---

## Caveats

**The projections in Tasks 3 and 4 are projections.** They extrapolate from the
marginal-sweep fit (`b = 1.19 ms`) and the measured 1-component proxy
(20.969 ms). They are not measurements of the proposed design, which does not
exist yet. The measured facts are the rows in [Results](#results).

**An earlier estimate in this session put Task 3's endpoint near 1.6x.** That
was wrong — it applied the hand-written `CONTROL-B` time to the library. The
measured 1-component proxy gives ~2.45x.

**GPU numbers are sm_80 PTX-JIT'd onto H100 NVL.** Consistent with the rest of
this repo's benchmarking, but not a native Hopper build.

**`E` matters for the diagnostic rows only.** Runs above use `E = 2500000`
rather than the default 12582912 to keep the wall clock reasonable; both
compared implementations clear the 10 ms timing floor at this size.

**Reproduce with:**

```
module load cudatoolkit/13.2
cmake --build build --target bench_sem_stiffness -j 8
./build/bench_sem_stiffness 2500000 3 1        # GPU
./build_serial/bench_sem_stiffness 16384 3 1   # CPU
```

The decomposition rows are printed by the benchmark itself as of Task 1; the
`bench_diag.cpp` scratchpad harness they came from is no longer needed.
