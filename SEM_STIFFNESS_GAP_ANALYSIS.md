# SEM stiffness: where the library's 4.26x gap actually comes from

**Status:** measured 2026-08-01 on `feat/fused-combine-operands`, H100 NVL (sm_80
build, PTX JIT) and Serial/AVX-512. All numbers below are `min` of 3 reps after
1 warmup, `E = 2500000` (GPU) / `E = 16384` (CPU). Tasks 1 and 2 are **done**;
the controls now live in the benchmark and the residual is attributed.

This document answers one question — *is redundant gradient recompute the
bottleneck?* — and turns the answer into an ordered work plan.

**Short answer: no, not on its own.** Gradient recompute is the largest single
line item (~50% of library runtime, worth 1.62x), but it is one of three
comparable factors that multiply together. Removing it entirely leaves the
library ~2.6x slower than hand-written code.

**Updated 2026-08-03.** `CONTROL-C` now measures the broader change that
*subsumes* gradient recompute — full fan-out deduplication, i.e. computing each
distinct gradient once and merging both components into one launch. That is
worth **2.23x on GPU / 2.19x on CPU**, taking the projected gap to ~1.90x /
~1.64x. It is the largest measured algorithmic item, it does not depend on
Task 3, and the feasibility work re-sized it from XL to L. See
[Task 4](#task-4--fan-out-deduplication-cse-of-shared-subtrees).

**The other two factors, in one line each.** The rest of the redundancy is 2
launches and 4x stress evaluation (1.79x), fixable by multi-output combine
(Task 3). The remaining 1.48x is **address arithmetic** — the library issues
5.65x the integer instructions of an equivalent hand-written kernel for
identical floating-point work (Task 2, now attributed; Task 5 acts on it).

> **Two corrections to earlier revisions of this document, both from actually
> running what it had only estimated.** (1) The CPU and GPU residuals do *not*
> agree — that claim came from substituting a FLOP ratio for a measurement, and
> it had wrongly closed the team-utilisation question. (2) The residual is *not*
> diffuse, and it is *not* barriers, bank conflicts, or memory traffic — all
> three were suspected here and all three are now ruled out by profile.

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
CONTROL-C (DAG work profile)       12.626         2103.6           2863.9
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
CONTROL-C (DAG work profile)       62.726            2.8              3.8
```

> `CONTROL-C` was added 2026-08-03; the other rows in both tables are the
> 2026-08-01 figures and reproduced within noise on the same run
> (GPU: 9.748 / 41.538 / 28.123 / 17.443).

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
| **CONTROL-C** | **1** | **4** | **4** | 4 | 4x |
| library | 2 | 16 | 4 | 4 | 4x |

`CONTROL-B` is the identical kernel with one `constexpr bool` flipped, hoisting
the gradient stage out of the F-node loop so it runs once per launch instead of
twice. That single difference isolates the intra-kernel gradient recompute.

`CONTROL-C` goes one step further and is a separate kernel: the four *distinct*
gradients computed once and shared by every consumer, both force components in
one launch — the work profile a **fan-out-deduplicating (DAG) evaluator** would
produce. It still runs four single-output F nodes, because sharing a subtree
does not merge four combines into one multi-output node, so the 4x stress and
4x aux traffic survive. Its four F passes are deliberately four separate
`parallel_for`s: fusing them would let the compiler share one stress tensor
across the slots, which is the redundancy the row exists to preserve.

All three controls validate against the host reference (`max rel diff
0.00e+00`), so they are doing the real physics, not a stripped-down imitation.

- `baseline -> CONTROL` = the price of the redundancy itself.
- `CONTROL -> library` = everything else the library costs.
- `CONTROL-C -> CONTROL` = **the ceiling on what fan-out dedup can buy.**
- `baseline -> CONTROL-C` = the redundancy dedup *cannot* remove on its own.

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
the library takes 28.19 ms -> 41.49 ms. **Attributed as of Task 2: it is
address arithmetic** — see [below](#attributed-2026-08-01-task-2-ncu-on-h100)
for the profile. The timing-level evidence that framed the search:

- It is **not** the GEMM kernel. A gradient contraction runs at 1290 GF/s
  standalone, and the library's marginal gradient sweep (1.19 ms) is *cheaper*
  than the hand kernel's (`(28.190-17.401)/8 = 1.35 ms`).
- It is **not** the fused-operand passthrough. A divergence GEMM with a
  materialized `View` operand (2.003 ms) is within noise of a gradient GEMM
  (1.984 ms) of the same shape.
- Budgeting the library's 41.49 ms as `16 x 1.19` (gradients) `+ 4 x ~1.2`
  (divergences) `+ 4 x 0.78` (aux/stress) `≈ 30 ms` leaves **~11 ms
  unexplained**, which looks fixed-per-tree rather than proportional to work.

### Attributed 2026-08-01 (Task 2, `ncu` on H100)

Profiled the library's fused force kernel against `CONTROL` — same FLOPs, same
global traffic — at `E = 65536`. Both launch 16384 blocks of 128 threads.
`ncu` needs `srun --account=rse --gres=gpu:1`, and the kernel filter must use
`--kernel-name-base demangled` (the default matches only the Kokkos wrapper).

| | library | CONTROL | ratio |
|---|---|---|---|
| duration | 994.5 us | 639.4 us | **1.56x** |
| **warp instructions** | **278.5 M** | **82.6 M** | **3.37x** |
| integer thread-inst | 4,517 M | 799 M | **5.65x** |
| fp32 thread-inst | 870 M | 487 M | 1.79x |
| **integer : fp32** | **5.19** | **1.64** | — |
| registers/thread | 96 | 32 | 3.0x |
| dynamic shared/block | 25,352 B | 9,296 B | 2.73x |
| achieved occupancy | **30.8%** | **95.8%** | 0.32x |
| L1/TEX throughput | 47.8% | **98.8%** | — |
| SM (issue) throughput | **51.2%** | 26.2% | 1.96x |
| barrier stall ratio | 1.73 | 12.64 | 0.14x |
| shared bank conflicts | 18.7 M | 60.8 M | 0.31x |
| local (spill) sectors | 50.1 M | **0** | ∞ |

**The residual is address arithmetic.** The library issues **5.65x the integer
instructions** — 3.7 *billion* extra thread-instructions — for identical
floating-point work. It spends 5.19 integer instructions per fp32 instruction
where the hand kernel spends 1.64. This is the tiled-layout index math
(`subview_tile`, per-mode stride and offset computation) being re-evaluated at
every element access instead of being strength-reduced into induction variables
across the staging and GEMM loops.

Instruction count is essentially the whole story: the library retires 3.37x the
warp-instructions and takes only 1.56x the time, because its *issue efficiency
is nearly twice CONTROL's* (51.2% vs 26.2% of peak). At equal instruction count
and its own issue rate it would finish in ~295 us, comfortably beating CONTROL.

**Three things it is NOT**, each previously suspected:

- **Not barriers.** The library's barrier stall ratio is 1.73 against CONTROL's
  12.64. The ~20 serialized stages are not what costs.
- **Not bank conflicts.** The library has *one third* CONTROL's shared-memory
  bank conflicts. Consistent with [[swizzle-not-bank-conflict-bound]].
- **Not global memory.** 0.37x CONTROL's global load instructions, L1/TEX at
  47.8% against CONTROL's 98.8%. **CONTROL is the one at a roofline** — it is
  L1-throttled (`mio_throttle` stall ratio 32.1). The library has bandwidth to
  spare and is issue-limited.

**Two real but secondary mechanisms:**

- **Occupancy, 30.8% vs 95.8%.** Registers (96/thread) and shared memory
  (25,352 B/block) *each independently* cap residency at 5 blocks/SM — the two
  limits bind at exactly the same number, so relieving only one moves nothing.
  This caps the issue rate, but at 51.2% it is not the binding constraint.
- **Register spills**, 50.1 M local-memory sectors against CONTROL's **zero** —
  direct confirmation that per-thread state exceeds the register budget. Only
  3.2% of warp instructions, so it is a symptom worth watching rather than the
  cost itself.

**Spilling does not compound with tree depth**, which was the going hypothesis.
Across three library kernels of increasing depth, spill traffic stays a flat
~8% of instructions; what grows is registers and shared memory, hence occupancy:

| kernel | regs | shared/block | occupancy | inst | local ld sectors |
|---|---|---|---|---|---|
| 1 gradient / 1 divergence | 40 | 3,368 B | 70.7% | 43.0 M | 3.7 M |
| 1 F node (4 grad + stress) | 39 | 11,384 B | 72.1% | 118.1 M | 9.2 M |
| full force tree | 96 | 25,352 B | **30.8%** | 278.5 M | 21.2 M |

**Why the CPU residual is only 1.11x** now follows: a CPU has far more integer
throughput relative to its FP throughput, the scalar loops let the compiler
hoist and strength-reduce the index math, and with `team_size == 1` there is no
occupancy term at all. The GPU-specific ~1.33x is the part of the address
arithmetic that the CUDA backend cannot hide.

> **Caveat.** `CONTROL` reproduces the library's *work profile*, not its exact
> instruction stream: it retires ~1.5x fewer FLOPs than the library
> (876.7 M vs 1,315 M by ffma/fadd/fmul weighting), because the library routes
> the divergence through the general GEMM. The headline comparison is unaffected
> — the integer excess (5.65x) is more than three times the fp32 excess (1.79x)
> — but do not read the fp32 row as a defect.

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

### Task 2 — Attribute the 1.48x residual  ✅ **DONE 2026-08-01**

Attributed to **address arithmetic**: the library issues 5.65x the integer
instructions and 3.37x the total warp-instructions of `CONTROL` for identical
floating-point work, at an integer:fp32 ratio of 5.19 against 1.64. Occupancy
(30.8% vs 95.8%, co-limited by 96 registers and 25,352 B shared) and register
spills (50.1 M local sectors vs zero) are real but secondary. Barriers, bank
conflicts and global-memory traffic are all ruled out — the library is *not* at
any roofline, `CONTROL` is. Full evidence in
[Factor 2](#factor-2--library-overhead-148x).

**This reorders the plan.** The residual is a *single named mechanism*, not the
diffuse cost the "worst case" anticipated, and index math is attackable without
touching the graph semantics. Task 5 is therefore promoted and scoped below.

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

> ### ⚠ CORRECTED 2026-08-01 — the payoff below does NOT follow from this task alone.
>
> The projection assumes the two divergence contractions that consume `Cxi[0]`
> and `Cxi[1]` **share one evaluation of `Cxi`**. They do not. `Graph` performs
> no memoization, and each operand's `ScratchAllocator` holds its own
> `inner_eval_t` by value (`ScratchAllocator.hpp:92`), so `Cxi` is instantiated
> once per consumer — each copy recomputing all four of its gradients and its
> own stress evaluation. The merged tree therefore still runs **16 gradient sums
> and 4 stress evals**, exactly today's count. `plans/specfem-kernel-graph.md:187`
> already recorded this ("phase 2 does not fix it"); this document lost it.
>
> **Measured, host-side, no kernel** (the spike this task's entry asked for):
>
> | shape | scratch/team | GPU cap 49,152 | CPU cap 32,768 | blocks/SM |
> |---|---|---|---|---|
> | today, 1 component × 2 launches | 24,312 B | fits | fits | **5** |
> | this task as scoped | **48,624 B** (2.000×) | fits by **528 B** | **OVER** | **2** |
> | with shared subtrees (below) | **27,408 B** (1.13×) | fits | fits | **4** |
>
> Scratch doubles *exactly*, which is the sharing failure made visible. The naive
> merge clears the GPU cap by 1.1%, loses the Serial backend, and drops occupancy
> from 30.8% to ~12% — while doing identical work, for one saved launch. It is a
> regression. **Do not implement the selector on its own.**
>
> **What to build instead**, neither of which needs general memoization:
> - **(B) multi-output contraction** — when a contraction's B operand is a
>   multi-output combine, evaluate the combine **once** and run M GEMMs over it.
>   `alloc_c_` (`Evaluator/Team.hpp:385`) becomes an array exactly as
>   `out_allocs_` (`:975`) already is. This is what collapses 16 gradients → 8.
> - **(C) multi-output operand expansion in a combine's operand pack** — an
>   operand that is multi-output is evaluated once and contributes M values to
>   `fn`'s arguments. Needed because after (B) the root combine would otherwise
>   consume outputs 0 and 1 of the same contraction through two operand slots,
>   reintroducing the duplication one level up. Touches `gather_vals` (`:1104`).
>
> Full analysis: `~/.claude/plans/feasibility-multi-output-combine-operand-2026-08-01.md`.

- **Projected payoff (with B+C, not as scoped):** the measured 1-component run
  (20.969 ms) is the closest available proxy — one launch, 8 gradient sums, 2
  stress evals, 2 divergences. Adding back 2 divergence sweeps (~1.2 ms each)
  and a second force write (~0.4 ms) projects **~24 ms, a 1.75x speedup, gap
  4.26x -> ~2.45x.** As scoped, the payoff is one saved kernel launch, against
  a large occupancy regression.
- **Effort:** the substantial one. Multi-output selection, scratch layout for M
  result tiles, and operand plumbing — plus (B) and (C) above.
- **Depends on:** Task 2's verdict (see above). Task 1 for measurement.
- **Risk:** medium. The mode-order constraint is a real trap — a single
  4-output node instead of two 2-output nodes would force a B-slot reorder and
  could *lose* 2-8x. Guard this with a test that asserts the identity
  permutation on both fused operands.
- **Done when:** the SEM graph runs in one launch with 8 gradient sums, still
  `PASS` against the host reference, and the benchmark shows <=2.6x.

---

### Task 4 — Fan-out deduplication (CSE) of shared subtrees

> ### ✅ DONE 2026-08-03. Built, measured, and its residual attributed by `ncu`.
>
> The SEM graph runs as a 14-node DAG: 4 gradients, 4 stress integrands, 4 divergences,
> 2 weighted sums, **one launch, both components**. `max rel diff 0.00e+00` on both backends.
>
> | | tree | DAG | |
> |---|---|---|---|
> | GPU (E=2.5M) | 41.6 ms | **18.313 ms** | **2.27x faster** |
> | CPU (E=4096) | 37.7 ms | **19.001 ms** | **1.98x faster** |
> | scratch/team | 24,312 B *per component* | **20,688 B for both** | |
> | gap to baseline, GPU | 4.26x | **1.88x** | |
> | gap to baseline, CPU | 3.66x | **1.82x** | |
>
> #### ⚠ RETRACTED: "the residual grew". It did not — that was a team-size artefact.
>
> The first measurement of this task reported a GPU gap of 2.66x and a residual of 2.06x
> against `CONTROL-C`, and concluded that the residual *grows* under restructuring, with
> per-thread state (14 live evaluators) the suspect. **Both halves were wrong**, and `ncu`
> disproved them:
>
> - **Registers are IDENTICAL to the hand-written control: 32 vs 32.** The tree's were 96. Far
>   from having more per-thread state, the DAG has a third of the tree's.
> - The 2.66x was **`Kokkos::AUTO` picking 512 threads for a tile of 256 points**, so every
>   `TeamVectorRange` left half the team idle. Forcing 128 gives **18.31 ms** — a 1.42x swing
>   from one parameter. Team size is now exposed as `DagOutputs::team_size(n)`.
>
> With a sane team size the residual is **1.45x**, against the tree's 1.48x. **The projection
> was right: the residual does survive the restructuring**, essentially unchanged.
>
> #### The attribution (`ncu`, H100, E=65536, DAG at team_size 128 vs `CONTROL-C`)
>
> | | DAG | CONTROL-C | ratio |
> |---|---|---|---|
> | duration | 1,156 us | 688 us | 1.68x |
> | **warp instructions** | **232.8 M** | **110.4 M** | **2.11x** |
> | **integer thread-inst** | **2,605 M** | **960 M** | **2.71x** |
> | fp32 thread-inst | 854 M | 570 M | 1.50x |
> | **integer : fp32** | **3.05** | **1.68** | — |
> | registers/thread | **32** | **32** | **1.00x** |
> | dynamic shared/block | 21,728 B | 12,384 B | 1.75x |
> | achieved occupancy | 43.0% | 97.2% | 0.44x |
> | local (spill) ld sectors | 20.4 M | **0** | ∞ |
> | L1/TEX throughput | 58.5% | **98.6%** | — |
> | SM issue throughput | **45.0%** | 35.1% | 1.28x |
>
> **It is instruction count, exactly as for the tree.** 2.71x the integer thread-instructions
> for 1.50x the floating-point work, at an integer:fp32 ratio of 3.05 against 1.68 — the same
> address-arithmetic mechanism Task 2 named. The DAG *improved* that ratio over the tree's 5.19
> (fewer duplicated subtrees is less duplicated index math) without closing it.
>
> **Two things it is NOT, both of which the tree's profile had implicated:**
>
> - **Not registers.** 32 vs 32, identical to the hand-written control, and a third of the
>   tree's 96. The occupancy co-limit Task 2 found is simply gone.
> - **Not occupancy.** The DAG runs at **43%** and beats its own **98%**-occupancy configuration
>   by 1.42x. Occupancy was never the objective; useful work per resident thread was, and
>   `Kokkos::AUTO` optimises the wrong one.
>
> Spills persist (20.4 M local sectors against zero) at only 32 registers — the compiler has
> capped registers and pushed the remainder to local memory. As with the tree, this is a symptom
> of the instruction-count story rather than a separate mechanism.
>
> **What comes next is unchanged and now better-founded:** the residual is address arithmetic,
> which is [Task 5](#task-5--strength-reduce-the-tiled-layout-index-arithmetic), and it is the
> single remaining named mechanism on the GPU.
>
> Still not collapsed: the four stress integrands each rebuild the stress tensor and re-read all
> seven auxiliary arrays. Only multi-output combine (Task 3) removes that, and a DAG node cannot
> yet be multi-output.

> ### ⬆ PROMOTED 2026-08-03, on two findings. **Now the highest-value algorithmic task, and it no longer depends on Task 3.**

**Priority: HIGH.** Previously "MEDIUM — real, but smaller and much harder than
Task 3". Both halves of that were wrong.

The four gradients are shared across the whole tree: `Cxi`'s `d(u0)/dxi` (modes
`{q,e,j}`) and `Cgm`'s (modes `{i,e,q}`) are the same tensor under
alpha-renaming — both are `(x-role, e, z-role)`, in the same axis order, under
the same tile. Computing them once collapses 16 gradient sums to **4** and
2 launches to 1.

**Finding 1 — the payoff is measured, not projected.** `CONTROL-C` (added
2026-08-03) is exactly this work profile at hand-written efficiency:

| | GPU | CPU |
|---|---|---|
| removable by sharing (`CONTROL-C -> CONTROL`) | **2.23x** | **2.19x** |
| *not* removable by sharing (`baseline -> CONTROL-C`) | 1.30x | 1.50x |
| projected library on a DAG (`CONTROL-C x residual`) | 18.65 ms | 68.6 ms |
| projected gap | **1.90x** (from 4.26x) | **1.64x** (from 3.69x) |

The last two rows are a projection — they assume the measured residual (1.48x
GPU / 1.11x CPU) survives the restructuring unchanged. The first two rows are
measurements, and they are the ones that decide the ordering.

**Finding 2 — the effort is L, not XL, if the sharing is DECLARED.** The XL
sizing came from "needs node identity up to label renaming". It does not, if
the *caller* names the slot each operand reads: the user asserts the identity
and the library never infers it. That turns the hard half of this task into an
API question. Full analysis, including the two invariants it breaks, in
`~/.claude/plans/feasibility-dag-topological-evaluator-2026-08-03.md`.

- **Effort:** L (XL only if sharing must be inferred). New: a slot-reference
  node kind, a relabel node, a flat topologically-ordered driver.
- **Depends on:** nothing. `CONTROL-C` carries *four* stress evals, i.e. it
  measures the DAG **without** Task 3 — the 2.23x is standalone.
- **Risk:** medium. Two invariants break, both nameable: (1) three sites
  (`Evaluator/Team.hpp:1380`, `:1382`, `:1299`) mutate a producer's own scratch
  in place, safe today only because each fused operand has exactly one consumer;
  (2) a shared slot is sound only if every consumer demands it at the same tile
  index, which holds in the SEM graph only because every contracted axis fits in
  one k-tile. Both are enforceable at compile time; neither is checked today.
- **Done when:** the SEM graph runs 4 gradient sums in one launch, still `PASS`
  against the host reference, at scratch ≤ 24,312 B.

**Relationship to Task 3, inverted.** Task 3 (multi-output combine, options
B+C) collapses 16 gradients → 8; Task 4 collapses them → 4, and subsumes the
launch merge. What Task 3 still uniquely buys is the *other* factor: `baseline
-> CONTROL-C` is 1.30x on GPU, and that residue is 4x stress + 4x aux traffic —
exactly what multi-output combine removes and sharing cannot. **The two are
complementary and independent; Task 4 is the larger and should go first.**

Also worth correcting: `plans/specfem-kernel-graph.md:23-43` argues that
relabeling is free *because* `Graph` performs no memoization. Landing this task
invalidates that reasoning and makes a relabel node necessary. Fix that section
in place rather than leaving it to mislead.

---

### Task 5 — Strength-reduce the tiled-layout index arithmetic

**Priority: HIGH — promoted from DEFERRED now that Task 2 has named the cause.**

The library spends 5.19 integer instructions per fp32 instruction; the hand
kernel spends 1.64. The staging and GEMM loops recompute a full
`TiledLayout` index→offset map per element access, where the hand kernel walks
a pointer. Make the inner loops walk induction variables instead: hoist the
per-tile base offset out of the loop and advance by a constant stride, so the
per-element cost is an add rather than a multiply-accumulate chain over
per-mode extents.

- **Projected payoff:** not yet bounded, but the ceiling is large. Instruction
  count is 3.37x CONTROL's while time is only 1.56x, so the library already
  issues at ~2x CONTROL's efficiency; removing integer instructions converts
  almost directly into time. Halving the integer count is worth roughly 1.25x
  end-to-end on the residual.
- **Effort:** medium, and *unlike Tasks 3 and 4 it does not touch graph
  semantics* — it is a codegen change inside the evaluator's loops, verifiable
  with the `cuobjdump -sass` discipline in [[team-perf-verify-sass-diff]].
- **Depends on:** nothing. Can run in parallel with Task 3.
- **Risk:** low-medium. The failure mode is that the index math is already as
  hoisted as the type system permits and the excess is inherent to
  `OrderedSubviewLayout` composition, in which case this becomes a layout
  redesign and should stop.
- **Done when:** the integer:fp32 ratio drops materially from 5.19 and the
  benchmark shows it.

**Secondary, cheaper:** the 96-register / 25,352-byte co-limit pins occupancy at
5 blocks/SM. Both limits bind at exactly 5, so **both** must move to gain
anything — a reason to treat this as one item and not two.

**Sequencing note.** This is now arguably better value than Task 3: it is a
smaller change, it does not risk the mode-order trap, and it applies
multiplicatively to everything Tasks 3 and 4 deliver. Task 3 remains the larger
single win in absolute terms.

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

**Reproduce the Task 2 profile with:**

```
srun --account=rse --gres=gpu:1 -t 25 ncu \
  --kernel-name-base demangled \
  -k "regex:execute_one_output_team|control_recompute_force" -c 6 \
  --metrics smsp__inst_executed.sum,\
sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__sass_thread_inst_executed_op_fp32_pred_on.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum \
  --csv ./build/bench_sem_stiffness 65536 1 0
```

Two gotchas that cost time: `ncu` needs a real allocation (`srun`), even though
plain GPU runs do not on the della-rse login node; and **`--kernel-name-base
demangled` is required** — by default `-k` matches only the outer Kokkos wrapper
`cuda_parallel_launch_local_memory`, so any regex on the inner kernel name
silently profiles nothing and reports "No kernels were profiled".
