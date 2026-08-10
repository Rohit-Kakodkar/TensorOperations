# Staging a non-canonical fused contraction operand: three approaches measured

**Question.** Commit `1162ba5` ("Allow non-canonical fused contraction operands") stages a permuted fused operand by reordering its own C scratch **in place**, via a sequence of parallel axis-pair transpositions. Two alternatives were never evaluated:

1. **In place** (what we do today) — `Impl::reorder_scratch_in_place`, T transposition passes over the operand's own buffer. No extra scratch, but T passes and T barriers, and it only works when every transposed axis pair has equal extents.
2. **Copy** — allocate a separate canonical staging buffer and do one reorder+copy pass into it (this is Specialization 8's existing `else` branch, the one input operands already take). One pass instead of T, no shape restriction — but it costs one extra operand tile of team scratch, which forces smaller tiles, which means more recompute.
3. **Native strides** — don't reorder at all. Teach the GEMM to read the operand in its non-canonical layout by changing the index→offset map.

This document reports what each actually costs on an H100, and — more importantly — *why*, in enough detail that you can predict the answer for a shape we didn't measure.

---

## TL;DR

**Adopt a hybrid: approach 3 for the A slot, keep approach 1 (in place) for B.** Do not adopt approach 2 anywhere.

| | reorder cost | extra scratch | lifts equal-extent rule? | verdict |
|---|---|---|---|---|
| **1. In place** (today) | **10–15%** of end-to-end throughput | none | no | keep for the **B** slot |
| **2. Copy to staging buffer** | 0–55% cheaper than #1, shape-dependent | +1 operand tile | yes | **reject** — nets ≈0%, with a −20% tail |
| **3. Native strides** | **zero** | none | yes | **adopt for the A slot**; catastrophic for B |

The single most important finding is an asymmetry that has nothing to do with permutations per se: **in this kernel, operand A is broadcast-read across a warp while operand B's read address strides with the lane index.** That makes A's memory layout nearly irrelevant to performance and B's layout critical. Approach 3 is therefore free in the A slot and a 2–8× disaster in the B slot.

A secondary finding, worth chasing independently: the canonical A layout is itself suboptimal. Reading A "transposed" (contracted-major) is **1.44–1.83× faster** than the canonical layout whenever a warp spans more than one register row-block. See [Bonus finding](#bonus-finding-the-canonical-a-layout-is-leaving-performance-on-the-table).

---

## Table of contents

- [Part 0 — The CUDA background you need](#part-0--the-cuda-background-you-need)
- [Part 1 — What the reorder actually costs (benchmark 1)](#part-1--what-the-reorder-actually-costs-benchmark-1)
- [Part 2 — End-to-end, and the cost of approach 2's extra buffer (benchmark 2)](#part-2--end-to-end-and-the-cost-of-approach-2s-extra-buffer-benchmark-2)
- [Part 3 — Approach 3: reading operands at native strides (benchmark 3)](#part-3--approach-3-reading-operands-at-native-strides-benchmark-3)
- [Part 4 — The ledger](#part-4--the-ledger)
- [Part 5 — Recommendation and implementation notes](#part-5--recommendation-and-implementation-notes)
- [Bonus finding](#bonus-finding-the-canonical-a-layout-is-leaving-performance-on-the-table)
- [Reproducing](#reproducing)
- [Caveats](#caveats)

---

## Part 0 — The CUDA background you need

Everything in this document follows from three mechanisms. If you already know shared-memory banking and occupancy, skip to Part 1.

### 0.1 Shared memory is 32 banks wide, and that is the whole story

GPU shared memory (what Kokkos calls team scratch) is physically split into **32 banks**, each 4 bytes wide. For a `float` array, element `e` lives in bank `e mod 32`.

A **warp** is 32 threads that execute in lockstep. When a warp issues one shared memory load, the hardware services it in one cycle *if and only if* the 32 addresses don't collide:

- **All 32 lanes read the same address** → **broadcast**, one cycle. Free.
- **All 32 lanes hit distinct banks** → one cycle. Free.
- **N lanes hit different addresses in the same bank** → **N-way conflict**, the access is serialized into N cycles.

The number that matters is the **stride between consecutive lanes**. If lane `l` reads element `base + l*σ`, then the lanes touch `32/gcd(σ,32)` distinct banks, so the conflict degree is:

```
conflict_degree = gcd(σ, 32)          (σ = per-lane stride, in floats)
```

| σ | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|----|----|-----|-----|
| **conflict** | 1× | 2× | 4× | 8× | 16× | **32×** | **32×** | **32×** | **32×** |

The killer case is **σ a multiple of 32**: every lane lands in the same bank and the access is fully serialized, 32× slower. And note that *any* multiple of 32 is equally bad — which is why nice round tile extents like 16 and 32 are exactly the ones that hurt.

One important refinement: the formula assumes the 32 lanes hold 32 **distinct** addresses. Lanes that share an address are merged by the broadcast path for free, so the real cost is

```
conflict_degree = min( gcd(σ, 32), number of distinct addresses in the warp )
```

That refinement is doing real work in Part 3: operand A's address depends only on `bi`, of which a warp sees just 1, 2 or 4 values, so even when σ is a multiple of 32 the conflict is only 1-, 2- or 4-way. Operand B's address depends on `bj`, which is distinct on all 32 lanes — so B pays the full 32×.

### 0.2 How this kernel maps warp lanes to work

This is the part that makes A and B behave differently, so it's worth being precise.

The GEMM in `Evaluator/Team.hpp` (Specialization 4, `accumulate_block`) is:

```cpp
Kokkos::parallel_for(
    Kokkos::TeamThreadRange(team, (SA / MT) * (SB / NR)), [=](int t) {
      const int bi = t / (SB / NR);   // C tile-block ROW
      const int bj = t % (SB / NR);   // C tile-block COLUMN
      ...
        br[n]  = b_reg(k0, k, bj, n);   // B: address depends on bj
        a_i    = a_reg(bi, i, k0, k);   // A: address depends on bi
```

with `MT=4, NT=2, NR=2` on GPU.

**Measured fact:** for every configuration in this study, Kokkos picks `vector_length == 1` (probed directly; team sizes came out 128/256/512/1024). With `blockDim.x == 1`, a `TeamThreadRange` iteration index `t` maps to `threadIdx.y`, so **a warp is 32 consecutive values of `t`.**

Let `G = SB/NR` be the number of distinct `bj` values. Within one warp, `bj` cycles through `G` values, and `bi` is **constant** if `G ≥ 32`, or takes `32/G` distinct values if `G < 32`.

This gives the fundamental asymmetry:

| operand | address depends on | across a warp |
|---|---|---|
| **A** | `bi` | constant, or a *few* values → mostly **broadcast** |
| **B** | `bj` | **all 32 lanes differ** → stride-sensitive |

**A is broadcast-read. B is lane-strided.** That one sentence explains every result in Part 3.

### 0.3 Occupancy: `min(thread limit, shared-memory limit)`

An SM (streaming multiprocessor) runs several thread blocks ("teams") concurrently. More resident blocks means more warps to hide memory latency with. On an H100, per SM, the limits are **2048 threads**, **233,472 bytes** of shared memory, and 32 blocks (never binding here). So:

```
blocks_per_SM = min( 2048 / team_size ,  233472 / scratch_bytes_per_team )
```

**This is why approach 2's extra buffer is sometimes free and sometimes catastrophic.** If the *thread* limit is already the binding constraint, extra shared memory costs literally nothing. If you're at the shared-memory limit, one more buffer can drop you a whole residency step — and residency is an integer, so the penalty is a **cliff, not a gradient**. Part 2 shows this predicting all eight measured rows.

---

## Part 1 — What the reorder actually costs (benchmark 1)

`bench_reorder.cpp` calls the library's own primitives exactly as Specialization 8 calls them (real team scratch, real barriers), and times the two mechanisms in isolation. Each team runs R reorders chained so nothing can be optimized away; verified flat across R = 8/16/32.

Times are wall-clock microseconds for **one** reorder pass across 8192 teams.

```
tile / perm             elems  swaps   inplace us  cpy(src)  cpy(dst)  ip/bestcpy
---------------------- ------ ------  ----------- --------- ---------  ----------
R2 32x32   [1,0]         1024      1         36.9      37.1      37.1      1.00x
R2 64x64   [1,0]         4096      1        151.3     147.3     147.4      1.03x
R3 8^3     [1,0,2]        512      1          5.3       4.7       5.2      1.11x
R3 8^3     [2,1,0]        512      1          5.6       5.0       5.2      1.12x
R3 8^3     [1,2,0]        512      2         11.2       4.8       5.2      2.32x
R3 8^3     [2,0,1]        512      2         10.8       4.9       5.2      2.19x
R3 16^3    [1,0,2]       4096      1         38.7      37.9      42.1      1.02x
R3 16^3    [2,0,1]       4096      2         79.8      74.3      74.5      1.07x   <-- copy gains nothing
R3 12^3    [2,0,1]       1728      2         40.3      20.4      20.9      1.97x
R3 20^3    [2,0,1]       8000      2        177.6      98.5      98.6      1.80x
R3 32x8x8  [1,0,2]       2048      1      ILLEGAL      18.9      20.8        n/a
R3 16x16x8 [2,0,1]       2048      2      ILLEGAL      19.7      20.8        n/a
R4 8^4     [1,0,2,3]     4096      1         40.7      50.1      50.1      0.81x   <-- copy LOSES
R4 8^4     [1,0,3,2]     4096      2         92.0      49.8      49.8      1.85x
R4 8^4     [3,2,1,0]     4096      2         97.0     146.3     146.4      0.66x   <-- copy loses BADLY
R4 8^4     [1,2,3,0]     4096      3        154.6      50.7      49.5      3.12x
R4 8^4     [3,0,1,2]     4096      3        137.0      50.5      49.7      2.76x
R4 4^4     [1,2,3,0]      256      3         10.2       3.2       3.1      3.27x
R4 16x8^3  [1,2,3,0]     8192      3      ILLEGAL     152.0     155.0        n/a
```

`cpy(src)` vs `cpy(dst)` is a tuning knob I tested and can report as a dead end: driving `team_for_each_coord` from the contiguous destination instead of the strided source makes **no difference** (within noise). Not a lever.

### Why: it's pass count × per-pass conflict degree

The naive model — "in place costs T passes, copy costs 1, so copy wins by T" — is right only when the passes are equally fast. They frequently aren't. The per-pass cost is set by §0.1, and you can compute it by hand.

**Setup.** The scratch tile is `LayoutRight` with extents `E` and strides `S` (`S[R-1] == 1`, the last axis is fastest). Both mechanisms walk the tile with a `TeamVectorRange` over the flat index, so — again with `vector_length == 1` — a warp covers **32 consecutive flat indices**, i.e. consecutive values of the **last (fastest) axis**.

**In place.** `swap_axes(a,b)` reads `data[s]` (stride 1, always clean) and `data[d]` where `d` is `s` with axes `a` and `b` exchanged. So a transposition `(a,b)` is **conflict-free if neither `a` nor `b` is the fastest axis**. If it does touch the fastest axis, the lane stride becomes `S[min(a,b)]`, and the conflict degree is `gcd(S[min(a,b)], 32)`.

**Copy.** One pass, reading the native buffer at the offset the canonical fastest axis maps to. Lane stride is `S_native[perm[R-1]]`, conflict degree `gcd(that, 32)`.

Worked examples, which reproduce the table:

**`R2 32x32 [1,0]`** — the swap *is* on the fastest axis. Stride `S[0] = 32` → `gcd(32,32) = 32` → **32-way conflict**. Compare `R3 16^3 [1,0,2]`: 4× more elements, same 38 µs, because there the swap is on axes 0 and 1 and the fastest axis 2 is untouched → conflict-free. A 2-D transpose is the worst case for this mechanism.

**`R3 16^3 [2,0,1]` (in place 79.8 vs copy 74.3 — copy gains nothing).** `transposition_plan` yields pairs `(0,1),(1,2)`, applied back-to-front: `swap(1,2)` then `swap(0,1)`. `swap(1,2)` touches the fastest axis → stride `S[1] = 16` → `gcd(16,32) = 16` → **16-way conflict**. `swap(0,1)` is clean. So in place = 1 bad pass + 1 clean pass. Meanwhile the copy reads at native offset `c1*256 + c2*16 + c0`; the canonical fastest axis is `c2`, stride 16 → **also 16-way conflicted**. One 16-way pass ≈ one 16-way pass plus one clean pass. Hence the 1.07× — the copy's "one pass instead of two" advantage is entirely eaten by the fact that its one pass is the bad one.

**`R4 8^4 [3,2,1,0]` (copy loses 0.66×).** Plan is `(0,3),(1,2)` applied as `swap(1,2)` then `swap(0,3)`. `swap(1,2)` doesn't touch axis 3 → clean. `swap(0,3)` does → stride `S[0] = 512` → `gcd(512,32) = 32` → 32-way. So in place = clean + 32-way. The copy reads at `c3*512 + c2*64 + c1*8 + c0`; fastest canonical axis `c3` has stride 512 → **32-way conflicted on every element**. In place wins because it gets one of its two passes for free.

**`R3 8^3 [2,0,1]` (copy wins 2.19×) and `R4 4^4 [1,2,3,0]` (3.27×).** At small tiles the ratio simply tracks the swap count (2.19 ≈ 2, 3.27 ≈ 3). These tiles are small enough that per-pass fixed cost (barrier, loop setup) dominates over bank behaviour, so the first-order "T passes vs 1" model holds.

**Takeaway.** Copy is *not* reliably cheaper. It wins by ≈T on small tiles, wins modestly on large non-power-of-two tiles, ties on `16^3`, and **loses** when the permutation moves the fastest axis by a stride that is a multiple of 32 — which is common, because tile extents are usually powers of two.

---

## Part 2 — End-to-end, and the cost of approach 2's extra buffer (benchmark 2)

`bench_fused3.cpp` runs a real rank-3 fused chain with a genuinely non-canonical fused operand, validated against a host reference:

```
T1{i,j,m} = sum_k A{i,j,k} B{k,m}            (inner contraction, canonical output)
E {m,l}   = sum_{i,j} T1{i,j,m} D{i,j,l}     (parent, contracts i and j)
```

The parent's canonical A order is `freeA ++ contracted = [m,i,j]`, but T1 hands its scratch back as `[i,j,m]`. So **permA = `[2,0,1]`**, a 3-cycle → **two** transpositions, and the in-place rule forces T1's output tile to be a cube.

Three builds of the identical benchmark were compared. **base** is as shipped. **noreorder** has `reorder_scratch_in_place` compiled out — results are wrong by design, but this isolates the reorder's cost exactly, with every other shape, buffer and barrier held fixed. **a2** has the per-team scratch request inflated by one canonical A tile and nothing else changed, isolating approach 2's *footprint* cost.

### 2.1 The in-place reorder costs 10–15% end-to-end

| TC/TK/TL | base G/s | noreorder G/s | **reorder cost** |
|---|---|---|---|
| 8/8/16 | 1307.8 | 1460.5 | 10.5% |
| 8/8/32 | 1870.7 | 2086.6 | 10.3% |
| 8/16/32 | 1541.4 | 1715.1 | 10.1% |
| 12/8/16 | 1543.7 | 1744.8 | 11.5% |
| 16/8/16 | 2083.0 | 2447.4 | **14.9%** |
| 16/8/32 | 2855.0 | 3216.7 | 11.2% |
| 16/16/32 | 2475.5 | 2775.9 | 10.8% |
| 20/8/32 | 2668.5 | 3113.3 | **14.3%** |

This is the number to beat. It is not a rounding error, and it scales with the transposition count — a rank-4 3-cycle would cost roughly half again as much.

### 2.2 Approach 2's extra buffer: free five times out of eight, then a cliff

| TC/TK/TL | base G/s | a2 G/s | **penalty** |
|---|---|---|---|
| 8/8/16 | 1307.8 | 1313.9 | +0.5% |
| 8/8/32 | 1870.7 | 1859.9 | −0.6% |
| 8/16/32 | 1541.4 | 1535.1 | −0.4% |
| 12/8/16 | 1543.7 | 1559.3 | +1.0% |
| 16/8/16 | 2083.0 | 1622.6 | **−22.1%** |
| 16/8/32 | 2855.0 | 2839.7 | −0.5% |
| 16/16/32 | 2475.5 | 2451.3 | −1.0% |
| 20/8/32 | 2668.5 | 1983.9 | **−25.7%** |

Note that **58 KB → 74 KB is free while 41 KB → 57 KB costs 22%.** The penalty is not monotone in bytes. It is entirely explained by §0.3.

### Why: the occupancy formula predicts all eight rows

Using measured team sizes and measured scratch sizes, with `blocks = min(2048/team_size, 233472/scratch_bytes)`:

| TC/TK/TL | team | scratch B | thread lim | shmem lim | **blocks** | +A2 | shmem lim' | **blocks'** | predicted | measured |
|---|---|---|---|---|---|---|---|---|---|---|
| 8/8/16 | 128 | 9,000 | 16 | 25 | **16** | +2,048 | 21 | **16** | 0% | +0.5% |
| 8/8/32 | 256 | 13,608 | 8 | 17 | **8** | +2,048 | 14 | **8** | 0% | −0.6% |
| 8/16/32 | 256 | 15,912 | 8 | 14 | **8** | +2,048 | 12 | **8** | 0% | −0.4% |
| 12/8/16 | 256 | 21,928 | 8 | 10 | **8** | +6,912 | 8 | **8** | 0% | +1.0% |
| 16/8/16 | 512 | 42,536 | 4 | 5 | **4** | +16,384 | 3 | **3** | **−25%** | **−22.1%** |
| 16/8/32 | 1024 | 59,944 | 2 | 3 | **2** | +16,384 | 3 | **2** | 0% | −0.5% |
| 16/16/32 | 1024 | 68,648 | 2 | 3 | **2** | +16,384 | 2 | **2** | 0% | −1.0% |
| 20/8/32 | 1024 | 99,240 | 2 | 2 | **2** | +32,000 | 1 | **1** | −50% | **−25.7%** |

The model gets the **sign right on every row**, and the magnitude nearly exactly on the 16/8/16 case. The 20/8/32 row loses less than predicted — with 1024 threads per block there is enough instruction-level parallelism inside a single block to recover about half the loss, so occupancy isn't the only thing keeping the SM busy.

Read the "blocks" column carefully. In five of the eight rows the **thread limit is already binding** (`2048/team_size` ≤ `233472/bytes`), so extra shared memory is literally free — you were leaving that shared memory unused anyway. In the other three the shared-memory limit binds, and whether you pay depends on whether the extra buffer crosses an integer boundary.

**Practical consequence:** you cannot reason about approach 2's cost from a percentage-of-scratch figure. "+25% scratch" costs 0% in one config and 22% in another. It has to be measured per configuration — which is itself a strong argument against adopting it, because it makes performance discontinuous in the tile size the user picks.

---

## Part 3 — Approach 3: reading operands at native strides (benchmark 3)

`bench_gemm_stride.cpp` replicates the library's register-blocked GEMM verbatim (same `MT=4, NT=2, NR=2`, same `TeamThreadRange` blocking) and varies **only** the map from a logical `(row, col)` to a shared-memory offset:

| variant | A offset | B offset | represents |
|---|---|---|---|
| `canon` | `i*SK + k` | `k*SB + n` | today's staged canonical tile |
| `trans` | `k*SA + i` | `n*SK + k` | native buffer with free/contracted groups swapped — the common permuted-fused case, e.g. permA `[2,0,1]` |
| `inter` | free and contracted modes **interleaved** | ditto | the general case: **not a matrix under any 2-D reinterpretation**, needs a full multi-dim decode per access |

Aggregate GFLOP/s, 4096 teams, shared-memory-resident (no global traffic):

```
  SA    SK   SB   canon G/s   A-trans        B-trans        A-inter        B-inter
---- ----- ----  ---------  --------      --------       --------       --------
  32    32   32    10410.4    1.54x          0.27x          1.00x          0.44x
  64    32   64    12869.3    1.00x          0.13x          1.00x          0.24x
  16   256   32    11390.1    1.44x          0.25x          1.00x          0.25x
   8    64   32    10658.7    1.51x          0.27x          0.99x          0.44x
 256     8   16     4443.1    1.83x          0.81x          1.83x          1.00x
  64    64   32    11048.6    1.60x          0.26x          1.00x          0.43x
```

Two results here, and they point in opposite directions.

### 3.1 The multi-dim decode is FREE (`A-inter` = 1.00×)

This is the result that makes approach 3 viable at all. The `inter` variants replace a single multiply-add with a divide, a modulo, and a two-term dot product with compile-time strides — and it costs **nothing**. There are two reasons.

First, **the divisors are compile-time constants**, so nvcc lowers `k / C1` and `k % C1` into a multiply-and-shift. No integer division instruction is ever issued. Integer division on GPU is genuinely expensive, tens of cycles, which is exactly why `Impl::ceil_div<Divisor>` exists in this codebase — but it only bites when the divisor is a *runtime* value.

Second, **the kernel is memory-latency bound, not ALU bound.** The extra integer ops issue in slots the SM was spending waiting on shared-memory loads anyway.

So the standard objection to approach 3 — "you'll pay for address arithmetic in the innermost loop" — is empirically false here. Approach 3's cost is *entirely* about memory access patterns.

### 3.2 A is broadcast, B is lane-strided — hence the asymmetry

Now apply §0.1 and §0.2. Let `G = SB/NR` be the number of distinct `bj` per warp.

**A, canonical:** `a[(bi*MT+i)*SK + kk]`. Depends on the lane only through `bi`, which steps by `MT*SK`.

| shape | G | bi values/warp | lane stride σ | `gcd(σ,32)` | conflict (capped by #addrs) | predicted | **measured** |
|---|---|---|---|---|---|---|---|
| SA=64,SK=32,**SB=64** | 32 | 1 | — | — | broadcast, 1-way | **1.00×** | **1.00×** |
| SA=32,SK=32,**SB=32** | 16 | 2 | `4*32 = 128` | 32 | **2-way** | up to 2× | **1.54×** |
| SA=16,SK=256,**SB=32** | 16 | 2 | `4*256 = 1024` | 32 | **2-way** | up to 2× | **1.44×** |
| SA=256,SK=8,**SB=16** | 8 | 4 | `4*8 = 32` | 32 | **4-way** | up to 4× | **1.83×** |

**A, transposed:** `a[kk*SA + (bi*MT+i)]`. Lane stride is `MT = 4` regardless of shape → `gcd(4,32) = 4`, only 4 addresses in play, always distinct banks → **conflict-free**.

That is the whole explanation. Canonical A puts the `bi` stride at `MT*SK = 4*SK`, which is a multiple of 32 for **any `SK` that is a multiple of 8** → every distinct `bi` lands in the same bank. Transposed A puts it at `MT = 4` → clean. And when `G ≥ 32` the warp only ever sees one `bi`, so it's a broadcast either way and the ratio is exactly 1.00 — **which is precisely what the `SB=64` row shows.** Three distinct `SB` values, monotone in `G`, prediction confirmed at both ends.

**B, canonical:** `b[kk*SB + bj*NR + n]`. Lane stride `NR = 2` → `gcd(2,32) = 2`, 16 distinct banks → mild, fine.

**B, transposed:** `b[(bj*NR+n)*SK + kk]`. Lane stride `NR*SK`:

| shape | lane stride | `gcd(σ,32)` | predicted | **measured** |
|---|---|---|---|---|
| SB=32, SK=32 | `2*32 = 64` | **32** → 16 lanes, same bank | 16-way | **0.27×** (3.7× slower) |
| SB=64, SK=32 | `2*32 = 64` | **32** → 32 lanes, same bank | **32-way** | **0.13×** (7.7× slower) |
| SB=16, SK=8 | `2*8 = 16` | 16 → 8 lanes | 8-way | **0.81×** |

The measured slowdowns are smaller than the raw conflict degree because B loads are only a fraction of the kernel's total work — but the *ordering* is exactly as predicted, including that the 32-lane case (`SB=64`) is worse than the 16-lane case.

**B, interleaved:** lane stride `NR*C1`. For `SB=32, SK=32, C0=4` → `C1=8` → stride 16 → `gcd(16,32)=16` → 8-way conflict, less bad than B-trans's 32. Measured **0.44×**, duly between B-trans (0.27×) and canonical.

**A, interleaved:** lane stride `MT*C1`. For `SA=32,SK=32,C0=4` → `C1=8` → stride 32 → same bank → 2-way, *identical to canonical A*. Measured **1.00×** relative to canon. And for `SA=256,SK=8,C0=4` → `C1=2` → stride 8 → clean, *identical to transposed A*. Measured **1.83×**, exactly matching A-trans's 1.83×.

The model predicts the direction and relative magnitude of all twelve numbers.

### 3.3 What this means for approach 3

**A slot: adopt it.** Any permutation, including the fully interleaved case that isn't a matrix under any 2-D view, costs **0%** — and often *gains*, because it accidentally fixes a pre-existing bank problem (see the Bonus finding). No extra scratch. No equal-extent restriction.

**B slot: never.** 2–8× slower. The B operand's free dimension must stay lane-contiguous, which means it must be physically reordered into canonical order. Keep approach 1 there.

This asymmetry is structural, not incidental: it follows from the kernel parallelizing lanes over `bj`. It would only change if the register blocking itself changed.

---

## Part 4 — The ledger

Combining the three measurements. Approach 2's net is `reorder_cost × (1 − copy/inplace) + footprint_penalty`, using the matching cube tile from Part 1.

| TC/TK/TL | #1 in place (baseline) | #2 copy | #3 native strides (A slot) |
|---|---|---|---|
| 8/8/16 | — | +5.7% | **+10.5%** |
| 8/8/32 | — | +5.6% | **+10.3%** |
| 8/16/32 | — | +5.5% | **+10.1%** |
| 12/8/16 | — | +6.7% | **+11.5%** |
| 16/8/16 | — | **−21%** | **+14.9%** |
| 16/8/32 | — | +0.3% | **+11.2%** |
| 16/16/32 | — | −0.3% | **+10.8%** |
| 20/8/32 | — | **−19%** | **+14.3%** |

Approach 3's column is the reorder cost recovered in full, since it does no reorder at all. It is a floor, not a ceiling — Part 3 suggests the GEMM itself gets faster too for shapes with `SB/NR < 32`.

Approach 2 is a coin flip: +5–7% on small tiles, ≈0% at medium, −20% at the cliffs. Median gain ≈ 0, with a bad tail and discontinuous behaviour. Not worth the extra buffer, the extra code path, or the tile-size sensitivity it introduces.

---

## Part 5 — Recommendation and implementation notes

### Do this

**Hybrid: approach 3 for the A slot, keep approach 1 (in place) for B.** The benefits:

- Recovers the full **10–15%** on A-slot permuted fused operands — the case `FusedPermutedOperandATeam` covers.
- **Zero extra scratch**, so no occupancy cliff, and no new sensitivity to tile size.
- **Lifts `Impl::operand_stageable_v` entirely for A.** The shape-changing permutations currently rejected (the `SkewedT1` 32×64 case asserted against in `test_graph.cpp`) simply work, because nothing is being swapped inside a fixed buffer any more.
- B keeps today's behaviour exactly, including its equal-extent rule.

### Scope of the change

The register kernel's *structure* does not change. What changes is how A's element address is computed:

```cpp
// today — requires a contiguous canonical tile, so the operand must be
// physically reordered first:
auto a     = Impl::as_matrix<FreeA>(staged_a.storage_, a_tile_);  // [SA,SK]
auto a_reg = tile_view(a, RegA{});
...  a_reg(bi, i, k0, k)

// approach 3 — a compile-time-strided index map over the operand's NATIVE
// buffer; no reshape, no reorder:
...  a_addr(bi * MT + i, k0 * NT + k)
```

`as_matrix`/`reshape` cannot express this — a permuted view has strides that don't collapse to 2-D in general — which is why it has to be an index map rather than a view transformation. But the change is contained: **B and C paths are untouched** and keep `as_matrix` + `tile_view`; the **`MT/NT/NR` register blocking is untouched**; the strides are compile-time (`StaticTile`), so §3.1 says the arithmetic is free; and `ScratchAllocator`'s ContractionTag specialization already hands back the inner evaluator's C scratch zero-copy, so approach 3 just stops reordering it afterwards.

The risk to guard against is letting the A path regress the *canonical* case. The canonical map (`i*SK + k`) must stay a special case that generates identical code, ideally verified with the SASS-diff technique already used in this repo.

### Don't do this

- **Approach 2 anywhere.** Nets ≈0% with a −20% tail.
- **Native strides in the B slot.** 2–8× slower.
- **Retuning the copy branch's traversal direction.** Measured; no effect.

---

## Bonus finding: the canonical A layout is leaving performance on the table

This is independent of permutations and probably worth more than the whole question above.

`A-trans` beating `canon` by **1.44–1.83×** is not a property of permuted operands. It says the layout we deliberately stage A into is the *worse* of the two options. Canonical `a[(bi*MT+i)*SK + kk]` gives a `bi` lane stride of `MT*SK = 4*SK`, a multiple of 32 whenever `SK` is a multiple of 8 → **every distinct `bi` in one bank**. Transposed `a[kk*SA + (bi*MT+i)]` gives a `bi` lane stride of `MT = 4` → **always conflict-free**.

The effect appears exactly when a warp spans more than one `bi` (i.e. `SB/NR < 32`) and vanishes at `SB=64` where it doesn't — confirmed at three `SB` values.

If A were staged **contracted-major** for *all* operands, permuted or not, there is a potential double-digit GEMM win on every shape with `SB/NR < 32`; and a "transpose-like" permA would then already be in the preferred order, so it would need **no reorder at all**, solving the original problem for free as a side effect.

Two caveats before anyone acts on this. First, these are shared-memory-resident micro-benchmark numbers; the end-to-end share depends on how much of the runtime is the GEMM inner loop. Second, note the interaction with the existing `NR = 2` tuning, recorded in `accumulate_block`'s comment as +20% matmul on A100 — both are bank-conflict fixes on different operands, so they should compose, but that's an assumption, not a measurement. Worth a dedicated experiment.

---

## Reproducing

Three standalone benchmarks were written for this study. They live in the session scratchpad rather than the repo:

```
bench_reorder.cpp       Part 1 — the two staging mechanisms in isolation,
                        using the library's own primitives
bench_fused3.cpp        Part 2 — end-to-end rank-3 fused chain with a
                        non-canonical fused operand, host-validated
bench_gemm_stride.cpp   Part 3 — the register GEMM with varying operand
                        index->offset maps
probe.cpp               team sizes, vector length, real scratch sizes
```

They compile against the existing build tree's Kokkos with the same flags CMake generates for `bench_multilevel_contraction` (see `build.sh` in the scratchpad). Run `module load cudatoolkit/13.2` first.

Part 2's `noreorder` and `a2` builds were produced against a **patched copy** of `include/` in the scratchpad — the repository itself was never modified. The two patches were, in `Evaluator/Team.hpp`: wrap Specialization 8's `reorder_scratch_in_place` call in `#ifndef TENSOR_BENCH_NO_REORDER`; and in Specialization 4's `scratch_size_per_team`, under `#ifdef TENSOR_BENCH_APPROACH2`, add one canonical operand tile for each fused operand with a non-identity perm.

`bench_fused3.cpp` is worth promoting into `benchmarks/` — there is currently no benchmark covering the fused *permuted*-operand path, and it is what caught the 22% occupancy cliff that pure reasoning would have missed.

## Caveats

**GPU only.** Consistent with the team-policy GPU priority. The CPU (Serial / AVX-512) path has a completely different cost model — no shared-memory banks, no occupancy — so none of Part 0 transfers. The reorder cost there is plain cache traffic and the answer may well differ.

**fp32, no tensor cores.** The register kernel is scalar FMA.

**sm_80 build JIT'd onto H100 NVL.** Bank behaviour (§0.1, Part 3) is architecture-independent; the *occupancy* numbers in Part 2 are Hopper's (2048 threads, 233,472 B shared per SM) and would shift on A100 (2048 threads, 167,936 B).

**The `canonical` control column in `bench_fused3` is a rough comparison only.** Its inner GEMM is the mirror image of the permuted variant's (M and N swapped), so it is not a matched control. Every reorder-cost number in this document comes from the `noreorder` build, which *is* matched, not from that column.

**`noreorder` produces incorrect results by design.** It exists purely to price the reorder. Do not read its correctness check as meaningful.
