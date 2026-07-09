# Where the Time (and Energy) Goes: A Precise Accounting of the NPU Encoder

This is a from-first-principles breakdown of where an AMD XDNA2 NPU spends its
time and energy running a Conformer speech encoder. The motivating question is
uncomfortable: the NPU encoder merely *ties* the CPU on wall-clock time (~662 ms
on the NPU pool versus ~563 ms on the CPU). Is that tie because the NPU is doing
real, irreducible compute, or because it is drowning in avoidable overhead?

**The answer: ~93% of the NPU's on-device time is avoidable overhead**
(context-switching + DMA + launch), **not compute.** A properly-fused backend has
roughly **5x NPU-pool latency headroom and ~3x energy headroom**, which would turn
today's tie into a clear ~1.7-1.9x latency win and ~3x energy win over the CPU.

Measured inputs used throughout:

- NPU pool 409-427 ms, host glue 211-244 ms, 144 dispatches.
- Context switch cost 2.67 ms/switch; a no-switch dispatch = write 0.13 ms + run 0.41 ms + read 0.13 ms = 0.66 ms.
- ~182 GFLOP of useful encoder work; an 18-91 ms fused compute floor.
- ~25 TFLOPS bf16 peak (datasheet).
- Track energy: 671 ms wall, 11.55 J/inference package, ~17 W average package power.

Every figure below is re-derived independently from first principles and
cross-checked against the raw measurements.

---

## 0. The shape sequence (the load-bearing structure)

The default shipping path (`chained_ffn`) has each of the 16 Conformer blocks
issue **9 whole-array matmul dispatches**, with M padded 400 -> 512 (`PAD_M=512`),
in this order, across **4 distinct xclbins** (= 4 distinct hardware contexts, since
a whole-array xclbin occupies all 8 columns):

| # | op       | xclbin (shape, epilogue) | switch vs prev? |
|---|----------|--------------------------|-----------------|
| 1 | ffn1.mm1 | `512x768x3072_silu`      | (block start)   |
| 2 | ffn1.mm2 | `512x3072x768` plain     | switch          |
| 3 | qk       | `512x768x1536_bias`      | switch          |
| 4 | v        | `512x768x768_bias`       | switch          |
| 5 | o        | `512x768x768_bias`       | no (== v)       |
| 6 | pw1      | `512x768x1536_bias`      | switch          |
| 7 | pw2      | `512x768x768_bias`       | switch          |
| 8 | ffn2.mm1 | `512x768x3072_silu`      | switch          |
| 9 | ffn2.mm2 | `512x3072x768` plain     | switch          |

That is **7 intra-block switches** (op #5, o, reuses v's shape, so it is free).
Across the block boundary, #9 (plain) -> #1 (silu) is another switch. Over 16
blocks: **16x7 + 15 boundary = 127 hardware-context switches.**

This independently reproduces the "~128 switches / ~340 ms" figure. Dispatch count
= 16x9 = **144**. The distinct-shape count, not the dispatch count, is what will
turn out to matter.

---

## 1. TIME budget: the ~427 ms NPU pool

| Term | ms | Arithmetic | Avoidable? |
|------|---:|------------|------------|
| **(a) context-reload** | **339.1** | 127 switches x 2.67 ms | **YES** (fewer distinct shapes) |
| **(b) DMA in/out** | 37.4 | 144 x (0.13 write + 0.13 read) | mostly (buffer-object chaining) |
| **(c) run+wait launch overhead** | ~30 | 144 x 0.41 minus genuine compute | YES (launch/sync stall) |
| **(d) genuine on-array compute** | **~29** (14-43) | 144 x ~0.2 ms | **NO, irreducible** |
| **sum** | **435.9** | vs 427 measured (within 2%, noise) | |

So of the 427 ms NPU pool, **~398 ms (93%) is avoidable overhead** and **only
~29 ms (~7%) is real compute.** Within the overhead, context-reload alone is
**~80%** (339/427). The dispatch count and the FLOP count are *not* the
bottleneck; the **distinct-shape count** is. This is the single most important
finding.

Note that the run+wait term (0.41 ms/dispatch) bundles launch latency together
with on-array compute. Splitting it by the compute floor (Section 2) leaves
~30 ms of pure fixed launch/sync stall, which is why (c) and (d) are separated
above.

---

## 2. Roofline floor and utilization (re-derived independently)

Useful model work = **182 GFLOP** (independently recomputed: 173.6 GFLOP of block
matmuls at M=400, plus 7.9 GFLOP attention = 181.5). The array actually computes
**222 GFLOP** because M is padded 400 -> 512.

| Reference rate | floor for 182 GFLOP |
|----------------|--------------------:|
| bf16 **peak** 25 TFLOPS | **7.3 ms** |
| **measured genuine compute** (0.1-0.3 ms/dispatch) | **14-43 ms (~29 ms mid)** |

The 29 ms genuine-compute figure implies an *active* compute throughput of
**~5-15 TFLOPS** (222 padded GFLOP / 0.029 s = 7.7 TFLOPS, about 31% of peak)
**during the compute window** - i.e. when the array is actually computing, it runs
respectably. The problem is that it computes for only ~29 ms out of 427.

**Utilization, two independent derivations:**

- Over the **full 427 ms pool**: 182 GFLOP / 0.427 s = **426 GFLOP/s = 1.7% of the 25 TFLOPS peak.**
- An older "~0.6%" figure is a *different* number: a single-matmul high-water mark
  from an unfused, single-core, host-orchestrated path ("47 GFLOP/s effective",
  1 of 8 columns). That path was ~98% DMA overhead. The current fused path is
  better (1.7%), still abysmal. **Both say the same thing: under 2% array
  utilization; the array is essentially idle.**

One caution on roofline numbers: a "128 GFLOP/s at 4 columns" rate that appears in
older notes is a *dispatch-bound* rate from the old host-orchestrated path, not the
current kernel's compute rate. At 128 GFLOP/s a 512^3 matmul would take 2.1 ms,
which contradicts the measured 0.41 ms run+wait. The whole-array kernel's real
compute rate is the 5-15 TFLOPS band above.

**Floor versus current: 7-29 ms of compute hiding inside a 427 ms pool = 15-58x of pure overhead.**

---

## 3. Energy budget: the 11.55 J/inference

Package RAPL, idle ~6.5 W, average ~17 W, wall 671 ms. (Caveat: the NPU-in-package
draw is *assumed*, not isolated - see Section 6.)

| Component | J | basis |
|-----------|--:|-------|
| **idle-baseline floor** | **4.36** | 6.5 W x 671 ms (burned doing nothing) |
| active (work-attributable) | 7.19 | 11.55 minus 4.36 |

Allocating the full 11.55 J by **wall-time fraction** (the package is powered the
whole window):

| Window slice | ms | J (time-share) | % window |
|--------------|---:|---------------:|---------:|
| context-reload | 339 | **5.84** | 50.5% |
| host glue (CPU) | 244 | 4.20 | 36.4% |
| DMA in/out | 37 | 0.64 | 5.5% |
| run+wait launch | 30 | 0.52 | 4.5% |
| **genuine NPU compute** | **29** | **0.50** | **4.3%** |

**The energy story mirrors the time story:** only **~0.5 J (~4%)** of the 11.55 J
is spent in genuine NPU compute. The largest single energy sink is
**context-reload (~5.8 J)** - the array is fully powered (all 8 columns resident)
while it reprograms itself instead of computing. Another ~4.2 J is the host CPU
doing layernorm, RoPE, GLU, softmax, and attention glue. If compute is only ~4% of
the time, the energy goes into keeping a powered array idle through 339 ms of
reprogramming plus 244 ms of host work - almost all of it avoidable on the NPU
side. Even the 4.36 J idle floor is "avoidable" only by finishing faster (shrinking
the window).

---

## 4. Headroom: current versus floor versus CPU

The compute floor is ~7 ms (peak) / 14-43 ms (measured) - call it **~29 ms /
~0.5 J** of irreducible NPU work. The current NPU pool is **427 ms / ~7 J** of
NPU-side energy (5.8 switch + 0.6 DMA + 0.5 launch + 0.5 compute, excluding host).

A realistic well-utilized backend (kills the switch wall via a small-shape kernel
set / same-shape grouping; keeps buffer-object chaining; minimal DMA) lands the
**NPU pool at ~81 ms** (29 compute + 37 DMA + ~15 residual launch, switches ~0):

| | NPU pool | end-to-end wall | J/inf |
|---|---:|---:|---:|
| **current NPU impl** | 427 ms | 671 ms | 11.55 J |
| **realistic fused NPU** | **~81 ms (5.3x)** | **~270-325 ms** | **~4 J (2.9x)** |
| compute-only floor | ~29 ms (15x) | host-bound | - |
| **CPU (onnxruntime)** | n/a | 563 ms | ~12 J |

- **NPU-pool latency headroom: ~5x** (427 -> ~81 ms); up to ~15x against the bare compute floor.
- **End-to-end:** once the NPU pool drops below ~244 ms, the **host glue (244 ms)
  becomes the new wall.** The realistic fused wall of ~270-325 ms (host-bound) is
  **~1.7-1.9x faster than the 563 ms CPU**, versus today's near-tie. (Further host
  parallelism could push lower, but that is a separate lever.)
- **Energy headroom: ~3x** (11.55 -> ~4 J), making the NPU **~3x more
  energy-efficient than the CPU** (~12 J) instead of roughly equal.

**The crisp answer:** today's NPU "tie with CPU" is almost entirely avoidable
overhead - ~93% of the NPU's on-device time and ~96% of its compute energy. The
genuine compute is only ~29 ms / ~0.5 J. A backend that eliminated the
context-switch wall would be ~5x faster on the device and ~3x more energy-efficient,
decisively beating the CPU on both latency and energy.

---

## 5. The catch: why this headroom is hard to bank

The headroom is real, but the specific lever to capture it is expensive, and the
cheap versions have already been proven not to work:

- **The switch wall is the prize, but the obvious fixes fail.** Collapsing to 2
  contexts is *measured net-negative*: forcing N-padding to unify shapes 4x'd the
  MACs, and the f32 readback added +70 ms of host time, netting **+25 ms slower**.
  A batched/async submit approach is **dead (-0.1 ms)** - host-side batching cannot
  recover an *on-device* reprogram cost. So "switches -> 0" is not free.

- **The ~81 ms target requires a substantial build.** On-chip GEMM -> GEMM fusion
  via the naive all-to-all DMA mapping is blocked by the **2-input-DMA-channel
  wall** (it fails at n_cols=2). For the softmax-free FFN/GEMM K-reduce, that wall
  is *escaped* by the cascade accumulator (core to core, not a DMA channel) -
  built silicon-correct in a phase-0 gate that cleared correctness but did not yet
  win on perf. Fused attention still needs more than 2 DMA inputs per tile, since
  the cascade cannot span a softmax. The merge-heads-on-MemTile plus hard-M-tiling
  dataflow remains a dedicated per-shape build, not a refactor.

- **The realistic intermediate** is a deliberately co-designed **small static-shape
  kernel set** that maximizes same-shape adjacency, capturing *most* of the switch
  saving without the N-padding penalty - but it is a model-shaped design effort, not
  a quick win.

- **So the accurate verdict is two-layered.** (1) *Analytically*, the NPU is ~93%
  overhead and has ~5x device headroom; the "tie" is not a compute limit. (2)
  *Pragmatically*, capturing it needs a shape-set redesign or the multi-week fusion
  build; the easy levers are exhausted, which is why ~648-671 ms is the current
  honest stopping point. The headroom is a **design-debt number** (what a
  from-scratch kernel set could hit), not a quick optimization left on the table.

---

## 6. Flagged assumptions

- **Genuine compute = 29 ms** uses a 0.1-0.3 ms/dispatch midpoint; the range is
  14-43 ms. The fused-NPU and headroom multipliers scale with this (5.3x at 29 ms;
  4x-10x across the range).
- **Energy time-share allocation** assumes package power is roughly flat across the
  window (~17 W average). In reality the array likely draws more during compute and
  reprogram than during host-only glue, so the true compute/switch energy shares
  could differ. But compute being a tiny *time* slice bounds its energy share
  regardless. Directionally robust, not exact per-slice.
- **NPU-in-package RAPL** is assumed, not isolated. Every J/inf carries that caveat.
- **25 TFLOPS bf16 peak** is the larger-bin figure; the measured box is a smaller
  bin. The utilization percentage would look better if the real peak is lower, but
  the numbers are dispatch/switch-bound far below any plausible peak, so this is not
  load-bearing.
- **"Realistic fused NPU ~81 ms"** is a model (compute + DMA + residual launch,
  switch -> 0), not a measured result; Section 5 explains why it is not trivially
  reachable.
- **CPU 563 ms / ~12 J**: the 12 J is an estimate (the CPU path was not separately
  RAPL-measured here). Treat the ~3x energy ratio as order-of-magnitude.
