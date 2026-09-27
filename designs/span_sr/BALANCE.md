# Balancing the gate stages onto the idle cores

Phase 1b of `2026-09-27-npu-any-game-realtime.md`, lever 4. The 6 gate stages (`b{1..6}c3`, kind
`gate` in `net_layout.STAGES`) each run `conv3x3_i8_gate`: a 3x3 conv, Cin=Cout=48, plus the SPAB
epilogue. Isolated rate ~415-433 cyc/px (`probe_span_core_rates.py`); in-chain (`b3c3`) 406.2 cyc/px
compute, unaffected by any fifo-depth lever tried so far (TRACE_RESULTS.md). 22 of 32 cores used,
10 idle.

## Option (a): output-channel split

Two cores each compute 24 of 48 output channels. Each still needs the FULL 3-row input window
(Cin=48 unchanged -- a 3x3 conv over 48 input channels producing any output-channel subset still
reads all 48 in channels) and half the weights.

The kernel (`conv3x3_u8.cc::conv3x3_core`) loops output-channel blocks (`ob = 0..OCB-1`, OCB =
COUT/8) reading its own weight/bias/mult slice per block; `COUT` is a compile-time macro. So a
24-channel half is just `-DCONV3X3_COUT=24` plus a params blob sliced by ob-range -- a host-side
`numpy` slice of `golden.pack_params`'s three concatenated sections (weights, then bias, then
mult, each ordered by `ob`), not a runtime cost. The gate epilogue's second input `x` (the block's
own input, carried in-band after the conv window) is channel-aligned with the conv output, so it
slices the same way: each half reads its own 24-channel sub-range of the SAME in-band `x` block --
no extra DMA, just a pointer offset in the shim.

**Rejoin.** The row layout is channel-block-planar (`[OCB][W+2*PAD][8]`), so 24 low channels + 24
high channels concatenated in memory IS a valid 48-channel row -- exactly the join mechanism
`net_design.py` already uses for `conv_cat`'s 4-source join (`ObjectFifo.prod().join(offsets, ...)`),
just with 2 sources and ordinary main-path depth, not a skip-sized ring. No core spent on the join;
it is 2 S2MM writes + 1 MM2S read on a MemTile buffer sized `lay[stage].out_bytes * depth` -- the
SAME total MemTile bytes as the unsplit single-producer fifo (2 sources of half the width each,
same depth).

**Budget count per split stage** (against this part's per-core/per-MemTile DMA limits, 2 in/2 out
per core and 6 S2MM/6 MM2S per MemTile):
- cores: +1 (2 instead of 1)
- per-core DMA: each half core uses 2 in (broadcast activation window, its own weight slice) + 1
  out (into the join) -- within the 2 in / 2 out per-core limit, unchanged from the unsplit gate
  core's own usage
- input broadcast: the previous stage's output fifo gains a second consumer (`.cons()` called
  twice) -- the SAME mechanism already used for every skip source (e.g. `conv_1`'s output is
  consumed by both `b1c1` and the `conv_cat` join today)
- MemTile: +1 small 2-source join (2 S2MM + 1 MM2S, well under the 6/6 per-MemTile channel limit);
  weight-group entity count rises by 1 (23 names instead of 22 across ~4 groups of <=6) -- no
  channel-count change per group
- L1: unaffected -- each half core's window/weight/stack footprint is the same shape as the
  unsplit gate core's, just COUT=24 instead of 48 (smaller weight blob, smaller output buffer)

Expected stage rate: roughly half of 406-433 cyc/px, i.e. ~200-220 cyc/px, since MAC count halves
and the epilogue (LUT fetch, gate arithmetic) is per-16-lane-group and scales with COUT too.
Skip ring: `b1c3` is a `CAT_SOURCES` entry; splitting it needs `out["b1c3"]` to ALSO carry the
`PROD_DEPTH=2` skip-broadcast producer once the join reassembles it, which this implementation
does not build (out of scope for `b2c3`; flagged below for the extend-to-all-6 step).

## Option (b): row interleave

Two cores take alternate output rows, each keeping its own 3x3 sliding window, full Cin/Cout
weights (no weight split). Doctrine framing checked against the actual kernel: computing row `y`
needs input rows `y-1, y, y+1`. If core A owns even output rows and core B owns odd, row 1 (needed
by A for rows 0 and 2) belongs to B's stripe and vice versa -- so EITHER both cores read every
input row anyway (no reduction in per-row DMA/compute pressure, since the windowed kernel still
processes one input row per iteration; a "half the input reads" hope does not materialize), or the
design gets a cross-core row hand-off at every boundary (halo of 1 row between two independently
scheduled cores) which is materially more plumbing than option (a)'s single small join, for a
scheme that does not reduce any per-row MAC count (Cin=Cout=48 unchanged, weights unsplit) -- it
only trades "one core stalled on chain latency" for "two cores each stalled on chain latency",
which does not address the compute imbalance TRACE_RESULTS.md measured (`b3c3` computing
406.2 cyc/px in-chain regardless of stall). Rejected: no weight/compute reduction for a rolling-
window stage, and the halo hand-off is strictly more MemTile/DMA machinery than option (a)'s join.

**Correction, found while implementing.** `conv3x3_core`'s output loop processes channel blocks
in PAIRS (`for (ob = 0; ob < OCB; ob += 2)`, `static_assert(COUT % 16 == 0)`), so 48 has no even
24/24 split point -- the only two valid split points are 16/32 and 32/16. Implemented as 32/16
(`net_design.GATE_SPLIT_LO = 32`): the heavier half does 2/3 of the original MACs, the lighter
1/3, rather than an even half each.

## Decision

**Option (a), output-channel split.** 6 stages x 1 extra core each = 28 of 32 cores if extended to
all 6 gates, leaving 4 idle. Implemented for `b2c3` only first, per task scope; if it measures a
win, extend to the other 5 (`b1c3` needs the skip-broadcast wiring noted above; `b3c3..b6c3` are
plain main-path like `b2c3`).

**On the remaining 4 cores, once the gate split lands:** the SiLU stages (kind `silu`/`silu_x`/
`silu16`/`silu_i16`, ~240-320 cyc/px measured) become the next-heaviest class once gate drops to
~200-220 -- above the ~107 cyc/px average this pipeline needs for 30 fps at 540p (per the plan's
lever-4 framing). Splitting the 1-2 heaviest SiLU stages (block 1's `silu16`/`silu_i16`, the widest
at 3h) the same way -- output-channel split, same join mechanism -- would use exactly those 4
cores. Not attempted here: out of this task's scope, and the gate split's whole-net effect must be
measured first (below) since TRACE_RESULTS.md's own dominant finding is that the skip-ring
latency, not raw per-core compute, sets the pace -- a compute win at `b2c3` can be masked by that
same structural stall.

## Implementation

`net_layout.split_gate_params(p, cin, cout, lo_channels=32)` slices a gate stage's
`net_params()` dict into the 32/16-channel halves (weights/bias/mult sections each slice at the
same ob boundary; ga/gb/gs1/gc/gs2/tables/pre/shift are per-net scalars, shared). `net_design.build`
takes `split_gate={"b2c3"}`: the two half-cores read the SAME full 3-row window (broadcast --
`out[prev].cons()` called twice, the same mechanism every skip source already uses) and their
24-vs-24... 32-vs-16-channel outputs rejoin into one `out["b2c3"]` fifo via `ObjectFifo.join()` (2
sources, ordinary main-path depth) -- so every downstream consumer of `out["b2c3"]` is unchanged.
`net_design.weights_blob(NP, names, split_gate=...)` produces the matching wire layout (a split
stage contributes its lo blob then its hi blob).

## Gate: bit-exact on device

`verify_span_split_gate.py`, W=32 H=64, `split_gate={"b2c3"}`, full net (23 cores) upto `up`:
**32768/32768 exact, PASS** -- same gate `verify_span_net.py` runs for the unsplit net.

## A/B: whole-net and up-to-b2c3 chain rate, split vs unsplit

Same-process, alternated per height, fitted slope (`probe_span_split_gate.py`,
`probe_span_split_gate_upto.py`), W=32, power mode not pinned (device shared, same caveat as every
prior measurement in TRACE_RESULTS.md):

| probe | unsplit | split b2c3 |
|---|---|---|
| whole net (up to `up`, 23 cores) | 675 cyc/px | 757 cyc/px |
| chain up to `b2c3` (7 cores, no conv_cat/skip ring) | 542 cyc/px | 640 cyc/px |

**No measured win; both fits show the split slightly SLOWER, but non-monotonic per-height medians
in both probes** (e.g. up-to-b2c3: split beats unsplit at h=128 and h=192, loses at h=64 and h=256)
-- the same box-contention signature TRACE_RESULTS.md flags on every wall-clock fit in this file.
Even the up-to-`b2c3` sub-chain has NO conv_cat and NO skip ring (`b2c3` is not a `CAT_SOURCES`
entry), yet still shows no win -- consistent with TRACE_RESULTS.md's other finding that ordinary
main-path hops run near-lockstep at `MAIN_DEPTH=4` (1 free slot), so a stage's own compute cost
barely reaches the measured period regardless of the skip ring. The gate split is CORRECT (bit-
exact) and reduces `b2c3`'s own compute per BALANCE.md's arithmetic, but on this device, at this
`MAIN_DEPTH`, that reduction is not visible in wall-clock cyc/px -- it is masked the same way the
prior MAIN_DEPTH+1 and b1c1-depth+1 experiments were masked, one level up the stack. Extending to
the other 5 gate stages is not warranted until the per-row lockstep / stall-attribution work in
`feat/span-stall` lands; re-measure this A/B after that, not before.
