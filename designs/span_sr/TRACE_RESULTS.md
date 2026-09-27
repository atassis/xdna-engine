# SPAN net 3.5x attribution: per-core trace results

Lane A of `2026-09-27-npu-any-game-realtime.md`. Whole-net rate (untraced, `probe_span_net_rate.py`):
**1528 cyc/px** (W=32, fitted slope, alternated). Slowest isolated single core (`probe_span_core_rates.py`,
gate class, 48->48, W=32): **~415-433 cyc/px**. This note attributes the ~3.5x gap.

Method: `aie_kernels/_test/trace_span_net.py`, built on `net_design.py`'s new `trace_stages=`/
`trace_config=` (event0()/event1() bracket + `Program.enable_trace`, same mechanism as
`bricklib._build_streamed_traced`/`trace_conv_dispatch.py`). One stage traced per device dispatch
(22-core net, W=32, H=128 rows, 5 of 8 core trace events: INSTR_EVENT_0/1 + LOCK_STALL +
MEMORY_STALL + STREAM_STALL). Clock assumed 1.8 GHz (canonical measured), power mode NOT pinned
for these runs (device shared under heavy contention -- see caveat below).

Raw summaries: `/mnt/data/xdna/traces/span/trace_span_net_<stage>_summary.json` (mirrored off
tmpfs; `/tmp` does not survive and is not shared across sessions).

## Results so far

| stage | role | compute cyc/px | gap cyc/px | compute+gap | LOCK_STALL % of span | verdict |
|---|---|---|---|---|---|---|
| conv_1 | skip source (depth 22), first stage | 40.42 (3.14%) | 1256.15 (96.77%) | 1296.6 | 96.68% | victim -- waits on the chain, not the cause |
| conv_cat | join emission (4-source iterate_bds) | 58.23 (4.08%) | 1246.58 (86.59%) | 1304.8 | 95.80% | victim -- also waits, own compute+join overhead is small |
| b3c3 (gate, main path, not a skip source) | 406.20 (29.52%) | 919.72 (66.32%) | 1325.9 | 70.30% | STILL a victim, despite compute landing right at its isolated rate (~415-433 cyc/px) |
| up (last stage) | 26.68 (1.87%) | 1273.05 (88.4%) | 1299.7 | 97.99% | victim |

All four measured stages sit at compute+gap ~1300-1330 cyc/px -- close to each other regardless of
each stage's own compute cost, and below the untraced net's 1528 cyc/px (tracing/bracketing overhead
plus H=128 vs. the net-rate probe's fitted range; lead/trail excluded from gap so ~4-10% of span is
unaccounted per stage, see `by_event`/summarize() in `trace_span_net.py`). All four are dominated by
LOCK_STALL (70-98%), never STREAM_STALL or MEMORY_STALL -- objectFifo acquire-wait, not a DMA/
bandwidth stall.

**This is the decisive result.** `b3c3` is the heaviest single core in the network (measured
isolated at ~415-433 cyc/px, `probe_span_core_rates.py`) and in-chain its own compute (406.20 cyc/px)
lands right at that isolated rate -- so embedding it in the pipeline costs it nothing directly. But it
is STILL LOCK_STALLed 70.3% of the time, only somewhat less than the near-idle skip/join stages
(~96-98%). No traced stage shows LOW LOCK_STALL at ~1528 cyc/px compute; the heaviest core does not
run near-saturated while everyone else stalls on it. Per the coordinator's framing, this rules out
"one slow core sets the pace" and points at a STRUCTURAL cause common to every core.

Read initially as a possible shallow-objectFifo bug (conv_1's own skip-broadcast producer depth is
`PROD_DEPTH=2` while the join's consumer-side depth for that same skip is `skip_depths['conv_1']=22`,
a big asymmetry). Checked against `mlir-aie/python/iron/dataflow/objectfifo.py` (`prod()`/`cons()`/
`join()`): `join()` builds an independent `ObjectFifo` per skip source with its own producer/consumer
depth pair -- prod=2 (core-local double buffer, DMA-fed) vs cons=22 (MemTile ring) is asymmetric BY
DESIGN, not obviously the collapse bug seen once before in M1 (mismatched declared depths on the SAME
endpoint). So a fast, cheap stage (conv_1, conv_cat) sitting almost entirely in LOCK_STALL is exactly
what a CORRECT pipeline looks like when the real bottleneck is downstream -- this does not by itself
localize the 3.5x.

## Leading hypothesis and the one cheap experiment

`windowed()`'s main loop (`net_design.py`) does `fi.acquire(3)` (a 3-row sliding window: y-1,y,y+1)
then `fi.release(1)` per iteration, against an input objectFIFO of `depth[prev]` = `MAIN_DEPTH` = 4
for every ordinary main-path hop. That leaves exactly **1 free slot** for the upstream producer to
run ahead into. With only 1 row of slack at each of ~20 main-path hops, a stall anywhere propagates
backward almost immediately, forcing near-lockstep row-by-row synchronization across the whole
22-stage chain instead of letting the pipeline overlap once filled -- which is consistent with EVERY
traced stage (cheap or heavy) paying the same ~1300 cyc/px per-row tax dominated by LOCK_STALL.

**Experiment: raise MAIN_DEPTH. Blocked by L1, network-wide, not just on one hop.**
`net_design.build()` now takes `main_depth=` (default None -> the module constant, 4). Compile-only
sweep at W=32, H=64, full 22-core net:

| W | main_depth | result |
|---|---|---|
| 32 | 4 (baseline) | OK |
| 32 | 5 | FAIL -- `b6c2_out_buff_2` needs 4608 B, tile (4,3) already at ~62.4 KB/64 KB at depth=4 |
| 32 | 6, 8, 10, 12 | FAIL, same tile/class |
| 16 | 5 | OK |
| 16 | 6, 8, 10 | FAIL -- same `b6c2_out_buff_*` class, now ~3072 B/buffer |

The binding tile is NOT special to block 6: `STAGES` in `net_layout.py` gives blocks 2-6 the
IDENTICAL kind sequence (`silu_x`, `silu`, `gate`) -- so every `c2` stage (b2c2..b6c2) carries the
same 2h-wide input/output buffers and is equally tight (block 6 just happened to be reported first).
Block 1's `silu16`/`silu_i16` kinds are wider still (3h) and already carry an explicit `DEPTH`
override (b1c1=3, i.e. LESS than MAIN_DEPTH) precisely because they have no headroom at all. So this
is an L1 wall across most of the network, not a one-tile fix -- raising MAIN_DEPTH uniformly by even
1 is infeasible at the production strip width (32), and buys only +1 (4->5) at half that width (16).

**Ran the +1 test anyway, at W=16 (only depth increment L1 admits), alternated same-session against
the W=16 baseline:** `aie_kernels/_test/probe_span_net_depth.py` (7-trial median, fitted slope across
heights 64-320, the approved in-design repeat method, NOT the retired two-size marginal method).

| main_depth | slope | fitted cyc/px @ 1.8 GHz (W=16) |
|---|---|---|
| 4 (baseline) | 24.0 us/row | 2696 |
| 5 | 18.9 us/row | 2125 |

**21% lower cyc/px from a single +1 depth bump.** But the raw per-height medians are non-monotonic
(depth=5, h=256 < h=192 -- device was shared with other lanes throughout; the depth=4 fit's negative
intercept, -0.311 ms, is the same jitter), so this wall-clock number alone is not fully trustworthy --
exactly why a same-dispatch trace corroboration matters more here than the fit.

**Corroboration: re-traced `b3c3` at depth=4 vs. 5, W=16, H=128, same process (no inter-dispatch
jitter in the comparison).** `aie_kernels/_test/trace_span_net_depth.py`:

| main_depth | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|
| 4 | 415.85 (28.73%) | 976.69 (66.94%) | 70.92% | 1392.5 |
| 5 | 415.86 (28.73%) | 976.87 (66.97%) | 70.91% | 1392.7 |

**b3c3's own numbers are identical within noise -- the +1 depth bump on its immediate neighbours did
NOTHING to its stall fraction.** So the 21% whole-net improvement (if real) is not coming from
relieving b3c3's own hop -- it comes from somewhere else in the chain that also got +1 slack. b3c3
sits between two skip producers in the dependency graph (fed via the main path from b1c3's descendants,
feeding toward b6c1's ancestors), and neither `PROD_DEPTH` (2, the skip-broadcast producer depth) nor
the join's own `f_cat.cons(depth=2)` for the `cat` stage were touched by this experiment -- both are
strong candidates for a depth ceiling that `main_depth` cannot reach. **Revised picture: MAIN_DEPTH is
a real, small lever (deepening ordinary main-path hops helps SOME stages), but it is not sufficient by
itself, and does not touch the stage (b3c3) with the least idle time.** The next-best hypothesis --
already the coordinator's fallback -- is the skip-ring/join depth (`PROD_DEPTH=2` and/or the join's
fixed consumer depth), which is untouched by this experiment and remains the more likely structural
cause given b3c3's exact zero-movement result.

## b1c1/b1c2 zero-slack hypothesis: REFUTED as an alternation-sum, small effect on b3c3

Hypothesis: `DEPTH["b1c1"]=3` against `b1c2`'s `windowed()` 3-row acquire leaves b1c1 and b1c2
strictly alternating with zero producer slack, so their compute times ADD and set the whole
chain's pace. Predicted: each of b1c1/b1c2 shows substantial compute, and the two sum to
~1300 cyc/px.

Traced individually (one stage per dispatch, `trace_span_net.py --stages b1c1` /
`--stages b1c2`, W=32, H=128; `run.sh` does not forward CLI args to the wrapped script, so these
were dispatched by replicating its env setup directly with `--stages` passed through):

| stage | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|
| b1c1 | 319.87 (24.04%) | 1015.80 (75.76%) | 75.78% | 1335.7 |
| b1c2 | 322.79 (24.18%) | 1012.77 (75.28%) | 75.67% | 1335.6 |

**REFUTED as stated.** b1c1's compute (319.87) + b1c2's compute (322.79) = 642.66, far below the
~1300 cyc/px period -- they do not sum. Instead each lands INDIVIDUALLY at compute+gap ~1335.6-
1335.7, the same signature every other traced stage shows (conv_1 1296.6, conv_cat 1304.8, b3c3
1325.9, up 1299.7): a common ~1300-1335 cyc/px pace regardless of a stage's own compute, with
LOCK_STALL dominant. b1c1/b1c2's LOCK_STALL (75.7-75.8%) sits between the near-idle skip/join
stages (~96-98%) and the heaviest core b3c3 (70.3%) in roughly the order their own compute would
predict -- consistent with the STRUCTURAL per-row-lockstep picture already in force above, not
with a b1c1<->b1c2-specific alternation defect.

**The fix, tested anyway (task instruction): raise the b1c1->b1c2 fifo depth 3->4.**
Compile-only sweep (`design.compile()`, no device) confirms the L1 wall predicted by the module
docstring: at W=32, `depths={"b1c1": 4}` FAILS identically to the MAIN_DEPTH+1 case
(`b6c2_out_buff_3` needs 4608 B, same tile class -- raising b1c1's OWN fifo depth costs L1 on
tiles far downstream because `net_design.py`'s digest/placement is network-wide, not local to
b1c1's tile). At W=16, `depths={"b1c1": d}` compiles for d=4,5 and fails at d=6 -- same halved
headroom pattern as MAIN_DEPTH.

Ran `probe_span_net_depth_b1c1.py` (W=16, depth 3 vs 4, alternated per height, 7-trial median,
fitted slope, same method as the MAIN_DEPTH sweep):

| b1c1 depth | slope | fitted cyc/px @ 1.8 GHz (W=16) |
|---|---|---|
| 3 (baseline) | 25.6 us/row | 2878 |
| 4 | 12.7 us/row | 1426 |

A big apparent swing, but the per-height medians are badly non-monotonic (depth=4 at h=128 is
SLOWER than depth=3 at h=192; depth=3's fit has a negative intercept, -1.072 ms) -- the same box-
contention pattern already flagged for the MAIN_DEPTH wall-clock fit, not trustworthy alone.

**Corroboration: re-traced `b3c3` at b1c1-depth=3 vs. 4, W=16, H=128, same process**
(`trace_span_net_depth_b1c1.py`):

| b1c1 depth | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|
| 3 | 415.85 (28.73%) | 976.68 (66.94%) | 70.92% | 1392.5 |
| 4 | 416.37 (29.42%) | 955.61 (66.99%) | 70.22% | 1372.0 |

**Same outcome as the MAIN_DEPTH+1 experiment: b3c3 barely moves** (compute+gap 1392.5 -> 1372.0,
~1.5%; LOCK_STALL 70.92% -> 70.22%). The within-dispatch trace does not corroborate the wall-clock
fit's ~2x swing -- that swing is contention noise, not a real effect of this depth bump. So the
b1c1->b1c2 hop is not b3c3's ceiling either, matching the MAIN_DEPTH result: a small, real, LOCAL
lever (b1c1/b1c2's own compute+gap likely drops the same ~20-50 cyc/px a depth bump gave b3c3's
neighbours) that does not touch the whole-chain pace.

**PROD_DEPTH/join, from existing traces (task item 4): not implicated beyond the general pattern.**
conv_1 (a skip source, `PROD_DEPTH=2` producer) and conv_cat (the join) both show the SAME
signature as every other stage -- near-zero own compute, LOCK_STALL 95.8-96.7%, compute+gap
~1300 cyc/px -- indistinguishable from an ordinary main-path hop's stall profile. Nothing in the
traces singles out `PROD_DEPTH=2` or the join's fixed consumer depth as a DIFFERENT kind of
bottleneck from the structural per-row lockstep already identified; no targeted PROD_DEPTH/join-
depth bump was run (out of scope here), so this remains open rather than ruled in or out.

## Skip-ring latency-throttle hypothesis: CONFIRMED, dose-response

Hypothesis: `conv_1` is `rows_ahead("conv_1")=20` rows ahead of `conv_cat` on the main path, and
the join's `f_cat` ring holds only `rows_ahead + SKIP_SLACK` rows (`net_layout.skip_depth`).
`conv_1` cannot write row r+depth until `conv_cat` has consumed row r, and `conv_cat` consumes
row r only after the ~20-stage main path has carried row r+20 all the way through -- a temporal
latency L. The ring admits only ~SKIP_SLACK rows per L, so period T >= L/SKIP_SLACK(+ some other
buffering), throttling every stage identically regardless of its own compute -- matching every
prior trace in this file.

Added `net_design.build(skip_slack=...)`, overriding `NL.skip_depth`'s slack term without
touching `NL.SKIP_SLACK` itself (module stays byte-identical when the arg is omitted).

**Step 1 -- compile-only MemTile budget, W=32.** The join's shared pool = `lay["conv_cat"].in_bytes`
(`CAT_HALVES(5) x half(32)=2304` = **11520 B/row-slot**) x `depth(conv_1) = 20 + SKIP_SLACK`.
512 KB / 11520 = 45.5 rows. Swept `design.compile()` (no device) at slack in
{2,4,8,16,20,22,23,24,25,26,30}: **OK through slack=25** (depth=45, 506.2/512 KiB), **FAILS at
slack=26** (depth=46, 517.5 KiB) -- exact arithmetic match, and unlike MAIN_DEPTH/b1c1-depth this
lever is NOT L1-bound, so it is testable at full production width (no W=16 fallback needed).

**Step 2 -- device dose-response, same session, W=32, SKIP_SLACK in {2,4,8,16,25}**
(`aie_kernels/_test/probe_span_net_skipslack.py`, 4 heights x 5 trials, fitted slope, same
alternated-per-height method as the MAIN_DEPTH sweep). Power mode: `default` (UNPINNED) before
and after, same caveat as every prior measurement in this file.

| SKIP_SLACK | depth(conv_1) | fitted cyc/px @ 1.8 GHz (W=32) |
|---|---|---|
| 2 (baseline) | 22 | 2624 |
| 4 | 24 | 973 |
| 8 | 28 | 659 |
| 16 | 36 | 744 |
| 25 (max fitting) | 45 | 703 |

Sharp drop 2->4->8, then a **noise-bound plateau at ~660-750** through 16 and 25 -- 8 already
captures effectively all of this run's available win; 16/25 do not improve on 8 (non-monotonic,
consistent with the box-contention caveat, not a real degradation). The absolute slack=2 number
here (2624) differs from the untraced net-rate baseline elsewhere in this file (1528) -- same
box-contention caveat, not a regression; the SHAPE (dose-response, plateau) is what corroborates
the hypothesis, not the absolute baseline.

**Corroboration: b3c3 re-traced at slack=2 vs. 25, same process, W=32, H=128**
(`aie_kernels/_test/trace_span_net_skipslack.py`):

| SKIP_SLACK | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|
| 2 | 406.2 (29.51%) | 919.72 (66.31%) | 70.3% | 1325.9 |
| 25 | 406.2 (59.14%) | 224.92 (32.49%) | 40.49% | 631.1 |

**Compute is bit-for-bit identical (406.2) at both slacks -- only the gap moved.** Gap fell 76%
(919.72 -> 224.92); LOCK_STALL fell from 70.3% to 40.49%; compute's share of the span rose from
29.5% to 59.1% -- b3c3 is now doing more work than waiting, which no prior lever in this file
achieved (MAIN_DEPTH+1 and b1c1-depth+1 both left b3c3 at ~70% LOCK_STALL, near-zero movement).
The slack=25 trace total (631.1) also lands within 11% of the slack=25 wall-clock fit (703),
cross-checking the two methods.

**This is the decisive result -- rules in the skip-ring, not the main-path 1-slot windows** (those
remain small, real, local levers per the sections above, just not the ceiling). `b3c3` still shows
40.49% LOCK_STALL at slack=25, so the ring is not the ONLY throttle left -- MAIN_DEPTH/PROD_DEPTH
are still live secondary levers, matching the earlier finding that they move b3c3 by ~1.5-20 cyc/px
on their own, now stacked on top of a much smaller base.

**Step 3 -- quantitative check, T ~= L/(slack+c).** Fitting the traced-b3c3 gap (919.72 at slack=2,
224.92 at slack=25) to `gap = L/(slack+c)` gives **c ~= 5.4, L ~= 6847 cyc/px**. Only 6 of the
~20 main-path stages between conv_1 and conv_cat have been individually traced (conv_1 40.42,
conv_cat 58.23, b1c1 319.87, b1c2 322.79, b3c3 406.20, up 26.68 -- up and the two join/skip
stages are not ON the timed main-path span this L covers, so the comparison is against b1c1/
b1c2/b3c3), whose mean is ~349 cyc/px; **6847 / ~20 hops ~= 342 cyc/px**, matching that mean to
within 2%. Consistent with the model, not an independent confirmation (too few stages traced to
sum L directly) -- but the right order of magnitude from a completely different fit.

**Change: SKIP_SLACK default raised 2 -> 8** (`net_layout.py`), chosen as the point where the
dose-response plateaus in this run, well under the slack=25 MemTile ceiling (leaves headroom for
future MAIN_DEPTH/PROD_DEPTH work on the same pool). Re-ran `verify_span_net.py` at the new
default: **22/22 cores exact** (`up`, 32x64, 32768/32768 -- every stage upto it also exact),
same as the slack=2 baseline gate.

## Phase 1a: PROD_DEPTH, cat_cons_depth, and the untraced skip stages -- all NULL

Task (`2026-09-27-npu-any-game-realtime.md` phase 1a): at SKIP_SLACK=8, b3c3 still shows ~40%
LOCK_STALL, ~1.6x over its own isolated compute. Three untested suspects going in: (1) `PROD_DEPTH`
(the four skip sources' core-side producer depth, default 2 -- their output ObjectFifo has TWO
consumers, the next main-path core AND the join's MemTile ring, and a broadcast producer can only
run as far ahead as its slower consumer); (2) conv_cat's own `f_cat.cons(depth=2)` read-ahead into
the join ring; (3) the untraced skip-related stages b6c1 and conv_2.

`net_design.build()` gained `prod_depth=`/`cat_cons_depth=` overrides (default None -> the prior
hardcoded values, module byte-identical when omitted; same pattern as `main_depth=`/`skip_slack=`).
**Compile-only sweep, W=32, skip_slack=8:** `prod_depth` fits L1 up to at least 16 (unlike
MAIN_DEPTH, the skip-source cores are lightly loaded); `cat_cons_depth` fits at 3, fails at 4
(conv_cat's own L1, same class of wall as the MAIN_DEPTH tiles).

**Device: same-process b3c3 trace A/B, W=32, H=128, SKIP_SLACK=8 held fixed**
(`aie_kernels/_test/probe_span_stall_levers.py`). Power mode: `default` (UNPINNED), same caveat
as every prior measurement in this file.

| lever | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|
| prod_depth=2 (baseline) | 406.2 | 224.94 | 40.46% | 631.1 |
| prod_depth=8 | 406.2 | 224.94 | 40.47% | 631.1 |
| cat_cons_depth=2 (baseline) | 406.2 | 224.94 | 40.46% | 631.1 |
| cat_cons_depth=3 | 406.2 | 224.94 | 40.46% | 631.1 |
| prod_depth=8 + cat_cons_depth=3 (combined) | 406.2 | 224.93 | 40.47% | 631.1 |

**All three NULL -- bit-for-bit identical to the baseline within trace noise.** Neither the
skip-source producer's own buffering nor conv_cat's read-ahead moves b3c3 at all, at any value
either compiles. This also indirectly refutes the "throttled by its slower consumer" framing for
PROD_DEPTH: if the main-path consumer (fixed at main_depth=4, L1-bound, already refuted as a
b3c3 lever) were the ceiling, giving the producer more of its OWN buffer still wouldn't show up at
b3c3 -- consistent with what was measured, but not a positive confirmation of the mechanism either.

**Untraced stages, same session, defaults (prod_depth=2, cat_cons_depth=2, skip_slack=8):**

| stage | role | compute cyc/px | gap cyc/px | LOCK_STALL % of span | compute+gap |
|---|---|---|---|---|---|
| b6c1 | skip source, mid-chain | 250.39 (34.58%) | 366.31 (50.2%) | 65.13% | 616.7 |
| conv_2 | skip source, feeds join 1 row ahead | 83.29 (11.28%) | 526.58 (70.75%) | 88.44% | 609.9 |

Both land at compute+gap ~610-617, matching b3c3's 631.1 and the earlier-traced conv_1/conv_cat/
b1c1/b1c2/up signature (all "victims" of the same common pace, LOCK_STALL-dominated) -- no
different behaviour from being a skip source vs. an ordinary main-path hop, and no anomaly
localized to either stage.

**Net: all three named suspects are dead ends, and the plateau already reached at SKIP_SLACK=8 is
reproduced exactly** (631.1 cyc/px, 40.46-40.47% LOCK_STALL here vs. 631.1 cyc/px, 40.49% at
SKIP_SLACK=25 in the earlier trace) -- SKIP_SLACK=8 is already at the same floor as 25, corroborating
the plateau finding independently. **What the remaining stall correlates with:** every stage tried
so far -- skip source, join, ordinary main-path hop, the heaviest core -- converges to the SAME
~610-631 cyc/px pace regardless of its own compute or of any ObjectFifo depth knob touched (skip
ring, MAIN_DEPTH, b1c1->b1c2, PROD_DEPTH, cat_cons_depth all refuted at b3c3 specifically). That is
the same "structural, common-to-every-core" signature the original 3.5x attribution found, just at a
~2x lower floor after the SKIP_SLACK fix. No object-fifo depth lever tested touches it, which argues
the remaining ~40% is a LATENCY floor (fixed hop count/DMA round-trip through the array) rather than
a THROUGHPUT/buffering one -- consistent with `rows_ahead(b3c3)`-style pipeline fill latency, but
untested directly here (would need e.g. varying the main-path hop COUNT or tracing enough
intermediate stages to sum the fill latency directly, both out of scope for this phase). Not ruled
in or out; the object-fifo-depth search space this task named is now exhausted.

## Phase 1a follow-up (coordinator): all 22 stages traced, no pace-setter -- MemTile(4,1) is the
## shared resource

Coordinator's objection to the "latency floor" reading: latency only throttles a buffer-insensitive
pipeline, and PROD_DEPTH/cat_cons_depth (both buffers) were null. Alternative: the pace-setter is one
of the 14 stages not yet traced, or a non-core resource. Traced all 14 (`probe_span_all_stages.py`,
same conditions: W=32, H=128, SKIP_SLACK=8/default, prod_depth=2/default, cat_cons_depth=2/default,
one dispatch per stage).

**Full 22-stage table** (8 from earlier sections of this file, 14 new):

| stage | kind | col,row | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|---|
| conv_1 | conv1 | 0,2 | 40.42 | 1256.15* | 96.68%* | 1296.6* |
| b1c1 | silu16 | 0,3 | 319.87 | 1015.80* | 75.78%* | 1335.7* |
| b1c2 | silu_i16 | 0,4 | 322.79 | 1012.77* | 75.67%* | 1335.6* |
| b1c3 | gate | 0,5 | 406.22 | 236.03 | 37.46% | 642.2 |
| b2c1 | silu_x | 1,2 | 250.40 | 388.60 | 61.48% | 639.0 |
| b2c2 | silu | 1,3 | 250.44 | 386.84 | 61.81% | 637.3 |
| b2c3 | gate | 1,4 | 406.16 | 230.52 | 38.96% | 636.7 |
| b3c1 | silu_x | 1,5 | 250.42 | 383.00 | 62.52% | 633.4 |
| b3c2 | silu | 2,2 | 250.46 | 381.27 | 62.75% | 631.7 |
| b3c3 | gate | 2,3 | 406.20 | 224.94 | 40.46% | 631.1 |
| b4c1 | silu_x | 2,4 | 250.41 | 377.47 | 63.36% | 627.9 |
| b4c2 | silu | 2,5 | 250.43 | 374.35 | 63.65% | 624.8 |
| b4c3 | gate | 3,2 | 406.50 | 219.08 | 41.88% | 625.6 |
| b5c1 | silu_x | 3,3 | 250.52 | 371.80 | 64.24% | 622.3 |
| b5c2 | silu | 3,4 | 250.46 | 370.13 | 64.49% | 620.6 |
| b5c3 | gate | 3,5 | 406.17 | 213.80 | 43.34% | 620.0 |
| b6c1 | silu_x | 4,2 | 250.39 | 366.31 | 65.13% | 616.7 |
| b6c2 | silu | 4,3 | 250.50 | 364.54 | 65.31% | 615.0 |
| b6c3 | gate | 4,4 | 406.26 | 208.18 | 44.58% | 614.4 |
| conv_2 | plain | 4,5 | 83.29 | 526.58 | 88.44% | 609.9 |
| conv_cat | cat | 5,2 | 58.23 | 1246.58* | 95.80%* | 1304.8* |
| up | up | 5,3 | 26.68 | 1273.05* | 97.99%* | 1299.7* |

`col,row` from the compiled MLIR (aie.core/aie.tile ops) of the b1c3-traced build in this run --
placement is deterministic for that exact digest but not asserted identical across every possible
`trace_stages=` compile, so treat the *pattern* (grouping, MemTile sharing) as load-bearing, not the
exact column numbers for a different build. `*` = figures from the earlier per-lever/attribution
sections above, at SKIP_SLACK=2 (conv_1/b1c1/b1c2/conv_cat/up were traced before the SKIP_SLACK fix
and not re-traced at slack=8 in this pass -- they are NOT comparable to the slack=8 column at face
value; re-tracing them is the obvious next step if this needs closing out further).

**No pace-setter.** LOCK_STALL at slack=8 ranges 37.46% (b1c3) to 65.31% (b6c2) -- no stage anywhere
near the "low LOCK_STALL, compute near the pace" signature the coordinator predicted. Two clean
patterns instead: (1) **compute is fixed per KIND, not per position** -- every `gate` stage computes
406.2-406.5, every `silu_x`/`silu` stage computes 250.4-250.5, regardless of where in the chain it
sits; kernel work never drifts. (2) **compute+gap decreases smoothly and monotonically moving
downstream**, independent of kind: b1c3 642.2 -> b6c3 614.4 (gate kind, -27.8 over 5 block-hops),
b2c1 639.0 -> b6c2 615.0 (silu_x/silu kind, -24.0 over 4 block-hops) -- roughly -5 to -6 cyc/px per
block, both kinds tracking together. Since kernel compute is provably constant, this whole-chain
gradient is in the GAP only, and it correlates with chain POSITION, not with any per-stage
structural property (kind, fifo depth, MemTile). **Corrected per coordinator review:** the
simpler explanation is PIPELINE FILL/DRAIN inside the single H=128-row dispatch, not DVFS -- a
stage further downstream starts its own first traced row later (after more upstream hops fill),
so its measured window covers proportionally fewer of the run's fixed-cost fill/drain rows out of
its own n_rows-1 gap intervals, pulling its average down; this needs no clock-ramp assumption and
fits the smooth, position-ordered decrease directly. Do not lean on the DVFS framing.

**MemTile(4,1) is the shared resource the coordinator asked to check.** `net_layout.weight_groups`
(WEIGHT_GROUP=6) makes 4 weight-feed MemTiles; `join()`'s default `tile=AnyMemTile` places the
4-source `cat_in` join ring wherever the placer picks, and it landed on the SAME MemTile as weight
group 3 (b6c3/conv_2/conv_cat/up), not a dedicated one. Channel count (from the compiled
`aie.memtile_dma` blocks, a MemTile has 6 S2MM + 6 MM2S total):

| MemTile | S2MM used | MM2S used | total/12 | role |
|---|---|---|---|---|
| (0,1) | 1 | 6 | 7 | weight group 0 only |
| (2,1) | 1 | 6 | 7 | weight group 1 only |
| (3,1) | 1 | 6 | 7 | weight group 2 only |
| (4,1) | 5 | 5 | 10 | weight group 3 AND the entire 4-source join ring (cat_in) |

MemTile(4,1) is the busiest in the design (10/12 channels vs. 7/12 elsewhere) and the only one
carrying two logically distinct dataflows. Its 4 join inputs are NOT symmetric in placement: b6c1
(col 4, local) and conv_2 (col 4, local) sit on the SAME column as the join MemTile, but conv_1
(col 0) and b1c3 (col 0) must cross 4 columns through the stream-switch fabric to reach it.
conv_cat itself sits at col 5 -- one column PAST its own join MemTile -- so its main-path read of
the ring also crosses a column. No per-op DMA-hop-count or channel-occupancy trace was taken here
(would need MEMORY_STALL/STREAM_STALL attribution per hop, not just LOCK_STALL, and ideally a
pinned power mode first); this is a structural observation from the compiled MLIR, not a measured
attribution of the remaining stall to this MemTile specifically.

**Net for this follow-up:** every stage is a "victim" in the sense the coordinator meant (no
core runs near-saturated while others wait), and the compute+gap gradient across the whole chain
is better explained by pipeline fill/drain within the single dispatch than by position (see the
correction above). MemTile(4,1)'s channel load (10/12, hosting both the join and a weight group)
and the conv_1/b1c3 cross-column skip broadcasts are the concrete "shared resource" / "crosses
columns" candidates this MLIR surfaces, per the coordinator's fallback -- neither has been
measured against the remaining ~40% LOCK_STALL directly. The chain-length bisection below settles
which of these (main chain vs. join/tail) is worth pursuing further.

## Phase 1a follow-up 2 (coordinator): chain-length bisection -- the pace is set in the main chain,
## by the first 4 cores, not the join/tail

Key fact motivating this: the period (~610-640 traced, ~655-690 whole-net) exceeds every single
core's own compute (max 406.5, the gate kind) and even the LEAST-stalled traced core (b1c3,
37.46% LOCK_STALL) still stalls -- so the throttle is outside any one core's work, and (per
coordinator) latency alone cannot explain a lever-insensitive ceiling since PROD_DEPTH/
cat_cons_depth were both null. Discriminates main-chain vs. join/tail by chain length: whole-net
rate (repeat/fit probe, same method as the SKIP_SLACK dose-response) at SPAN_UPTO in {b1c3, b3c3,
b6c3, conv_2, conv_cat, up} -- everything up to and including conv_2 has NO join built at all
(`net_design.build()` only wires `f_cat`/the join when `"conv_cat"` is in the stage list), so this
isolates "main chain alone" from "main chain + join + tail" cleanly.

`aie_kernels/_test/probe_span_upto_bisect.py`, W=32, SKIP_SLACK=8/default, HEIGHTS
[64,128,192,256], 5 trials, alternated per height, fitted slope, same session. Power mode:
`default` (UNPINNED), same standing caveat.

| upto | cores | fitted cyc/px @ 1.8 GHz (W=32) |
|---|---|---|
| b1c3 | 4 | 663 |
| b3c3 | 10 | 760 |
| b6c3 | 19 | 666 |
| conv_2 | 20 (no join) | 655 |
| conv_cat | 21 (+join) | 688 |
| up | 22 (full net) | 679 |

**The pace is already fully present at 4 cores.** `upto=b1c3` (663) is within 2.4% of the full
22-core net (679) -- adding the remaining 18 cores, the entire join, AND the tail (conv_cat, up)
moves the rate by less than the run-to-run noise this file has repeatedly flagged (compare the
90/659/703/744 spread across the SKIP_SLACK dose-response's own trials). Removing the join and
tail entirely (`conv_2`, 655) does not raise the rate toward any single core's isolated compute
(250-433 cyc/px measured elsewhere) or lower it toward the full net's rate in a way that implicates
the join -- it's already indistinguishable from `up`. **This rules out the join/tail as the
throttle**: MemTile(4,1)'s channel sharing and the conv_1/b1c3 cross-column skip broadcasts (the
candidates flagged above) are NOT where the ~655-690 cyc/px pace comes from, since that pace exists
identically without them.

`b3c3` (760) is the outlier -- higher than BOTH `b1c3` (a shorter prefix) and `up` (the full net,
a longer prefix), which is not mechanistically sensible for a monotonically-accumulating chain and
reads as this file's usual wall-clock-fit noise (box shared throughout) rather than a real
mid-chain spike; the same-process TRACE of b3c3 IN THE FULL NET (compute+gap=631.1, well below
this truncated-build's 760) corroborates that 760 is a build/contention artifact of THIS specific
truncated design, not a property of b3c3 itself.

**Conclusion: the throttle is in the main chain, and it is already fully set within block 1 (the
first 4 cores: conv_1, b1c1, b1c2, b1c3)** -- not the join, not MemTile(4,1), not the tail. Given
b1c3's own compute is only 406.2-406.5 cyc/px (well under the ~660-680 pace) and PROD_DEPTH/
cat_cons_depth/MAIN_DEPTH/b1c1-b1c2-depth are all refuted as levers on it, the mechanism WITHIN
those first 4 cores is still open -- narrowing further (e.g. upto=conv_1, upto=b1c1, upto=b1c2 to
find exactly which hop within block 1 first reaches the pace) is the natural next bisection step,
not run here per "stop at the answer."

## Phase 1a follow-up 3 (coordinator): b1c1<->b1c2 alternation-sum, RE-CONFIRMED at SKIP_SLACK=8 --
## depth 3->4 collapses the gap, but does not fit L1 at W=32 on the full net

Coordinator's re-read of the earlier REFUTED section above: that test ran at SKIP_SLACK=2, where
the ring throttle set the pace network-wide and masked whatever the b1c1<->b1c2 link (`DEPTH =
{"b1c1": 3}` against b1c2's 3-row `windowed()` acquire, i.e. ZERO producer slack) was doing on its
own. Predicted compute-sum (320+323=643) already matched the chain-bisection's `upto=b1c3` pace
(663, prior section) suspiciously well. Re-tested at SKIP_SLACK=8 (`probe_span_b1c1b1c2_retest.py`,
`probe_span_b1c1b1c2_retest2.py`), W=32 unless noted, same-session methods throughout.

**Part 1 -- bisect upto in {conv_1, b1c1, b1c2, b1c3}, fitted rate:**

| upto | cores | fitted cyc/px |
|---|---|---|
| conv_1 | 1 | -10 (fit garbage -- too cheap, dominated by dispatch-overhead noise) |
| b1c1 | 2 | 448 |
| b1c2 | 3 | 647 |
| b1c3 | 4 | 813 (this session's own re-measurement; noisier than the prior bisection's 663 for the
same config -- see the standing box-contention caveat, not a regression) |

**b1c1 alone (448) matches its own isolated compute; b1c2 jumps to 647 -- within 0.6% of the
320+323=643 cyc/px alternation-sum prediction.** b1c3 is noisier (813) but the qualitative jump
already lands at b1c2, exactly as predicted.

**Part 2 -- same-process depth 3 vs 4 A/B, `upto=b1c3` (compile-only sweep found depth=4 fits L1
here, unlike the full net):**

| depth | fitted whole-net cyc/px |
|---|---|
| 3 (baseline) | 732 |
| 4 | 437 |

437 lands almost exactly on the gate-core isolated rate (~406-433). Same-process trace of b1c1 and
b1c2 individually, W=32 H=128:

| depth | stage | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|
| 3 | b1c1 | 319.94 | 320.95 | 49.72% | 640.9 |
| 3 | b1c2 | 322.75 | 318.06 | 49.70% | 640.8 |
| 4 | b1c1 | 319.52 | 85.74 | 20.89% | 405.3 |
| 4 | b1c2 | 323.14 | 84.11 | 21.28% | 407.2 |

**At depth=3 the two stages are near-perfectly symmetric: each one's own compute (~320-323) is
almost exactly half its own compute+gap (~640.8-640.9), and gap ~= the OTHER stage's compute --
the ping-pong signature the alternation-sum hypothesis predicts.** At depth=4, compute is
unchanged (confirms kernel work never moved) but gap collapses 74-76% (320.95->85.74,
318.06->84.11) and LOCK_STALL drops from ~49.7% to ~21% -- both stages land at the isolated gate
rate. **This is the strongest confirmation in this file**: same shape (compute pinned, gap
collapses) as the SKIP_SLACK fix, on a completely different link.

Tracing depth=4 on this short chain required a second lever: adding the trace bracket to b1c1
tipped the SAME tile over an L1 wall that non-traced depth=4 alone did not (see below) --
`net_design.build()` gained `data_sizes=` (aiecc's own suggested fix: `Worker(data_size=...)`,
default None -> no override, byte-identical when omitted) to reserve the LUT's static data
explicitly; `data_sizes={"b1c1": 4160}` fixed it for this short chain.

**Part 3 -- depth 3 vs 4 on the FULL NET (`upto=up`), W=16 (depth=4 does not fit L1 at W=32 on the
full net -- see below), fitted rate:**

| depth | fitted cyc/px (W=16) |
|---|---|
| 3 (baseline) | 781 |
| 4 | 631 |

19% lower, same direction as the short chain and the earlier corroborated levers in this file, but
a smaller relative win than the W=32 short-chain result (40%) -- consistent with this file's other
W=16 fallback sweeps (MAIN_DEPTH, b1c1-depth-at-slack=2) showing smaller/noisier wall-clock swings
than their W=32/trace counterparts.

**Does depth=4 fit L1 at W=32 on the full net? No, and there are TWO independent walls on the SAME
tile (0,3) = b1c1, not one:**

1. **b1c1's own static/constant data (its silu16 LUT table).** Without `data_sizes=`, aiecc's
   automatic buffer placement leaves too little room for the LUT after growing b1c1's own output
   buffer by one depth-4 slot: `ld.lld: error: section '.data' will not fit in region 'data':
   overflowed by 2624 bytes` / `aiecc: core main_core_0_3 needs space for 4160 bytes of static
   data ... but it may fit if you reserve it explicitly` -- exactly the fix `data_sizes=` now
   applies. **Confirmed fixable**, and fixed, for the short chain.
2. **Applying that fix on the full net does NOT close the gap -- it exposes a SECOND, independent
   wall on the exact same tile.** With `data_sizes={"b1c1": 4160}` set, compilation proceeds
   further (bank-aware allocation now fails only on `b6c2_out_buff_3`/`b3c2_out_buff_3`, which
   basic-sequential allocation recovers from as warnings, not fatal) and then hits a HARD error
   back on tile (0,3): `'aie.tile' op basic-sequential allocation failed. Core (0, 3) reserves
   4160 bytes for its static data ..., which has to fit alongside this tile's buffers` -- the
   specific buffer that fails to place is `conv_1_skip_1_cons_buff_2` (2304 B), the conv_1->b1c1
   MAIN-PATH INPUT fifo's own buffer (main_depth=4, 4 x 2304 B = 9216 B total), which is exactly
   the SAME broadcast objectfifo the PROD_DEPTH investigation named (conv_1's output has two
   consumers: the join, and this main-path hop into b1c1).

**What is on tile (0,3) and what would have to shrink** (from the compiled buffer sizes in
`net_layout.py`/`net_design.py`, W=32): LUT/static data 4160 B (fixed, kernel constant table) +
STACK["silu16"] 3584 B + conv_1->b1c1 input fifo 4 x 2304 B = 9216 B (main_depth=4, network-wide) +
b1c1's OWN output fifo D x 6912 B (silu16 out_bytes=6912 B; D=3 -> 20736 B, D=4 -> 27648 B, i.e.
raising D by 1 costs exactly +6912 B) + b1c1's weight sub-blob (model-fixed, `plen["b1c1"]`). D=3
already compiles with `net_design.py`'s existing `DEPTH={"b1c1": 3}` override -- the module's own
comment says this block "ha[s] no headroom at all" -- so the tile is already fully committed at
D=3, and D=4's marginal +6912 B is what overflows it. Candidates to shrink, none free: (a) the
LUT table itself (4160 B, narrower/fewer entries -- touches SiLU16 approximation quality, out of
scope here); (b) `STACK["silu16"]` (3584 B, only shrinkable if aiecc's own measured-stack-size
check confirms slack -- guessing is exactly what this codebase's hanging-numbers doctrine warns
against); (c) the conv_1->b1c1 input fifo's depth, specifically for this one hop rather than
`MAIN_DEPTH` network-wide (no per-hop input-depth override exists yet in `net_design.py`; would
need a new parameter, untested here). None attempted -- flagged for a follow-up, not closed.

**Decision: do NOT flip the default.** Confirmed at W=32 on the short chain (trace) and W=16 on
the full net (fitted rate); NOT confirmed to fit L1 at W=32 on the full net, which is this file's
explicit gate for adopting a new default. `verify_span_net.py` was not re-run since
`net_layout.DEPTH`/`net_design.MAIN_DEPTH` etc. are unchanged -- only the new opt-in `data_sizes=`
diagnostic parameter (default None, byte-identical when omitted) was added to `net_design.py`.

## Phase 1a follow-up 4 (coordinator): made DEPTH["b1c1"]=4 fit L1 at W=32 -- DEFAULT FLIPPED

Merged `main` (bee0720, "inline the gate epilogue") into this branch first, so all timings below
are current. That merge alone lowered the whole net's baseline pace network-wide (this session's
own baseline trace: b1c1/b1c2/b3c3 all ~575-587 cyc/px, down from ~610-690 in pre-merge sessions --
also visible in b1c1's own compute dropping from ~320 to ~266 cyc/px, unrelated to any lever in
this file). Net conflict in `net_design.py` (main added `split_gate`/output-channel splitting,
`aie_kernels/_test/BALANCE.md` epic) resolved by keeping both feature sets side by side; compile-
and device-verified after the merge before continuing.

Tried the coordinator's candidates in cost order, each measured against the compiled buffer map
(the exact byte accounting the prior section only estimated):

1. **Measured stack (tried, informative, NOT used in the final fix).** Deliberately under-sizing
   `stacks={"silu16": 1}` and reading aiecc's own error gives b1c1's MEASURED stack requirement:
   **1472 B** (vs. the reserved 3584 B -- 2112 B of apparent headroom). But the tile's true deficit
   at depth=4 (with the LUT fix below applied) is *also* 2112 B, so closing it via stack ALONE
   would need the bare measured value with ZERO margin (1472, no headroom for e.g. a future shim
   change) -- confirmed by compiling at `stacks={"silu16": 1600}` (measured + 128 B margin): still
   FAILS, over by exactly 128 B. Not used because lever 3 (below) closes the gap with margin to
   spare, without touching stack at all.
2. **Asymmetric producer/consumer depths through DMA -- not tried.** Lever 1 already gave a full
   byte-exact accounting (`net_design.py`'s error output prints tile (0,3)'s complete MemoryMap;
   see below) showing the b1c1<->b1c2 buffer lives ENTIRELY on b1c1's own tile (0,3), not split
   across b1c1/b1c2 -- an adjacent-core objectFifo link is placed as one buffer array on one tile,
   not two. b1c2's tile budget is untouched by this whole depth bump, so there is no producer/
   consumer split to make; the shared-pool bug this lever was meant to route around does not apply
   here (it is a JOIN pool issue, not an adjacent two-core link).
3. **conv_1->b1c1 input depth 3 (USED).** New `net_design.build(skip_cons_depths=...)` param
   (per-skip-source override of the hardcoded `main_depth` on that source's main-path consumer;
   default `{}`, merged over a new `SKIP_CONS_DEPTHS` module constant, same pattern as `DEPTH`).
   `skip_cons_depths={"conv_1": 3}` drops ONE `conv_1_skip_1_cons_buff` slot (2304 B) on tile (0,3).
   Combined with `data_sizes={"b1c1": 4160}` (the LUT-starvation fix from the prior section) and
   `depths={"b1c1": 4}`, **compiles at W=32 on the full net, with default `STACK["silu16"]`
   unchanged.**

**Exact byte accounting for tile (0,3), from the compiled buffer map** (forcing a deliberate small
overflow to print it, `aie_kernels/_test/probe_span_b1c1_w32_fit.py`'s sibling checks):

| item | bytes | note |
|---|---|---|
| stack | 3584 | STACK["silu16"], unchanged |
| p_b1c1_cons_buff_0 | 23040 | weight sub-blob, model-fixed |
| b1c1_out_buff_0..3 | 4 x 6912 = 27648 | DEPTH["b1c1"]=4 (was 3 x 6912 = 20736) |
| core data sections | 4160 | LUT table, data_sizes fix |
| conv_1_skip_1_cons_buff_0..2 | 3 x 2304 = 6912 | skip_cons_depths={"conv_1": 3} (was 4 x 2304 = 9216) |
| **total** | **65344** | **192 B free of 65536 (64 KB)** |

Without the skip_cons_depths fix (4 x 2304 = 9216), total = 67648, over by 2112 B -- matching lever
1's stack-alone deficit exactly. With it, the deficit closes with 192 B to spare, no LUT/weight/
stack change needed.

**Full-net (upto=up) W=32 A/B, alternated per height, 4 heights, medians**
(`aie_kernels/_test/probe_span_b1c1_w32_fit.py`). Power mode: `default` (UNPINNED).

| config | fitted cyc/px (W=32) |
|---|---|
| baseline (depth=3) | 553 |
| new (depth=4 + data_sizes + skip_cons_depths) | 561 |

**The wall-clock fit shows no clear win (553 vs 561, within this file's usual noise band) -- do
not read this as a null result; the same-process trace below contradicts it and is the
instrument this file has trusted throughout.**

**Same-process trace, W=32 H=128, at the new config, then re-traced at baseline for a clean
same-instrument comparison** (`probe_span_b1c1_w32_fit.py` + `probe_span_b1c1_w32_fit_baseline.py`):

| config | stage | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|
| baseline | b1c1 | 265.6 | 321.07 | 54.35% | 586.7 |
| baseline | b1c2 | 322.75 | 264.29 | 45.12% | 587.0 |
| baseline | b3c3 | 315.08 | 259.4 | 49.17% | 574.5 |
| new | b1c1 | 266.35 | 216.47 | 44.51% | 482.8 |
| new | b1c2 | 324.01 | 159.78 | 33.25% | 483.8 |
| new | b3c3 | 316.26 | 167.49 | 38.98% | 483.8 |

**Compute is unchanged between baseline and new (confirms the fix touches only synchronization,
not kernel work) and the new config's three stages land within 1 cyc/px of each other (482.8-
483.8) -- the tightest common-pace convergence in this whole file.** b3c3 (the previously-
identified heaviest core) drops 574.5 -> 483.8 (15.8%), LOCK_STALL 49.17% -> 38.98%; b1c1 drops
586.7 -> 482.8 (17.7%), LOCK_STALL 54.35% -> 44.51%. **This is a real, device-confirmed win on the
full production-width net, decisively shown by the trusted instrument even though the noisier
wall-clock A/B missed it** -- consistent with this file's standing caveat that same-dispatch trace
splits are trustworthy where cross-dispatch wall-clock timing is not.

**Gate: `verify_span_net.py` at the new default (no explicit overrides) -- 22/22 cores exact.**

**DEFAULT FLIPPED.** `net_design.py`: `DEPTH["b1c1"]` 3 -> 4; new module constants
`DATA_SIZES = {"b1c1": 4160}` and `SKIP_CONS_DEPTHS = {"conv_1": 3}`, both merged unconditionally
(same pattern as `DEPTH`) so a caller with no overrides gets the fixed design. `data_sizes=`/
`skip_cons_depths=` build() params still override per-call if needed.

## Phase 1c: fresh 22-stage trace at main's new defaults -- still no pace-setter, zero-slack
## census closed, split-gate trace lever landed at ~1.5%

Continuation on `main` d898bfe (the block-1 `DEPTH["b1c1"]=4` flip and the gate-epilogue inline
both already landed). `probe_span_all_stages_phase1c.py` re-traces all 22 stages fresh, one per
dispatch (`ALREADY_TRACED = set()`, was 8/22 before). Power mode: `default` (UNPINNED), standing
caveat.

| stage | kind | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|
| conv_1 | conv1 | 40.42 | 436.45 | 91.0% | 476.9 |
| b1c1 | silu16 | 266.35 | 216.5 | 44.53% | 482.9 |
| b1c2 | silu_i16 | 324.01 | 159.77 | 33.2% | 483.8 |
| b1c3 | gate | 315.29 | 168.83 | 35.75% | 484.1 |
| b2c1 | silu_x | 250.82 | 232.55 | 49.09% | 483.4 |
| b2c2 | silu | 251.09 | 232.41 | 49.56% | 483.5 |
| b2c3 | gate | 315.61 | 168.3 | 37.37% | 483.9 |
| b3c1 | silu_x | 250.72 | 232.44 | 50.59% | 483.2 |
| b3c2 | silu | 251.24 | 232.04 | 50.86% | 483.3 |
| b3c3 | gate | 316.28 | 167.47 | 38.97% | 483.8 |
| b4c1 | silu_x | 250.93 | 232.05 | 51.83% | 483.0 |
| b4c2 | silu | 250.66 | 231.77 | 52.33% | 482.4 |
| b4c3 | gate | 315.78 | 167.79 | 40.63% | 483.6 |
| b5c1 | silu_x | 251.47 | 231.34 | 52.98% | 482.8 |
| b5c2 | silu | 251.01 | 231.88 | 53.46% | 482.9 |
| b5c3 | gate | 315.4 | 168.0 | 42.31% | 483.3 |
| b6c1 | silu_x | 250.39 | 232.15 | 54.44% | 482.5 |
| b6c2 | silu | 251.23 | 231.5 | 54.56% | 482.7 |
| b6c3 | gate | 316.17 | 167.02 | 43.51% | 483.2 |
| conv_2 | plain | 83.5 | 397.65 | 84.87% | 481.1 |
| conv_cat | cat | 58.23 | 423.05 | 89.4% | 481.3 |
| up | up | 26.68 | 452.02 | 94.93% | 478.7 |

**Per-kind compute (fixed regardless of chain position, confirming the kind-not-position finding
still holds after the epilogue inline -- these numbers are unchanged from the last section's
"new" config, i.e. main's current computes ARE already the post-inline ones):** conv1=40.4,
silu16=266.4, silu_i16=324.0, gate=315-316, silu_x=250.4-251.5, silu=250.7-251.2, plain=83.5,
cat=58.2, up=26.7.

**Still no pace-setter.** Every stage converges to compute+gap 477-484, LOCK_STALL ranging
33.2% (b1c2, the highest-compute stage) to 94.93% (up, the cheapest) -- the same inverse
compute-vs-stall ordering as before the block-1 fix, just at a ~480 floor instead of ~610-690.

**Zero-slack census, from the compiled buffer map (not guessed), network-wide:** every ordinary
main-path hop's consumer window is 3 rows (`windowed()`'s steady-state `fi.acquire(3)`) against a
depth-4 fifo (`MAIN_DEPTH`), 1 row of slack; `conv_cat` acquires 1 row (`rowwise()`) against
`cat_cons_depth=2`, 1 row of slack. Exactly ONE link in the whole network has zero slack:
**conv_1 -> b1c1**, where `SKIP_CONS_DEPTHS = {"conv_1": 3}` (the fix that made `DEPTH["b1c1"]=4`
fit L1, landed in the prior section) drops that specific consumer depth to 3, matching the
3-row window exactly. Checked against the trace for the block-1 alternation-sum signature that
found the *previous* zero-slack link (b1c1<->b1c2, now fixed): conv_1's gap (436.45) does not
match b1c1's compute (266.35), and b1c1's gap (216.5) does not match conv_1's compute (40.42) --
**refuted as a coupled pair**. conv_1's own compute is small enough (40.4 cyc/px) that the zero
slack here does not visibly throttle b1c1; b1c1 sits at the common ~483 pace like everything else.
No other zero-slack link exists to check.

**Chain-length bisection re-run at the new baseline** (`probe_span_upto_bisect.py`, unchanged
script, W=32, same alternated-per-height fitted-slope method). Log:
`/mnt/data/xdna/traces/span/phase1c_bisect.log`.

| upto | cores | fitted cyc/px (W=32) |
|---|---|---|
| b1c3 | 4 | 572 |
| b3c3 | 10 | 704 |
| b6c3 | 19 | 353 |
| conv_2 | 20 (no join) | 236 |
| conv_cat | 21 (+join) | 582 |
| up | 22 (full net) | 616 |

Non-monotonic (b6c3/conv_2 below both a shorter prefix and the full net -- mechanistically
impossible for an accumulating chain, the same box-contention pattern flagged throughout this
file). `b1c3` (572) is within ~7% of `up` (616) -- **same conclusion as the pre-fix bisection: the
pace is already essentially set within block 1's first 4 cores**, just ~530-680 instead of
~660-680. No new lever found by this pass; the mechanism *within* block 1 (given b1c1's own
compute, 266.4, is well under the ~480-620 pace, and the sole zero-slack link there is refuted
above) remains open, as it was before this phase.

**Split-gate opt-in lever (`net_design.build(split_gate={"b2c3"})`), same-process trace of the
halves and neighbours** -- previously measured only by whole-net repeat/fit
(`probe_span_split_gate.py`); tracing the halves needed two source fixes to `net_design.py`
(both additive/default-off, applied and gated below):

1. `_shim_gate_half()` had no `bracket=` parameter at all, so a split gate's half-kernels could
   never carry the event0()/event1() trace bracket. Added `bracket=False` (default off, same
   convention as `_shim()`).
2. `design()`'s trace-worker lookup was `workers[names.index(n)]` -- broken for split gates,
   since `names` never contains the half-suffixed names (`"b2c3_lo"`/`"b2c3_hi"`) and the split
   branch appends 2 workers per one `names` entry, desyncing the index from `workers`. Replaced
   with a `worker_by_name` dict built alongside `workers` (keyed by the half suffix for split
   stages, by `n` otherwise); a strict superset -- identical lookup result whenever `split_gate`
   is empty.

**Found and worked around a real defect in `trace_span_net.summarize()`**: it aggregates every
traced-core event in a JSON by event NAME only, not by (pid, stage) -- so tracing 2 stages in one
dispatch (which the module's own docstring claims works, "at most 2 stages per run") silently
merges two cores' timelines. Reproduced directly: an unsplit `b2c2`+`b2c3` 2-stage dispatch gave
*identical* numbers for both stages, compute >100% of span, negative gap -- obviously wrong.
Every number in this file and its predecessor was, on inspection, traced ONE stage per dispatch
despite the docstring's claim; worked around here the same way (one stage per dispatch
throughout). `summarize()` needs a pid filter before a 2-stage trace is trusted again -- flagged,
not fixed, since fixing the instrument is out of scope for this phase.

Device trace, W=32 H=128, one stage per dispatch:

| build | stage | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|
| unsplit | b2c2 | 251.11 | 232.4 | 49.52% | 483.5 |
| unsplit | b2c3 | 315.61 | 168.33 | 37.41% | 483.9 |
| unsplit | b3c1 | 250.72 | 232.46 | 50.6% | 483.2 |
| split | b2c3_lo | 210.7 | 265.46 | 57.44% | 476.2 |
| split | b2c3_hi | 103.42 | 371.97 | 78.83% | 475.4 |
| split | b2c2 | 250.79 | 226.38 | 49.07% | 477.2 |
| split | b3c1 | 250.79 | 225.9 | 49.82% | 476.7 |

`b2c3_lo` + `b2c3_hi` compute = 314.12, matching unsplit `b2c3`'s 315.61 (splitting a 48-channel
gate into 32/16 preserves total compute, as expected). All four split-build stages converge at
~476-477, ~1.4-1.7% below the unsplit ~483 pace -- a small, real win, consistent with (not
contradicting) task 2/3's finding that `gate`'s own compute was never the pace-setter: splitting
it relieves its own LOCK_STALL (37.41% -> 57.44%/78.83% split across the two halves, each now
further from the pace) but barely moves the whole-chain floor, because the floor was never set
by gate's compute in the first place.

**Net for this phase:** the 22-stage census, zero-slack check, and re-bisection all corroborate
each other and the pre-fix pass -- the throttle is structural, common to every core, and already
fully present within block 1's first 4 cores, at a ~480-620 floor (down from ~610-690). No new
buffer-depth lever closes it; `split_gate={"b2c3"}` buys ~1.5% (expected, since it was never
implicated). The open item carried forward is unchanged: the mechanism *within* block 1's first
4 cores, given every buffer-depth knob tried on it (MAIN_DEPTH, PROD_DEPTH, cat_cons_depth,
b1c1<->b1c2, conv_1->b1c1) is now refuted or null.

## Phase 1c follow-up (coordinator): DMA-crossing-hop hypothesis -- REFUTED

Hypothesis: every main-path hop has depth 4 against a 3-row window (1 free slot). On a
SHARED-MEMORY hop (adjacent tiles, same column) the freed slot costs no copy. On a DMA-crossing
hop the freed slot must be refilled by an actual DMA transfer after the consumer releases a row,
so that latency sits on the critical path every row: pace ~= max compute + T_dma.

**Confirmed from the compiled MLIR (`span_net_physical.mlir`) which links actually cross tiles,
not assumed:** block 1 (`conv_1` (0,2), `b1c1` (0,3), `b1c2` (0,4), `b1c3` (0,5)) sits on ONE
column, and its objectFifo buffer for e.g. `b1c1<->b1c2` lives as ONE array on ONE tile (already
noted in the earlier byte-accounting section) -- shared memory, no DMA. The four column-crossing
main-path hops are `b1c3`(0,5)`->b2c1`(1,2), `b3c1`(1,5)`->b3c2`(2,2), `b4c2`(2,5)`->b4c3`(3,2),
`b5c3`(3,5)`->b6c1`(4,2) -- and for the one checked directly, `b1c3->b2c1`, the MLIR allocates
TWO separate buffer arrays: `b1c3_skip_buff_0/1` (2 buffers, PROD_DEPTH=2) on `b1c3`'s own tile
(0,5), and a SEPARATE `b1c3_skip_1_cons_buff_0..3` (4 buffers, matching `main_depth`) on `b2c1`'s
tile (1,2) -- confirming a real DMA copy backs this link, unlike the one-array-one-tile
shared-memory case.

`b1c3` is also a skip source (`CAT_SOURCES`), so its main-path consumer depth is keyed by
`skip_cons_depths`, not `depths`; the other three column-crossing hops are plain stages, keyed
by `depths`. Compile-only sweep, W=32: `depths={"b3c1": 5, "b4c2": 5, "b5c3": 5},
skip_cons_depths={"b1c3": 5}` -- **fits L1** (all four raised simultaneously, no per-hop
override needed beyond the existing `build()` parameters -- no new code required for this test).

**Device: same-process A/B trace of `conv_1`, `b1c2`, `b3c3`, W=32 H=128**
(`aie_kernels/_test/probe_span_dmahop_depth.py`). Power mode: `default` (UNPINNED), standing
caveat.

| config | stage | compute cyc/px | gap cyc/px | LOCK_STALL % | compute+gap |
|---|---|---|---|---|---|
| baseline | conv_1 | 40.42 | 436.32 | 90.99% | 476.7 |
| baseline | b1c2 | 324.02 | 159.78 | 33.21% | 483.8 |
| baseline | b3c3 | 316.27 | 167.48 | 38.92% | 483.8 |
| dmahop+1 | conv_1 | 40.42 | 436.63 | 91.0% | 477.1 |
| dmahop+1 | b1c2 | 324.04 | 159.74 | 33.24% | 483.8 |
| dmahop+1 | b3c3 | 316.29 | 167.43 | 38.93% | 483.7 |

**REFUTED -- bit-for-bit identical within trace noise (<0.1-0.4 cyc/px) at all three stages.**
Giving every column-crossing hop one more consumer slot does not move the pace at all, even
though the compile-only check confirmed it actually changed the buffer allocation (fits L1 only
because it grew). So the ~480-620 floor is not a per-row DMA-refill latency on these four links
either -- ruling out the specific mechanism this hypothesis named, on top of every buffer-depth
lever already refuted in this file (SKIP_SLACK past 8, MAIN_DEPTH, PROD_DEPTH, cat_cons_depth,
b1c1<->b1c2 at slack=8, conv_1->b1c1's zero-slack link). The DMA-vs-shared-memory distinction is
real (confirmed in the MLIR) but this experiment shows it is not where the remaining stall lives.

## `trace_span_net.summarize()` fixed: pid-filtered, no longer silently merges multi-core traces

`summarize()` gained a `pid=` parameter (default `None`): auto-detects the traced core from
`INSTR_EVENT_0`'s pid and, if more than one is present in the JSON, RAISES instead of silently
aggregating both cores' events under one event name (the defect found in the split-gate section
above). Every existing call site is single-stage-per-dispatch, so behavior is unchanged for all
of them; a future 2-stage-per-dispatch caller must now pass `pid=` explicitly or gets a loud
error instead of a wrong number.

## Phase 1d: fixed-per-row-cost hypothesis -- CONFIRMED IN SHAPE, REFUTED IN MAGNITUDE; the one
## named cheap fix is a documented device-refuted dead end

Hypothesis: T_row = a + b*W (pace cyc/px = b + a/W), with a rough two-point fit from older,
differently-configured runs guessing a ~= 4.7k cycles/row. Tested on the `SPAN_UPTO=b1c3` 4-core
prefix (main's defaults, no depth/skip overrides), the only chain that fits L1 across more than
one W.

**W compile sweep** (`probe_span_w_sweep_compile.py`, compile-only, no device): W=16 and 32 OK;
W=48 and 64 FAIL on tile (0,3) = b1c1 (`b1c1_out_buff`/`silu16` LUT, same class of wall this file's
byte-budget section already derived: `half(w)` grows past the tile's remaining headroom once
`half(w) > ~2317 B`, i.e. `w > ~32`). So only W=16/32 admit a device sweep at main's defaults --
48/64 are out of reach without also touching DEPTH/SKIP_CONS_DEPTHS, which is a different
experiment.

**Device: same-process trace of b1c2/b1c3 (one stage per dispatch, H=128) plus a wall-clock
fitted-slope cross-check of the whole b1c3 prefix, W=16 vs 32** (`probe_span_w_sweep_trace.py`).
Power mode: `default` (UNPINNED), standing caveat.

| W | stage | compute cyc/row | gap cyc/row | LOCK_STALL % | compute+gap/row | compute+gap cyc/px |
|---|---|---|---|---|---|---|
| 16 | b1c2 | 5372.3 | 86.1 | 2.44% | 5458.4 | 341.2 |
| 16 | b1c3 | 5192.3 | 269.3 | 7.51% | 5461.6 | 341.4 |
| 32 | b1c2 | 10344.4 | 86.1 | 1.97% | 10430.5 | 326.0 |
| 32 | b1c3 | 10096.0 | 349.4 | 5.91% | 10445.4 | 326.4 |

LOCK_STALL is tiny here (2.4-7.5%) versus the full net's 33-98% -- this prefix has no join/tail
contention, so it is not comparable to the full-net floor, exactly as expected from using it only
to hold L1 across widths.

**Two-point fit, T_row = a + b*W** (only 2 W admit, so this is an exact solve, not a regression):

| stage | compute a, b | gap a, b | compute+gap a, b |
|---|---|---|---|
| b1c2 | 400.2, 310.76 | 86.1, 0.00 | 486.3, 310.76 |
| b1c3 | 288.6, 306.48 | 189.2, 5.01 | 477.8, 311.49 |

Wall-clock cross-check (whole b1c3-prefix chain, fitted slope over heights [64,128,192,256], not
per-stage): 4926.6 cyc/row at W=16, 10875.8 at W=32 -- consistent in shape (compute-dominated,
near-linear in W) with the trace, though the 2-point a/b solve on this noisier series alone
(a=-1023, b=372) is not trustworthy on its own, same standing box-contention caveat as every
wall-clock fit in this file.

**CONFIRMED IN SHAPE: a > 0 (there is a real, W-independent per-row cost).** **REFUTED IN
MAGNITUDE: a ~= 400-486 cycles/row, not ~4.7k.** b (~307-311 cyc/px) lands right at the measured
per-kind compute for `gate`/`silu_i16` (306-324 cyc/px elsewhere in this file) -- at these widths
the pace is compute-dominated, not fixed-cost-dominated: a/W is only ~15-30 of the ~326-341 cyc/px
total (5-9%). b1c2's gap fit is the cleanest single result in this file: b=0.00 exactly -- its gap
is a pure W-independent constant (86.1 cycles), not a per-pixel term.

**Where `a` lives: mostly INSIDE the compute bracket, not the gap.** Of the ~478-486 total,
289-400 cycles sit inside compute (the traced kernel call itself) and only 86-189 sit in the gap
(handshake/lock latency, consistent with LOCK_STALL being small here). So the fixed cost is a
kernel-call overhead, not a synchronization one, on this prefix.

**Disassembly (Peano `llvm-objdump` from the pinned instance, `elfs_main_core_0_4.elf`,
`conv3x3_i16i8_lut` = b1c2's kernel):** the per-row call is `core_0_4`'s steady-state loop (`acq`
lock, `jl #0xf20` into the shim, `rel` lock) calling the shim (`wsweeptr32b1c2_b1c2_w32`, 0x140 B)
which brackets `event0()`/a call to `conv3x3_core<...>` (0x420, 0xb00 B)/`event1()`. The visible
prefix of `conv3x3_core` (30+ VLIW bundles before the tile loop, register-spill prologue +
`vbcst.8`/`vbcst.16`/`vbcst.32`/`vconv.fp32.bf16`/`vst bmll0`/`vst bmlh0`) IS per-call LUT/table
construction -- broadcasting and converting the SiLU lookup table into vector registers and
spilling it to the stack, every row, even though the table is a per-net constant. Order of
magnitude matches the measured ~289-400 cycle compute-side fixed cost.

**This exact hoist is a KNOWN, ALREADY-TRIED, DEVICE-REFUTED dead end, not an open opportunity.**
`aie_kernels/conv2d-3x3-u8/conv3x3_u8.cc:92-94` documents it directly: "The lookup object is built
per call: one built once at the top of the gate kernel and captured by reference gathered every
key as 0 (`probe_conv3x3_gate_stages.py`)." So the one concrete source this disassembly finds for
`a` cannot be hoisted with the current `aie::lut`/`parallel_lookup` API without reintroducing a
device-measured correctness bug -- confirmed from source, not re-derived from scratch (per this
file's own "search before re-deriving" standing rule).

**Decision: no prototype attempted.** The task's own gate for step 4 ("if `a` is large") is not
met -- measured `a` is ~400-486 cycles/row, two orders of magnitude under the ~4.7k the rough fit
guessed, and the one identified mechanism for it is a documented dead end. The "process 2 output
rows per kernel call" alternative was not attempted: it is a Worker/kernel-signature change (new
acquire/release counts, a wider LUT-amortization boundary), not a cheap one, and the small size of
`a` does not justify that risk here. **Implication for Phase 2:** widening W is a small, real lever
on this prefix (341.2 -> 326.0 cyc/px, W=16->32, ~4.5%) but is capped by L1 at W=32 with current
buffer depths (probe_span_w_sweep_compile.py); "more rows per call" remains open and untested.

## Phase 1e: trace-only re-bisection -- CORRECTS Phase 1a/1c: the pace jump is the JOIN, not block 1

**Correction to the two prior chain-length bisections in this file (Phase 1a follow-up 2 and its
Phase 1c re-run).** Both concluded "the pace is already essentially set within block 1's first 4
cores" from `probe_span_upto_bisect.py`'s WALL-CLOCK fitted-slope numbers (663/572 cyc/px at
`upto=b1c3`, within a few percent of the full net). That conclusion is WRONG. Phase 1d's
same-process TRACE of the identical `b1c3` prefix (`probe_span_w_sweep_trace.py`, W=32) measured
**326.0-326.4 cyc/px** -- a ~45-50% disagreement with the wall-clock bisection's own number for the
same prefix under the same conditions. Per this file's own standing caveat ("treat compute-vs-gap
SPLITS as trustworthy, absolute cyc/px versus the untraced net-rate probe as approximate"), the
wall-clock bisection's absolute numbers were never trustworthy; they were nonetheless read as
attribution evidence for WHERE the pace is set, which the caveat never licensed. Re-run below with
the trusted instrument only.

**Method** (`aie_kernels/_test/trace_span_upto_bisect.py`, new): for each `SPAN_UPTO` in {b1c3,
b2c3, b3c3, b4c3, b5c3, b6c3, conv_2, conv_cat, up}, same-process trace of `b1c2` (present at every
prefix from b1c3 onward -- a fixed cross-prefix probe) plus the prefix's own last stage, one stage
per dispatch, W=32, H=128. `SPAN_UPTO` below `conv_cat` builds no join at all (`net_design.build()`
only wires `f_cat` when `"conv_cat"` is in the truncated stage list), so b1c3..conv_2 isolate the
main chain alone; conv_cat/up add the join and the `up` tail. Power mode: `default` (UNPINNED),
standing caveat.

| upto | cores | join built? | b1c2 compute+gap | b1c2 LOCK% | last stage | last compute+gap | last LOCK% |
|---|---|---|---|---|---|---|---|
| b1c3 | 4 | no | 325.9 | 1.88% | b1c3 | 326.2 | 5.8% |
| b2c3 | 7 | no | 325.9 | 1.93% | b2c3 | 326.0 | 9.67% |
| b3c3 | 10 | no | 325.9 | 1.99% | b3c3 | 325.8 | 12.97% |
| b4c3 | 13 | no | 325.9 | 1.92% | b4c3 | 325.6 | 16.49% |
| b5c3 | 16 | no | 325.9 | 1.91% | b5c3 | 325.5 | 19.3% |
| b6c3 | 19 | no | 325.9 | 2.03% | b6c3 | 325.6 | 22.06% |
| conv_2 | 20 | no | 325.9 | 2.06% | conv_2 | 323.3 | 79.06% |
| **conv_cat** | **21** | **YES** | **483.8** | **33.19%** | conv_cat | 481.3 | 89.47% |
| up | 22 | yes | 483.8 | 33.22% | up | 478.7 | 94.93% |

**The pace steps from ~326 to ~483.8 cyc/px in exactly one place: between `conv_2` (20 cores, no
join) and `conv_cat` (21 cores, +join).** Every core-only prefix (4 through 20 cores) sits at
325.5-326.2 regardless of chain length -- 16 additional main-path cores (b2c3 through conv_2) buy
NOTHING, refuting "block 1 sets the pace" outright, not just its magnitude. Block 1 (and the whole
main chain) is compute-dominated and near its own rate (LOCK_STALL 1.9-2.1% on the fixed b1c2 probe
throughout); the throttle is not distributed across the chain length, it is introduced by ONE
structural element: whatever the join adds. `b1c2`'s own LOCK_STALL rises in lockstep with the
prefix's OWN last-stage LOCK_STALL as the prefix lengthens through the no-join range (1.88% at
b1c3's own last stage 5.8% -> conv_2's own last stage 79.06%) -- consistent with ordinary pipeline
fill/drain inside a fixed H=128 dispatch (per Phase 1a follow-up's correction), not with a
per-hop throttle; b1c2's own number stays flat at ~325.9 throughout, which is the signal that
matters here.

**What the join adds, from the compiled MLIR (`aie.objectfifo @cat_in`, `aie_kernels/_test/gen`):**
`cat_in` is built with `iterate_bds=True` and is the ONLY objectFifo in the entire 22-core network
built that way (grep confirms one hit). Per the commit that added it (282feac, "conv_cat joins via
iterate_bds"): the join's per-object BD lowering needs **182 MemTile BDs** against a **48-BD**
hard limit on that MemTile; `iterate_bds` replaces that with **8 BDs** by lowering to one
self-looping BD per (channel, segment) instead of one static BD per repeat -- a MOVEMENT-layer
mechanism switch, not merely a bigger buffer. Every ordinary main-path hop (`windowed()`,
`depth=main_depth=4`) and the skip-broadcast producers (`PROD_DEPTH=2`) still lower as static
per-slot BD chains; `cat_in` is qualitatively different DMA machinery, forced by the 48-BD cap,
not a policy choice this file's existing depth/slack knobs (MAIN_DEPTH, PROD_DEPTH, SKIP_SLACK,
cat_cons_depth -- all previously refuted at b3c3 in Phase 1a) ever touched.

**Targeted experiment: pin the join off the MemTile it shares with weight group 3.** Phase 1a's
follow-up found `cat_in`'s default placement (`join(tile=AnyMemTile)`, unconstrained) lands on the
SAME MemTile as weight group 3 -- 10/12 DMA channels used there vs. 7/12 on every other
weight-feed MemTile -- and flagged this as an untested "shared resource" candidate. Added
`net_design.build(join_tile=...)` (new param, default `None` -> `AnyMemTile`, byte-identical
placement when omitted) so the join can be pinned to a specific `Tile`. Compile-only check
(`aie_kernels/_test/compile_span_join_tile.py`) confirmed `Tile(1,1)` and `Tile(5,1)` (two MemTile
columns no weight group uses) both fit L1/BD budget cleanly, same as the default.

**Device: same-process trace of `conv_cat` and `b1c2`, full net (`upto=up`), W=32, H=128, one stage
per dispatch, baseline vs. both candidate tiles** (`aie_kernels/_test/trace_span_join_tile.py`).
Power mode: `default` (UNPINNED), standing caveat.

| join tile | conv_cat compute+gap | conv_cat LOCK% | b1c2 compute+gap | b1c2 LOCK% |
|---|---|---|---|---|
| baseline (AnyMemTile, lands on shared (4,1)) | 481.3 | 89.39% | 483.8 | 33.21% |
| Tile(1,1) (dedicated) | 481.4 | 89.4% | 483.9 | 33.26% |
| Tile(5,1) (dedicated) | 484.6 | 89.41% | 483.8 | 33.25% |

**NULL -- all three land within trace noise of each other (481.3-484.6, LOCK_STALL 89.4% flat).**
Moving the join off the shared MemTile, onto a tile with no weight-group DMA traffic at all, does
not move the pace. This refutes the specific "MemTile(4,1) channel-sharing" mechanism Phase 1a's
follow-up flagged as the concrete "shared resource" candidate -- the join's cost is not contention
for that MemTile's 12 DMA channels with weight group 3.

**Net: the pace-jump is now localized to the join precisely (conv_2 -> conv_cat, not "block 1" and
not "somewhere downstream"), but the MECHANISM within the join is still open.** MemTile
channel-sharing is refuted (this experiment). The remaining, MLIR-grounded candidate this phase's
budget did not reach is the `iterate_bds` self-looping-BD mechanism itself -- whether a
self-looping BD carries a per-iteration reload/resync cost that a static per-slot BD chain does
not, independent of which MemTile it sits on. Testing that needs an isolated join (few enough
skip sources to fit the 48-BD cap WITHOUT `iterate_bds`) traced against an equivalent
`iterate_bds`-lowered one at the same depth -- not attempted here (one experiment was this
phase's budget); flagged as the concrete next step rather than re-guessed.

**Gate: `verify_span_net.py`, default config (no `join_tile` override) -- all 8 checkpoints exact**
(conv_1 through up, 98304/98304 or 32768/32768 per stage) -- confirms the new opt-in `join_tile=`
parameter is behavior-preserving when omitted, as intended; no default was flipped.

## Caveat

Device is shared with other concurrent lanes (gemma4 prefill gates, this session's own
`wt-span-frame` full-frame lane) for the whole measurement window -- each dispatch waited through
multiple `npu_lock.sh` defer/retry cycles. No control for box thermal/DPM drift across dispatches
(power mode not queried per `npu_power_mode.py`); treat compute-vs-gap SPLITS (same-dispatch ratios)
as trustworthy, absolute cyc/px versus the untraced net-rate probe as approximate.

## Phase 1f: trace-confirm the iterate_bds mechanism itself -- REFUTED, join tax is NOT iterate_bds

Phase 1e's open item: the join step (+157.9 cyc/px, conv_2->conv_cat) is confirmed structural and
NOT MemTile(4,1) channel sharing (join_tile experiment, null), leaving the `iterate_bds`
self-looping-BD mechanism itself (vs. a static per-slot BD chain) as the one untested candidate.
`bench_iterate_bds.py`'s wall-clock A/B is this file's OWN flagged-unreliable instrument on this
box (single-source cases flipped sign) -- re-tested here with the trusted same-process TRACE
instrument instead.

Method: `aie_kernels/_test/trace_iterate_bds.py`, new. Same isolated-join shape as
`bench_iterate_bds.py` (NSRC producers -> one MemTile join pool -> 1 consumer, SPAN's own join
row size 2304 B), but the consumer's per-row touch is routed through a tiny ExternalFunction
kernel (`touch.cc`: `o[0]=l0[0]` bracketed by `event0()`/`event1()`) so it can carry a hardware
trace -- pure-Python IRON worker bodies have no event0()/event1() binding (only ExternalFunction
kernel bodies do, same as net_design.py's `_shim(bracket=True)` convention). Traced the consumer
core only, NSRC=4, depth in {4, 8}, ON/OFF alternated per repeat (3 repeats), W=32-equivalent
row=2304B, H=128. Power mode: `default` (UNPINNED), standing caveat.

Compile-only check first (`compile_trace_iterate_bds.py`, no device): depth=4 compiles both ways;
depth=8 compiles ON only (OFF needs 64 BDs > 48-BD cap, `'aie.memtile_dma' op has more than 48
blocks` -- confirms the cap arithmetic from the original commit, not just at NSRC=4/depth=22).

| nsrc | row | depth | iterate_bds | compute+gap cyc/row (median, n=3) | stall_union % of span |
|---|---|---|---|---|---|
| 4 | 2304B | 4 | ON | 1144.2 | 97.15-97.18% |
| 4 | 2304B | 4 | OFF | 1144.0 | 97.15-97.18% |
| 4 | 2304B | 8 | ON | 1144.0 | 97.15-97.16% |
| 4 | 2304B | 8 | OFF | compile FAILS (48-BD cap, expected) | -- |

**REFUTED -- ON and OFF land within 0.2 cyc/row of each other at depth=4 (both ~1144.0-1144.2,
run-to-run noise here is ~3 cyc/row), and depth=8 ON matches depth=4 ON exactly (1144.0 vs
1144.2/1144.0).** A real ~158 cyc/px tax (33% of SPAN's ~484 cyc/px join-step floor) would be
~35x this run's noise floor and would show up unmistakably; it does not. The self-looping-BD
lowering itself carries no measurable per-row cost over the static per-slot BD chain, at NSRC=4,
either depth tested, isolated from SPAN's own MemTile placement, weight-group sharing, and
kernel compute.

**Net: the join's `iterate_bds` MECHANISM is now also refuted, on top of MemTile channel sharing
(Phase 1e).** Both concrete candidates this file named for the join-step tax are dead ends. The
tax is real and localized (Phase 1e's bisection is not in question -- conv_2->conv_cat is still
where the whole-net pace jumps ~326->483.8 cyc/px) but its MECHANISM remains open: it is
something about the join's shape in situ (4 real producer cores each running SPAN's own kernel
work and objectFifo topology, not this isolated microbench's trivial producers) that this
isolated harness does not reproduce, or something outside the MemTile-DMA layer entirely (e.g.
stream-switch routing/arbitration across the 4 real producer cores' distinct source tiles, which
this microbench's producers do not share with SPAN's real skip-source placement). **No fork fix
is warranted**: the task's premise (iterate_bds itself carries a per-row tax) does not hold under
the trusted instrument, so there is nothing in the MemTile objectFifo lowering to change.
