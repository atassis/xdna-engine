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

## Caveat

Device is shared with other concurrent lanes (gemma4 prefill gates, this session's own
`wt-span-frame` full-frame lane) for the whole measurement window -- each dispatch waited through
multiple `npu_lock.sh` defer/retry cycles. No control for box thermal/DPM drift across dispatches
(power mode not queried per `npu_power_mode.py`); treat compute-vs-gap SPLITS (same-dispatch ratios)
as trustworthy, absolute cyc/px versus the untraced net-rate probe as approximate.
