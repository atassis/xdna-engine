# int8-block-1: re-measuring the docstring's "~2 dB" claim

CPU-only, `span_int.py`'s own methodology (`main()`): calib = LR left half, test = LR right half,
Y-PSNR vs HR, border=2. Export `spanx2_ch48`, demo `xdna-engine/artifacts/edsr/demo` (768x510 HR).
Probe: `_probe_int8_block1.py` (subclasses `Span`; `span_int.py` itself unchanged).

## Reproduction of the shipped figures

```
bicubic 36.50  fp 40.69  integer program (int16 block1) 39.74  (vs fp 45.99)
```
Matches the task brief's 39.74/40.69 exactly (`span_int.py` line 107's scheme, unmodified).

## PSNR table -- same calib/test split for every row

| scheme | S["b1.c1.silu"] | PSNR (dB) | cost vs int16 | cost vs fp |
|---|---|---|---|---|
| fp (no quant) | -- | 40.69 | -- | -- |
| int16 (shipped) | amax/32767 | 39.74 | 0 (ref) | 0.95 |
| (a) int8, per-tensor | amax/127 | 38.12 | **-1.61** | -2.57 |
| (b) int8, per-channel (48) | amax_c/127, c=0..47 | **39.70** | **-0.03** | -0.99 |
| (c) int8, per-tensor, pct=99.5 | p99.5(\|x\|)/127 | 39.40 | -0.33 | -1.29 |
| (c) int8, per-tensor, pct=99.9 | p99.9(\|x\|)/127 | 38.42 | -1.32 | -2.27 |
| (c) int8, per-tensor, pct=99.99 | p99.99(\|x\|)/127 | 35.40 | -4.33 | -5.29 |

Docstring's "~2 dB" is the right order of magnitude for the naive per-tensor int8 swap (a): measured
**-1.61 dB**, not -2. (b) essentially retires the claim: per-channel int8 costs **0.03 dB**, i.e.
within noise of the shipped int16 scheme, at half the bit width. The single best percentile variant
tried, p99.5 (c), beats plain per-tensor int8 by +1.28 dB but still trails per-channel by 0.30 dB --
looser clipping (p99.9, p99.99) gets WORSE, not better, and p99.99 is worse than even unclipped
max-abs (a): clipping a rare outlier of an *intermediate feature map* (not the final image) feeds a
hard-saturated value into 5 more conv/LUT/gate stages, and whatever that costs outweighs the finer
step size the tighter scale buys everywhere else. Non-monotonic, measured as-is -- not smoothed to
a story.

(b)'s mechanism: `conv_params`'s `s_in = np.broadcast_to(...)` (`span_int.py:116`) already folds a
per-input-channel scale into the weight before the requant multiplier is computed, so b1.c2's conv
needs no code change to consume a 48-vector `s_in` -- confirmed by reading it, per the task's ask,
not assumed. What (b) DOES need that isn't free: the SiLU LUT itself must vary per channel (48
different 256-entry tables, since one shared table can't emit 48 different output scales from the
same int8 code) -- `_probe_int8_block1.py`'s `lut_perchan` does this on CPU via `np.take_along_axis`,
but the shipped kernel (`conv3x3_i8_lut`, one `Look` table for all `COUT` channels,
`conv3x3_u8.cc:11`) has no per-channel table input. Getting (b)'s number on hardware needs a new
kernel entry point, not a parameter change -- flagged, not built here.

## Kernel-kind change (b1c1/b1c2), if variant (a) or (c) were adopted

Variant (b) is CPU-only until a per-channel-table kernel exists (see above); (a)/(c) need none --
same call shape as blocks 2-6, already shipped.

| stage | today (int16 block1) | int8-block1 (a/c) | source |
|---|---|---|---|
| b1c1 | `conv3x3_i8_lut16` (kind `silu16`) | `conv3x3_i8_lut` (kind `silu_x`, == blocks 2-6's c1) | `net_design.py:57,59`; `net_layout.STAGES:17` |
| b1c2 | `conv3x3_i16i8_lut` (kind `silu_i16`, PA=int16) | `conv3x3_i8_lut` (kind `silu`, PA=int8, == blocks 2-6's c2) | `net_design.py:58,60`; `conv3x3_u8.cc:12` |

Byte-layout side effect (`net_layout.layout()`, `half(w)=h`): b1c1's `out_bytes` drops 3h -> 2h
(kind `silu16`->`silu_x`, `net_layout.py:56,58`); b1c2's `in_bytes` drops 3h -> 2h (kind
`silu_i16`->`silu`, `net_layout.py:57,59`) -- one fewer h-unit crossing the b1c1/b1c2 boundary per
row, on top of the compute change below. Not separately re-measured; derived from the layout table.

## Cycle implication -- measured, both directions

`TRACE_RESULTS.md` Phase 1i/1j, device re-trace, full net W=32 H=128, one stage per dispatch
(cited, not invented):

| stage | kind | compute (Phase 1i) | compute (Phase 1j) | compute+gap (Phase 1j) |
|---|---|---|---|---|
| b1c1 | silu16 | 266.4 | **213.08** | 244.7 |
| b1c2 | silu_i16 | **242.4** | **228.67** | 231.58 |
| b2c2 | silu (== b1c2's target kind) | 251.1 | **178.81** | 231.06 |

(`TRACE_RESULTS.md:1308,1424,1426,1427`.) The task brief's "242 / 179 cyc/px" is Phase 1i's b1c2
figure against Phase 1j's b2c2 figure -- real numbers, but two different phases. Same-phase (1j),
apples to apples: **228.67 -> 178.81 cyc/px, -21.8%** on b1c2's own compute if it switches from
`silu_i16` to `silu`. `silu_x` (b1c1's target kind) is bit-identical C code to `silu` -- both call
`conv3x3_i8_lut` (`net_design.py:59-60`) -- but its Phase-1j-era rate was never re-traced
(`TRACE_RESULTS.md:1321` says so explicitly); its pre-fix rate (250.4-251.5, line 703/338-350) sat
level with `silu`'s pre-fix rate (250.7-251.2), so ~179 cyc/px is the plausible post-fix number, not
a measured one for b1c1 specifically.

**Whole-net effect today is near zero, and that's measured too.** b1c1/b1c2's compute+gap already
sit at 244.7/231.58, within noise of every other traced kind's floor (215-231.5,
`TRACE_RESULTS.md:1424-1427`) -- Phase 1h's join fix made the net gap/lockstep-bound at a common
~231 cyc/px pace, not compute-bound on any one stage. Cutting b1c2's *compute* 228.67->178.81 buys
essentially nothing at the whole-net level (231.58->231.06, -0.2%) until that shared gap floor moves
too. Same caveat as everywhere else in this file: device shared, power mode not pinned.

## Read (not a decision)

- (a), the literal "make it int8" swap the docstring is warning about: costs 1.6 dB, not the full
  2 dB claimed, but a real and visible loss, and buys no cyc/px win by itself today (gap-bound).
- (b) is the interesting result: 0.03 dB, i.e. free, on this test image -- but "free" is contingent
  on a kernel that doesn't exist yet (per-channel LUT), so it prices an unbuilt capability, not a
  drop-in change.
- (c) plain percentile clipping is not a free lunch here: best case (p99.5) still gives up 0.33 dB
  against int16 and 0.30 dB against (b), for no code the hardware doesn't already have -- closer to
  (a) than to (b) in what it buys.
- Whichever variant, the cyc/px payoff on the *current* build is small (gap-bound, ~0.2% net), so the
  case for int8-block1 today is about PSNR/precision economy (half the bytes moving b1c1->b1c2,
  int8 vs int16 tables/regs), not about frame rate -- that would change if/when the gap floor itself
  drops below ~215-230 cyc/px and compute becomes the pace-setter again.
