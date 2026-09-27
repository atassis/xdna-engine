# RIFE backward warp (bilinear grid_sample) on one AIE2P core -- step 2 prototype

Scope: `npu-rife-frame-generation` step 2 -- price the 18 full-res `grid_sample` calls/frame
(no sampler on AIE2P) that step 1 (`rife-sizing.md`) flagged as a confirmed movement-layer gap.

## Semantics (confirmed against Practical-RIFE, not assumed)

`model/warplayer.py::warp` (code at `/mnt/data/xdna/models/rife/code`): bilinear,
`padding_mode='border'`, `align_corners=True`, per-pixel flow offset in normalized coords,
converted internally to pixel-space. `cpu_ref.py::warp_bilinear_border` is a numpy
transliteration checked directly against `torch.nn.functional.grid_sample` on random data
(`/mnt/data/xdna/venvs/rife-sizing`): **rel_l2 9.0e-7, max abs diff 4.8e-6** -- float32-ULP
match, not an approximation.

## Flow magnitude distribution (measured, not assumed)

Ran the real 4.25.lite model's full flow pyramid (`scale_list=[32,16,8,4,1]`) on 39 triplets
from a public-domain 720p clip (Big Buck Bunny sample), frames scaled to 1920x1080 and padded
to 1920x1152 (the sizing doc's canvas), 172.5M flow-pixel samples total:

| percentile | \|flow\| (px) |
|---|---|
| p50 | 0.285 |
| p90 | 2.538 |
| p95 | 4.568 |
| p99 | 15.32 |
| p99.9 | 39.55 |
| p99.99 | 59.80 |
| max | 66.31 |

| halo | fraction of pixels exceeding it (fallback needed) |
|---|---|
| 8px | 2.68% |
| 16px | 0.93% |
| 24px | 0.26% |
| 32px | 0.15% |
| 64px | 0.0013% |

Caveat: this clip is a slow camera pan with limited motion; a game with fast camera turns or
sprite motion would push more mass into the tail. **halo=16px** (0.93% fallback) is the design
point used below; the fallback is border-clamp to the halo edge, the same operation the kernel
already does at the tile boundary, just with a different (nearer) clamp radius than the true
image edge -- not a special code path.

## Design (per the movement-brick gap, no data-dependent DMA addressing)

DMA brings one output TILE **plus a fixed HALO** into L1 as a static window (ordinary
strided/relayout DMA, ADDRESS-independent of the flow field). The core then does a per-pixel
**scalar gather**: computes integer base + fractional weight from the flow, and does 4 taps x
C channels of plain pointer-indexed loads from the already-resident L1 buffer. This is not a
DMA gather (which AIE2P cannot do at this granularity) -- it is ordinary scalar memory access
inside the core, unrestricted by DMA descriptor rules. `kernel/warp_kernel.cc`.

**Gather strategy: scalar loads, not vector shuffle/parallel_lookup.** `parallel_lookup` is a
<=32-lane 1-D table gather with a shared index register across lanes; here every output pixel
has an independent 2-D flow offset, so there is no shared address pattern across lanes to feed
it (unlike FSR1's EASU, where a fixed x3 phase table makes every lane's tap offset the same
constant -- see `designs/fsr1/kernel/fsr1_kernel_vec.cc`'s `load_tap`). A vector table lookup
buys nothing when the table index itself varies per lane in 2-D with no structure. Scalar
per-pixel loads are therefore the correct choice here, not a fallback taken for lack of time.

Two-input-DMA-channel limit forced fx/fy into one packed `flow` buffer (`[fx|fy]`
concatenated) -- a core tile has only 2 input objectFIFO channels, and `in_padded` + `fx` +
`fy` as three separate inputs hit `"tile requires 3 input DMA channels, only 2 available"`.

## Correctness: device-verified, PASS

`kernel/verify_warp.py` (bricklib `verify_oneshot` rail, real XRT run on aie2p,
npu_lock-serialized), reduced tile (16x4, halo=8 -- a C=4 crop at the full 32x8/halo=16 design
size overflows one core's 64KB L1 at objectFIFO depth 2; same kernel, same numerics, per
`method-aie2p-device-test-build.md`'s "verify a big brick at a reduced shape"):

| C | rel_l2 (note) | max abs diff | determinism (4 runs) | status |
|---|---|---|---|---|
| 3 | 9.19e-07 | 2.89e-06 | 0.0 | PASS |
| 4 | 1.06e-06 | 2.47e-06 | 0.0 | PASS |

Host-compiled (native g++) sanity check against `cpu_ref.warp_tile_halo` at the full design
size (32x8 tile, halo=16, C=4) ran first: rel_l2 1.8e-6, matching the device numbers -- the
device run is not a fluke of the smaller crop.

## Cycles/pixel, scalar kernel: the marginal two-tile-size method OVERSTATED (superseded)

`time_warp.py`'s marginal two-tile-size method (16x8 vs 16x16, 128px apart) gave 879.2/1499.3
ns/px. **This instrument does not reproduce** -- see the repeat-count re-measurement below,
which reads 1349/1728 ns/px on the same kernel, +54%/+15% higher. The 128px gap was too small
relative to host-dispatch jitter (~300-400us, run-to-run spread ~100+us between wall_min and
wall_med) for a two-point marginal to be trustworthy; it happened to look internally consistent
(both points agreed to within a few %) but that was not sufficient evidence, only necessary.
**Do not use `time_warp.py`'s numbers; they are kept in git history for the record, not as a
citable result.**

## Was the scalar cost gathering or soft-float? Pushback, and the fix (load-bearing)

The scalar numbers above were originally read as pricing the gather. Pushback: AIE2P has **no
scalar hardware FPU** -- plain `float` arithmetic on the scalar core is software-emulated (the
same reason FSR1's own scalar kernel measured ~25476 ns/px and its vectorized EASU 16x less on
identical hardware, `designs/fsr1/README.md`). So the scalar kernel's cycles/px price
soft-float library calls, not the gather. `kernel/warp_kernel.cc` is kept as the reference
point; `kernel/warp_kernel_vec.cc` is the fix -- every op except the 4 taps/channel/pixel moved
onto `aie::vector<T,16>` (native hardware ALU): position math, floor, clamp, and the bilinear
lerp (after taps are gathered into a buffer and vector-loaded). Two format choices:

- **Flow: fixed-point int16, Q8.7** (7 fractional bits). Measured max flow is 66.3px (p99.99
  59.8px, see above); Q8.7 covers +/-255.99px, >4x headroom. Floor is `aie::downshift`
  (arithmetic shift, exact for two's-complement) rather than `aie::to_fixed<float>`, which
  rounds-to-nearest on this target per its own doc comment in `aie_api-fork/include/aie_api/
  aie.hpp` -- a float floor would need its own correction step; the fixed-point shift doesn't.
- **Pixel data: bf16.** Keeps full float dynamic range; int8 was not chosen because
  Practical-RIFE's own int8 pass (`rife-sizing.md`) only quantized weights, leaving the
  warp/flow activation path's accuracy untested.

Two real `aie_api` misuses were caught by reading the fork's header before running, not by
device trial-and-error: `cast_to<T>` reinterprets bits at a FIXED TOTAL WIDTH (would have
shuffled pairs of int16 into garbage int32 on the flow widen) -- fixed to `aie::unpack`, the
documented value-preserving sign-extending widen; and `aie::to_float(v, shift)` divides by
2^shift directly (used for the Q8.7 fractional weight), not `cast_to<float>`, which would have
bit-reinterpreted the fixed-point integer as a float.

## Device-verified, vectorized kernel: PASS

`kernel/verify_warp_vec.py`, same rail, gated against a bf16/Q8.7-quantized golden
(`cpu_ref.warp_tile_halo_vec`) that mirrors the kernel's own integer arithmetic exactly:

| C | rel_l2 vs quantized golden (note) | rel_l2 vs fp32 reference (note, format cost) | determinism | status |
|---|---|---|---|---|
| 3 | 3.875e-03 | 6.451e-03 | 0.0 | PASS |
| 4 | 4.036e-03 | 5.954e-03 | 0.0 | PASS |

**The ~4e-3 gap vs the quantized golden was hypothesized as an output bf16-narrow
rounding-mode mismatch (device default vs. numpy round-to-nearest) and that hypothesis is
REFUTED, recorded not chased further.** Pinned `conv_even` rounding on the narrow
(`narrow_bf16`, matching `cast_f32_bf16.cc`'s idiom) and re-verified: rel_l2 unchanged to 4
significant figures. Confirmed this was a genuine recompile, not a stale cache serving the old
binary -- the post-fix ELF is a different `.text` size (3120 B vs 3008 B) and a different
bricklib design hash. So the fix compiled in and had negligible numeric effect; the real
source of the ~4e-3 gap is unidentified (most likely float32-vs-float64 precision in the lerp
itself, since the golden computes it in float64 -- unconfirmed, not chased further per scope).

## Cycles/pixel, repeat-count instrument (trusted result)

The marginal two-tile-size method breaks down when per-pixel cost is small relative to
dispatch jitter -- confirmed on the vectorized kernel, where a first attempt at 16x8 vs 16x16
returned a **negative** marginal for C=3 (-330 ns/px, pure noise). Fix, `kernel/
time_warp_reps.py`: bake a compile-time repeat count into the shim (`for(r<REPS) kernel(...)`),
so one host dispatch executes the kernel REPS times; `(T(K_hi)-T(K_lo))/(K_hi-K_lo)` isolates
device-side per-call time since only the compiled trip count differs between builds and host
dispatch overhead cancels. **Verified linear**, not just measured at two points: a 3rd K point
(4, 200, 800) gives slopes agreeing to within 0.1-1.7% pairwise (ratio 0.991-1.008), so the
instrument is trusted for both kernels, at tile 16x16, halo=8:

| kernel | C | ns/px | cycles/px @1.8GHz |
|---|---|---|---|
| scalar (control) | 3 | 1349.1 | 2428 |
| scalar (control) | 4 | 1727.7 | 3110 |
| vectorized | 3 | 192.8 | 347 |
| vectorized | 4 | 241.2 | 434 |

**Vectorizing everything but the gather buys ~7.0x (C=3) / ~7.2x (C=4).** The scalar control
did not reproduce the earlier (now-superseded) marginal-method numbers -- this instrument is
the one trusted going forward; the vec/scalar RATIO is the robust cross-check either way (both
old and new scalar readings give ~7x when compared to the vec numbers measured the same way).
Box on the shared single-tenant NPU, `fuser`-checked free before/after; canonical clock 1.8 GHz
per `decode-perop-aie-clock`; power profile not independently re-pinned, box's current default.

## Load-vs-math split (llvm-objdump, `-Oz`, Peano seed `llvm-aie-22.0.0.2026090801`)

`warp_kernel_vec`'s C=3 build is 437 disassembled instruction bundles. Classified by mnemonic
(grep-counted per bundle slot, not hand-verified per instruction): loads ~112, stores ~95,
vector arithmetic (`vadd.f`/`vmul.f`/`vsub.f`/`vconv`/`vsel`/`vshuffle`/`vsrs`) ~64, loop/address
control ~140, empty-slot `nop*` fillers ~137 (bundles not densely packed at `-Oz`).

**Splitting loads/stores by operand pattern (`[sp,...]` = stack spill vs. everything else =
gather/output): 92/112 loads (82%) and 88/95 stores (93%) are spill traffic, not the tap
gather.** Only ~20 loads and ~7 stores are real gather/kernel I/O. VW=16 with several live
int32/float vector temporaries (sx, sy, wx, wy, x0/x1/y0/y1 clamped, four taps x C channels)
does not fit in registers at `-Oz`.

**This kernel is spill-bound, and that is OURS, not SILICON** -- register pressure and bundle
packing are properties of this specific kernel's register allocation at this optimization
level, not a hardware limit. Next levers, not yet tried: smaller VW (8 instead of 16, halving
live-vector footprint), restructuring to keep fewer vectors live across the gather (e.g.
compute+store clamped indices before touching any tap, rather than interleaving), and `-O2`
instead of `-Oz` (trades code size for register allocation/bundle-packing quality -- the FSR1
scalar kernel needed `-Oz` just to FIT program memory, but the vectorized kernel here is far
under that budget and has not been checked against `-O2`).

## Projection: 18 calls/frame, 1920x1152 (10xC=3 + 8xC=4, 2,211,840 px/call)

Using the trusted (repeat-count) vectorized numbers:

| | per call (1 core) | 18 calls (1 core) | 18 calls (ideal 32 cores) |
|---|---|---|---|
| C=3 (x10) | 0.4264 s | 4.264 s | |
| C=4 (x8) | 0.5335 s | 4.268 s | |
| **total** | | **8.532 s/frame** | **0.267 s/frame** |

60fps needs 16.6 ms/frame. Ideal, contention-free 32-core scaling is **~16x over budget** --
much closer than the scalar kernel's ~86x, but still an order of magnitude short, and this is
the BEST case (zero multi-core overhead, zero contention from other tenants).

## Read: what this means for the design

Vectorizing everything but the gather bought ~7x, cutting the gap from ~86x to ~16x -- a real,
measured win, not a wash. But the kernel is now spill-bound (82%/93% of load/store traffic),
which is a register-allocation problem at this tile width, not a hardware ceiling: AIE2P's
vector float/int32 ALU is native and fast (proven by FSR1's own 16x scalar-to-vector jump and
reused here), the only genuinely hardware-forced scalar step is the 4 taps/channel/pixel gather
itself (no unit does data-dependent 2-D addressing at finer grain than a fixed DMA descriptor).
So the correct read is **OURS (spill/packing), stacked on top of a smaller genuine SILICON
floor (the gather), not a single monolithic SILICON wall** -- the ~86x-over verdict from the
scalar kernel was wrong to call SILICON, and this ~16x-over number should not be called SILICON
either until the OURS levers above (smaller VW, restructured live ranges, `-O2`) are tried and
the residual is re-measured.

**Recommendation, updated:** do not conclude "leave warp on the iGPU" from this pass -- that
conclusion was drawn from the scalar (soft-float-bound) number and is not supported by the
vectorized one. The honest state is: 7x won from format/vectorization work already done, ~16x
still open, with concrete untried levers (smaller VW, live-range restructuring, `-O2`) that
directly target the measured spill-bound cause. Revisit the iGPU-vs-NPU call after those levers
are tried and the residual gap is re-measured -- not before.

## Files
- `cpu_ref.py` -- numpy reference: exact-border (`warp_bilinear_border`, checked against torch
  grid_sample), tile-halo (`warp_tile_halo`, what the scalar kernel computes), and
  bf16/Q8.7-quantized (`warp_tile_halo_vec` + `quantize_flow_q87`, what the vec kernel computes).
- `kernel/warp_kernel.cc` -- scalar warp kernel (reference point, soft-float-bound).
- `kernel/warp_kernel_vec.cc` -- vectorized warp kernel (bf16 data, Q8.7 flow).
- `kernel/verify_warp.py`, `kernel/verify_warp_vec.py` -- device correctness gates (bricklib).
- `kernel/time_warp.py` -- superseded marginal two-tile-size timing (kept for the record, not
  a citable result -- see above).
- `kernel/time_warp_vec.py` -- superseded marginal two-tile-size timing for the vec kernel
  (also broke down; kept for the record).
- `kernel/time_warp_reps.py` -- the trusted repeat-count timing instrument, both kernels,
  3-point linearity check.
- `kernel/run.sh` -- npu_lock-wrapped runner (copy of `designs/fsr1/kernel/run.sh`).
