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

## Cycles/pixel (measured, marginal two-size method, `kernel/time_warp.py`)

Tile/halo shrunk to 16x{8,16}, halo=8 for the *timing* kernel too, to fit a C=4 buffer in
L1 at objectFIFO depth 2 (32x8/halo=16 as in the design write-up overflows for C=4, same as
verify). Halo size does not change the kernel's per-pixel instruction count (clamp bounds are
just different constants), so this does not bias the ns/px result -- only the L1 footprint.

| C | tile 16x8 | tile 16x16 | marginal ns/px | cycles/px @1.8GHz |
|---|---|---|---|---|
| 3, f32 | 3225.6 ns/px | 2052.4 ns/px | **879.2** | 1583 |
| 4, f32 | 3405.0 ns/px | 2452.1 ns/px | **1499.3** | 2699 |

Box was on the shared single-tenant NPU, `fuser`-checked free before and after each run
(nothing else was holding `/dev/accel/accel0`); canonical clock 1.8 GHz per
`decode-perop-aie-clock` (the AIE core clock is a DPM ladder value, not a fixed constant --
power profile not independently re-pinned this pass, taken as the box's current default).

bf16/int8 were **not device-measured** this pass. Reasoning for deferral, not an oversight:
the bottleneck here is 4 scalar L1 loads/pixel/channel plus branchy clamp arithmetic, not
arithmetic throughput or bytes moved -- a narrower storage format halves L1 bytes touched but
does not cut the load-instruction count, and Practical-RIFE's own int8 feasibility pass
(`rife-sizing.md`) only quantized weights, leaving the warp/flow activation path explicitly
untested and likely to need per-op accuracy work before it is worth device time here.

## Projection: 18 calls/frame, 1920x1152, this scalar kernel

10 calls at C=3, 8 at C=4, all at full padded resolution (2,211,840 px/call, per the sizing
doc):

| | per call (1 core) | 18 calls (1 core) | 18 calls (ideal 32 cores) |
|---|---|---|---|
| C=3 (x10) | 1.944 s | 19.44 s | |
| C=4 (x8) | 3.316 s | 26.53 s | |
| **total** | | **45.96 s/frame** | **1.436 s/frame** |

60fps needs 16.6 ms/frame. Even ideal, contention-free 32-core scaling is **~86x over
budget** -- this scalar kernel is not close, in the same way FSR1's first scalar EASU+RCAS
kernel was 3-4 orders of magnitude off before vectorization (`designs/fsr1/README.md` step 2).

## Read: what this means for the design, not just the kernel

Unlike FSR1's EASU, where a fixed upscale ratio gives every output pixel in a phase group the
*same* tap offset (so `aie_api` vector loads share addresses across lanes), RIFE's warp has an
**independent flow value per pixel** -- there is no shared address pattern to vectorize the
gather itself onto, on this hardware, at any format. Vectorizing the *arithmetic* (weight
computation, the final FMA) would help some, but the measured cost here is dominated by 4
scalar-indexed loads x C channels per pixel plus per-pixel floor/clamp branching, not by flops
-- the same movement-bound diagnosis the sizing doc predicted, now with a number attached
(~900-1500 ns/px, vs a ~11.4 ns/px budget implied by 16.6ms/2.2Mpx at C=3 ideal-32-core, an
~80-130x gap that a single format or vectorization lever will not close).

**Recommendation:** keep the warp on the iGPU (a real hardware bilinear sampler, effectively
free relative to the frame budget) and run only the conv/deconv-dominated CNN (94% of MACs,
16-37x arithmetic headroom per the sizing doc) on the NPU. This is a genuine SILICON gap, not
an unwritten-kernel TOOLCHAIN gap: AIE2P has no unit that does data-dependent 2-D addressing at
any granularity finer than a fixed DMA descriptor, and the measured scalar-gather cost here is
the honest price of emulating one in software. A multi-core row-split would divide the 45.96s
total by up to 32 (already reflected above) and still miss real-time by ~two orders of
magnitude; int8 gather would shrink bytes moved but not the load-instruction count that
dominates the measured cost. Neither closes the gap on its own.

## Files
- `cpu_ref.py` -- numpy reference, exact-border (`warp_bilinear_border`, checked against torch
  grid_sample) and tile-halo (`warp_tile_halo`, what the kernel computes).
- `kernel/warp_kernel.cc` -- scalar warp kernel, compile-time tile/halo/channel count.
- `kernel/verify_warp.py` -- device correctness gate (bricklib, PASS for C=3 and C=4).
- `kernel/time_warp.py` -- device cycles/pixel measurement, marginal two-size method.
- `kernel/run.sh` -- npu_lock-wrapped runner (copy of `designs/fsr1/kernel/run.sh`).
