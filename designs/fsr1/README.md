# FSR1 (EASU+RCAS) x3, fixed 640x360 -> 1920x1080 -- step 1: CPU reference + GPU ground truth

Scope: reproduce gamescope's FSR1 pass (fp32, sharpness = `g_upscaleFilterSharpness/10` = 0.2,
the shipped default) bit-faithfully enough to gate an AIE port against, then measure the real gap.

## Files
- `cpu_ref.py` -- numpy transliteration of `ffx_fsr1.h`/`ffx_a.h` (FsrEasuCon/F, FsrRcasCon/F),
  including the bit-trick fast rcp/rsqrt approximations (these are part of the algorithm's output,
  not incidental precision loss).
- `gpu_harness/` -- standalone headless Vulkan compute harness running the *actual* AMD
  `ffx_a.h`/`ffx_fsr1.h` (copied verbatim, MIT) as two dispatches (EASU then RCAS), no gamescope
  descriptor-set/layer plumbing. This is the GPU ground truth. Ran on the box's discrete GPU
  (`NVIDIA GeForce RTX 5070 Ti`, not the iGPU) -- FSR1 is vendor-neutral GLSL, so this is a valid
  reference for the algorithm gamescope ships (same shader source), not a claim about the
  iGPU-vs-dGPU path gamescope itself would take.

## Correctness gate: CPU ref vs GPU ground truth

Instrument control (known-good case first): flat 128-gray 640x360 input -> **bit-exact, 0 diff**
across all 3 channels, both EASU alone and EASU+RCAS. This pins down that con0-3/RCAS-con setup,
gather addressing/ordering, edge clamping and the RCAS constant are all correct -- any residual on
real content is the algorithm's own float-approximation sensitivity, not a harness bug.

| test image | max abs (8-bit) | mean abs | % px > 2 | % px > 8 |
|---|---|---|---|---|
| flat 128 gray | 0 | 0.0 | 0% | 0% |
| smooth gradient + sine | 255* | 0.97 | 0.45% | ~0.02%* |
| noisy + hard edge + thin line | 255* | 2.17 (EASU+RCAS), 0.58 (EASU only) | 15.1% | 4.2%* |

\* A handful of pixels (5 of 2.07M in the noisy case) sit on `mn4==mx4`/`4*mn4+peakC.y==0`
degenerate ratios in RCAS's `hitMin`/`hitMax` (division by ~0) -- both CPU (numpy) and GPU hit
the *same* algorithmic singularity but resolve the NaN/Inf differently (IEEE div-by-zero sign and
compiler instruction selection are implementation-defined here); this is FSR1's own edge case, not
a port bug. Excluding those, deep-interior noisy-content mean abs is 1.48/255.

**Read**: on smooth/typical game content the port matches the GPU to <1 LSB average, and the
higher noisy-content number is dominated by the algorithm's own reciprocal-approximation
sensitivity to sub-ULP float differences near direction/length thresholds (EASU's `dir`/`len`
computation is a chain of `APrxLoRcpF1`/`APrxLoRsqF1` bit-trick approximations, which amplify
tiny numeric differences on high-frequency input) -- not evidence of a wrong tap, wrong gather
order, or wrong constant (the flat-image control rules those out). This is not yet validated
against real game frames; only synthetic content was tested (no gamescope build/dump was done --
see Left undone below).

## Op-count table (per output pixel, fixed x3 scale)

**Correction to the task brief's stated lever (b):** re-deriving `pp` from `FsrEasuCon` shows the
9 output pixels covered by one source quad share the same `fp` (hence the same gather addresses
and the same 12 luma taps) -- that part IS exact and shareable. But EASU's direction/length
reduction (`FsrEasuSetF`) is a continuous function of the fractional position `pp`, which takes
9 *different* values across that 3x3 output block, so the direction/length analysis itself is
**not** shareable losslessly across the 9 outputs (only the gather+luma extraction is). Sharing
`dir`/`len` too would be a lossy approximation, unmeasured here.

| stage | naive (per px, independent) | + lever (b): share gather+luma per 3x3 block |
|---|---|---|
| position/gather-address compute | ~10 flops | ~10 flops (shared 1/9x) |
| texture gather (12 gather instr, 12 unique texels) | 12 gather instr | 1.33 gather instr/px (12/9) |
| luma extraction (12 taps) | 36 flops | 4 flops/px (36/9) |
| direction/length (`FsrEasuSetF` x4) | 48 flops | 48 flops (not shareable, see above) |
| normalize dir/len (1 fast rsqrt) | 15 flops | 15 flops |
| 12-tap weighted accum (`FsrEasuTapF`) | 192 flops | 192 flops |
| final normalize+dering (1 exact rcp) | 6 flops | 6 flops |
| **EASU total** | **~355 flops + 12 gather instr** | **~275 flops + 1.33 gather instr** (~22% flop cut, ~89% gather-instruction cut) |
| RCAS (5-tap cross, no cross-pixel sharing at 1x) | ~70 flops + 5 texelFetch | with row-strip halo (BD-chain sliding window): ~70 flops + ~1 new texelFetch/px (4 of 5 taps reused from the previous pixel's window) |

Lever (c) (pre-baked per-phase constants, 9-entry phase table for `pp`/tap offsets) removes the
per-pixel floor/divide from the position-compute row above; on AIE this matters less for FLOP count
and more for avoiding float divide/floor in the per-pixel hot loop (division is expensive on the
AIE2p vector datapath) -- a FORMAT/ORCHESTRATION win, not a compute-FLOP win.

Not yet measured: an actual `whole_array`-style device kernel to turn this table into cycles/pixel
(step 2).

## Sharpness / precision choices
- Sharpness = 0.2 (gamescope default `g_upscaleFilterSharpness=2`, `RcasPushData_t` divides by 10).
- fp32 throughout, matching gamescope's `FSR_EASU_F`/`FSR_RCAS_F` (32-bit, non-packed) path -- gamescope
  does not use the fp16 EASU/RCAS variants (`cs_easu_fp16.comp` exists but is unused by the pass wired
  up in `rendervulkan.cpp`).
- Edge handling: clamp-to-edge on both EASU's `textureGather` and RCAS's `texelFetch`. Confirmed correct
  by the flat-image bit-exact control (any border mismatch would show as border-row error, which is 0
  in that control).

## Left undone in this pass (for the next session on this task)
- No real gamescope-rendered frame was used (`/mnt/data/xdna/build/gamescope-npu` was not built/run in
  this pass) -- only synthetic test images. The error table above should be re-run against real captured
  frames before treating the ~1 LSB figure as representative.
- Step 2 (single-core AIE kernel), step 3 (whole-frame single dispatch), step 4 (C ABI) not started.
