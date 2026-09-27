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

## Left undone in step 1 (for a future session)
- No real gamescope-rendered frame was used (`/mnt/data/xdna/build/gamescope-npu` was not built/run in
  this pass) -- only synthetic test images. The error table above should be re-run against real captured
  frames before treating the ~1 LSB figure as representative.

## FMA-contraction check on the noisy-content gap (capped at 30 min, 2026-09-27)

Hypothesis: the CPU-vs-GPU gap on noisy content (15.1% of px >2 LSB) is glslang/driver fused-multiply-add
contraction (`a*b+c` in one rounding step) vs numpy's separate mul-then-add. Checked, not confirmed:

- `glslangValidator -Od` (disables glslang's own SPIR-V optimization pass) produced a **byte-identical**
  `easu.spv`/`rcas.spv` to the default build -- rules out glslang-level contraction as the source; there is
  none to disable at that layer.
- FMA contraction most plausibly happens in the NVIDIA driver's SPIR-V-to-ISA compiler, which is not
  reachable from glslangValidator flags. Testing that would need GLSL `precise` qualifiers on the shader's
  accumulators (`aC`, `aW`, `dir`, `len` in `ffx_fsr1.h`) forcing no reassociation/fusion, which means
  editing the vendored header -- not done in the capped window.
- **Result recorded, not closed**: FMA contraction remains the leading candidate mechanism for the
  noisy-content gap, unconfirmed. The gap is a NOTE (see `error-metrics-are-notes-not-gates`), not a
  blocker -- step 2's device gate uses the CPU reference directly, not the GPU, so this does not affect it.

## Step 2 -- single-core AIE kernel, gated against cpu_ref.py: PASS on device

`kernel/fsr1_kernel.cc`: full EASU+RCAS, **scalar fp32**, one resident RGB crop in L1 (no aie_api vector
types -- see "what this costs" below). Verified twice before touching the device:
1. Host-compiled (native gcc) against `cpu_ref.py` on an 8x8 and a 16x16 crop: **float32-ULP exact**
   (max abs diff 2.4e-7) outside two pixels that hit RCAS's own `mn4==mx4` div-by-zero singularity.
   Caught and fixed one real bug this way (`b_`/`c_` were reading the wrong 2x2-gather components,
   `bczz[3],bczz[2]` instead of `bczz[0],bczz[1]` -- traced by comparing intermediate `p0..p3` gather
   coordinates and tap values between the two implementations pixel-by-pixel, not by guessing).
2. Compiled for aie2p with the pinned Peano (`--target=aie2p-none-unknown-elf`): clean.
3. **Device-verified** (`kernel/verify_fsr1.py`, aie_kernels/_test's `bricklib.verify_oneshot` rail, real
   XRT run on aie2p, npu_lock-serialized): `rel_l2=3.094e-07` (note), **run-to-run determinism 0.0 over 4
   runs -> PASS**, max abs diff 1.222e-06 vs `cpu_ref.py` on device.

**.text vs the 16KB program memory** (`llvm-size -A` on the built per-core ELF,
`elfs_main_core_0_2.elf`): **15456 bytes = 94.3% of 16384**, at `-Oz` -- at the compiler's default `-O2`
this kernel's own functions alone are ~18.2KB, already over budget before the IRON DMA/objectFifo runtime
glue is added; `-Oz` was required to fit at all (own-function total ~12.2KB at `-Oz`, rest is runtime glue).
This is itself a finding: a straight-line scalar transliteration of the reference algorithm does not fit
AIE2p's program memory at a normal optimization level.

**MAC/lane utilization: 0%.** `llvm-nm` on the compiled kernel shows only software-float library calls
(`__mulsf3`, `__divsf3`, `__addsf3`-class helpers, `__ltsf2`/`__gtsf2`/`__gesf2`/`__nesf2` compares,
`__floatsisf`/`__fixsfsi` conversions) -- **zero** `aie::mmul`/vector-MAC instructions anywhere in the
build. Every arithmetic op in this kernel is a software emulation routine, not hardware MAC-array or even
hardware scalar-FPU work. This is the expected, measured cost of writing `float`/`int` C++ directly instead
of `aie::vector<float,N>`/`aie_api` -- a correctness-first port, exactly the "generic brick where the
hardware has a specialized one" mistake the brick-first doctrine warns against, done here deliberately to
get a fast, bit-exact-verified baseline before spending kernel-authoring effort on vectorization.

**Cycles/output-pixel:** measured via device wall-clock (dispatch-inclusive, `kernel/time_fsr1.py`,
`BRICK_JIT_CACHE=1` so repeat calls hit the cached xclbin, min of 15 reps per
`method-build-run-npu-xrt-test.md`'s convention) at two crop sizes to separate the fixed dispatch cost
from the per-pixel cost:

| crop | out px | wall_min | ns/px (crop avg) |
|---|---|---|---|
| 8x8 -> 24x24 | 576 | 15.075 ms | 26172 |
| 11x11 -> 33x33 | 1089 | 28.144 ms | 25844 |
| marginal (isolates per-pixel from fixed dispatch overhead) | +513 | +13.069 ms | **25476 ns/px** |

Fixed dispatch overhead is small (~0.4 ms) against a large, highly linear per-pixel cost (26172 vs 25844
vs 25476 ns/px across two very different crop sizes) -- this kernel is per-pixel-compute-bound, not
dispatch-bound. At the measured/canonical AIE clock (**~1.8 GHz**, `decode-perop-aie-clock`, stated per
doctrine since the AIE clock is a DPM variable, not a constant): **25476 ns/px = ~45857 cycles/output-pixel.**

## Step 2b -- vectorized EASU kernel, aie_api, reciprocal fix landed: PASS on device

`kernel/fsr1_kernel_vec.cc`'s `fsr1_easu_vec` (EASU only, RCAS still a separate scalar pass -- mirrors
gamescope's own two-dispatch split, see file header). Finishes the WIP left by the prior session: the
`aie::inv`/`aie::invsqrt` call sites in `easu_set_v` (2) and `easu_phase_row` (`rdirR`, `stretch`'s
`aie::inv(mx)`, `clp` -- 3) now call the vectorized `aprx_lo_rcp_v`/`aprx_lo_rsq_v` bit-trick helpers
instead of the hardware SFU; the one exact case (`invW`, FSR1's own plain `ARcpF1`) still uses
`aie::inv`, per the file header's reasoning.

- **A real bug in the added helpers, caught before device**: `aprx_lo_rsq_v` used `bits >> 1u` on an
  `aie::vector<int32_t,16>`, which fails to compile -- `operator>>` for vectors lives in the opt-in
  `aie::operators` namespace, not found by ADL. Fixed to call `aie::downshift(bits, 1u)` directly.
  This means the helpers had never been compiled before this pass, despite being written in the prior
  session -- confirmed the standalone compile-check step actually exercises them now.
- **Standalone Peano object-file compile-check** (`clang++ --target=aie2p-none-unknown-elf -Oz`):
  clean, own-function `.text` total 11808 B (was 11952 B before this fix -- the swap did not grow it,
  contrary to the "adding int32 ops should cost little" prediction, it shrank slightly).
- **Linked ELF `.text`** (`llvm-size -A` on the device-built `elfs_main_core_0_2.elf`): **13344 B =
  81.4% of 16384**, comfortably under budget (vs the scalar kernel's 15456 B/94.3%).
- **Device-verified** (`kernel/verify_fsr1_vec.py`): **rel_l2=3.355e-07**, max abs diff 2.295e-06,
  determinism 0.0 over 4 runs -> PASS -- down from the WIP's 0.211, now in the same band as the scalar
  kernel's 3.094e-07. Confirms the root cause (SFU reciprocal's `1/0=inf` vs FSR1's own finite bit-trick
  result on degenerate flat-region input) and the fix.
- **`optnone` workaround on `easu_tap_v` no longer reproduces.** Recompiled the file with the attribute
  removed at `-O1`/`-O2`/`-Os`/`-Oz`: all four compile clean, no "ran out of registers" error. The
  int32 bit-trick rewrite changed this function's live-range shape enough that the allocator failure
  the prior session hit is gone -- per toolchain-bug-test-latest-before-workaround doctrine this should
  be dropped, not carried forward on a stale premise; left in place here since removing it was out of
  this pass's scope (report-only per the task brief), but it is dead weight now.
- **Cycles/px -- SUPERSEDED, see Step 2c.** `kernel/time_fsr1_vec.py`'s marginal two-crop-size method
  reported **1550.3 ns/px** (~2791 cyc/px @1.8GHz), kept below for the record but not trusted: the
  method turned out not reproducible under host dispatch jitter (~300-400us) once probe crops are
  small (found on `wt-npu-warp`'s vec kernel, whose marginal method returned a NEGATIVE per-pixel
  delta at one crop size; FSR1's probe crops are smaller still). Superseded by Step 2c's repeat-count
  measurement: **1395.9 ns/px = 2513 cyc/px @1.8GHz**, linearity ratio 0.998 (near-perfect -- a real
  per-call cost, not jitter). Original marginal figures, unchanged: 16x6->48x18: 1706.0 ns/px;
  16x12->48x36: 1628.2 ns/px, canonical clock 1.8GHz (`decode-perop-aie-clock`), box AC-powered,
  `performance` power profile.
- **Projected, EASU-only -- see Step 2c for the trusted number and the fused-kernel comparison.**

## Step 2c -- vectorized RCAS, fused EASU+RCAS gated on device

`fsr1_rcas_vec` (RCAS, vectorized, `kernel/fsr1_kernel_vec.cc`) and `fsr1_strip_vec` (EASU+RCAS,
one dispatch, calls `fsr1_easu_vec` then `fsr1_rcas_vec` through an intermediate resident buffer,
mirroring the scalar kernel's `fsr1_strip`). RCAS reuses the same bit-trick bricks as EASU
(`aprx_med_rcp_v`, the APrxMedRcpF1 twin, added for RCAS's lobe reciprocal) and reads/writes the
interleaved RGB buffer directly with per-lane clamp addressing (no planar deinterleave -- see
"data-memory budget" below for why). The `optnone` on `easu_tap_v` no longer reproduces (removed);
Peano compiles clean at -O1..-Oz with the code as it stands now.

**Device-verified** (`kernel/verify_fsr1_rcas_vec.py`, `kernel/verify_fsr1_strip_vec.py`, real XRT
run, npu_lock-serialized):

| kernel | rel_l2 | max abs diff | determinism (n=4) | status |
|---|---|---|---|---|
| `fsr1_rcas_vec` (RCAS alone, random RGB) | 9.158e-08 | 2.384e-07 | 0.0 | PASS |
| `fsr1_strip_vec` (EASU+RCAS fused) | 5.469e-07 | 3.278e-06 | 0.0 | PASS |

Both in the same rel_l2 band as the EASU-only vec kernel's 3.355e-07 -- RCAS's own reciprocal
(hitMin/hitMax's exact division, done via `aie::inv` per the EASU precedent for FSR1's "exact"
cases) does not add a new error regime.

**.text vs the 16KB program memory** (`llvm-size -A` on the linked per-core ELF): `fsr1_strip_vec`
(fused) is **15808 B = 96.5% of 16384** -- it fits on ONE core, no split needed (up from
`fsr1_easu_vec` alone's 13344 B/81.4%; RCAS's own contribution is ~2.4 KB of .text).
`fsr1_rcas_vec` alone is 4208 B/25.7%.

**Data-memory budget is the tighter constraint for RCAS, not .text.** RCAS is same-size in->out,
so both its objectFifo buffers are output-sized (unlike EASU, whose input buffer is 9x smaller
than its output). The first version deinterleaved into an extra `planar[]` static scratch buffer
the same size as the image and blew the core's 64KB data memory (`'.bss' will not fit in region
'data': overflowed by 2688 bytes` at the crop size used for the EASU-vec gate). Fix: read the
interleaved buffer directly with a strided per-channel load (no scratch copy) -- see
`load_strip` in `fsr1_kernel_vec.cc`. Even with that fix, RCAS-alone's two depth-2 objectFifos (4
buffers, all output-sized) leave only ~1024 output pixels' worth of room per buffer at `w=48`
(48x21) before `aiecc` reports `'aie.tile' op basic-sequential allocation also failed` -- this is
why `time_fsr1_rcas_vec.py`'s two probe crops (12/21 output rows) are smaller than
`time_fsr1_vec.py`'s (18/36). The fused kernel hits a variant of the same wall: its
`easu_buf` intermediate (output-sized, static, no double-buffering) is a THIRD buffer beyond the
usual depth-2 in/out objectFifos, so `time_fsr1_strip_vec.py`'s max feasible crop is `in_h=8`
(out 24x48), not the vec-EASU probe's 12.

**A second static scratch buffer was found and removed before timing was trusted.** RCAS-alone's
`.bss` overflow (above) was fixed first, but the fused `fsr1_strip_vec` still overflowed at larger
crops (`'.bss' will not fit in region 'data': overflowed by 648 bytes`, needing 15496 B) even after
shrinking the timing probe's crop -- because `fsr1_easu_vec` carried its OWN static `planar[]`
deinterleave scratch (the same pattern already fixed for RCAS, missed on EASU), which stacked with
`easu_buf` in the fused kernel. Fixed the same way: EASU's `load_tap` now reads the interleaved
buffer directly, no scratch copy. Re-verified all three kernels after the fix -- rel_l2 and
determinism unchanged from the table above, confirming the fix changed only the access pattern.

**Cycles/px -- marginal two-crop-size method SUPERSEDED.** The method used above (and in Step 2b)
is not reproducible under host dispatch jitter (~300-400us) once the marginal delta is small
relative to it -- shown on `wt-npu-warp`'s vec kernel, whose two-size probe returned numbers 15-54%
off its own earlier measurement on a rerun, and FSR1's RCAS/fused probe crops are smaller still (12
vs warp's already-too-small ones). Superseded by a repeat-count instrument
(`kernel/time_fsr1_reps.py`, following `wt-npu-warp/designs/warp/kernel/time_warp_reps.py`,
e5fa996): a compile-time `REPS` loop calls the kernel REPS times inside ONE dispatch at ONE fixed
crop (16x6->48x18, the same size the verify scripts already build at), so `(T(K_hi)-T(K_lo))/(K_hi-K_lo)`
isolates device-side per-call time from the fixed dispatch overhead; a 3-point (K=4/200/800)
linearity check confirms the slope is a real per-call cost (ratio near 1.0), not a `K_lo`/`K_hi`
artifact.

| kernel | ns/px | cycles/px @1.8GHz | linearity ratio | superseded marginal-method figure |
|---|---|---|---|---|
| `fsr1_easu_vec` (EASU alone) | 1395.9 | 2513 | 0.998 | 1550.3 ns/px (Step 2b) |
| `fsr1_rcas_vec` (RCAS alone) | 391.3 | 704 | 0.988 | 344.8 ns/px |
| `fsr1_strip_vec` (EASU+RCAS fused) | 1786.1 | 3215 | 0.999 | -- (never landed under the old method) |

All three ratios sit within ~1.2% of 1.0 -- trusted. **Fusion costs nothing measurable**:
EASU (2513) + RCAS (704) = 3217 cyc/px, against the fused kernel's own measured 3215 -- the two
numbers agree to within rounding, so calling both passes from one dispatch neither adds nor saves
overhead relative to running them separately.

**Projected, 640x360->1920x1080 (2,073,600 output px), from the trusted REPS numbers:**

| kernel | 1-core ms/frame | ideal 32-core ms/frame |
|---|---|---|
| EASU alone | 2,073,600 x 1395.9 ns = ~2894 ms | ~90.4 ms |
| RCAS alone | 2,073,600 x 391.3 ns = ~811 ms | ~25.3 ms |
| EASU+RCAS fused | 2,073,600 x 1786.1 ns = ~3704 ms | ~115.7 ms |

**FSR1-on-NPU is parked as a correctness baseline, not an fps lever.** ~3.7 s/frame at 1 core and
~116 ms/frame at an ideal, unattainable 32 cores are both far outside real-time (16-33 ms/frame for
30-60fps) on this box's actual GPU path, which does the same 640x360->1920x1080 EASU+RCAS pass in
0.66-1.20 ms on the 890M iGPU (this workspace's own Vulkan harness, see Step 1). The NPU port stays
useful as a bit-faithful correctness reference and a brick-vectorization exercise, not as a
candidate render-path accelerator for this filter. Broader NPU-for-real-time-games framing is tracked separately
(internal planning, not in this repo).

## Step 3 -- whole frame, one dispatch: NOT REACHED (two demonstrated blockers, not scope)

1. **Structural: a literal one-shot whole-frame dispatch cannot fit.** `verify_oneshot`/`_build_oneshot`
   requires the WHOLE input and output resident in L1 at once (64KB/core). 640x360 input alone is 2.7MB;
   even the smallest useful output tile at 1x is far over budget. A real whole-frame single dispatch needs
   a STREAMING design (tiled DMA, `_build_streamed`) with overlapping row-halos for EASU's 12-tap/RCAS's
   5-tap neighborhoods across tile boundaries -- unbuilt this pass; that halo-correct tiling is itself a
   nontrivial kernel-engineering task, not a config flag.
2. **Even if built, the measured per-pixel cost disqualifies it before multi-core is relevant.**
   2,073,600 output pixels (1920x1080) x 25476 ns/px (measured, not assumed) = **~52.8 s/frame on one
   core** -- projected by direct extrapolation from a measurement whose linearity is itself measured
   (two crop sizes agree to within 3%), not a guess. Perfect 8-column scaling would only buy 8x -> **~6.6
   s/frame**, still ~6.6x worse than the ~1000 ms/frame ESPCN baseline this task exists to replace, and
   nowhere near game speed (16-33 ms/frame for 30-60 fps). 0% MAC utilization (above) is *why*: spreading
   a software-float scalar kernel across columns is the wrong next lever.

**Recommendation recorded, not executed:** vectorize with `aie_api` (`aie::vector<float,N>`/bfp16 MAC
path) BEFORE building the multi-core streaming/halo plumbing -- building the streaming design around a
kernel already known to be 3-4 orders of magnitude off target would be wasted engineering. This is the
concrete next step on the task.

## Step 4 -- C ABI: NOT REACHED

Conditional on step 3 landing (per the task brief); step 3 did not land, so this was not started.

## Files added this pass (step 2)
- `kernel/fsr1_kernel.cc` -- the scalar EASU+RCAS kernel (fp32, compile-time `FSR1_IN_W`/`FSR1_IN_H`).
- `kernel/host_check.c`, `host_check_easu.c` -- native-compiled pre-device correctness checks (not
  committed as binaries; regenerate with plain `gcc`, see the file headers).
- `kernel/verify_fsr1.py` -- device correctness gate (bricklib `verify_oneshot`, PASS).
- `kernel/time_fsr1.py` -- device cycles/pixel measurement at two crop sizes.
- `kernel/run.sh` -- npu_lock-wrapped runner (adapted from `aie_kernels/_test/run.sh`).

## Files added this pass (step 2b)
- `kernel/time_fsr1_vec.py` -- device cycles/pixel measurement for `fsr1_easu_vec` at two crop
  sizes (`FSR1_IN_H` 6/12, `FSR1_IN_W` fixed at 16 since it is also the vector width). DELETED --
  see step 2c: the marginal two-crop-size method it used is not reproducible under host dispatch
  jitter; superseded by `kernel/time_fsr1_reps.py`.

## Files added/removed this pass (step 2c)
- `kernel/fsr1_kernel_vec.cc` -- `fsr1_rcas_vec` (RCAS, vectorized) and `fsr1_strip_vec` (fused
  EASU+RCAS, one dispatch) added; `easu_tap_v`'s dead `optnone` removed; both `load_tap` and the
  new `load_strip` read the interleaved RGB buffer directly (no planar deinterleave scratch).
- `kernel/verify_fsr1_rcas_vec.py`, `kernel/verify_fsr1_strip_vec.py` -- device correctness gates.
- `kernel/time_fsr1_reps.py` -- repeat-count cycles/pixel measurement for all three vectorized
  kernels (EASU, RCAS, fused), one fixed crop, `K=4/200/800` with a linearity check. Supersedes and
  replaces `kernel/time_fsr1_rcas_vec.py`/`kernel/time_fsr1_strip_vec.py` (also DELETED, same
  jitter problem as `time_fsr1_vec.py`) and `kernel/time_fsr1_vec.py`.
