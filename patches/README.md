# Vendored toolchain patches

Out-of-tree changes against the AMD toolchains this repo builds on.

**The default model is the fork integration branch, not a patch file.** `toolchain.lock` pins a
commit of each fork, and `scripts/bump_upstream.sh` rebases those branches forward and drops work
that merged upstream. A `.patch` here is the exception: a change carried against a checkout the lock
does not pin, or one deliberately kept opt-in.

## Carried now

| Patch | Target | Applied by | What it does |
|---|---|---|---|
| `mlir-aie-kernel-compile-speedups.patch` | the built mlir-aie instance (`$INST/src`) | `scripts/toolchain_up.sh`, automatically | Kernel-object cache plus a shared precompiled header for standalone kernel `.cc` compiles. Touches `python/utils/compile/` only. Eviction takes the same per-key lock the compile path holds and skips a busy entry, and PCH construction is single-flight across processes rather than per-process. Source: fork branch `perf/kernel-compile-pch`. |
| `iron-mha-noncausal.patch` | `amd/IRON` (`mha.cc`) | by hand; `scripts/p0b_mha_noncausal_test.py` documents the flow | Gates the four unconditional causal-mask sites behind `MHA_NONCAUSAL`, so the same kernel serves an encoder, which needs no causal mask. |

`toolchain_up.sh` applies the speedups patch idempotently and **non-fatally**: it is skipped when
the pin already carries it, and a pin that moves under it produces a loud warning naming the fix
rather than a failed build. A build without it is slow, not wrong. Kill switches, bluntest last:
`AIE_KERNEL_PCH=0` disables just the PCH; `XDNA_NO_KERNEL_COMPILE_PATCH=1` skips the patch entirely.

## Drafts and PR bodies

Not applied by anything; staging for upstream work.

| File | For |
|---|---|
| `ndn-build-cap-channel-balance.DRAFT.diff`, `…v2-compiles.DRAFT.diff` | Channel-balance work against the build cap; v2 is the one that compiles. |
| `mlir-air-dmatochannel-pr2-body.md` | PR body: `AIRDmaToChannel` iterator invalidation when hoisting broadcast DMAs out of deeply-nested herds. |

## Retired

Until 2026-06-29 this directory carried fourteen tethered patches, and this file was a table
describing each one. Commit `091e1aa` migrated to the fork-branch model above and deleted all of
them, reaching zero carried patch files; the table outlived them by some months.

Recording where they went, since a merged patch is a provenance record:

- **Merged upstream.** The O(n^2) and lowering fixes -- materialize-lower-once, slice-strip,
  datawords-cache, shimdma-symbol-cache, MaterializeBDChains -- landed as
  Xilinx/mlir-aie #3178, #3211, #3212, #3216 and #3316, and are in the pinned toolchain. (That is
  the set as a set; this file never recorded which PR carried which patch.)
- **Parked as a design record**, re-measured marginal on the post-merge base: `core-elf-cache`
  (0 wall at B=128), `compilecores-kernel-cache` (~1-3s, needs a rewrite), and the `pass-timing` /
  `phase-timers` instrumentation (diagnostic only). Source on fork branch `xdna2-asr`.
- **Superseded by the fork branches**: the `cachyos` and `aiecc-jobs` build fixes, and the four IRON
  operator patches -- `transpose-num-batches`, `gemm-fusion-prefix`, `gemv-coalesce-batch-dma` and
  `ops-trace`.

Each retired patch's full description, measurements and apply instructions are recoverable with
`git show 091e1aa~1:route_b_kernels/patches/README.md` -- the table lived under `route_b_kernels/`
until that tree was retired into `designs/`.
