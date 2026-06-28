# route_b_kernels — custom XDNA2 AIE kernels & IRON designs (canonical source)

These are the hand-written kernels and mlir-aie/IRON designs built in Stage 3 (docs/08–11).
**This directory is the single source of truth for OUR code.** The `mlir-aie/` tree is a
**pinned git submodule** (upstream commit `8373e49`; see internal notes) used as
a disposable build sandbox; `scripts/sync_kernels.sh` copies these files **forward** into it
(one-directional → no drift). **Edit here, never in `mlir-aie/`.**

The delta on top of pristine upstream is split by kind:
- **Our new files** (everything here except `patches/`) → copied-forward as real files (`sync_kernels.sh`).
- **Edits to upstream mlir-aie files** (the CachyOS build fixes, the bf16 `mm.cc` microkernel, the aiecc
  perf/build patches) → now carried as COMMITS on the **fork integration branch**
  `atassis/mlir-aie:xdna2-asr` (base 8373e49 + 14 patches), checked out by `setup_route_b.sh`. There is no
  `.patch`/apply step for mlir-aie anymore. `toolchain.lock` pins the branch commit; `toolchain_up.sh` builds
  the toolchain instance from a clean worktree of it. (IRON/mlir-air still use the `patches/*.series` model
  until they migrate.)

Workflow:
```
scripts/setup_route_b.sh         # submodule init + pinned wheels + checkout xdna2-asr + sync these in
scripts/sync_kernels.sh          # re-copy after editing anything here
scripts/build_kernels.sh         # sync + build all xclbins (CPU; no NPU)
scripts/test_repro_vendoring.sh  # prove a fresh clone reproduces the pinned build
```

| file | role | docs |
|---|---|---|
| `dwconv1d/` (`.cc`,`.py`,`Makefile`) | depthwise-conv1d k=5 — last missing Conformer primitive | 08 |
| `aie_kernels/mm_silu_epilogue.cc` | on-chip bias+SiLU / f32→bf16 narrow epilogue | 10 |
| `whole_array_fused/` (`whole_array_silu_iron.py`,`Makefile.silu`) | 8-col whole-array matmul + epilogue (the fast fused matmul) | 10 |
| `ffn_gemm2/` (`ffn_gemm2_iron.py`,`Makefile.ffn`) | fused GEMM→GEMM, intermediate on-chip (proven N_div_n=1) | 10 |
| `softmax400/` (`softmax400.py`,`Makefile`) | per-row softmax over length 400 (pad→416 + −∞) | 10 |
| upstream mlir-aie edits (cachyos, mm.cc, aiecc perf) | now COMMITS on `atassis/mlir-aie:xdna2-asr`, not `.patch` files | 11 |

Why **copy-forward for our kernels, patch for upstream edits**: symlinks break mlir-aie's
`realpath`-based `srcdir`, and folding our kernels into a patch-blob would hurt reading/editing
live kernel source — so our new files stay as real files copied one-directionally (first-class +
tracked, satisfying mlir-aie's in-tree build convention). The 3 *upstream-file* fixes are a
genuine delta against upstream, so they live as a single tethered patch (was: idempotent seds)
that `setup_route_b.sh` applies — see internal notes.
