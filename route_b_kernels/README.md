# route_b_kernels — custom XDNA2 AIE kernels & IRON designs (canonical source)

These are the hand-written kernels and mlir-aie/IRON designs built in Stage 3 (docs/08–11).
**This directory is the single source of truth for OUR code.** The `mlir-aie/` tree is a
**pinned git submodule** (upstream commit `8373e49`; see internal notes) used as
a disposable build sandbox; `scripts/sync_kernels.sh` copies these files **forward** into it
(one-directional → no drift). **Edit here, never in `mlir-aie/`.**

The delta on top of pristine upstream is split by kind:
- **Our new files** (everything here except `patches/`) → copied-forward as real files.
- **Edits to 3 upstream files** → `patches/mlir-aie-cachyos.patch`, a single patch tethered to the
  pinned submodule SHA, applied by `setup_route_b.sh`. (Keeping our kernels as copy-forward real
  files rather than folding them into the patch keeps them first-class/readable; the patch is
  reserved for the genuine upstream delta.)

Workflow:
```
scripts/setup_route_b.sh         # submodule init + pinned wheels + apply patch + sync these in
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
| `patches/mlir-aie-cachyos.patch` | the 3 upstream-file CachyOS fixes, tethered to the pinned SHA | 11 |

Why **copy-forward for our kernels, patch for upstream edits**: symlinks break mlir-aie's
`realpath`-based `srcdir`, and folding our kernels into a patch-blob would hurt reading/editing
live kernel source — so our new files stay as real files copied one-directionally (first-class +
tracked, satisfying mlir-aie's in-tree build convention). The 3 *upstream-file* fixes are a
genuine delta against upstream, so they live as a single tethered patch (was: idempotent seds)
that `setup_route_b.sh` applies — see internal notes.
