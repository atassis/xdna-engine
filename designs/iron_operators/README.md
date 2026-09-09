# designs/iron_operators/ -- our IRON operators, mirrored from the fork

The engine's fused decode build (`scripts/build_llm_decode.sh` -> `designs/decode_fused/`)
composes IRON operators that **amd/IRON does not have**: its log line `[gen] fused arm
qkv_head_dp: on / swiglu_mlp_dp: on` names two of them. Until 2026-09-09 those operators existed
only as commits on `atassis/IRON` branches, none of which was pushed. A tracked, shipped build
depended on source nothing in this repo tracked -- the same shape as `conveyor_proto`, and the
reason this directory exists.

## What is here

Every file on IRON `integration-stack` that is **absent from `upstream/devel`**, computed rather
than chosen: 16 files, kept at their IRON-relative paths so the subtree maps onto a checkout
one-to-one.

| Path | What |
|---|---|
| `iron/operators/qkv_head_dp/` | decode QKV head, data-parallel across columns |
| `iron/operators/swiglu_mlp_dp/` | decode SwiGLU MLP, every core runs every stage on 1/N |
| `iron/operators/tmatvec/` | transposed mat-vec |
| `iron/operators/gemv/quant.py` | the gemv quantised path |
| `aie_kernels/generic/mv_quant.cc`, `mv_taccum.cc` | kernels those operators compile |
| `iron/tests/infrastructure/element_size.py` | test helper they import |

## What this is NOT

**Not a build input.** Nothing here is compiled from this directory; `IRON_DIR` still resolves to
an IRON checkout and that is what builds. This is the tracked SOURCE OF TRUTH, so the work survives
a branch reset, a lost disk, or a checkout pointed somewhere else -- the failure modes that made it
worth copying.

**Not the history.** Mirrored as ONE commit from `integration-stack` at
`f35c772e7624` (on `upstream/devel` deb6e1e7c). The 39-commit stack and the ~20 lane branches
(`prefill/*`, `int4-*`, `fuse/*`) are deliberately **left intact on the fork** -- they carry the
measurements and the dead ends, and which of them is still live is an open decision, not something
this mirror settles.

## Restoring into a checkout

```sh
cp -a designs/iron_operators/iron        <iron-checkout>/
cp -a designs/iron_operators/aie_kernels <iron-checkout>/
```

Re-mirror after landing operator work on the fork; there is no automatic sync, deliberately --
copying back over a checkout that has newer work would silently revert it.
