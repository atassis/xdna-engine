# designs/ -- IRON dataflow graphs built from aie_kernels/

A design is a multi-core AIE dataflow graph compiled from one or more `aie_kernels/` kernel
sources into one artifact (xclbin+insts, or an ELF for the fused-decode path). "design" is
AMD's own noun for this (1399 uses in mlir-aie, their `aie_design.py` convention).

This directory, `aie_kernels/`, `experiments/` and `patches/` are what `route_b_kernels/`
split into in `81e513d` ("designs: retire route_b_kernels..."), 2026-09-05 -- 16 shipping
designs here, 7 one-off studies to `experiments/`, the mlir-air PR drafts to `patches/`. That
commit deleted `route_b_kernels/README.md` (40 lines) with no replacement; this file is it.

## Build models -- three, not two, and checked by counting `Makefile*` per dir

- **sync+Makefile** (11 of 16: `conveyor_proto`, `ctx_ln`, `decode_norm_gemv`, `dwconv1d`, `ffn_gemm2`,
  `mha_decode`, `m_stationary`, `relpos_mha`, `silu`, `softmax400`, `whole_array_fused`): a
  `*_iron.py` generator emits MLIR; `scripts/sync_kernels.sh` copies it, plus the
  `aie_kernels/` sources it needs, into the mlir-aie build sandbox; a Makefile drives the OLD
  generator -> `.mlir` -> `aiecc` flow. `design_override.mk` exists because the toolchain's
  own upstream `matrix_multiplication/makefile-common` moved to a `@iron.jit` compile-only
  flow that our MLIR-only generators can't drive -- see the `WHY` comment at the top of that
  file. `toolchain_stamp.mk` (included by these Makefiles) makes every build target depend on
  the toolchain instance's identity, not just file mtimes.
- **direct python/IRON, no Makefile** (2 of 16: `decode_fused`, `codec_block`): a `gen_*.py`
  builds an `OperatorSequence` (or, for `codec_block`, an `aie.iron.Program`) and drives
  IRON's own build path directly; a `scripts/build_*.sh` wrapper sets up
  `PYTHONPATH`/`AIECC_PATH` and runs it. See `scripts/build_llm_decode.sh`.
- **neither** (3 of 16: `iron_operators`, `cascade_ffn`, `subsample_conv2d`):
  `iron_operators` is not a design at all -- it is our IRON operator set mirrored out of the
  fork so the operators `decode_fused` composes are readable in this tree (see its own
  README); it has no generator and nothing builds it here. The other two are
  pre-design-stage: `cascade_ffn` builds through **mlir-air**
  (`aircc`, sourced via a "mlir-air airenv"), a different toolchain from mlir-aie/IRON
  entirely -- see `build_cascade_ffn.sh`. `subsample_conv2d` has no dataflow-graph generator
  at all yet: `build_check.sh` is a bare Peano kernel compile-check plus a numpy golden, which
  matches its own README calling it a "CPU/build-only draft".

Kernel sources are never built independently: `aie_kernels/*/` holds only `.cc` + `golden.py`,
no Makefile of its own (checked: zero `Makefile*` under `aie_kernels/` anywhere). A
sync+Makefile design points `lib_kernels_dir`/`KERNEL_CC` at `aie_kernels/`; a direct-python
design's `Kernel(...)` call references the `.cc` path directly.

## Layout

"Engine dispatch" below is what a targeted grep of `rust/` for each design's name actually
turned up, not the design's intent -- two entries below (`dwconv1d`, `cascade_ffn`) looked
wired from a surface grep and were NOT, once traced to a call site. Treat a "yes" as evidence
found, not a runtime guarantee for every code path or feature flag.

| Design | Build model | What it is | Engine dispatch |
|---|---|---|---|
| `decode_fused/` | direct python/IRON | fused Qwen3/Gemma decode step: QKV, MLP, attention, argmax, cross-attn | **yes** -- `rust/npu-gemma`, `rust/npu-engine/src/llm/npu_decode.rs`, the Whisper decoder's `verify_llm_decode.py`/`gen_projout.py` path |
| `whole_array_fused/` | sync+Makefile | 8-column whole-array GEMM family: plain/modal/silu/gelu/int8 epilogues | **yes** -- `rust/npu-asr/src/block.rs` (`WAEpilogue::new(..,"silu",..)`), `ctx2.rs`, `engines.rs` |
| `mha_decode/` | sync+Makefile | resident single-query streaming/flash MHA | **yes** -- `rust/npu-asr/src/ctx_decode.rs::attn()` (public call site, not just a loader comment) |
| `ctx_ln/` | sync+Makefile | two-pass on-chip LayerNorm (normalize-only; host applies affine) | **yes, opt-in** -- `rust/npu-asr/src/ctx_ln.rs::CtxLn`, wired through `block.rs` behind the `two_ctx` feature |
| `relpos_mha/` | sync+Makefile | relative-position MHA (Parakeet FastConformer) | **yes** -- `rust/npu-parakeet/src/encoder.rs`, `npu.rs` |
| `dwconv1d/` | sync+Makefile | depthwise conv1d k=5 (Conformer front-end) | **no, currently** -- `kernel_registry.rs` still knows its artifact-name grammar and `engines.rs::dwconv()` is a live NPU method, but nothing calls it: `block.rs:309` runs dwconv on the HOST instead ("cheap 5-tap FIR, parallelized"). Design retained, not on the live path. |
| `cascade_ffn/` | mlir-air/`aircc` (`build_cascade_ffn.sh`), not mlir-aie/IRON | on-chip fc1->GELU->fc2 cascade-accumulate reduction (K=3072), avoids a host readback between the two GEMMs | **no, not yet** -- `ctx2.rs:1370` names it as the thing a TRUE on-device fc1->fc2 hand-off "needs", i.e. future work; the current code takes the host-resident fallback |
| `silu/` | sync+Makefile | standalone SiLU epilogue kernel + IRON design | **unclear** -- every `silu` reference actually found in `rust/npu-asr` is the `whole_array_fused` `modalsilu` epilogue variant, not this directory; no rust reference to `designs/silu` specifically |
| `decode_norm_gemv/` | sync+Makefile | resident-norm + GEMV prologue study | no rust reference found -- research |
| `ffn_gemm2/` | sync+Makefile | fused GEMM->GEMM with the intermediate kept on-chip | no rust reference found -- research |
| `m_stationary/` | sync+Makefile | M-stationary GEMM variant (plain + LN-fused) | no rust reference found -- research |
| `softmax400/` | sync+Makefile | fixed-length-400 softmax (pad to 416 + -inf) | no rust reference found -- research |
| `codec_block/` | direct python/IRON, no Makefile | audio-codec decoder chain: head -> stage1..4 -> tail -> tanh, via `window_driver` | no rust reference found -- research |
| `subsample_conv2d/` | bare Peano kernel compile-check, no dataflow-graph generator yet | Parakeet conv2d/8 front-end reformulated as im2col -> `aie::mmul` GEMM | no -- its own `README.md` calls it "Task A5 (CPU/build-only draft)" |

## Also in this tree

- **`aie_kernels/`** (51 dirs: 50 kernels + `_test/`) -- the hand-written kernel sources these
  designs compile in. Per-kernel index, golden coverage and device status:
  [`aie_kernels/INDEX.md`](../aie_kernels/INDEX.md).
- **`aie_kernels/_test/`** -- device-verify harness for kernels in isolation (40 `verify_*.py`
  gates plus bisect probes), not per-design; see its own `README.md`.
- **`experiments/`** (7 dirs: `exp2_ab`, `ffn_bfp16`, `lpddr_bw`, `occupancy`, `phase_probe`,
  `probes`, `tanh_ab`) -- one-off studies and A/B probes, not dispatched by the engine. Build
  model varies per dir (4 of 7 have their own `Makefile`).
- **`patches/`** -- carried upstream toolchain patches (mlir-aie, IRON, a draft mlir-air PR),
  unrelated to building these designs; see its own `README.md`.
