# scripts/

194 entries, flat on disk, doing eight unrelated jobs. This index groups them by job so you
can find the one you want without reading filenames. Descriptions are each script's own
header line, not a re-description.

**Device discipline:** the NPU is single-tenant. Anything marked **[NPU]** takes the device -
announce it, stop `npu-serve` / `npu-asr`, and check `fuser -v /dev/accel/accel0` first.
Everything else is CPU-only and safe to run any time.

**The Rust-side siblings of these live in `rust/npu-probes`** (53 device probes, parity checks
and benchmarks). Several scripts here drive them: `cargo run -p npu-probes --bin <name>`.

---

## 1. Environment and toolchain

Bring up or move the pinned AIE toolchain. `toolchain.lock` is the pin; `toolchain_up.sh` is
the entry point. See also the FORK-ONLY rule in `AGENTS.md` - never build against the wheel.

| Script | What it does |
| --- | --- |
| `toolchain_up.sh` | Build (or locate) the mlir-aie-with-bindings toolchain INSTANCE for the current pin. **Entry point.** |
| `toolchain_smoke.sh` | CPU smoke gate (no NPU): the modal generator must emit `logical_tile` and place. |
| `toolchain_wire.sh` | Point `.venv-iron` at an instance's python (off the wheel), reversibly, via `aie.pth`. |
| `toolchain_refresh.sh` | The "stay on latest upstream" doctor. |
| `toolchain_gc.sh` | Instance-dir GC for the content-addressed toolchain instances. |
| `iron_env.sh` | Source to build/run mlir-aie examples on this box. |
| `air_env.sh` | Sourceable env for the vendored mlir-air toolchain (`airenv` 3.12 venv). |
| `amd_paths.sh` | Single relocatable anchor for the AMD/Xilinx upstream checkouts. |
| `cache_env.sh` | Single relocatable anchor for the build CACHE (toolchain instances, ...). |
| `fast_build_env.sh` | Cross-cutting fast-build levers for the AIE toolchain. |
| `kernel_sandbox.sh` | Kernel build-sandbox freshness stamp. Sourced by the kernel-build scripts. |
| `setup_route_b.sh` | Reproduce the Route B (open mlir-aie/Peano) build environment. |
| `setup_amd_toolchains.sh` | Single entry point: apply ALL AMD-toolchain patch series onto their targets. |
| `setup_export_venv.sh` | Create the py3.12 model-EXPORT venv used by the ONNX export/convert scripts. |
| `setup_decode_env.sh` | Recreate the whisper deep-C DECODE build environment from a clean workspace. |
| `apply_patches.sh` | Generic patch-series applier. |
| `sync_kernels.sh` | Copy our canonical kernels/designs FORWARD into the mlir-aie build sandbox. |
| `fetch_mlir_distro.sh` | Provision the pinned MLIR core distro (prebuilt LLVM/MLIR wheel). |
| `build_peano_fast.sh` | Build/rebuild a patched Peano (llvm-aie) fast enough to iterate. |
| `install_peano_local.sh` | Install a locally-built Peano into a usable dist dir. |
| `build_pyxrt_py314.sh` | Build a py3.14 pyxrt with the ctrl-scratchpad binding into `.venv-iron`. |
| `build_wheelhouse.sh` | Rebuild the local wheelhouse from the uv archive cache. |
| `bump_upstream.sh` | Bump the mlir-aie fork integration branch onto latest upstream. |
| `integration_refresh.sh` | Rebuild per-fork `integration` branches from the manifest. |
| `peano_upstream_status.sh` | Fetch live state of the upstream Peano int->float miscompile PR cluster. |
| `profile_aiecc.sh` | Profile the running aiecc final-ELF phase. Needs sudo (`ptrace_scope`). |
| `amd_toolchains.env.example` | Template for the toolchain env file. |

## 2. Kernel builds (produce xclbins)

CPU-only unless noted - these run aiecc, they do not touch the device.

| Script | What it does |
| --- | --- |
| `build_kernels.sh` | Build ALL NPU xclbins the `npu_asr` encoder needs. |
| `build_parakeet_kernels.sh` | Build the Parakeet NPU matmul xclbins. Production engine. |
| `build_parakeet_modal_kernels.sh` | Build the Parakeet MODAL resident xclbin + per-N instruction streams. |
| `build_decode_kernels.sh` | Build the thin-M GEMV xclbin the on-NPU decoder needs. |
| `build_decode_profile.sh` | Canonical fused-decode build entrypoint; encodes the exact dims each consumer needs. |
| `build_batched_decode.sh` | Build batched fused-decode block ELFs (vector-b). |
| `build_deepc_decode.sh` | Build the deep-C (constant ELF + runtime scratchpad params) fused Whisper decode. |
| `build_conv_kernels.sh` | Per-channel-band M-stationary conv xclbins (ResNet-18 on NPU). |
| `build_mha_decode.sh` | On-chip single-query MHA-decode xclbin. |
| `build_esm_native_kernels.sh` | NATIVE (real-K) whole-array xclbins for ESM-2. |
| `build_gemm_probe.sh` | Lever-3 batching probe ELFs (single fc1 GEMM at a sweep of N). |
| `build_projout_elf.sh` | Standalone proj_out (lm-head) GEMV ELF. |
| `build_lpddr_bw_microbench.sh` | Pure-DMA LPDDR bandwidth microbench xclbins for a transfer-size sweep. |
| `build_parakeet_dma_sweep.sh` | Extra sweep points for the DMA-occupancy regression. |
| `build_parakeet_occupancy_stub.sh` | DATA_MOVEMENT_ONLY stub twin of the Parakeet resident kernel. |
| `conveyor_prebuild.sh` | 8-head relpos-MHA CONVEYOR xclbin. |
| `conveyor_bd_prebuild.sh` | H=4 BD-ON-CHIP conveyor xclbin. |
| `relpos_prebuild.sh` | Resident relpos-MHA xclbins (STEP=8) + template instruction streams. |
| `golden_backup.sh` | Back up known-good xclbins outside the gitignored build dir, with shas. |
| `gate_aiecc_only.sh` | Fast isolated aiecc byte-gate on the frozen input. |
| `gate_build_decode_l1.sh` | Byte-gate harness for the build-chain attack. |
| `gate_freeze_and_build.sh` | Establish the aiecc byte-gate from generated fused-decode MLIR. |
| `ksweep_buildtime.sh` | Build the fused decode at several layer counts, fully optimized. |
| `npu_xrt_dynseq_harness.sh` | Build + run the dyn-seq dynamic whole-array harness. |
| `npu_xrt_dynseq_bench.cpp` | C++ source for the above. |

## 3. Model export and conversion

Turn a Hugging Face / ONNX checkpoint into the engine's artifact layout. CPU-only. Need the
export venv (`setup_export_venv.sh`).

| Script | Model |
| --- | --- |
| `fetch_models.sh` | Download the model inputs the export pipeline consumes into the HF hub cache. |
| `export_gigaam_encoder.py` | GigaAM-v3 RNNT encoder to clean static-shape ONNX. |
| `extract_encoder.py` | FULL GigaAM-v3 encoder: all 16 blocks' weights + pre_encode. |
| `extract_block0.py` | GigaAM-v3 block-0 weights + ONNX reference intermediates. |
| `extract_blocks.py` | Weights + output refs for blocks 0..N-1 (default 3). |
| `extract_parakeet_encoder.py` | Parakeet-tdt-0.6b-v3 FastConformer encoder (24 blocks). |
| `extract_whisper_encoder.py` | whisper-small encoder weights + golden activations. |
| `extract_whisper_decoder.py` | whisper-small DECODER weights from the exported ONNX. |
| `export_whisper_preproc.py` | Whisper log-mel front-end to ONNX. |
| `export_bge.py` / `convert_bge.py` | bge-base-en-v1.5. |
| `export_minilm.py` | sentence-transformers BERT-family encoder. |
| `convert_modernbert.py` | answerdotai/ModernBERT-base. |
| `export_esm.py` | ESM-2 + an ONNX oracle + golden fixture. |
| `convert_opt125m.py` | facebook/opt-125m. |
| `convert_vit.py` / `convert_dinov2.py` / `convert_clip.py` | ViT-base, DINOv2-base, CLIP-ViT-B-32. |
| `export_resnet.py` | ResNet-18 fixture for the general-conv2d proof. |
| `export_patch_stem.py` | Fixture for the conv2d-kit patch-embed verifier. |
| `export_espcn.py` / `export_edsr.py` | npu-sr nets (ESPCN, EDSR) + whole-net gate. |
| `quantize_encoder_static.py` | Static int8 quantization of the GigaAM-v3 encoder with mel calibration. |
| `gemv_coalesce_packer.py` | Canonical batch-coalesce BD packer for the IRON GEMV (single source of truth). |
| `fetch_wer_clips.py` | Pull FLEURS dev clips (RU + EN) for the WER set. |
| `requirements-export.txt` | Pins for the export venv. |

## 4. Device runners **[NPU]**

Drive one kernel or one path on the device and check it. These take the NPU.

| Script | What it validates |
| --- | --- |
| `run_npu_matmul.py`, `run_npu_matmul_bf16.py`, `run_npu_matmul_wholearray.py` | single_core / bf16 / whole-array GEMM. |
| `run_npu_matrix_scalar_add.py` | The `matrix_scalar_add` design (first rung). |
| `run_npu_mm_silu.py`, `run_npu_mm_silu_wa.py` | Fused matmul+bias+SiLU, single and 8-column. |
| `run_npu_ffn_gemm2.py`, `run_npu_wa_gemm2.py` | Fused two-matmul chain `C = (A@W1)@W2`. |
| `run_npu_layernorm.py`, `run_npu_ctxln.py` | LayerNorm [400,768]; f32 two-pass ctxLN. |
| `run_npu_softmax.py`, `run_npu_softmax400.py` | bf16 softmax; per-row length-400. |
| `run_npu_silu.py`, `run_silu_probe.py` | SiLU/Swish; the standalone SiLU brick. |
| `run_npu_cast.py` | f32 -> bf16 cast (resident-rails seam primitive). |
| `run_npu_dwconv1d.py`, `run_npu_dwconv1d_k9.py` | Depthwise conv1d k=5 and vectorized k=9 (`sliding_mul`). |
| `run_dwconv_silu_fused_probe.py`, `run_dwconv_silu_tmajor_probe.py` | Fused dwconv->silu, channel- and time-major. |
| `run_npu_conf_epi_silu.py` | Fused SiLU epilogue (`conformer_epilogues.cc`). |
| `run_npu_relpos_scores.py` | Standalone rel-pos scores->softmax (step 1). |
| `run_npu_relpos_ac_scores.py` | Composed rel-pos block: on-chip AC matmul (step 2). |
| `run_npu_relpos_qkp.py` | Resident rel-pos block, both score matmuls (step 3). |
| `run_npu_relpos_ctx.py` | Context matmul `ctx = probs @ V` (step 4). |
| `run_npu_relpos_rowtiled.py` | Row-tiled, MemTile-staged rel-pos MHA block (step 6). |
| `run_npu_block0.py` | Host-orchestrated GigaAM-v3 Conformer block 0 with heavy ops on the NPU. |
| `run_transpose_probe.py` | On-chip compute-tile transpose xclbin. |
| `p0b_mha_noncausal_test.py` | Non-causal MHA patch vs a non-causal golden. |
| `run_ra_spill_repro.py` | The minimal RA/spill-around-call repro. |
| `repro_ndma_multicontext.py` | n-D output DMA hang repro under multi-context. |
| `qos_priority_probe.py` | Does amdxdna QoS `priority` affect scheduling or only the DPM clock? |
| `sweep_control_registers.py` | Sweep AIE kernels for lossy ops on an unset global control register. |
| `decode_trace_probe.py`, `perop_trace_measure.py` | Per-op on-NPU hardware trace. |
| `run_lpddr_bw_microbench.sh`, `lpddr_bw_microbench_harness.py` | LPDDR bandwidth sweep. |
| `run_parakeet_occupancy.sh`, `parakeet_occupancy_harness.py` | Per-op occupancy A/B. |
| `run_parakeet_dma_occupancy.sh`, `parakeet_dma_occupancy_harness.py` | Split DMA-wait from dispatch/BD/lock/stall. |
| `run_glue_contention.sh` | Host-glue bandwidth-contention probe. |
| `run_fused_decode_testsuite.sh` | Comprehensive fused-decode + full-ASR e2e suite. |
| `prototype_ln_cast_resident.py` | Device-side BO hand-off across TWO xclbins (resident-rails feasibility). |
| `proto_ffn_chain.py` | Keep the FFN intermediate H on-device between two matmul dispatches. |
| `npu_power_mode.py` | NPU power-mode gate for timed runs. |
| `reset_npu.sh` | Recover a wedged NPU without a reboot. |

## 5. Verification, parity and goldens

Numerical correctness. CPU-only unless the name says `verify_*` against a device path.

| Script | Gate |
| --- | --- |
| `encoder_parity.py` | **The device-change gate.** rel-L2 parity; replaces the chaotic 17-clip greedy WER. |
| `verify_encoder.py` | `npu_asr` Encoder end-to-end vs static-ONNX reference tensors. |
| `verify_fused_encoder.py`, `verify_fused_attn.py`, `verify_fused_conv.py`, `verify_fused_ffn.py` | Fused GigaAM sub-blocks vs ONNX. |
| `verify_bge_parity.py` | NPU embeddings vs HF f32 reference (mean-pooled, L2-normalized). |
| `block0_numpy.py`, `stack_blocks.py` | Pure-numpy Conformer block 0, verified op-by-op; then stacked. |
| `parakeet_ref_encoder.py`, `parakeet_relpos_mha_golden.py`, `parakeet_tdt_decoder_ref.py` | Parakeet references. |
| `parakeet_occupancy_golden.py` | Per-op occupancy CPU golden + roofline. |
| `whisper_ffn_resident_golden.py` | Resident-intermediate Whisper FFN draft. |
| `bge_reference.py`, `vit_reference.py`, `opt125m_reference.py` | HF goldens. |
| `gemma_ref_generate.py`, `gemma_sampling_ref.py` | Gemma references. |
| `relpos_block_model_check.py`, `relpos_stream_packing_check.py`, `relpos_synth_ref_check.py` | Rel-pos numpy models. |
| `tap_equivalence_check.py`, `tap_equivalence_transpose.py`, `tap_equivalence_transpose_multitile.py` | Offline DMA-coalesce correctness (no NPU, no toolchain). |
| `ln_onepass_vs_twopass.py` | Is one-pass f32 LayerNorm variance accurate enough? |
| `mha_precision_sim.py` | Which bf16 quantity dominates encoder-MHA error (no device). |
| `conveyor_bd_precision_check.py` | SPLITP / BD-carriage precision gate. |
| `count_tile_resources.py` | Per-tile hardware resources a design actually configures. |

## 6. WER and accuracy evaluation

| Script | What it scores |
| --- | --- |
| `wer_eval.py` | WER harness for the NPU-ASR service. |
| `asr_oracle.py` | Ground-truth oracle + model export for the GigaAM-v3 RNNT pipeline. |
| `parakeet_wer_eval.py` | Parakeet CPU oracle WER on the wer_clips set. |
| `parakeet_npu_wer.py` | Parakeet encoder WER via the encoder-swap method. |
| `whisper_cpu_oracle.py` | whisper-small full-CPU WER over the 17 FLEURS clips. |
| `whisper_npu_wer.py` | NPU encoder -> ONNX decoder -> WER over the same clips. |
| `whisper_decode_wer.py` | WER via the live service. |
| `whisper_dump_mels.py` | Dump log-mel features for the WER clips. |
| `_whisper_decode_wer_run.sh`, `_whisper_decode_attn_wer_run.sh` | Single-tenant head-to-heads. **[NPU]** |
| `wer_batched_decode.sh`, `_score_batched_wer.py`, `_score_m1_wer.py` | Batched-decode WER gate and scorers. |
| `int8_wer_eval.py` | int8-quantization WER for the GigaAM-v3 encoder (CPU only). |
| `burst_sensitivity.py`, `burst_to_transcript.py` | How big must an encoder error burst be to change the transcript? |
| `edsr_before_after.py` | EDSR ship-net before/after on a real image. |

## 7. Benchmarks and measurement **[NPU]**

Timed runs. Read `docs/benchmark-methodology.md` first - these want a quiesced box.

| Script | Measurement |
| --- | --- |
| `idle_perf_sweep.sh` | Idle full-NPU ASR perf sweep on a quiesced box. |
| `measure_phase2_sweep.sh` | Phase-2 sweep; stops `npu-asr`/vox first. |
| `measure_int8_energy.sh` | Batched energy + timing A/B of int8 variants vs baseline. |
| `measure_arrayfill.sh` | Per-token dispatch at B=16 vs B=128. |
| `bench_batched_decode.sh` | Batched (B=16) vs M=1 decode J/token + tok/s. |
| `bench_npu_block.py` | Latency reality-check for the host-orchestrated GigaAM-v3 encoder. |
| `gemm_probe_sweep.sh` | Batched-GEMM dispatch-amortisation sweep. |
| `lever3_coalesce_ab.sh`, `lever3_isolate.sh`, `coalesce_e2e_ab.sh` | Coalesced-dispatch A/B and correctness isolation. |
| `parakeet_crossover.py` | CPU/NPU encoder crossover vs audio length. |
| `_esm_latency.sh` | ESM e2e latency via the service (release, idle). |

## 8. Service, install and repo hygiene

| Script | What it does |
| --- | --- |
| `test_install.sh` | Verify the installed service end-to-end: health, model list, a real transcription. |
| `uninstall.sh` | Stop + remove the service unit and binary. Keeps the onnxruntime dir. |
| `test_npu_pipeline.py` | End-to-end test of the NPU encode path. |
| `ci_gate.sh` | **The Rust gate.** Run before pushing anything that touches `rust/`. |
| `check_no_private_refs.sh` | Audit the PUBLIC tree for anything that should not be here. |
| `test_repro_vendoring.sh` | End-to-end reproducibility test for the pinned-submodule + tethered-patch vendoring. |
| `prune_artifacts.sh` | Dry-run-by-default reclaim of regenerable `artifacts/` contents. |
| `CONVEYOR_INTEGRATION_RUNBOOK.md` | Runbook for integrating the conveyor kernel. |
| `tests/` | Script-level tests. |

## Retired

Still on disk, referenced by nothing that runs:

| Entry | Status |
| --- | --- |
| `install.sh` | Deprecated shim to `../install.sh`. Use the repo-root one. |
| `asr_service.py` | The Python HTTP ASR service the Rust `npu serve` replaced. Nothing in the engine references it. |
