# Model support

I get this list from the code, not from what the top-level README says works: for every
model below I checked whether it has a shipped `scenarios/*.toml`, whether its encoder or
decoder actually opens the NPU device and dispatches an xclbin, or whether the only thing
in the tree is a weight-conversion transform and a host-CPU reference. A few models the
README lists as working never touch the NPU in this checkout; I call those out explicitly.

Three labels, and I use them narrowly:

- **Supported** -- has a shipped scenario/config path, and its core forward pass runs on
  the NPU end to end (some per-domain glue -- tokenization, log-mel framing, RNNT/TDT
  decode -- legitimately stays on the host; that is a design choice documented in
  [general-engine.md](general-engine.md), not a support gap).
- **Experimental** -- a real, working NPU dispatch exists, but only behind an opt-in
  flag, or only for part of the model; it is not what you get by default.
- **Host-only** -- never dispatches to the NPU anywhere in this tree. Some of these have
  a full host-CPU forward pass checked against a reference; some have nothing but a
  weight-conversion (baking) transform checked against a Python oracle, meaning no
  forward pass -- host or NPU -- has ever been run at all.

## Speech recognition

| Model | Status | Evidence |
| --- | --- | --- |
| GigaAM-v3 (Russian) | Supported | `scenarios/asr-gigaam.toml`; encoder is `ConformerEncoder` in `rust/npu-engine/src/asr/mod.rs`, wrapping `npu_asr::encoder::Encoder`, which opens the NPU device |
| Parakeet-TDT-0.6B-v3 (multilingual) | Supported | `scenarios/asr.toml`; encoder in `rust/npu-engine/src/asr/parakeet.rs`, same NPU-device pattern |
| Whisper-small | Supported | `scenarios/asr-whisper-small.toml`; encoder in `rust/npu-whisper` runs on the NPU unconditionally |
| Whisper-large-v3-turbo | Supported | `scenarios/asr-whisper-turbo.toml`; same encoder path, 32 encoder / 4 decoder layers |

For all four, only the **encoder** is unconditionally on the NPU. Decode differs by
model and is worth stating precisely:

- GigaAM and Parakeet decode via an RNNT/TDT predictor + joint network run through
  `onnxruntime` on the host (`rust/npu-engine/src/asr/mod.rs`'s `run_decoder`/`run_joint`,
  `rust/npu-engine/src/asr/parakeet.rs`). This is not a gap to close -- it is a different
  decode algorithm than Whisper's, by necessity (`general-engine.md`: "ASR decode is
  per-model, not merely per-capability").
- Whisper decodes through `onnxruntime` (`decoder_model.onnx` / `decoder_with_past_model.onnx`)
  by default. Two environment variables in `rust/npu-engine/src/asr/whisper.rs` route the
  decoder onto the NPU instead: `NPU_DECODE` (per-op matmuls on the NPU, ~72 dispatches
  per token) and `NPU_DECODE_FUSED` (the entire decoder stack fused into **one** ELF
  dispatch per token, which takes precedence over `NPU_DECODE`). Both are real, working
  device paths -- `scripts/lever3_determinism_gate.sh` gates the fused path by checking
  its output is byte-identical to the ONNX reference across the 17-clip evaluation set --
  but neither is the default `npu serve` takes, so I label **on-NPU Whisper decode
  Experimental** even though the encoder above it is Supported.

## Embeddings

| Model | Status | Evidence |
| --- | --- | --- |
| BGE-base-en-v1.5 | Supported | `scenarios/bge-base.toml`; `BertEncoder` in `rust/npu-engine/src/bert/encoder.rs` ("BERT encoder on the NPU") |
| ESM2-8M, ESM2-35M (protein) | Supported | `scenarios/esm2-{8m,35m}[-native].toml`; `rust/npu-engine/src/esm/encoder.rs` |
| all-MiniLM-L6-v2, e5-small-v2, multilingual-e5-small | Host-only | weight-conversion parity only, see below |
| ModernBERT-base | Host-only | weight-conversion parity only, see below |

ESM ships two kernel paths, both on the NPU: the default `zeropad` path
(`rust/npu-engine/src/esm/encoder.rs`, zero-pads ESM's 320/480-wide hidden state onto the
kernel's fixed 768-wide matmul shape) and a `native` path
(`rust/npu-engine/src/esm/native.rs`, its own header calls it a "research/comparison
path" that runs the real, unpadded K). Both are Supported; `native` is not the default.

**MiniLM / e5 / multilingual-e5 / ModernBERT** have no scenario file, and I could not find
any code path that runs them on the NPU or even on the host end to end. What exists is a
`npu_weights::arch::Arch` transform (`rust/npu-weights/src/arch/bert.rs`,
`rust/npu-weights/src/arch/modernbert.rs`) and a matching parity test
(`rust/npu-weights/tests/parity_minilm.rs`, `parity_modernbert.rs`) that bakes the HF
checkpoint and diffs every tensor against a Python-oracle `.npy` -- that proves the
*weight conversion* is correct, nothing about inference. ModernBERT specifically could
not reuse the shipped `BertEncoder` even if a scenario were added: that encoder is a
post-norm, GELU-MLP, bias-carrying BERT block, while ModernBERT is bias-free with GeGLU
and RoPE (`rust/npu-weights/src/arch/modernbert.rs`'s own header) -- ops the shipped
encoder does not implement.

## Text generation (decoder LLMs)

| Model | Status | Evidence |
| --- | --- | --- |
| Qwen3-0.6B | Supported | `scenarios/generate-qwen3-0.6b.toml`; `rust/npu-engine/src/llm/npu_decode.rs`'s `NpuDecodeStep` drives a real `ElfResident`/`FusedArena` device dispatch, one fused ELF per token |
| opt-125m | Host-only | see below |
| Gemma 3 (270m, e2b bring-up) | Host-only | see below |

**opt-125m** (`rust/npu-probes/src/bin/opt125m_decode.rs`) is a greedy-decode host
reference validated token-for-token against the HF golden (`scripts/opt125m_reference.py`),
with host-simulated int8/int4 weight quantization for a bandwidth study. It never opens
an NPU device -- the file's own comment puts the decode on host f32 first and leaves the NPU LLM-decode
path as a follow-on reusing the K=768 ctx_decode primitives.
It is dimension-identical to Whisper-small (768/12/12/3072/64), so wiring it to the
existing on-NPU decode kernels is plausible future work, but it has not happened in this
tree.

**Gemma 3** (`rust/npu-gemma`) is, in its own `Cargo.toml` description, a "Phase 0
scaffold (host-CPU reference + the NPU port map)." The host reference
(`scripts/gemma_ref_generate.py`) reproduces `transformers`' output on CPU. The `npu`
Cargo feature (off by default) compiles a routing skeleton, but
`rust/npu-gemma/src/npu.rs`'s `GemmaNpuDecoder::step` unconditionally returns
`NpuError::DeviceBackendUnlinked` -- there is no device backend to link yet. Nothing in
the engine depends on this crate and it depends on nothing else in the workspace.

The top-level README currently describes both of these as "reusing the resident-FFN +
fused-decode + KV primitives" alongside Qwen3, which reads as one tier of NPU support
across all three. From the code, only Qwen3 is there; opt-125m and Gemma 3 are host-side
research/scaffolding toward that goal, not instances of it yet.

## Vision

| Model | Status | Evidence |
| --- | --- | --- |
| ViT-base (google/vit-base-patch16-224) | Host-only | `rust/npu-probes/src/bin/vit_embed.rs`, own comment: "(host f32)" |
| DINOv2 (facebook/dinov2-base) | Host-only | weight-conversion parity only |
| ResNet-18 (microsoft/resnet-18) | Host-only | `rust/npu-probes/src/bin/verify_resnet.rs` |
| CLIP | Host-only | weight-conversion parity only |

None of these reach the NPU as a full model. What each one actually has:

- **ViT-base**: a host-f32 forward pass, validated by cosine + argmax agreement against
  the HF golden on a fixed random image (`scripts/vit_reference.py`).
- **ResNet-18**: a host-f32 forward pass via im2col + `ndarray` GEMM, validated against
  an ONNX Runtime oracle. The file's own comment: "Proves the conv->im2col->GEMM LOWERING
  is correct end-to-end before any NPU kernel exists."
- **DINOv2** and **CLIP**: only a `npu_weights::arch::Arch` transform
  (`rust/npu-weights/src/arch/dinov2.rs`, `clip.rs`) plus a weight-conversion parity test
  (`tests/parity_dinov2.rs`, `parity_clip.rs`). No forward pass of any kind -- host or
  NPU -- exists for either in this tree.

One real NPU component does exist for this family: `rust/npu-probes/src/bin/verify_patch_embed.rs`
has an `--npu` mode that runs the ViT/DINOv2 patch-embed convolution (the stem, not the
transformer body) on the device in bf16 and gates it at rel-L2 <= 0.08 against an ONNX
Conv2d oracle, across three configs (`vit_b16`, `vit_l16`, `dinov2_b14`). That is one op,
not a model, and it does not appear in the table above for that reason. The README's
"Vision - ViT, DINOv2, and ResNet-18 through a general conv2d path" reads as NPU support
for full models; what actually runs on the NPU is this one shared stem op.

## Video super-resolution

| Model | Status | Evidence |
| --- | --- | --- |
| ESPCN | Supported | `rust/npu-sr/tests/npu_gate.rs`; 31.79 dB Y-PSNR (kodim23, x3) per `rust/npu-sr/README.md` |
| EDSR | Supported | `rust/npu-sr/tests/edsr_npu_gate.rs`; 34.29 dB Y-PSNR, same benchmark |

Both nets run a real conv -> im2col -> whole-array bf16 GEMM dispatch on the NPU, gated
against both the CPU frontier and a PyTorch oracle by relative L2 (`edsr_npu_gate.rs`
requires < 1.5e-2 vs CPU and < 2.0e-2 vs the oracle). This is not reached through
`npu serve` or the HTTP API at all -- it is a separate engine (`npu-sr`) with its own
frame-in/frame-out ABI, driven by the `xdna-sr` CLI or the `vf_xdna_sr` ffmpeg filter
(`rust/npu-sr/README.md`). The top-level README's "What works today" list omits this
entirely, even though it is a real, NPU-verified, shipped capability. One honest
limitation from `npu-sr`'s own README: the current NPU frontier uses a general
whole-array GEMM that is "correct but not yet size-optimized for these shapes," so
reported per-frame latency is a correctness baseline, not a real-time figure -- real-time
playback is "measured as headroom, not shipped in v1."

## Diarization

| Model | Status | Evidence |
| --- | --- | --- |
| pyannote/speaker-diarization-3.1 | Host-only | `scenarios/diarize-pyannote-3.1.toml` |
| pyannote/speaker-diarization-community-1 | Host-only | `scenarios/diarize-pyannote-community-1.toml` |

This one is host-only by design, not by omission, and the tree says so twice:
`rust/npu-engine/src/registry.rs`'s comment ("v1 diarization is host-only") and
`rust/npu-engine/src/diarize/mod.rs`'s own doc comment ("Host-only in v1; the NPU
embedder swaps in behind `SpeakerEmbedder` without touching this file"). `registry::try_build`
does not even open the NPU device for a diarize scenario, specifically so it cannot take
a hardware context away from a co-resident ASR model. Segmentation (PyanNet) and speaker
embedding (WeSpeaker ResNet34) both run through `onnxruntime` on the host.

## What I could not substantiate, and left out

- Any accuracy/throughput number for opt-125m or Gemma 3 on the NPU -- there is no NPU
  path to measure.
- Any claim that MiniLM/e5/ModernBERT/DINOv2/CLIP "work" in any sense beyond weight
  conversion -- I found no forward-pass code for any of them.
- Whether Whisper's `NPU_DECODE_FUSED` path generalizes to Whisper-large-v3-turbo's
  4-layer decoder the same way it does to Whisper-small's 12-layer one -- the mechanism
  in `rust/npu-engine/src/asr/whisper.rs` is written generically over decoder depth, but
  I did not find a turbo-specific fused-decode artifact or gate run in this tree, so I
  did not assert it works for turbo specifically.
