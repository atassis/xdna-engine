# Architecture

xdna-engine is a Rust workspace of focused crates layered over a kit of hand-written AIE
kernels. The design goal is a *general* engine: one execution pipeline that any
transformer or conv model plugs into, rather than a per-model stack.

## Layers

```
   models (ASR / embeddings / LLM / vision)     npu-sr (video super-resolution)
        |  Frontend / Encoder / Head traits          |  frame in / frame out
   npu-engine ......... general multi-model pipeline |
        |                                            |
   npu-runtime ........ control plane: desired-state config, reconcile, one device actor
   npu-weights ........ bake HF safetensors/ONNX -> mmap bf16 weight checkpoint
        |                                            |
   npu-xrt ............ safe Rust bindings over a C++ XRT shim -> the NPU
        |
   route_b_kernels .... hand-written AIE kernels (GEMM/GEMV/cascade/MHA/conv/LN/...)
   mlir-aie (submodule) the open AIE toolchain (kernel build + place-tiles)
```

`npu-sr` sits BESIDE `npu-engine`, not under it: it drives `npu-xrt` directly and has its own
frame-in/frame-out ABI. The two stacks share the device and the weight loader but not the
request path. Unifying them is open work, not a shipped property; see "Known seams" below.

## Crates

14 workspace members (`rust/Cargo.toml`).

| Crate | Responsibility |
| --- | --- |
| `npu-xrt` | Safe Rust bindings to drive the XDNA2 NPU via a thin C++ XRT shim. |
| `npu-engine` | General multi-model engine over the kernel kit: a `Frontend / Encoder / Head` pipeline serving ASR and embeddings. |
| `npu-runtime` | Control plane over `npu-engine`: desired-state config, reconcile, and a single device actor that serializes NPU work. |
| `npu-weights` | Rust-native weight loader: bakes HF safetensors / ONNX into an mmap-able bf16 checkpoint with a content fingerprint and parity gate. |
| `npu-onnx` | Runs ONNX graphs from Rust via a thin C shim over the system onnxruntime (oracles + fallback). |
| `npu-asr` / `npu-asr-host` | GigaAM-v3 encoder on the NPU (`npu-asr`) and its pure host-CPU reference math (`npu-asr-host`). |
| `npu-parakeet` | Parakeet-TDT FastConformer encoder (rel-pos attention, depthwise conv1d k=9, /8 conv2D subsample). |
| `npu-whisper` | Whisper-small encoder + decoder reference and the on-NPU decode path. |
| `npu-gemma` | Gemma 3 small-LLM decoder. SCAFFOLD: reference math, a `Brick` decode schedule and sampling, but no serving path - nothing depends on it and it depends on no other crate here. |
| `npu-sr` | Super-resolution engine: frame in / frame out video upscaling (ESPCN, EDSR). Own schedule JSON, own `SrEngine` ABI, drives `npu-xrt` directly. |
| `npu-sr-capi` | C ABI over `npu-sr` (`libxdna_sr.so`) for the ffmpeg `vf_xdna_sr` filter and other embedders. |
| `npu-capi` | C ABI over `npu-engine` (cdylib + staticlib, cbindgen header) for in-process embedding from any language. |
| `npu-cli` | `npu` multitool: serve, transcribe, embed, models, config, reload, bake. |

## Dataflow

A request enters through a `Frontend` (tokenize / feature-extract), runs the model's
`Encoder` (and, for autoregressive models, a decode loop) on the NPU, and finishes in a
`Head` (pooling, projection, argmax). The device actor in `npu-runtime` serializes all NPU
dispatches, since the NPU is single-tenant.

## Model residency

Which models hold the device is a property of the request, not of the config file. A request
naming a model that is not resident loads it on demand, evicting the least-recently-used
model when `max_resident` is already full; a model that has served nothing for
`idle_unload_s` is unloaded and gives the device back. Both happen on the device actor thread,
between commands, so residency never changes underneath a request in flight and the design
stays one owner, one thread, no lock. `[server]` keys: `max_resident` (slots),
`idle_unload_s` (idle window, `0` disables), `sweep_interval_s` (how often the actor looks),
`evict_policy` (`lru`, or `none` to refuse instead of evicting). `GET /v1/models` reports each
model's `state` and `idle_s`.

`max_resident` counts SLOTS, not resources. The accountant's `bo_bytes()` currently reports a
fixed `0` (`npu-runtime/src/loader.rs`), and the NPU's 8 columns of hw_context budget are not
tracked at all, so eviction cannot yet reason about the two things that are actually scarce:
LPDDR bytes and hw_contexts. Making that real is open work.

## Known seams

Where the tree does not yet match the story above. Named here so a reader is not misled:

- **Model placement has no single rule.** GigaAM lives in `npu-asr` (a model-specific crate
  under a generic name), Parakeet in `npu-parakeet`, but BERT and ESM live as modules INSIDE
  `npu-engine`, and Whisper is split across both `npu-whisper` and `npu-engine/src/asr/`.
- **`npu-engine` is mostly not the engine.** Its generic core (`api`, `config`, `lib`,
  `pipeline`, `registry`, `tuning_profile`) is ~460 lines; the other ~5500 are the ASR, BERT
  and ESM model implementations plus probe binaries that happen to share its manifest.
- **The capability set is closed.** `ModelKind { Asr, Embed }` is threaded through
  `api.rs`, `pipeline.rs`, `loader.rs`, `actor.rs` and `select.rs`, so a third modality means
  editing all five. This is why `npu-sr` and `npu-gemma` sit outside the request path.
- **Kernel binaries are selected by hardcoded path.** Model crates name xclbins as string
  literals with shape, tile, column count and variant encoded in the filename (for example
  `final_512x1024x4096_64x32x128_8c_modalsilu.xclbin`). There is no machine-readable
  statement of which ops a model needs, which is what a dynamically composed binary would want.
- **`&self` on the model entry points is not the truth.** `transcribe(&self, ..)` mutates
  device-resident state through `RefCell` (the whisper decoder's KV cache; seven cache fields
  in `npu-parakeet/src/npu.rs`). Only `npu-sr` declares `&mut self` honestly.

The performance work lives below this seam. The recurring theme is **eliminating data
movement**, not speeding up arithmetic:

- **Resident dataflow** - keep weights, KV, and intermediate activations on-chip across
  steps instead of re-streaming them from LPDDR.
- **Dispatch collapse** - fuse an op sequence (e.g. a whole 12-layer decoder) into one ELF
  dispatch to remove per-op host round-trips and shape-reloads.
- **Op-count reduction** - fold biases and activations into GEMVs; replace per-batch DMA
  unrolls with on-chip BD-chain iteration.
- **Precision** - bf16 for parity, bfp16/int8 where accuracy allows, to cut bytes moved.

## Kernels

`route_b_kernels/` holds the AIE kernels the engine dispatches: whole-array GEMM, resident
GEMV, cascade FFN, single-query flash MHA, depthwise/2D conv, LayerNorm, softmax, and a
transpose path. They are built through the pinned `mlir-aie` toolchain (place-tiles model)
and validated against NumPy/ONNX goldens before use. The kernel-selection map - which
brick for which node and regime - is documented in `docs/`.
