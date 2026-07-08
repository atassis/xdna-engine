# Architecture

xdna-engine is a Rust workspace of focused crates layered over a kit of hand-written AIE
kernels. The design goal is a *general* engine: one execution pipeline that any
transformer or conv model plugs into, rather than a per-model stack.

## Layers

```
   models (ASR / embeddings / LLM / vision)
        |  Frontend / Encoder / Head traits
   npu-engine ......... general multi-model pipeline
        |
   npu-runtime ........ control plane: desired-state config, reconcile, one device actor
   npu-weights ........ bake HF safetensors/ONNX -> mmap bf16 weight arena
        |
   npu-xrt ............ safe Rust bindings over a C++ XRT shim -> the NPU
        |
   route_b_kernels .... hand-written AIE kernels (GEMM/GEMV/cascade/MHA/conv/LN/...)
   mlir-aie (submodule) the open AIE toolchain (kernel build + place-tiles)
```

## Crates

| Crate | Responsibility |
| --- | --- |
| `npu-xrt` | Safe Rust bindings to drive the XDNA2 NPU via a thin C++ XRT shim. |
| `npu-engine` | General multi-model engine over the kernel kit: a `Frontend / Encoder / Head` pipeline serving ASR and embeddings. |
| `npu-runtime` | Control plane over `npu-engine`: desired-state config, reconcile, and a single device actor that serializes NPU work. |
| `npu-weights` | Rust-native weight loader: bakes HF safetensors / ONNX into an mmap-able bf16 arena with a content fingerprint and parity gate. |
| `npu-onnx` | Runs ONNX graphs from Rust via a thin C shim over the system onnxruntime (oracles + fallback). |
| `npu-asr` / `npu-asr-host` | GigaAM-v3 encoder on the NPU (`npu-asr`) and its pure host-CPU reference math (`npu-asr-host`). |
| `npu-parakeet` | Parakeet-TDT FastConformer encoder (rel-pos attention, depthwise conv1d k=9, /8 conv2D subsample). |
| `npu-whisper` | Whisper-small encoder + decoder reference and the on-NPU decode path. |
| `npu-gemma` | Gemma 3 small-LLM decoder bring-up: the run-any-model proof, reusing the resident-FFN / fused-decode / KV primitives. |
| `npu-capi` | C ABI over `npu-engine` (cdylib + staticlib, cbindgen header) for in-process embedding from any language. |
| `npu-cli` | `npu` multitool: serve, transcribe, embed, models, config, reload, bake. |

## Dataflow

A request enters through a `Frontend` (tokenize / feature-extract), runs the model's
`Encoder` (and, for autoregressive models, a decode loop) on the NPU, and finishes in a
`Head` (pooling, projection, argmax). The device actor in `npu-runtime` serializes all NPU
dispatches, since the NPU is single-tenant.

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
