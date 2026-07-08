# xdna-engine

A general inference engine for the **AMD XDNA2 (Strix) NPU**, written in Rust with
hand-written AIE kernels. It runs transformer and conv models - ASR, embeddings, small
LLMs, and vision - on the NPU under Linux via the open MLIR-AIE / IRON kernel stack,
with a host-CPU fallback for ops that are not yet on-device.

ASR was the first target, but the engine is not ASR-specific: the same primitives
(resident dataflow, fused decode, KV cache, multi-precision GEMM/GEMV) serve every
front through one `Frontend / Encoder / Head` pipeline.

## Why the NPU

This pipeline is **data-movement-bound, not compute-bound**. The NPU's cores sit mostly
idle; the cost is bytes streamed from LPDDR and array shape-reloads. The engine is built
around that fact: keep weights and activations on-chip, fuse op sequences into few
dispatches, and quantize to cut the bytes moved. The payoff is latency, energy, and
freeing the CPU - see [docs/data-movement-thesis.md](docs/data-movement-thesis.md).

## What works today

- **ASR** - GigaAM-v3 and Parakeet FastConformer encoders on the NPU; Whisper-small
  encoder + a full 12-layer decoder fused into a single ELF dispatch.
- **Embeddings** - BGE / MiniLM / E5 / ModernBERT BERT encoders on the NPU, served over
  an OpenAI-compatible `/v1/embeddings` endpoint.
- **Small LLMs** - opt-125m and a Gemma 3 bring-up reusing the resident-FFN + fused-decode
  + KV primitives (weight-bandwidth-bound; int8 is the sweet spot).
- **Vision** - ViT, DINOv2, and ResNet-18 through a general conv2d path.
- **Precision** - selectable bf16 / bfp16 / int8, per-op, gated on WER/accuracy.

Representative measured results (host: AMD Ryzen AI 9 465, XDNA2, Linux):

| Result | Number |
| --- | --- |
| GigaAM encoder, NPU vs CPU | 651 ms vs 890 ms |
| Parakeet resident engine | 4.0 s -> 0.70-0.92 s / clip, WER-lossless |
| BGE embeddings, NPU vs host | 2.5-4x |
| opt-125m decode, int8 | 92 -> 47 ms/token, golden-exact |
| aiecc kernel build | 536 s -> ~7 min cold, < 10 s warm |

## Build and run

The AIE toolchain is pinned in `toolchain.lock`. `install.sh` sets up the engine and a
control-plane service.

    ./install.sh                 # build + install the service
    npu serve                    # start the engine
    npu transcribe audio.wav     # run ASR
    npu embed "some text"        # run embeddings
    npu models                   # list loaded models

Weight arenas are baked from Hugging Face checkpoints with `npu bake` (see
`rust/npu-weights`). Model export/convert scripts live in `scripts/`.

## Layout

- `rust/` - the engine (12 crates; see [ARCHITECTURE.md](ARCHITECTURE.md))
- `route_b_kernels/` - hand-written AIE kernels (GEMM, GEMV, cascade FFN, MHA, conv, LayerNorm, ...)
- `scripts/` - model export/convert + kernel build helpers
- `bench/` - latency/energy benchmark harness
- `docs/` - engineering deep-dives (data-movement thesis, brick catalog, benchmark methodology, ...)
- `mlir-aie/` - pinned submodule (the open AIE toolchain)

## Hardware

AMD Ryzen AI 9 465 (Krackan, XDNA2), Linux with the `amdxdna` driver and `/dev/accel/accel0`.
The open IRON / MLIR-AIE path is distro-agnostic.

## License

Apache-2.0. See [LICENSE](LICENSE).
