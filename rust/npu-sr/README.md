# npu-sr: video super-resolution on the AMD XDNA2 NPU

Offload video upscaling to the NPU on your AMD laptop, on Linux, keeping the GPU free for the game or app
that needs it. `npu-sr` upscales a video file (for example 1080p to 4K) by running a small
super-resolution network on the XDNA2 NPU. It is positioned on efficiency, not on beating a large GPU
model: the NPU sits idle most of the time, and super-resolution is compute-bound, which is the one regime
where the NPU is a good fit.

This is the engine library plus two thin adapters:

- a Rust library (`npu_sr`) with a stable frame-in / frame-out API,
- an `xdna-sr` command-line tool that upscales a whole video file,
- a standalone FFmpeg video filter, `vf_xdna_sr`, so any FFmpeg pipeline can use it.

## What it produces

Two networks ship as data (a schedule plus baked weights), both running on the same conv / pixel-shuffle
/ residual rails:

| net    | size  | quality (kodim23, x3, Y-PSNR) | role |
|--------|-------|-------------------------------|------|
| espcn  | ~20k  | 31.79 dB (+0.43 over bicubic) | the fast path and the pipeline proof |
| edsr   | 1.55M | 34.29 dB (+2.93 over bicubic) | the shipped quality |

The NPU reproduces the reference (PyTorch) output to within about 0.3% relative L2, so the on-device
result matches what the network produces on the host.

## Build

Prerequisites: the Rust toolchain, FFmpeg (the `ffmpeg`/`ffprobe` binaries on PATH; the pipeline shells
out to them for decode and encode), and the AIE toolchain for building the NPU kernel (see the repo's
`scripts/toolchain_up.sh`).

    # engine + CLI + C ABI
    cd rust
    cargo build --release -p npu-sr -p npu-sr-capi

Bake the weights arena for the network you want (uses the declarative weight intake; the source is a
pretrained model, converted by the `espcn` / `edsr` arch):

    # from the repo root; needs a Python env with onnx/torch/super_image for the oracle export
    <venv>/bin/python scripts/export_edsr.py --scale 3     # writes artifacts/edsr/{edsr.json, edsr_base.safetensors}
    cargo run -p npu-weights -- bake \
      --source path:artifacts/edsr/edsr_base.safetensors --arch edsr \
      --arena target/test-arenas/edsr.safetensors --force

Build the whole-array GEMM kernel the NPU frontier uses (once; aie2p / NPU2, bf16):

    cd mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
    make M=512 K=576 N=256 n_aie_cols=8 NPU2=1 dtype_in=bf16 dtype_out=f32 \
         build/final_512x576x256_32x32x32_8c.xclbin
    cp build/insts_512x576x256_32x32x32_8c.bin build/insts_512x576x256_32x32x32_8c.txt

## Use it

Command line (run from the repo root so the schedule's arena path resolves):

    # NPU by default if a device is present; --cpu forces the reference CPU path
    xdna-sr input.mp4 output.mp4 --net edsr
    xdna-sr input.mp4 output.mp4 --net edsr --bench       # also prints ms/frame + extrapolated fps
    xdna-sr input.mp4 output.mp4 --net espcn --cpu        # fast net, CPU

FFmpeg filter (compiles the filter into a local FFmpeg build; see `ffmpeg/README-xdna-sr-filter.md`):

    bash ffmpeg/apply.sh
    LD_LIBRARY_PATH=rust/target/release target/ffmpeg-xdna/ffmpeg \
      -i input.mp4 -vf "xdna_sr=schedule=artifacts/edsr/edsr.json:npu=1" output.mp4

## How it works

The engine is the durable piece: a library that owns the schedule load, the resident NPU frontier, the
pixel-format and colorspace conversion, and the decode / upscale / encode pipeline. A network is data, an
ordered list of typed ops (`conv2d`, `pixel_shuffle`, residual `save` / `add`) over a small brick
vocabulary, plus a weights arena. Adding a network is writing its schedule and, only if needed, one new
op-type; the two shipped nets share the same rails. The CLI and the FFmpeg filter are thin adapters over
the library's frame API.

A convolution lowers to im2col followed by a whole-array GEMM on the NPU (int8 or bf16), which reuses the
engine's matmul path; pixel-shuffle is a strided DMA rearrange; residual connections are host-side adds.
Y-only nets (espcn) upscale the luma and resample chroma with bicubic; RGB nets (edsr) run all three
channels.

## Status and limitations

Offline file-to-file upscaling is the v1 product. The current NPU frontier uses a general whole-array
GEMM that is correct but not yet size-optimized for these shapes, so the reported ms/frame is a
single-hardware baseline, not the real-time figure; the follow-up is an on-chip windowed-DMA im2col that
keeps the feature map resident. Real-time playback is measured as headroom, not shipped in v1. The FFmpeg
filter is built out-of-tree against a local FFmpeg checkout.

## License

AGPL-3.0. See the repository root.
