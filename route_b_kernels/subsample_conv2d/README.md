# subsample_conv2d -- Parakeet conv2D /8 front-end as im2col -> mmul GEMM

Task A5 (CPU/build-only draft). Reformulates the FastConformer `dw_striding`
subsample (mel `[1,128,T]` -> `[T/8, 1024]`) into the **patch-embed idiom**:
each conv2d = im2col (gather receptive-field patches) -> **`aie::mmul` GEMM**
(COMPUTE brick, not a mac+reduce scalar loop) -> optional **fused ReLU epilogue**.

## Files
- `subsample_patch_embed.cc` -- AIE2P kernel. Two extern-C entries over pre-tiled
  mmul blocks (bf16 operands, **f32 accumulate**, bf16 out):
  - `patch_embed_relu_bf16` = `relu(A@B + bias)` (conv.0, conv.3, conv.6).
  - `patch_embed_bf16` = `A@B + bias` (depthwise blocks + out-projection).
  Modeled on IRON `aie2p/mm.cc` (2x2 m/n mmul expansion). Bias folded via host
  K-augmentation (core takes only A,B = 2 DMA inputs). ReLU is per-element so it
  is layout-independent w.r.t. the mmul-blocked C storage.
- `golden_subsample.py` -- numpy golden. Gates: (1) im2col->GEMM == reference
  conv2d (f64, rel 8e-16); (2) f64 e2e+out-proj vs `block_in` (2.9e-7);
  (3) **bf16-operand/f32-acc e2e vs `block_in` = 5.0e-3 <= 0.08** (the node gate).
- `build_check.sh` -- runs the golden + Peano compile-check (4x8x8 bf16 emul AND
  bfp16 8x8x8 true-systolic). Both compile clean.

## Per-conv lowering (host pre-tiling recipe for the NPU wire-up)
M = output positions (Hout*Wout), K = patch size (Cin*kh*kw, +1 for K-aug bias),
N = Cout. Host pads K to a multiple of s=8 and M,N to multiples of 2r,2t.

| conv | kind | stride/pad/k | M (Hout*Wout) | K (Cin*kh*kw) | N (Cout) | act |
|---|---|---|---|---|---|---|
| conv.0 | dense (1->256) | 2 / 1 / 3 | 128*64 = 8192 | 1*9 = 9 | 256 | ReLU |
| conv.2 | depthwise g=256 | 2 / 1 / 3 | 64*32 = 2048 | 256*9 (block-diag) | 256 | - |
| conv.3 | pointwise 1x1 | 1 / 0 / 1 | 64*32 = 2048 | 256 | 256 | ReLU |
| conv.5 | depthwise g=256 | 2 / 1 / 3 | 32*16 = 512 | 256*9 (block-diag) | 256 | - |
| conv.6 | pointwise 1x1 | 1 / 0 / 1 | 32*16 = 512 | 256 | 256 | ReLU |
| out    | dense proj | - | T/8 = 32 | 4096 | 1024 | - |

im2col layout is Cin-major (each channel's kh*kw contiguous on K), so the
depthwise weight is block-diagonal (`golden_subsample.py:conv2d_gemm`).

## Brick notes / follow-ups
- The **pointwise (1x1)** convs are a degenerate im2col (K=Cin) = a pure GEMM --
  the cleanest patch-embed; conv.0 is the classic 1->256 patch embed.
- The **depthwise** convs are expressed here as a block-diagonal GEMM only to keep
  the whole chain GEMM-shaped for the golden; their on-device hot path is
  **`sliding_mul`** (task A1, the dwconv brick), not this dense mmul.
- bfp16 8x8x8 (the ~4x true-systolic format) compiles; WER-gate before shipping.
- NOT YET wired into an IRON/air generator (no NPU this phase). Next: host im2col
  ObjectFifo (the `dma_bd` n-D strided gather does im2col for free) + the K-aug
  bias + tile sizing per the table -> route through `npu-xrt`.
