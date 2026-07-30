#!/usr/bin/env python3
"""Host numpy reference for the S2 codec's QUANTIZER decode path -- the segment nobody had ported:

    codes [10, T]  ->  RVQ lookup (10 codebooks)  ->  quantizer.post_module transformer (8 layers)
                    ->  quantizer.upsample x2 (factor 2 each, 4x total)  ->  latent [1024, 4T]

codec_decoder_ref.py's own docstring says exactly what it left out: "the quantizer half (RVQ
lookup, post_module transformer, quantizer upsamples) is taken from the oracle as codec_latent, and
only the decoder half is reimplemented here." This script is that missing half. Chained with
codec_decoder_ref.decode(), it takes codec_codes.bin all the way to codec_audio.bin on host with no
oracle dump in between -- spec G1's harness (see __main__: it runs both stages and reports both
rel-L2s).

Graph (s2.cpp/src/s2_codec.cpp):
    build_decode_codes_stage_backend  (~648)  RVQ lookup: gather + out_proj, summed over 10 codebooks
    build_quantizer_decode_stage      (~463)  calls build_transformer then build_quantizer_stage_up x2
    build_transformer                 (~290)  generic pre-norm transformer: RMSNorm/RoPE/MHA/SwiGLU-FFN
    build_quantizer_stage_up          (~451)  causal_conv_transpose_1d (UNCROPPED) + build_convnext_block
    build_convnext_block              (~395)  depthwise conv + LayerNorm + pwconv1 + GELU(erf) + pwconv2

All config values below (dims, n_layer, rope_base, eps, window_size, downsample factors, codebook
sizes) were read directly from the GGUF's `fish_speech.codec.*` key-value metadata, not re-derived
or guessed -- see the constants block. quantizer_downsample_factor = [2, 2] is why T -> 4T (measured
on real dumps: codes [10, 55] -> latent [1024, 220], and [10, 65] -> [1024, 260]).

THREE LAYOUT DETAILS THAT SILENTLY PRODUCE A WRONG-BUT-SHAPED ANSWER IF MISSED:

1. codec_codes.bin is INT32, not float32. codec_decoder_ref.load_dump hardcodes
   `np.fromfile(..., dtype=np.float32)` because every OTHER dump in this pipeline is f32 -- reusing
   it here would reinterpret int32 code-ID bit patterns as floats. load_codes() below reads it right,
   and needs no [C,T]-transpose either: codec_codes.bin is a raw C host array `codes[cb*n_frames+t]`
   (row-major [n_cb, n_frames] already), unlike the ggml-tensor dumps (codec_latent.bin,
   codec_audio.bin) whose ne[0]-fastest memory order needs the reshape-then-transpose dance
   codec_decoder_ref.py does in its own main().

2. RoPE here is ggml mode=0 = GGML_ROPE_TYPE_NORMAL: INTERLEAVED pairs (x[2i], x[2i+1]), NOT the
   NeoX-style half-split pairing (x[i], x[i+d/2]) that e.g. Llama/GPT-NeoX-family ports usually
   default to. Confirmed by reading the ggml_rope_ext call site (mode arg = 0) and ggml.h's own
   mode-table comment (`GGML_ROPE_TYPE_NORMAL n_dims=8 --> [cscscscs]`). Get this backwards and the
   attention output is wrong in a way that looks numerically plausible (RoPE is still a rotation,
   just of the wrong coordinate pairs), not a shape or NaN error.

3. The quantizer's causal_conv_transpose_1d is called with crop_right=0 (no cropping at all) --
   different from the DECODER's upsample, which crops crop_right=stride. Conveniently, kernel_size
   == stride == factor for every quantizer upsample layer (k=2, stride=2), so the uncropped output
   length is exactly T*factor with no fractional remainder to think about -- reuse
   codec_decoder_ref.conv_transpose_1d(..., crop_right=0) verbatim, don't write a new conv-transpose.

Verified vs inferred: every tensor this script reads is f16 (checked via gguf_shapes.py against
s2-pro-q6_k.gguf -- a q6_k FILE but the codec sub-tree is f16, not quantized). gguf_extract.py only
handles f32/f16/bf16, so if a future checkpoint ships a quantized codec tree, this script raises
NotImplementedError from gguf_extract rather than silently producing garbage -- that is the existing
gguf_extract.py behavior, not something this script special-cases.

INTENTIONALLY OUT OF SCOPE: `quantizer.pre_module` and every codebook's `in_proj.{weight,bias}` exist
in the GGUF but are ENCODE-path only (project_1x1 + quantize_with_vq, s2_codec.cpp ~570-620) -- the
decode path build_decode_codes_stage_backend calls never touch them (confirmed by reading both call
sites), and AudioCodec::decode() likewise never runs pre_module. This script only implements decode,
so neither is read here; do not add them "for completeness" without a reason tied to the encode path.

    python3 scripts/codec_quantizer_ref.py <dump-dir> [--gguf <path>]

============================================================================================
KERNEL MAP -- one row per op in this segment: existing route_b_kernels/bricks/ brick, or a gap.
============================================================================================

Two measured constraints from the ALREADY-PORTED decoder segment (route_b_kernels/codec_block/
residual_unit_bf16.cc's frontier table) that this map is judged against:
  (a) a streamed brick design takes ONE streamed operand + ONE resident operand;
  (b) the resident operand is the dominant L1 term -- at c=96 channels, f32 [c,64] fits, [c,128]
      overflows .bss (measured, not estimated).

THE SINGLE BIGGEST STRUCTURAL FACT ABOUT THIS SEGMENT: it runs at c=1024 (quantizer_input_dim /
rvq_transformer.dim) almost throughout, not c<=96..768 like the decoder conv stack. A resident f32
[1024, 64] tile is 262144 B (256 KB) -- 4x a whole L1 (64 KB) and half a whole MemTile COLUMN
(512 KB) for ONE buffer, before any scratch, before the streamed operand, before the second matmul
operand. Every dense op below (RVQ out_proj, RMSNorm/LayerNorm, all 5 transformer GEMMs, both
ConvNeXt pwconvs) therefore MUST tile over the channel axis and stage full-channel reductions
(RMSNorm's mean-of-squares, a GEMM's K-reduction) through PSUM/MemTile accumulation -- it cannot
hold a whole-channel [1024, t] tile resident in L1 the way parts of the c<=96 decoder stack can.
Design for this FIRST; it is not a detail to discover mid-port.

  op                                    existing brick                    notes / operand shapes
  ------------------------------------  ---------------------------------  --------------------------
  RVQ codebook gather (x10)             NONE -- see AWKWARD #1 below       resident table [size,8]
                                                                            (size=4096 semantic /
                                                                            1024 residual), T indices
                                                                            streamed, [T,8] out.
  RVQ out_proj GEMM+accumulate (x10)    gemm-bf16xbfp16 / -bfp16-ebs8      K=8 == ONE native
                                                                            aie::mmul<8,8,8> tile,
                                                                            no K-loop needed. resident
                                                                            weight [1024,8] bf16=16KB.
  attention_norm / ffn_norm (x16)       rmsnorm (beta=None variant)        brick's [rows=T,cols=1024]
                                                                            is THIS script's [C,T]
                                                                            transposed -- see the
                                                                            transpose-dma row below.
  wqkv projection (x8)                  gemm-bf16xbfp16                   [3072,1024]x[1024,T]; K=1024
                                                                            =128 tiles; weight alone is
                                                                            6MB bf16 / ~3MB bfp16ebs8 --
                                                                            MemTile-staged, nowhere
                                                                            near L1-resident.
  RoPE on Q,K (x8)                      rope-lut -- BLOCKED + WRONG LAYOUT see AWKWARD #2 below.
  causal(+window) softmax attention(x8) NONE -- no softmax brick exists    see AWKWARD #3. [T,T] score
                                         (grepped route_b_kernels/bricks;                    matrix is
                                         only inline use is inside                          SMALL here
                                         moe_topk_router.cc's top-k, not                    (T<=65 in the
                                         a generic row-softmax)                              test clips,
                                                                                              <=17KB) --
                                                                                              not the L1
                                                                                              pressure
                                                                                              point.
  QK^T and attn@V GEMMs (x8, x16 heads) gemm-bf16xbfp16 / gemv-int8 family [T,64]x[64,T] and
                                                                            [64,T]x[T,T]; K=64=8 tiles,
                                                                            or K=T (small, variable).
  wo / w1 / w3 / w2 projections (x8)    gemm-bf16xbfp16                   [1024,1024] and [3072,1024]
                                                                            x2 -- same MemTile-staged
                                                                            regime as wqkv.
  SwiGLU  silu(gate)*up (x8)            swiglu -- DIRECT, exact formula   [3072,T] elementwise; no
                                         match, no adaptation needed       adaptation needed.
  upsample causal_conv_transpose_1d(x2) conv-transpose-1d (already used   crop_right=0 (a call-site
                                         + verified for the decoder)      parameter, not new code);
                                                                            k==stride==2 so no
                                                                            fractional-crop case.
  ConvNeXt depthwise conv (x2)          dwconv1d_same_scalar<T,K,P,BIAS>  NOT a gap -- AWKWARD #4 is
                                         (route_b_kernels/dwconv1d/,      CORRECTED below. Reuse via
                                         Parakeet's shipped kernel)       P=K-1, device-green
                                                                            2026-07-31.
  ConvNeXt LayerNorm affine (x2)        layernorm (layernorm_ln_affine)   exact 2-pass mean-centered
                                                                            match. [1024,T] -> tiled.
  ConvNeXt pwconv1 / pwconv2 (x2)       gemm-bf16xbfp16                   1x1 convs = linear layers,
                                                                            K=1024 / K=4096.
  ConvNeXt GELU (erf-exact) (x2)        NONE, see AWKWARD #5              geglu.cc is TANH-approx
                                                                            GELU and is GATED (2
                                                                            inputs) -- wrong formula
                                                                            AND wrong arity for a
                                                                            plain single-input erf
                                                                            GELU.
  [C,T] <-> brick-native [T,C] layout   transpose-dma                    pure DMA relayout, zero
                                                                            compute cost -- same
                                                                            [C,T]<->[T,C] cliff this
                                                                            repo already crosses
                                                                            elsewhere.

REGIME NOTE. T
(code-frame count) runs from the tens to low hundreds in real utterances, i.e. M=T is almost always
>= 8 -- this is the COMPUTE-BOUND batched/encoder regime, not the M=1 decode/overhead regime. That is
why every GEMM above is mapped to gemm-bf16xbfp16 (a COMPUTE/FORMAT brick) rather than a gemv-int8
decode brick: the regime rule says COMPUTE/FORMAT bricks win at M>=8, and this segment runs once per
utterance over the WHOLE code-frame sequence, never per-token. (The rule of thumb: compute- and
format-oriented primitives pay at M>=8, while at M=1 the wins come from data movement and op count
instead.)

AWKWARD ON-NPU (deliverable 4):

  #1 RVQ codebook gather is a genuine indexed row-gather (T runtime indices -> T rows of width 8 out
     of a resident table with up to 4096 rows), and NOTHING in the current catalog does this. Do not
     confuse it with rope-lut's/sin's `aie::lut<4>::parallel_lookup` gather (probed extensively in
     bricks/_verify/probe_gather_{known,width,ramp}.py) -- that is a 256-entry SCALAR hardware LUT
     (one bf16 value per int8 key), an entirely different mechanism from an N-row VECTOR-embedding
     gather. Two honest options: (a) a new indexed-DMA gather brick (unverified whether AIE2P's BD
     engine supports data-dependent offsets at all -- would need its own probe, analogous to but
     distinct from the existing lut probes); (b) do the gather on HOST -- it is tiny (10 codebooks x
     T frames x 8 f32 = a few thousand floats for a whole utterance) and stream the gathered [8,T]
     per-codebook result to the NPU for the out_proj GEMM. (b) is the pragmatic near-term path.

  #2 RoPE needs rope-lut, and rope-lut is BLOCKED on an unresolved `aie::lut<4>` ab/cd duplication
     layout question (route_b_kernels/bricks/sin/sin.cc's header cites
     log/2026-07/2026-07-25-rope-lut-root-cause-bank-granularity.md, "the layout question is OPEN",
     plus bricks/_verify/probe_linear_approx_abcd.py still returning permuted entries). This
     transformer's attention NEEDS RoPE (confirmed: build_transformer calls ggml_rope_ext on both Q
     and K unconditionally), so this segment inherits that already-known-bad dependency directly --
     it is not a new blocker, but it is now a REAL one, not a hypothetical one, the moment this
     segment is ported. On top of the layout bug, rope-lut's own golden.py implements NeoX-style
     half-split pairing, while this transformer needs INTERLEAVED pairing (see layout detail #2
     above) -- so even a fixed rope-lut would need a pairing-mode parameter it does not have today.

  #3 No softmax brick exists at all (see kernel map row). Small in this segment (T<=65 keys), but
     still unwritten: exp (SFU) + causal-masked reduce-sum over the key axis + reciprocal-multiply.

  #4 CORRECTED 2026-07-31, DEVICE-GREEN -- this was never a gap. The original claim ("no depthwise
     brick exists in this catalog") was true only as scoped to bricks/: route_b_kernels/dwconv1d/
     dwconv1d.cc, Parakeet's shipped depthwise-1d kernel, lives OUTSIDE bricks/ and is directly
     reusable here by reparameterization -- the same reuse class as mha_decode HD=64->128.

     dwconv1d_same_scalar<T,K,P,BIAS> (dwconv1d.cc:114-129) computes
     out[t] = bias + sum_p w[p]*in[t-P+p], with the padding offset P a FREE TEMPLATE PARAMETER.
     Parakeet instantiates P=(K-1)/2 (centered 'same') at every call site (dwconv1d.cc:296-322) --
     a call-site choice, not a kernel property. Causal, which is what dwconv_1d_causal below needs
     (shift = K-1-j), is simply P=K-1.

     That distinction is load-bearing and is exactly the silent-shift failure class this project has
     paid for twice (the conv-1d weight transpose, the RoPE pairing convention): host-checked, P=K-1
     matches dwconv_1d_causal at rel-L2 3.1e-08 while the stock centered P=(K-1)/2 gives 0.92, i.e.
     unrelated output that no self-consistent gate would catch.

     S2's real weight (c.quantizer.upsample.{0,1}.1.dwconv.conv.weight, read from the shipped GGUF)
     is K=7, C=1024, and T is data-dependent (110, 220, ...). Use the SCALAR FIR form: the vectorized
     dwconv1d_shift hardcodes static_assert(K==9) (dwconv1d.cc:152,:214) and requires T%16==0
     (dwconv1d.cc:153), neither of which holds here; dwconv1d_same_scalar asserts neither, so K=7 and
     an arbitrary T compile unmodified. It also has no fused SiLU (that lives behind -DDWCONV_SILU on
     the shift variant, default off), which is what this block wants -- bias-add only.

     Device-gated by bricks/_verify/verify_dwconv_causal.py: upsample.0 (T=110) rel-L2 4.185e-03 and
     upsample.1 (T=220) 4.031e-03, run2run 0, three fresh builds. L1 is a non-issue: the kernel is
     per-channel-row, so ~1.8 KB/core at depth 2, independent of C.

     Which FIR form is FASTER (scalar vs the vectorized shift path, with K zero-padded to 9 and T
     padded to a multiple of 16) is an open, unmeasured perf question -- not settled here.

  #5 No plain (single-input, ungated) GELU brick exists, exact-erf or otherwise; geglu.cc is both the
     wrong formula (tanh-approx, not erf) and the wrong arity (gated, needs 2 inputs) for this block's
     ungated pwconv1->GELU->pwconv2. Whether tanh-approx GELU's error vs erf-GELU is small enough to
     repurpose geglu with the gate branch forced off is unverified -- would need its own numeric check
     before relying on it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402
import codec_decoder_ref as decoder_ref  # noqa: E402 -- reused for conv_transpose_1d + the G1 chain

DEFAULT_GGUF = codec_paths.gguf()
QPREFIX = "c.quantizer"

# --- constants, read directly from the GGUF's fish_speech.codec.* KV metadata, not re-derived ---
SEMANTIC_CODEBOOK_SIZE = 4096     # fish_speech.codec.quantizer_semantic_codebook_size
RESIDUAL_CODEBOOK_SIZE = 1024     # fish_speech.codec.quantizer_residual_codebook_size
QUANTIZER_INPUT_DIM = 1024        # fish_speech.codec.quantizer_input_dim (== latent_dim)
DOWNSAMPLE_FACTORS = [2, 2]       # fish_speech.codec.quantizer_downsample_factor
RVQ_HEAD_DIM = 64                 # fish_speech.codec.rvq_transformer.head_dim
RVQ_ROPE_BASE = 10000.0           # fish_speech.codec.rvq_transformer.rope_freq_base
RVQ_NORM_EPS = 1e-5               # fish_speech.codec.rvq_transformer.layer_norm_rms_eps (~1e-5)
RVQ_WINDOW_SIZE = 128             # fish_speech.codec.rvq_transformer.window_size
CONVNEXT_LN_EPS = 1e-6            # matches build_convnext_block's layer_norm_affine(..., 1e-6f)


def load_codes(dump_dir):
    """codec_codes.bin is int32 (see module docstring, layout detail #1) -- do NOT reuse
    decoder_ref.load_dump, which hardcodes float32 for every other (ggml-tensor) dump in this
    pipeline. Also unlike those dumps, codes is a raw host C array [n_cb, n_frames] row-major
    already (s2_codec.cpp indexes it as codes[cb*n_frames+t]) -- no reshape-then-transpose needed."""
    d = Path(dump_dir)
    shape = [int(v) for v in (d / "codec_codes.shape").read_text().split()]
    flat = np.fromfile(d / "codec_codes.bin", dtype=np.int32)
    n_cb, n_frames = shape
    return flat.reshape(n_cb, n_frames)


def rms_norm(x, weight, eps):
    """x [C,T], weight [C] -> [C,T]. f64 mean-of-squares reduction (kb/bf16-norm-numerics-and-
    accumulation-guards: summing a norm reduction in low precision loses the norm; this is a host
    reference so there is no reason not to just do it right)."""
    xf = x.astype(np.float64)
    ms = np.mean(xf * xf, axis=0, keepdims=True)
    y = xf / np.sqrt(ms + eps)
    return (y * weight.reshape(-1, 1).astype(np.float64)).astype(np.float32)


def layer_norm_affine(x, weight, bias, eps):
    """x [C,T], weight/bias [C] -> [C,T]. Standard 2-pass mean-centered LayerNorm over the channel
    axis, matching ggml_norm + affine (layer_norm_affine in s2_codec.cpp)."""
    xf = x.astype(np.float64)
    mean = xf.mean(axis=0, keepdims=True)
    var = np.mean((xf - mean) ** 2, axis=0, keepdims=True)
    y = (xf - mean) / np.sqrt(var + eps)
    y = y * weight.reshape(-1, 1).astype(np.float64) + bias.reshape(-1, 1).astype(np.float64)
    return y.astype(np.float32)


def _erf(x):
    """erf via Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7, i.e. at f32 machine eps --
    negligible next to this script's ~1e-3 rel-L2 gate). Written out rather than importing
    scipy.special.erf to keep this script numpy-only, matching every other script in this tree."""
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def gelu_erf(x):
    """Exact erf-based GELU: 0.5*x*(1+erf(x/sqrt(2))), matching ggml_vec_gelu_erf_f32 exactly (see
    module docstring AWKWARD #5 -- no on-device brick implements this today; geglu.cc is a tanh
    approximation of a DIFFERENT, gated, op)."""
    xf = x.astype(np.float64)
    return (0.5 * xf * (1.0 + _erf(xf / np.sqrt(2.0)))).astype(np.float32)


def silu(x):
    xf = x.astype(np.float64)
    return xf / (1.0 + np.exp(-xf))


def dwconv_1d_causal(x, w, bias):
    """x [C,T], w [C,1,K] (depthwise: one length-K kernel per channel, ggml_conv_1d_dw layout),
    bias [C] -> [C,T]. Same causal shift-and-accumulate structure as codec_decoder_ref.conv_1d_causal,
    but multiply-per-channel instead of a matmul (no cross-channel mixing) -- see kernel map AWKWARD
    #4: nothing in the brick catalog does this natively."""
    c, one, k = w.shape
    assert one == 1
    t = x.shape[1]
    y = np.zeros((c, t), np.float64)
    xf = x.astype(np.float64)
    wf = w.astype(np.float64)
    for j in range(k):
        shift = k - 1 - j
        if shift < t:
            y[:, shift:] += wf[:, 0, j:j + 1] * xf[:, :t - shift]
    return (y + bias.reshape(-1, 1).astype(np.float64)).astype(np.float32)


def convnext_block(x, W, prefix):
    """depthwise conv -> LayerNorm(affine) -> pwconv1 -> GELU(erf) -> pwconv2 -> gamma-scale ->
    residual. Matches build_convnext_block exactly (s2_codec.cpp ~395)."""
    dw_w = W(f"{prefix}.dwconv.conv.weight")
    dw_b = W(f"{prefix}.dwconv.conv.bias").reshape(-1)
    y = dwconv_1d_causal(x, dw_w, dw_b)
    y = layer_norm_affine(y, W(f"{prefix}.norm.weight").reshape(-1),
                          W(f"{prefix}.norm.bias").reshape(-1), CONVNEXT_LN_EPS)
    pw1_w = W(f"{prefix}.pwconv1.weight").reshape(-1, y.shape[0])   # [4096,1024]
    pw1_b = W(f"{prefix}.pwconv1.bias").reshape(-1)
    y = (pw1_w.astype(np.float64) @ y.astype(np.float64) + pw1_b.reshape(-1, 1)).astype(np.float32)
    y = gelu_erf(y)
    pw2_w = W(f"{prefix}.pwconv2.weight").reshape(-1, y.shape[0])   # [1024,4096]
    pw2_b = W(f"{prefix}.pwconv2.bias").reshape(-1)
    y = (pw2_w.astype(np.float64) @ y.astype(np.float64) + pw2_b.reshape(-1, 1)).astype(np.float32)
    gamma = W(f"{prefix}.gamma").reshape(-1, 1)
    y = y * gamma
    return (x + y).astype(np.float32)


def quantizer_stage_up(x, W, prefix, factor):
    """causal_conv_transpose_1d(stride=factor, crop_right=0) + build_convnext_block. Reuses
    decoder_ref.conv_transpose_1d verbatim (see module docstring layout detail #3: crop_right=0 is
    just a call-site argument that function already supports, not new conv-transpose code)."""
    w = W(f"{prefix}.0.conv.weight")
    b = W(f"{prefix}.0.conv.bias").reshape(-1)
    x = decoder_ref.conv_transpose_1d(x, w, b, factor, crop_right=0)
    x = convnext_block(x, W, f"{prefix}.1")
    return x


def _causal_window_mask(seq_len, window_size):
    """Additive mask [T_k, T_q]: 0.0 where key k is allowed for query q, -1e9 otherwise. Matches
    prepare_transformer_inputs exactly: sliding window IFF window_size>0 and window_size<seq_len,
    else plain causal (the branch this model's CPU path actually takes for T <= 128, since
    RVQ_WINDOW_SIZE=128 and every test clip has T well under that -- implemented generally anyway
    since a longer utterance would cross that threshold)."""
    q = np.arange(seq_len)[None, :]
    k = np.arange(seq_len)[:, None]
    allowed = k <= q
    if 0 < window_size < seq_len:
        allowed = allowed & (k >= (q - window_size + 1))
    return np.where(allowed, 0.0, -1e9)


def _rope_normal(x, positions, base):
    """x [n_head, head_dim, T]. GGML_ROPE_TYPE_NORMAL (mode=0): INTERLEAVED pairs (x[2i], x[2i+1]),
    NOT the NeoX half-split layout -- see module docstring layout detail #2, this is the exact
    hazard the rope-lut brick's golden.py (NeoX-style) does not match."""
    n_head, head_dim, t = x.shape
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = base ** (-(2.0 * i) / head_dim)              # [half]
    theta = positions[None, :] * inv_freq[:, None]          # [half, T]
    cos = np.cos(theta)[None, :, :]                          # [1, half, T]
    sin = np.sin(theta)[None, :, :]
    xf = x.astype(np.float64)
    x_even = xf[:, 0::2, :]
    x_odd = xf[:, 1::2, :]
    out = np.empty_like(xf)
    out[:, 0::2, :] = x_even * cos - x_odd * sin
    out[:, 1::2, :] = x_even * sin + x_odd * cos
    return out.astype(np.float32)


def _softmax_over_keys(scores):
    """scores [n_head, T_k, T_q] (already masked) -> softmax over axis=1 (keys), f64 for stability
    at the small T this segment runs at (no brick exists for this -- kernel map AWKWARD #3)."""
    s = scores.astype(np.float64)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def rvq_transformer(x, W, prefix):
    """The generic pre-norm transformer build_transformer implements, specialised to this model's
    verified config (no GQA: rvq_transformer.n_local_heads=-1 in the GGUF means "use n_head", so
    K/V have the same head count as Q -- repeat_interleave_heads is a no-op here and is not
    implemented; a model that actually used GQA would need it added back).

    x [dim=1024, T] -> [dim, T]. Layer count discovered dynamically (probes tensor names until one
    is missing), matching build_transformer's own `for (i=0;;++i) { ...; if (!wqkv) break; }`
    rather than hard-coding n_layer=8.
    """
    dim, t = x.shape
    n_head = dim // RVQ_HEAD_DIM
    positions = np.arange(t, dtype=np.float64)
    mask = _causal_window_mask(t, RVQ_WINDOW_SIZE)

    layer = 0
    while True:
        stem = f"{prefix}.layers.{layer}"
        try:
            wqkv = W(f"{stem}.attention.wqkv.weight")   # [3*dim_or_less, dim]
        except KeyError:
            break

        q_size = dim
        kv_size = (wqkv.shape[0] - q_size) // 2
        assert kv_size == dim, (
            f"layer {layer}: kv_size={kv_size} != dim={dim} -- this model has no GQA "
            "(n_local_heads=-1 in the GGUF); a GQA model would need repeat_interleave_heads added")

        wo = W(f"{stem}.attention.wo.weight")
        w1 = W(f"{stem}.feed_forward.w1.weight")
        w2 = W(f"{stem}.feed_forward.w2.weight")
        w3 = W(f"{stem}.feed_forward.w3.weight")
        ffn_norm = W(f"{stem}.ffn_norm.weight").reshape(-1)
        attn_norm = W(f"{stem}.attention_norm.weight").reshape(-1)
        attn_gamma = W(f"{stem}.attention_layer_scale.gamma").reshape(-1, 1)
        ffn_gamma = W(f"{stem}.ffn_layer_scale.gamma").reshape(-1, 1)

        attn_in = rms_norm(x, attn_norm, RVQ_NORM_EPS)
        qkv = (wqkv.astype(np.float64) @ attn_in.astype(np.float64))   # [3*dim, T]
        q = qkv[0:q_size]
        k = qkv[q_size:q_size + kv_size]
        v = qkv[q_size + kv_size:q_size + 2 * kv_size]

        q3 = q.reshape(n_head, RVQ_HEAD_DIM, t)
        k3 = k.reshape(n_head, RVQ_HEAD_DIM, t)
        v3 = v.reshape(n_head, RVQ_HEAD_DIM, t)

        q3 = _rope_normal(q3, positions, RVQ_ROPE_BASE)
        k3 = _rope_normal(k3, positions, RVQ_ROPE_BASE)

        scores = np.einsum("hdk,hdq->hkq", k3.astype(np.float64), q3.astype(np.float64))
        scores = scores / np.sqrt(RVQ_HEAD_DIM)
        scores = scores + mask[None, :, :]
        probs = _softmax_over_keys(scores)                             # [n_head, T_k, T_q]
        attn = np.einsum("hdk,hkq->hdq", v3.astype(np.float64), probs)  # [n_head, head_dim, T]
        attn2d = attn.reshape(dim, t).astype(np.float32)

        attn_out = (wo.astype(np.float64) @ attn2d.astype(np.float64)).astype(np.float32)
        attn_sc = attn_out * attn_gamma
        h = x + attn_sc

        ff_in = rms_norm(h, ffn_norm, RVQ_NORM_EPS)
        gate = (w1.astype(np.float64) @ ff_in.astype(np.float64))
        up = (w3.astype(np.float64) @ ff_in.astype(np.float64))
        ff_h = (silu(gate) * up).astype(np.float32)
        ff_out = (w2.astype(np.float64) @ ff_h.astype(np.float64)).astype(np.float32)
        ff_sc = ff_out * ffn_gamma
        x = (h + ff_sc).astype(np.float32)

        layer += 1

    norm_w = W(f"{prefix}.norm.weight").reshape(-1)
    return rms_norm(x, norm_w, RVQ_NORM_EPS)


def rvq_lookup(codes, W, prefix):
    """codes [10, T] int -> stage [1024, T]. Gather + out_proj per codebook, summed. Matches
    build_decode_codes_stage_backend: row 0 is the semantic codebook (size 4096), rows 1..9 are the
    9 residual codebooks (size 1024 each). See module docstring / kernel map AWKWARD #1: this gather
    has no existing brick."""
    n_cb, t = codes.shape
    stage = np.zeros((QUANTIZER_INPUT_DIM, t), np.float64)
    for cb in range(n_cb):
        if cb == 0:
            sub = f"{prefix}.semantic_quantizer.quantizers.0"
            size = SEMANTIC_CODEBOOK_SIZE
        else:
            sub = f"{prefix}.quantizer.quantizers.{cb - 1}"
            size = RESIDUAL_CODEBOOK_SIZE
        codebook = W(f"{sub}.codebook.weight")                  # [size, codebook_dim]
        out_w = W(f"{sub}.out_proj.weight").reshape(QUANTIZER_INPUT_DIM, -1)  # [1024, codebook_dim]
        out_b = W(f"{sub}.out_proj.bias").reshape(-1)
        # clamp matches AudioCodec::decode's inline clamp_code lambda exactly (clip to [0,size-1]);
        # a no-op in practice since oracle codes are already valid, kept for parity with the C++.
        idx = np.clip(codes[cb].astype(np.int64), 0, size - 1)
        gathered = codebook[idx]                                 # [T, codebook_dim]
        proj = out_w.astype(np.float64) @ gathered.T.astype(np.float64) + out_b.reshape(-1, 1)
        stage += proj
    return stage.astype(np.float32)


def decode(codes, W, verbose=False):
    def note(msg):
        if verbose:
            print(f"    {msg}", flush=True)

    stage = rvq_lookup(codes, W, QPREFIX)
    note(f"RVQ lookup ({codes.shape[0]} codebooks) -> {stage.shape}")

    x = rvq_transformer(stage, W, f"{QPREFIX}.post_module")
    note(f"post_module transformer -> {x.shape}")

    for i, factor in enumerate(reversed(DOWNSAMPLE_FACTORS)):
        x = quantizer_stage_up(x, W, f"{QPREFIX}.upsample.{i}", factor)
        note(f"upsample {i} x{factor} -> {x.shape}")

    return x


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den != 0.0 else float(num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--gate", type=float, default=3e-2,
                    help="rel-L2 gate, applied to both the latent check and the chained audio check")
    args = ap.parse_args()

    cache = {}

    def W(name):
        if name not in cache:
            cache[name] = gx.load(args.gguf, name).astype(np.float32)
        return cache[name]

    codes = load_codes(args.dump_dir)
    print(f"codes {codes.shape}")

    latent = decode(codes, W, verbose=True)
    print(f"computed latent {latent.shape}")

    zf, zshape = decoder_ref.load_dump(args.dump_dir, "codec_latent")
    oracle_latent = zf.reshape(zshape[1], zshape[0]).T.copy()
    print(f"oracle latent   {oracle_latent.shape}")

    assert latent.shape == oracle_latent.shape, \
        f"shape mismatch: {latent.shape} vs {oracle_latent.shape}"
    latent_rl2 = rel_l2(latent, oracle_latent)
    print(f"  rel-L2 vs codec_latent.bin : {latent_rl2:.6e}   (gate {args.gate:.1e})")
    latent_ok = latent_rl2 <= args.gate
    print("PASS" if latent_ok else "FAIL")

    # --- spec G1's host harness: chain into codec_decoder_ref and go all the way to audio ---
    print("\n[G1 chain] codes -> (this script) -> latent -> (codec_decoder_ref) -> audio")
    audio = decoder_ref.decode(latent, W, verbose=True).reshape(-1)

    af, ashape = decoder_ref.load_dump(args.dump_dir, "codec_audio")
    oracle_audio = af.reshape(-1)
    assert audio.shape == oracle_audio.shape, \
        f"audio shape mismatch: {audio.shape} vs {oracle_audio.shape}"
    audio_rl2 = rel_l2(audio, oracle_audio)
    print(f"  rel-L2 vs codec_audio.bin  : {audio_rl2:.6e}   (gate {args.gate:.1e})")
    audio_ok = audio_rl2 <= args.gate
    print("PASS" if audio_ok else "FAIL")

    return 0 if (latent_ok and audio_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
