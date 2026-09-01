#!/usr/bin/env python3
"""Per-op S2 quantizer shapes, read from the real GGUF, and the L1-fit + TDR-time arithmetic that
follows -- the SIBLING of stage_shapes.py (decoder), not an extension of it: the decoder's stages
are all `snake -> conv_transpose_1d(k=2*stride) -> 3x residual_unit(k=7)`; this segment is a
transformer (RMSNorm/wqkv/RoPE/attention/SwiGLU) followed by `conv_transpose_1d(k==stride) ->
ConvNeXt(dwconv/LN/pwconv/GELU)`, and BOTH the op mix and the k/stride relationship differ, so the
formulas below are re-derived rather than inherited (the L1/TIME byte-counting primitives
themselves ARE shared -- imported from stage_shapes, not copied, since they don't depend on which
model owns the bytes).

Like stage_shapes.py, this file does NOT import window_driver/bricklib/aie.iron -- it stays a pure
shape/arithmetic utility (gguf reads + numpy-free integer math) so it runs standalone, device-free,
with no toolchain dependency. window_driver.T is therefore MIRRORED below as RES_T, not imported;
if window_driver.T ever changes, change RES_T here too (same convention stage_shapes.py's own
header states for its RES_T/UP_T mirrors).

THE ONE STRUCTURAL FACT THIS FILE IS BUILT AROUND: c_in==c_out==1024 (QUANTIZER_INPUT_DIM) through
nearly the whole segment -- the transformer body, the RVQ out_proj target, and BOTH conv_transpose
upsamples all land on 1024. Only the FFN's inner width (3072) and the ConvNeXt pwconv's inner width
(4096) go wider. That is a much narrower channel range than the decoder's 96..1536 sweep, so a
single `proj_plan`/`ct_plan` pair (self-checking ci_chunk against both L1 and TDR-time) covers
every projection and the conv_transpose alike, rather than a per-stage table.

    python3 quantizer_shapes.py        # print the shape + chunk-plan + dispatch-count table
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402
import stage_shapes as ss  # noqa: E402

GGUF = codec_paths.gguf()
QPREFIX = "c.quantizer"
DIM = 1024          # QUANTIZER_INPUT_DIM (codec_quantizer_ref.py)
HD = 64              # RVQ_HEAD_DIM
RVQ_D = 8            # RVQ codebook_dim (both semantic size=4096 and residual size=1024 rows)
SEMANTIC_ROWS = 4096
RESIDUAL_ROWS = 1024
GATHER_T_TILE = 16   # gather_rows.cc's GATHER_T_TILE; codes padded to a multiple of this
PREFILL_VL = 16      # prefill_attn.cc's softmax_core<VL> / prefill_attn_chunk key-chunk width

# Mirrored from scripts/codec_quantizer_ref.py's own constants block (that file's own comment:
# "read directly from the GGUF's fish_speech.codec.* KV metadata, not re-derived or guessed"),
# NOT imported -- quantizer_driver.py stays independent of the golden/reference module, the same
# separation decoder_chain.py already keeps from codec_decoder_ref.py. If codec_quantizer_ref.py's
# values ever change, change these too.
RVQ_ROPE_BASE = 10000.0
RVQ_NORM_EPS = 1e-5
RVQ_WINDOW_SIZE = 128
CONVNEXT_LN_EPS = 1e-6
DOWNSAMPLE_FACTORS = [2, 2]

# Mirror window_driver.py's T=64 constant (see module docstring -- this file stays aie.iron-free).
RES_T = 64

L1_BUDGET = ss.L1_BUDGET
F32 = ss.F32
TDR_MARGIN = ss.TDR_MARGIN
STREAM_MS_PER_KIB = ss.STREAM_MS_PER_KIB
CT_STREAM_MS_PER_KIB = ss.CT_STREAM_MS_PER_KIB
TDR_TIMEOUT_MS = ss.TDR_TIMEOUT_MS


# ---- shapes, read from the real GGUF -------------------------------------------------------

def n_layers():
    """post_module transformer layer count, discovered the same way codec_quantizer_ref.rvq_
    transformer does (probe tensor names until one is missing) rather than hardcoded -- the real
    GGUF has 8 (checked at import time below)."""
    i = 0
    while True:
        try:
            gx.load(GGUF, f"{QPREFIX}.post_module.layers.{i}.attention.wqkv.weight")
        except KeyError:
            break
        i += 1
    return i


def n_residual_codebooks():
    i = 0
    while True:
        try:
            gx.load(GGUF, f"{QPREFIX}.quantizer.quantizers.{i}.codebook.weight")
        except KeyError:
            break
        i += 1
    return i


def n_heads():
    assert DIM % HD == 0
    return DIM // HD


def proj_shape(name):
    """(c_out, c_in) of a plain 2D linear weight (wqkv/wo/w1/w2/w3/pwconv1/pwconv2 -- the GGUF
    stores these as ggml_mul_mat matrices, no trailing k=1 axis, unlike the conv-shaped tensors
    below)."""
    w = gx.load(GGUF, name)
    assert w.ndim == 2, f"{name}: expected a plain 2D linear weight, got {w.shape}"
    return int(w.shape[0]), int(w.shape[1])


def layer_proj_shapes(layer):
    p = f"{QPREFIX}.post_module.layers.{layer}"
    return dict(
        wqkv=proj_shape(f"{p}.attention.wqkv.weight"),
        wo=proj_shape(f"{p}.attention.wo.weight"),
        w1=proj_shape(f"{p}.feed_forward.w1.weight"),
        w2=proj_shape(f"{p}.feed_forward.w2.weight"),
        w3=proj_shape(f"{p}.feed_forward.w3.weight"),
    )


def rvq_out_proj_shape():
    """(c_out, c_in) for ANY codebook's out_proj -- semantic and residual out_proj weights are
    both [1024, 8, 1] (only the codebook row count differs, not out_proj's shape)."""
    w = gx.load(GGUF, f"{QPREFIX}.semantic_quantizer.quantizers.0.out_proj.weight")
    assert w.shape == (DIM, RVQ_D, 1), f"unexpected RVQ out_proj shape {w.shape}"
    w2 = gx.load(GGUF, f"{QPREFIX}.quantizer.quantizers.0.out_proj.weight")
    assert w2.shape == (DIM, RVQ_D, 1), f"unexpected RVQ out_proj shape {w2.shape}"
    return DIM, RVQ_D


def upsample_shape(stage):
    """(c_in, c_out, k, stride) of upsample.{stage}.0.conv (conv_transpose_1d). Asserts the
    k==stride==2 invariant quantizer_driver.py's windowing (ctx=0, see its module docstring) is
    built on -- if a future checkpoint ever ships k!=stride here, this fails loud instead of
    silently mis-windowing (the exact "hanging number" failure class this project's doctrine
    warns about)."""
    w = gx.load(GGUF, f"{QPREFIX}.upsample.{stage}.0.conv.weight")
    c_in, c_out, k = w.shape
    stride = 2  # DOWNSAMPLE_FACTORS = [2, 2] (codec_quantizer_ref.py) for both quantizer stages
    assert k == stride and c_in == c_out == DIM, (
        f"upsample.{stage}: conv_transpose shape {w.shape} breaks the k==stride==DIM assumption "
        "quantizer_driver's ctx=0 windowing relies on -- re-derive before reusing it")
    return int(c_in), int(c_out), int(k), stride


def convnext_shapes(stage):
    p = f"{QPREFIX}.upsample.{stage}.1"
    dw = gx.load(GGUF, f"{p}.dwconv.conv.weight")
    assert dw.shape[1] == 1, f"dwconv weight {dw.shape}: expected depthwise [C,1,K]"
    pw1 = proj_shape(f"{p}.pwconv1.weight")
    pw2 = proj_shape(f"{p}.pwconv2.weight")
    assert pw1 == (4096, DIM) and pw2 == (DIM, 4096), (stage, pw1, pw2)
    return dict(dwconv_c=int(dw.shape[0]), dwconv_k=int(dw.shape[2]), pwconv1=pw1, pwconv2=pw2)


# ---- sequence lengths, forward from T (== codes.shape[1]) -----------------------------------
# post_module transformer runs at T; each upsample stage doubles it (factor 2, uncropped -- see
# quantizer_driver.py). Pure arithmetic, no GGUF read needed.

def stage_t(t0, stage):
    """T at the INPUT of upsample.{stage} (stage 0 or 1): t0 for stage 0, 2*t0 for stage 1."""
    return t0 * (2 ** stage)


# ---- L1 + TDR-time arithmetic for a k=1 projection (window_driver.conv), self-checked ---------
# Reuses stage_shapes.py's byte-counting primitives (conv_l1, stream_bytes, max_ci_chunk,
# max_ci_chunk_time, max_stream_bytes, _snap_pow2) verbatim -- those formulas don't depend on
# which model owns the bytes, only on (c, k, t, resident_depth), so re-deriving them here would be
# the exact duplication build-methodology's "generic primitives" rule warns against.

def proj_plan(c_out, c_in, k=1, t=RES_T, resident_depth=1, name=""):
    """ci_chunk for a window_driver.conv(k=1) call over `c_in` input channels, `c_out` output
    channels, self-checked against BOTH the L1 ceiling and the TDR-time budget (same two-cap,
    take-the-tighter discipline as stage_shapes.head_plan()) -- a plan that fits L1 but blows the
    2000 ms watchdog is exactly the bug that broke the decoder head the first time
    (stage_shapes.py's own TIME ARITHMETIC section)."""
    l1_cap = ss.max_ci_chunk(c_in, lambda c: ss.conv_l1(c, k, t=t, resident_depth=resident_depth,
                                                        has_add=False))
    t_cap = ss.max_ci_chunk_time(c_out, k, has_add=False, t=t)
    caps = [c for c in (l1_cap, t_cap, c_in) if c is not None]
    ci_chunk = ss._snap_pow2(min(caps))
    assert ci_chunk >= 1, f"{name}: no ci_chunk fits (c_in={c_in} c_out={c_out} k={k} t={t})"

    total_l1 = ss.conv_l1(ci_chunk, k, t=t, resident_depth=resident_depth, has_add=False)[-1]
    assert total_l1 <= L1_BUDGET, f"{name}: ci_chunk={ci_chunk} L1 {total_l1} > {L1_BUDGET}"
    sb = ss.stream_bytes(c_out, ci_chunk, k, has_add=False, t=t)
    cap = ss.max_stream_bytes()
    assert sb <= cap, f"{name}: ci_chunk={ci_chunk} streams {sb} B > TDR cap {cap} B"

    n_ci_chunks = -(-c_in // ci_chunk)
    return dict(ci_chunk=(None if ci_chunk >= c_in else ci_chunk), resident_depth=resident_depth,
                n_ci_chunks=n_ci_chunks, t=t, l1_bytes=total_l1, stream_bytes=sb,
                l1_pct=total_l1 / L1_BUDGET * 100, time_ms=sb / 1024 * STREAM_MS_PER_KIB)


def n_t_windows(t_seq, t=RES_T):
    """window_driver's own windowing step for k=1/dilation=1 is `t - ctx = t` (ctx=0) -- so
    consecutive windows do NOT overlap and this is a plain ceiling division, unlike the decoder's
    dilated-conv case (stage_shapes.py carries no such helper because every decoder call there
    windows over a MUCH longer stream where the difference is immaterial to report separately)."""
    return -(-t_seq // t)


def proj_dispatches(plan, t_seq, t=RES_T):
    return plan["n_ci_chunks"] * n_t_windows(t_seq, t)


# ---- L1 + TDR-time arithmetic for the k==stride conv_transpose (NOT window_driver.conv_
# transpose -- see quantizer_driver.py's module docstring for why: that function hardcodes
# k=2*stride internally, which does not hold here). conv_transpose_channel.cc's own byte shape is
# IDENTICAL to the decoder's (one streamed weight row per c_out, one resident [c_in,t] activation
# window) -- only the WINDOWING (ctx=0 here, vs stage_shapes.UP_CTX=2 there) differs, and that
# lives in quantizer_driver.py, not in the byte-counting arithmetic below. So stage_shapes.
# conv_transpose_l1 is reused verbatim (same reasoning as proj_plan above).

def ct_plan(c_in, c_out, k, stride, t, resident_depth=1, name=""):
    l1_cap = ss.max_ci_chunk(
        c_in, lambda c: ss.conv_transpose_l1(c, k, stride, t=t, resident_depth=resident_depth))
    t_cap = ss.max_ci_chunk_time(c_out, k, has_add=False, t=t, ms_per_kib=CT_STREAM_MS_PER_KIB)
    caps = [c for c in (l1_cap, t_cap, c_in) if c is not None]
    ci_chunk = ss._snap_pow2(min(caps))
    assert ci_chunk >= 1, f"{name}: no ci_chunk fits (c_in={c_in} c_out={c_out} k={k} t={t})"

    total_l1 = ss.conv_transpose_l1(ci_chunk, k, stride, t=t, resident_depth=resident_depth)[-1]
    assert total_l1 <= L1_BUDGET, f"{name}: ci_chunk={ci_chunk} L1 {total_l1} > {L1_BUDGET}"
    sb = ss.stream_bytes(c_out, ci_chunk, k, has_add=False, t=t)
    cap = ss.max_stream_bytes(ms_per_kib=CT_STREAM_MS_PER_KIB)
    assert sb <= cap, f"{name}: ci_chunk={ci_chunk} streams {sb} B > TDR cap {cap} B"

    n_ci_chunks = -(-c_in // ci_chunk)
    return dict(ci_chunk=(None if ci_chunk >= c_in else ci_chunk), resident_depth=resident_depth,
                n_ci_chunks=n_ci_chunks, t=t, l1_bytes=total_l1, stream_bytes=sb,
                l1_pct=total_l1 / L1_BUDGET * 100, time_ms=sb / 1024 * CT_STREAM_MS_PER_KIB)


def ct_dispatches(plan, t_seq):
    """ctx=0 (see quantizer_driver.py): windows do NOT overlap, step == t exactly."""
    return plan["n_ci_chunks"] * n_t_windows(t_seq, plan["t"])


# ---- RVQ gather-rows: fixed shape, no chunking question (32 KB resident, proven) ---------------

def gather_n_tiles(t_seq, t_tile=GATHER_T_TILE):
    return -(-t_seq // t_tile)


# ---- rope-interleaved: ONE-SHOT kernel (whole [M,D] resident in one call), so M itself must be
# chunked to fit L1 -- unlike the streamed rail above, there is no on-device loop to hide behind.
# ROPE_M=32 reuses the EXACT shape route_b_kernels/bricks/_verify/verify_rope_interleaved.py
# already gates green (D=ROT=64, M=32) rather than deriving a new (untested) chunk size.

ROPE_M = 32


def rope_l1(m, d=HD, resident_depth=2):
    """qk (bf16, depth 2) + cossin (f32, depth 2) + out (bf16, depth 2) -- _build_oneshot's
    ObjectFifo defaults (aie/iron/dataflow/objectfifo.py: depth=2) apply to every buffer, unlike
    the streamed-rail depth=1 lever proj_plan/ct_plan use for a large RESIDENT operand; rope's
    inputs are the whole per-call operand, so there is no separate resident/streamed distinction
    to exploit here."""
    qk = m * d * 2 * resident_depth
    cossin = m * d * F32 * resident_depth
    out = m * d * 2 * resident_depth
    return qk, cossin, out, qk + cossin + out


assert rope_l1(ROPE_M)[-1] <= L1_BUDGET, "ROPE_M=32 no longer fits L1 -- re-derive"


def rope_dispatches(n_rows):
    return -(-n_rows // ROPE_M)


# ---- prefill_attn_chunk: L1 is not the binding constraint (checked below); TIME is UNMEASURED
# for this kernel specifically (STREAM_MS_PER_KIB/CT_STREAM_MS_PER_KIB are conv-1d/conv-transpose
# rates, not this kernel's -- doctrine: "Rates are PER KERNEL", stage_shapes.py's TIME ARITHMETIC
# section). ONE dispatch runs the WHOLE (t_tokens x n_chunks) sweep on-device (range_(t_tokens)
# device loop, n_chunks Python-unrolled call sites) -- see quantizer_driver.py.

def prefill_chunk_l1(hd, n_chunks):
    qm = (hd + PREFILL_VL) * 2 * 2       # bf16, depth 2
    kv = 2 * PREFILL_VL * hd * 2 * 2     # bf16, depth 2
    state = (hd + 2) * F32 * 2           # f32, depth 2
    return qm, kv, state, qm + kv + state


def prefill_n_chunks(t_seq, vl=PREFILL_VL):
    return -(-t_seq // vl)


# ---- elementwise (GELU/SwiGLU) row-batching against the TDR time budget ------------------------
# Mirrors quantizer_driver._elementwise_row_batch's formula exactly (kept here too so this file's
# _report() prints the dispatch count the driver ACTUALLY issues, not the single-dispatch count an
# earlier draft assumed -- a single GELU call over ConvNeXt upsample.1's full [4096,264] output
# streams ~4224 KiB, ~4.1s at the conv-1d proxy rate, over the RAW 2000 ms watchdog, not just the
# 70%-margin convention every other op here already respects).

def elementwise_row_batch(bytes_per_row, margin=TDR_MARGIN, ms_per_kib=STREAM_MS_PER_KIB):
    cap = ss.max_stream_bytes(margin, ms_per_kib)
    return max(1, cap // bytes_per_row)


def gelu_dispatches(c, t_seq):
    m = -(-(c * t_seq) // 16)
    return -(-m // elementwise_row_batch(16 * F32))


def swiglu_dispatches(c, t_seq, cols=1024):
    m = -(-(c * t_seq) // cols)
    return -(-m // elementwise_row_batch(2 * cols * F32))


# ---- the chosen policy for THIS segment's actual shapes, self-checked at import (not merely
# asserted in a comment) -- mirrors stage_shapes.py's own "chosen policy" section. ------------

assert n_layers() == 8, f"expected 8 post_module layers, found {n_layers()}"
assert n_residual_codebooks() == 9, f"expected 9 residual codebooks, found {n_residual_codebooks()}"
assert n_heads() == 16

PROJ_RESIDENT_DEPTH = 1   # every projection call: c_in=1024/3072/4096 is always the large operand
CT_RESIDENT_DEPTH = 1


def _report():
    print(f"GGUF {GGUF}")
    print(f"L1 budget {L1_BUDGET} bytes ({L1_BUDGET / 1024:.0f} KiB), RES_T={RES_T}, DIM={DIM}, "
          f"HD={HD}, n_heads={n_heads()}, n_layers={n_layers()}\n")

    T0 = 66  # real captured clip (dump80: codec_codes.bin int32 [10,66])
    t_by_stage = {"post_module": T0, "upsample.0": stage_t(T0, 0), "upsample.1": stage_t(T0, 1)}
    print(f"sequence lengths for T0={T0}: post_module T={t_by_stage['post_module']}, "
          f"upsample.0 (pre-CT) T={t_by_stage['upsample.0']} -> post-CT T={2*t_by_stage['upsample.0']}, "
          f"upsample.1 (pre-CT) T={t_by_stage['upsample.1']} -> post-CT T={2*t_by_stage['upsample.1']}\n")

    print("=== PROJECTIONS (window_driver.conv, k=1), one layer's shapes ===")
    total_proj_dispatches = 0
    ls = layer_proj_shapes(0)
    for opname, (c_out, c_in) in ls.items():
        p = proj_plan(c_out, c_in, name=opname)
        t_seq = t_by_stage["post_module"]
        nd = proj_dispatches(p, t_seq)
        total_proj_dispatches += nd
        print(f"  {opname:6s} c_out={c_out:4d} c_in={c_in:4d}  ci_chunk={p['ci_chunk'] or c_in:4d} "
              f"n_ci_chunks={p['n_ci_chunks']:2d}  L1={p['l1_pct']:5.1f}%  "
              f"stream/dispatch={p['stream_bytes']/1024:7.1f} KiB ({p['time_ms']:6.1f} ms est)  "
              f"T-windows={n_t_windows(t_seq)}  dispatches={nd}")
    print(f"  -> per-layer projection dispatches: {total_proj_dispatches}  "
          f"(x8 layers = {total_proj_dispatches*8})")

    rvq_c_out, rvq_c_in = rvq_out_proj_shape()
    rp = proj_plan(rvq_c_out, rvq_c_in, name="rvq_out_proj")
    rvq_nd = proj_dispatches(rp, T0) * 10  # 10 codebooks
    print(f"\n  rvq_out_proj c_out={rvq_c_out} c_in={rvq_c_in}  ci_chunk={rp['ci_chunk'] or rvq_c_in} "
          f"dispatches/codebook={proj_dispatches(rp, T0)}  x10 codebooks = {rvq_nd}")

    print("\n=== RVQ gather-rows ===")
    gt = gather_n_tiles(T0)
    print(f"  T={T0} T_TILE={GATHER_T_TILE} -> n_tiles={gt} (one dispatch/codebook, "
          f"+3 extra per-chunk dispatches for the semantic codebook's 4x1024 split)")
    gather_dispatches = 9 * 1 + 4  # 9 residual (1 dispatch each) + 4 semantic chunks
    print(f"  -> gather dispatches: {gather_dispatches}")

    print("\n=== rope-interleaved (ROPE_M=32 one-shot chunks) ===")
    qk, cossin, out, tot = rope_l1(ROPE_M)
    print(f"  per-call L1: qk={qk}B cossin={cossin}B out={out}B total={tot}B "
          f"({tot/L1_BUDGET*100:.1f}% of {L1_BUDGET}B)")
    rows_per_layer = 2 * n_heads() * T0  # Q + K
    rd = rope_dispatches(rows_per_layer)
    print(f"  rows/layer (Q+K, {n_heads()} heads x T={T0} x2) = {rows_per_layer}  "
          f"-> dispatches/layer={rd}  (x8 layers = {rd*8})")

    print("\n=== prefill_attn_chunk (post_module attention) ===")
    nchunks = prefill_n_chunks(T0)
    qm, kv, state, tot = prefill_chunk_l1(HD, nchunks)
    print(f"  T={T0} n_chunks={nchunks} (window={128}, degenerates to plain causal since T<window)")
    print(f"  per-call L1: qm={qm}B kv={kv}B state={state}B total={tot}B "
          f"({tot/L1_BUDGET*100:.1f}% of {L1_BUDGET}B) -- L1 is NOT the binding constraint")
    print(f"  ONE dispatch runs the whole (T x n_chunks) sweep for one (head, layer) -- "
          f"{n_heads()} heads x 8 layers = {n_heads()*8} dispatches total")
    print(f"  TIME per dispatch is UNMEASURED for this kernel (no probe_*device_ms.py run exists "
          f"for prefill_attn_chunk) -- flagged, not estimated")

    print("\n=== upsample conv_transpose (ctx=0, custom windowing -- see quantizer_driver.py) ===")
    for stage in (0, 1):
        c_in, c_out, k, stride = upsample_shape(stage)
        t_in = t_by_stage[f"upsample.{stage}"]
        p = ct_plan(c_in, c_out, k, stride, t=RES_T, name=f"upsample.{stage}")
        nd = ct_dispatches(p, t_in)
        print(f"  upsample.{stage}: c_in={c_in} c_out={c_out} k={k} stride={stride} T_in={t_in} "
              f"ci_chunk={p['ci_chunk'] or c_in} n_ci_chunks={p['n_ci_chunks']} "
              f"L1={p['l1_pct']:.1f}% stream/dispatch={p['stream_bytes']/1024:.1f} KiB "
              f"({p['time_ms']:.1f} ms est) T-windows={n_t_windows(t_in, p['t'])} dispatches={nd}")

    print("\n=== ConvNeXt per upsample stage ===")
    convnext_total = 0
    for stage in (0, 1):
        cs = convnext_shapes(stage)
        t_out = 2 * t_by_stage[f"upsample.{stage}"]  # post-CT length, what convnext runs at
        pw1 = proj_plan(*cs["pwconv1"], name=f"upsample.{stage}.pwconv1")
        pw2 = proj_plan(*cs["pwconv2"], name=f"upsample.{stage}.pwconv2")
        pw1_nd = proj_dispatches(pw1, t_out)
        pw2_nd = proj_dispatches(pw2, t_out)
        gelu_nd = gelu_dispatches(cs["pwconv1"][0], t_out)
        print(f"  upsample.{stage}: T_out={t_out} dwconv C={cs['dwconv_c']} K={cs['dwconv_k']} "
              f"(1 dispatch, packed [x_row|w] per channel)")
        print(f"    LN: 1 dispatch (custom eps=1e-6 shim, normalize-only)")
        print(f"    pwconv1 dispatches={pw1_nd}  pwconv2 dispatches={pw2_nd}  "
              f"GELU dispatches={gelu_nd} (TDR-batched, see elementwise_row_batch)")
        convnext_total += 1 + 1 + pw1_nd + pw2_nd + gelu_nd

    print("\n=== TOTAL estimated dispatches at T0=66 (gated chain path; run2run doubles this) ===")
    proj_per_layer = sum(proj_dispatches(proj_plan(c_out, c_in, name=n), T0)
                         for n, (c_out, c_in) in layer_proj_shapes(0).items())
    rvq_proj_nd = proj_dispatches(proj_plan(*rvq_out_proj_shape(), name="rvq_out_proj"), T0) * 10
    gather_nd = 9 * 1 + 4
    rope_nd = rope_dispatches(2 * n_heads() * T0) * n_layers()
    prefill_nd = n_heads() * n_layers()
    swiglu_nd = swiglu_dispatches(3072, T0) * n_layers()
    norm_nd = 2 * n_layers() + 1
    ct_nd = sum(ct_dispatches(ct_plan(*upsample_shape(s)[:2], upsample_shape(s)[2],
                                      upsample_shape(s)[3], t=RES_T, name=f"up{s}"),
                              t_by_stage[f"upsample.{s}"]) for s in (0, 1))
    total = (proj_per_layer * n_layers() + rvq_proj_nd + gather_nd + rope_nd + prefill_nd
            + swiglu_nd + norm_nd + ct_nd + convnext_total)
    print(f"  projections (per-layer x {n_layers()})     : {proj_per_layer * n_layers()}")
    print(f"  rvq_out_proj (x10 codebooks)               : {rvq_proj_nd}")
    print(f"  gather-rows                                : {gather_nd}")
    print(f"  rope-interleaved (Q+K, x{n_layers()} layers): {rope_nd}")
    print(f"  prefill_attn_chunk ({n_heads()} heads x {n_layers()} layers): {prefill_nd}")
    print(f"  swiglu (x{n_layers()} layers)               : {swiglu_nd}")
    print(f"  rmsnorm (attn+ffn x{n_layers()} + final)    : {norm_nd}")
    print(f"  upsample conv_transpose                    : {ct_nd}")
    print(f"  convnext (dwconv+LN+pwconv1/2+gelu, x2)     : {convnext_total}")
    print(f"  TOTAL                                      : {total}")


if __name__ == "__main__":
    _report()
