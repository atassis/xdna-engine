#!/usr/bin/env python3
"""The S2 codec QUANTIZER segment on device: codes [10,T] -> latent [1024,4T]. M1.

    rvq_lookup (gather + out_proj x10)  ->  post_module (8-layer transformer)
        ->  quantizer_upsample x2 (conv_transpose + ConvNeXt)  ->  latent

Sibling of decoder_chain.py (read that first -- same weight-cache convention `W(name)`, same
op-by-op-gateable structure), not an extension of it: the decoder is convs end to end and drives
everything through window_driver.py's `_run`; this segment adds five op-families window_driver never
needed (indexed gather, rowwise norm, RoPE, chunked/flash attention, elementwise activation), each of
which needs its own dispatch shape. `quantizer_shapes.py` carries the chunk-size arithmetic; this file
is the dispatch code that consumes those plans.

FIVE OP-FAMILIES, FIVE DISPATCH SHAPES -- know which one an op is before reading its function:
  1. LINEAR PROJECTIONS (RVQ out_proj, wqkv/wo/w1/w2/w3, ConvNeXt pwconv1/pwconv2) are k=1 convs,
     dispatched through `window_driver.conv()` UNCHANGED -- Fact #1 (device-established): a
     transformer projection is a k=1 conv, already device-green at c_in=1024. No new code for this
     class; `wd.conv(..., ci_chunk=plan['ci_chunk'])` is the whole call.
  2. UPSAMPLE CONV_TRANSPOSE reuses conv_transpose_channel.cc's kernel (same file
     window_driver.conv_transpose already links) but NOT window_driver.conv_transpose's own Python
     wrapper: that wrapper hardcodes `k = 2*stride` internally (the decoder's own invariant,
     stage_shapes.py `assert k == 2*stride`), but the quantizer's upsample weights are k==stride==2
     (confirmed against the real GGUF: c.quantizer.upsample.{0,1}.0.conv.weight is (1024,1024,2), not
     (1024,1024,4)). At k==stride there is NO cross-window overlap at all: output position
     ti*stride+j depends on input ti ALONE (codec_decoder_ref.conv_transpose_1d's own scatter,
     `y[:, j::stride][:, :t] += contrib[j]`, has zero overlap between taps when k==stride) -- so
     `_conv_transpose_ctx0` below windows with ctx=0, step==t, no crop, rather than adapting
     window_driver's ctx=2/crop=stride machinery to a case where cropping isn't happening (out_len =
     t*stride already equals the natural uncropped length). Consequence for the driver: it does NOT
     need conv_transpose_channel_core to be `k/stride`-general in the abstract -- it only needs k<=
     stride, which the kernel's per-tap guard `if (q < out_len)` already satisfies for any k<=stride
     without any cropping logic at all.
  3. GATHER / DWCONV / ROWWISE-NORM / SWIGLU / GELU all go through a dtype-parameterized clone of
     window_driver's own `_run` (below), because `window_driver._run` hardcodes f32 in/out and these
     ops are int32 (gather indices), bf16 (dwconv, matching dwconv1d.cc's own dtype), or need a
     resident=None two-streamed-operand shape window_driver never uses (swiglu/gelu pack two operands
     -- or, for gelu, need none at all -- into one streamed tile the same way window_driver.snake
     packs alpha into its tile).
  4. ROPE (rope_interleaved_prologue) is a ONE-SHOT kernel: the whole [M,D] tile is resident for one
     call, no on-device streaming loop to hide behind (unlike every op above). M is therefore chunked
     host-side at ROPE_M=32 (quantizer_shapes.ROPE_M, the exact shape
     bricks/_verify/verify_rope_interleaved.py already gates), dispatched via `bricklib.verify_oneshot`
     directly rather than `_run`.
  5. PREFILL_ATTN_CHUNK's ABI (2 streamed inputs -- qm_row, kv_chunk -- plus ONE RESIDENT-PER-ROW
     buffer that DOUBLES as online-softmax state, plus a per-call compile-time-literal chunk_idx) fits
     none of bricklib's generic builders (`_check_symbol_arity` wants exactly `n_buffers` pointer
     params, no scalar, no read-write resident). `_flash_design` below is `_build_flash_design` from
     route_b_kernels/bricks/_verify/verify_prefill_attn.py, adapted only to be MEMOIZED (that script
     builds once per test case; this driver reuses one compiled design across all 16 heads x 8 layers
     for a given T, since the compiled program depends on (t_tokens, n_chunks, hd) only, never on
     which head/layer is being run).

WEIGHT FOLDING (the task's named optimization: "deletes op-types rather than adding them"). Every
attention/FFN `layer_scale.gamma`, the ConvNeXt `gamma`, and RMSNorm's own affine weight are pure
per-channel scales with NO nonlinearity between them and the adjacent linear op, so they fold
losslessly into that op's weight -- see `fold_weights()` for the exact algebra per fold and
`_selftest_weight_folds()` for the host-side exactness check the task asks for. ONE fold needed a
correction while authoring this file: the ConvNeXt `gamma` fold is OUTPUT-side (into pwconv2's
weight ROWS) and pwconv2 HAS a bias, so the bias must be scaled by gamma too
(`(pw2 @ y + b) * gamma == (gamma*pw2) @ y + gamma*b`) -- omitting that scaled the weight only and
landed at rel-L2 3.59e-02, inside this project's own 3e-2 gate margin on some inputs and outside it on
others: a bug that would have shown up as an intermittent gate flake, not a clean failure. Every other
fold here has NO bias on the folded side (wqkv/wo/w1/w2/w3 carry no bias in this checkpoint, confirmed
against the real GGUF) so this correction is specific to pwconv2.

`post_module.norm` is algebraically foldable into upsample[0]'s conv_transpose the same way
(RMSNorm immediately followed by a linear op, no nonlinearity between) -- but is DELIBERATELY NOT
folded here, unlike the within-layer folds above. Folding it would make `post_module()`'s return
value a different tensor than `cq.rvq_transformer`'s (the true post-norm output vs. a gamma=1
normalize-only intermediate whose gamma has been pushed one stage downstream), which breaks the
task's own per-stage-gateable requirement: `verify_quantizer_segment.py` compares `post_module()`
directly against `cq.rvq_transformer(...)`, and a cross-stage fold means that comparison is no
longer checking the same quantity, not a numerics bug you could gate around. The within-LAYER folds
above don't have this problem because their stage boundary (one full `post_module_layer` call) is
never independently gated -- only the whole 8-layer `post_module()` output is. So: apply the real
gamma directly in `post_module()`'s final norm call, and leave upsample[0]'s conv weight unfolded.
ConvNeXt's own LayerNorm-into-pwconv1 fold is ALSO algebraically exact and ALSO not taken, for the
simpler reason given below at its call site: `layernorm_ln_affine_f32` is already a proven brick, so
there is no op-count reduction to buy from folding it, only bookkeeping -- no gating-boundary issue
either way, it is just not worth the complexity.

QUANTIZER-STAGE-UP LAYOUT: `x` is channel-major `[C,T]` throughout (matching window_driver.conv's own
convention and codec_quantizer_ref.py's), while the rowwise bricks (rmsnorm/layernorm) want `[T,C]`.
There is no transpose-dma kernel dispatch anywhere in this file for that -- `x.T` between two host-
materialized ops is a free numpy view (this driver, like window_driver.py, round-trips every
intermediate through host memory between dispatches; see window_driver.py's own module docstring,
"OP-MAJOR... every intermediate materialises in host memory" -- transpose-dma exists for the
DIFFERENT, L2-resident-never-touches-host regime this driver does not use).

    python3 quantizer_driver.py <dump-dir>   # decode() one clip's codes, report shape + dispatch count
"""
import contextlib
import io
import re
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "route_b_kernels" / "bricks" / "_verify"))
sys.path.insert(0, str(ROOT / "scripts"))

import bricklib  # noqa: E402
import window_driver as wd  # noqa: E402
import quantizer_shapes as qs  # noqa: E402

import aie.iron as iron  # noqa: E402
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402

_bf16 = ml_dtypes.bfloat16

QPREFIX = qs.QPREFIX
BRICKS = ROOT / "route_b_kernels" / "bricks"
GATHER_CC = (BRICKS / "gather-rows" / "gather_rows.cc").resolve()
ROPE_CC = (BRICKS / "rope-interleaved" / "rope_interleaved.cc").resolve()
PREFILL_CC = (BRICKS / "prefill-attn" / "prefill_attn.cc").resolve()
RMSNORM_CC = (BRICKS / "rmsnorm" / "rmsnorm.cc").resolve()
LAYERNORM_CC = (BRICKS / "layernorm" / "layernorm.cc").resolve()
SWIGLU_CC = (BRICKS / "swiglu" / "swiglu.cc").resolve()
GELU_CC = (BRICKS / "gelu-erf" / "gelu_erf.cc").resolve()
DWCONV_CC = (ROOT / "route_b_kernels" / "dwconv1d" / "dwconv1d.cc").resolve()

# Shapes discovered from the real GGUF at import time (device-free), mirroring quantizer_shapes.py's
# own self-checking-at-import discipline rather than hardcoding numbers this file could drift from.
DIM = qs.DIM
HD = qs.HD
N_HEAD = qs.n_heads()
N_LAYERS = qs.n_layers()
FFN_DIM = qs.layer_proj_shapes(0)["w1"][0]
CONVNEXT_HIDDEN = qs.convnext_shapes(0)["pwconv1"][0]

# Per-dispatch tile batching for the two elementwise ops (swiglu, gelu-erf), via
# quantizer_shapes.elementwise_row_batch -- the SAME TIME-budget cap every other op in this file
# self-checks against (STREAM_MS_PER_KIB, a conv-1d proxy: see quantizer_shapes.py's TIME-BUDGET
# CAVEAT, UNVERIFIED for these two kernels specifically), rather than a hand-picked constant. SWIGLU
# rows are 1024 elements (matches bricks/_verify/verify_f1.py's own do_swiglu test width); GELU tiles
# are the kernel's fixed 16 elements.
SWIGLU_ROW = 1024
SWIGLU_TILES_PER_DISPATCH = qs.elementwise_row_batch(2 * SWIGLU_ROW * qs.F32)
GELU_TILES_PER_DISPATCH = qs.elementwise_row_batch(16 * qs.F32)

_stats = {"dispatches": 0}


def reset_stats():
    _stats["dispatches"] = 0


def stats():
    return dict(_stats)


def _safe(tag):
    """Turn an arbitrary driver tag into a valid (if ugly) C identifier fragment."""
    return re.sub(r"[^0-9A-Za-z_]", "_", tag)


def _resident_dtype(resident, resident_dt):
    """f32 is the default when a resident is supplied and no dtype is named.

    window_driver._run hardcodes f32 for in/out/resident; generalising it over dtype here dropped
    that DEFAULT along with the hardcoding, so a call site passing a resident but no resident_dt
    handed bricklib None, and iron.tensor then fell back to float64 and rejected the f32 array.
    Every resident in this file is f32 except where a call site says otherwise."""
    if resident is None:
        return None
    return resident_dt if resident_dt is not None else np.float32


def _run(name, shim_text, symbol, tiles, out_numel, resident, in_dt=np.float32, out_dt=np.float32,
         resident_dt=None, flags=None, resident_depth=2):
    """window_driver._run, generalized over dtype (that function hardcodes f32 in/out/resident,
    which every op in THIS file except the k=1 projections and conv_transpose violates: gather's
    indices are int32, dwconv/rope/prefill are bf16). Same ungated-intermediate swallow (golden
    zeros, gate inf, stdout redirected) -- every call here is a driver-internal dispatch, not a gate;
    verify_quantizer_segment.py does the gating, against the real oracle, per stage."""
    shim = bricklib.GEN / f"qd_{name}_shim.cc"
    shim.write_text(shim_text)
    with contextlib.redirect_stdout(io.StringIO()):
        r = bricklib.verify_streamed(
            name=name, shim=shim, symbol=symbol, in_tiles=tiles, out_tile_numel=out_numel,
            resident=(None if resident is None else np.asarray(resident).reshape(-1)),
            unpack=lambda d: np.asarray(d), golden=np.zeros((tiles.shape[0], out_numel)),
            gate=np.inf, in_dt=in_dt, out_dt=out_dt, compile_flags=flags,
            resident_dt=(_resident_dtype(resident, resident_dt)), resident_depth=resident_depth)
    _stats["dispatches"] += 1
    return np.asarray(r["got"])


# =====================================================================================================
# 1. RVQ GATHER-ROWS. codebook RESIDENT (depth=1 -- REQUIRED, not a nicety: the 1024x8 f32 codebook is
# 32 KB, and depth 2 fails to build at this exact shape, measured in gather-rows' own verify script),
# indices streamed as GATHER_T_TILE=16 tiles, ALL of T's tiles in ONE dispatch (device-side n_tiles
# loop). The semantic codebook (4096 rows) does not fit resident in one piece -- served as FOUR
# dispatches against 1024-row chunks of the SAME compiled kernel shape, with the host doing a disjoint
# select by chunk_of[t] = idx_clamped[t] // 1024 -- gather-rows' own verify script's proven pattern.
# =====================================================================================================

def _gather_rows(idx_i32, codebook_f32, tag):
    n_rows, d = codebook_f32.shape
    assert len(idx_i32) % qs.GATHER_T_TILE == 0, "caller must pad idx to a GATHER_T_TILE multiple"
    n_tiles = len(idx_i32) // qs.GATHER_T_TILE
    shim = (f"#include <stdint.h>\n"
            f"#define GATHER_N_ROWS {n_rows}\n#define GATHER_D {d}\n"
            f"#define GATHER_T_TILE {qs.GATHER_T_TILE}\n"
            f'#include "{GATHER_CC}"\n')
    tiles = idx_i32.reshape(n_tiles, qs.GATHER_T_TILE)
    got = _run(f"gather_{tag}", shim, "gather_rows_f32", tiles, qs.GATHER_T_TILE * d,
               codebook_f32, in_dt=np.int32, out_dt=np.float32, resident_dt=np.float32,
               resident_depth=1)
    return got.reshape(n_tiles * qs.GATHER_T_TILE, d)


def rvq_lookup(codes, W, tag="rvq"):
    """codes [10,T] int -> [1024,T] f32. Gather + out_proj (k=1 conv) per codebook, summed on host
    (f64 accumulation, matching codec_quantizer_ref.rvq_lookup's own accumulation dtype)."""
    n_cb, T = codes.shape
    T_pad = qs.gather_n_tiles(T) * qs.GATHER_T_TILE
    stage = np.zeros((DIM, T), np.float64)
    for cb in range(n_cb):
        if cb == 0:
            sub = f"{QPREFIX}.semantic_quantizer.quantizers.0"
            size = 4096
        else:
            sub = f"{QPREFIX}.quantizer.quantizers.{cb - 1}"
            size = 1024
        codebook = W(f"{sub}.codebook.weight")            # [size,8]
        out_w = W(f"{sub}.out_proj.weight")                # [1024,8,1] -- window_driver.conv's own
        out_b = W(f"{sub}.out_proj.bias").reshape(-1)       # native [c_out,c_in,k] layout, no reshape

        idx = np.clip(codes[cb].astype(np.int64), 0, size - 1).astype(np.int32)
        idx_pad = np.zeros(T_pad, np.int32)
        idx_pad[:T] = idx  # tail padding is discarded below, value is irrelevant (0 is in-range)

        if size == 1024:
            gathered = _gather_rows(idx_pad, codebook, f"{tag}{cb}")[:T]
        else:
            local_idx = (idx_pad.astype(np.int64) % 1024).astype(np.int32)
            chunk_of = idx_pad.astype(np.int64) // 1024
            chunks = [_gather_rows(local_idx, codebook[c * 1024:(c + 1) * 1024], f"{tag}{cb}c{c}")
                      for c in range(4)]
            gathered = np.empty((T_pad, qs.RVQ_D), np.float32)
            for t in range(T_pad):
                gathered[t] = chunks[chunk_of[t]][t]
            gathered = gathered[:T]

        gathered_ct = np.ascontiguousarray(gathered.T)  # [8,T], window_driver.conv's [c_in,L]
        plan = qs.proj_plan(DIM, qs.RVQ_D, name=f"rvq_out_proj{cb}")
        proj = wd.conv(gathered_ct, out_w, out_b, 1, 1, f"{tag}proj{cb}",
                       ci_chunk=plan["ci_chunk"], resident_depth=plan["resident_depth"])
        stage += proj.astype(np.float64)
    return stage.astype(np.float32)


# =====================================================================================================
# 2. ROPE (rope_interleaved_prologue, ONE-SHOT). ROPE_M=32 rows/call (quantizer_shapes.ROPE_M -- the
# exact shape already gated green), ADJACENT-PAIR convention matching codec_quantizer_ref._rope_normal
# exactly (cos/sin computed host-side in f64, cast to f32 -- the device performs no trig at all, only
# the rotation). Cast to bf16 going in, matching rope_interleaved_prologue's own dtype.
# =====================================================================================================

def _rope_cossin(positions, d, base=qs.RVQ_ROPE_BASE):
    """positions: (rows,) f64 -> cossin [rows,d] f32, packed [cos(0..half-1)|sin(0..half-1)] per row
    -- codec_quantizer_ref._rope_normal's exact inv_freq/theta formula, reproduced here rather than
    imported (quantizer_shapes.py's own convention: this driver stays independent of the reference
    module, mirroring its constants instead)."""
    half = d // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = base ** (-(2.0 * i) / d)
    theta = positions[:, None] * inv_freq[None, :]  # [rows, half]
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def _rope_head(x_head, positions, tag):
    """x_head [HD,T] f32 (one head's Q or K slice) -> rotated [HD,T] f32. Windows T in ROPE_M chunks
    (zero-padded on the last chunk; padded rows are computed and discarded, harmless since rotation
    has no cross-row term)."""
    Hd, T = x_head.shape
    x_tok = np.ascontiguousarray(x_head.T)  # [T,HD] -- rope's native row-major-per-token layout
    out_tok = np.zeros((T, Hd), np.float32)
    M = qs.ROPE_M
    for o in range(0, T, M):
        n = min(M, T - o)
        qk_pad = np.zeros((M, Hd), np.float32)
        qk_pad[:n] = x_tok[o:o + n]
        pos_pad = np.zeros(M, np.float64)
        pos_pad[:n] = positions[o:o + n]
        cossin = _rope_cossin(pos_pad, Hd)

        # NOT per-offset: the body is identical across windows and the position data
        # arrives at runtime in `cossin`, so a per-offset symbol/name only splits the
        # design key and forces a rebuild per window. See window_driver._run.
        sym = f"qd_rope_{_safe(tag)}"
        shim_body = (f'extern "C" void {sym}(bfloat16 *qk_in, float *cossin, bfloat16 *qk_out) {{\n'
                    f"  for (unsigned i = 0; i < (unsigned)(ROPE_M*ROPE_D); ++i) qk_out[i]=qk_in[i];\n"
                    f"  rope_interleaved_prologue(qk_out, cossin);\n}}\n")
        with contextlib.redirect_stdout(io.StringIO()):
            r = bricklib.verify_oneshot(
                name=f"rope_{tag}", brick_cc=ROPE_CC, shim_body=shim_body, symbol=sym,
                inputs=[(qk_pad.astype(_bf16).reshape(-1), _bf16), (cossin.reshape(-1), np.float32)],
                out_numel=M * Hd, out_shape=(M, Hd),
                unpack=lambda d: np.asarray(d, np.float32).reshape(M, Hd),
                golden=np.zeros((M, Hd)), gate=np.inf, out_dt=_bf16,
                compile_flags=[f"-DROPE_D={Hd}", f"-DROPE_ROT={Hd}", f"-DROPE_M={M}"])
        _stats["dispatches"] += 1
        out_tok[o:o + n] = np.asarray(r["got"], np.float32)[:n]
    return np.ascontiguousarray(out_tok.T)


# =====================================================================================================
# 3. PREFILL_ATTN_CHUNK (flash/chunked attention). `_flash_design` is verify_prefill_attn.py's own
# `_build_flash_design`, UNCHANGED except for memoization: the compiled program depends only on
# (t_tokens, n_chunks, hd), never on which head or layer, so this driver builds it ONCE per T and
# reuses it for all N_HEAD x N_LAYERS calls, rather than rebuilding (use_cache=False, ~150ms each)
# 128 times over. Packing (`_pack_flash_head`) is golden.pack_head_chunks's own convention, transcribed
# rather than imported (same reasoning as _rope_cossin above).
# =====================================================================================================

_flash_design_cache = {}


def _flash_design(t_tokens, n_chunks, hd):
    key = (t_tokens, n_chunks, hd)
    if key in _flash_design_cache:
        return _flash_design_cache[key]

    compile_flags = bricklib._aie_api_include() + [f"-DPREFILL2_HD={hd}", f"-DPREFILL2_NCHUNKS={n_chunks}"]
    qm_bf16 = hd + qs.PREFILL_VL
    kv_bf16 = 2 * qs.PREFILL_VL * hd
    state_f32 = hd + 2

    qm_ty = np.ndarray[(qm_bf16,), np.dtype[_bf16]]
    kv_ty = np.ndarray[(kv_bf16,), np.dtype[_bf16]]
    state_ty = np.ndarray[(state_f32,), np.dtype[np.float32]]
    qm_full_ty = np.ndarray[(t_tokens * n_chunks * qm_bf16,), np.dtype[_bf16]]
    kv_full_ty = np.ndarray[(t_tokens * n_chunks * kv_bf16,), np.dtype[_bf16]]
    state_full_ty = np.ndarray[(t_tokens * state_f32,), np.dtype[np.float32]]

    def design(qm_in: In, kv_in: In, state_out: Out):
        kern = ExternalFunction(
            "prefill_attn_chunk", source_file=str(PREFILL_CC),
            arg_types=[qm_ty, kv_ty, state_ty, np.int32], compile_flags=compile_flags)
        of_qm = ObjectFifo(qm_ty, name="qm_in", depth=2)
        of_kv = ObjectFifo(kv_ty, name="kv_in", depth=2)
        of_state = ObjectFifo(state_ty, name="state_out", depth=2)

        def core(qm_cons, kv_cons, state_prod, kern):
            for _ in range_(t_tokens):        # genuine device dispatch loop, ONE call-site body
                es = state_prod.acquire(1)    # resident for this row's whole chunk sweep
                for c in range(n_chunks):     # Python-unrolled, n_chunks literal call sites TOTAL
                    eqm = qm_cons.acquire(1)
                    ekv = kv_cons.acquire(1)
                    kern(eqm, ekv, es, c)
                    kv_cons.release(1)
                    qm_cons.release(1)
                state_prod.release(1)

        worker = Worker(core, fn_args=[of_qm.cons(), of_kv.cons(), of_state.prod(), kern],
                        stack_size=0xD00)

        def sequence(qm, kv, st, qm_h, kv_h, st_h):
            qm_h.fill(qm)
            kv_h.fill(kv)
            st_h.drain(st, wait=True)

        rt = Runtime(sequence, [qm_full_ty, kv_full_ty, state_full_ty,
                                of_qm.prod(), of_kv.prod(), of_state.cons()])
        return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

    base = bricklib._design_key(
        "prefill_attn_chunk", compile_flags, bricklib._include_closure_digest(PREFILL_CC, compile_flags))
    design.__name__ = design.__qualname__ = f"{base}_hd{hd}_nchunks{n_chunks}_t{t_tokens}"
    built = iron.jit(design, use_cache=False)
    _flash_design_cache[key] = built
    return built


def _pack_flash_head(q_head, k_head, v_head, mask, n_chunks):
    """q/k/v_head: [HD,T] f32. mask: [Tk,Tq] additive (0/-1e9), already length T on the key axis --
    padded here to n_chunks*VL with -1e9 (padding is masked unconditionally, matching prefill_attn.cc's
    own SPAD-padding convention). Returns (qm_tiles, kv_tiles) in (row,chunk) order, matching
    _flash_design's core loop order exactly."""
    hd, T = q_head.shape
    VL = qs.PREFILL_VL
    n_keys_padded = n_chunks * VL
    mask_padded = np.full((n_keys_padded, T), -1.0e9, np.float32)
    mask_padded[:T, :] = mask.astype(np.float32)
    k_pad = np.zeros((hd, n_keys_padded), np.float32)
    v_pad = np.zeros((hd, n_keys_padded), np.float32)
    k_pad[:, :T] = k_head
    v_pad[:, :T] = v_head

    qm_tiles = np.zeros((T * n_chunks, hd + VL), _bf16)
    kv_tiles = np.zeros((T * n_chunks, 2 * VL * hd), _bf16)
    idx = 0
    for qi in range(T):
        q_row = q_head[:, qi].astype(_bf16)
        for c in range(n_chunks):
            mask_chunk = mask_padded[c * VL:(c + 1) * VL, qi].astype(_bf16)
            qm_tiles[idx] = np.concatenate([q_row, mask_chunk])
            k_chunk = k_pad[:, c * VL:(c + 1) * VL].T.astype(_bf16)
            v_chunk = v_pad[:, c * VL:(c + 1) * VL].T.astype(_bf16)
            kv_tiles[idx] = np.concatenate([k_chunk.reshape(-1), v_chunk.reshape(-1)])
            idx += 1
    return qm_tiles, kv_tiles


def _prefill_attn_head(q_head, k_head, v_head, mask):
    hd, T = q_head.shape
    n_chunks = qs.prefill_n_chunks(T)
    design = _flash_design(T, n_chunks, hd)
    qm_tiles, kv_tiles = _pack_flash_head(q_head, k_head, v_head, mask, n_chunks)

    qm_t = iron.tensor(np.ascontiguousarray(qm_tiles.reshape(-1)), dtype=_bf16, device="npu")
    kv_t = iron.tensor(np.ascontiguousarray(kv_tiles.reshape(-1)), dtype=_bf16, device="npu")
    st_t = iron.zeros((T * (hd + 2),), dtype=np.float32, device="npu")
    design(qm_t, kv_t, st_t)
    ctx = st_t.numpy().reshape(T, hd + 2)[:, :hd].copy()  # drop the 2 running max/sum scratch scalars
    _stats["dispatches"] += 1
    return ctx  # [T,HD] f32


def _causal_window_mask(seq_len, window_size):
    """Reproduces codec_quantizer_ref._causal_window_mask exactly (see that module's docstring) --
    duplicated rather than imported, same policy as _rope_cossin above."""
    q = np.arange(seq_len)[None, :]
    k = np.arange(seq_len)[:, None]
    allowed = k <= q
    if 0 < window_size < seq_len:
        allowed = allowed & (k >= (q - window_size + 1))
    return np.where(allowed, 0.0, -1e9).astype(np.float32)


# =====================================================================================================
# 4. DWCONV (ConvNeXt depthwise causal, K=7, C=1024). dwconv1d_same_scalar<T,K,P=K-1,BIAS> is truly
# per-channel (no cross-channel term, no resident operand shared across channels -- see
# verify_dwconv_causal.py's TEST STRUCTURE note), so BOTH the per-channel signal and its taps vary per
# channel: they are packed into ONE streamed tile [x(T)|taps(K)|bias(1)] per channel, resident=None,
# ALL C=1024 channels in ONE dispatch (device-side n_tiles=C loop). bf16 in/out, matching the kernel's
# own dtype -- the only bf16 op-family outside RoPE/prefill-attn in this driver.
# =====================================================================================================

def _aie_kernels_aie2p_include():
    """dwconv1d.cc:34 `#include "../aie_kernel_utils.h"` resolves against dwconv1d.cc's own directory
    -- a shim #include'ing it from bricklib.GEN needs this extra -I. Same instance-derivation trick
    verify_dwconv_causal.py's own helper uses."""
    import aie
    inst = Path(aie.__file__).resolve().parent.parent.parent
    cand = inst / "src" / "aie_kernels" / "aie2p"
    if (cand / ".." / "aie_kernel_utils.h").resolve().exists():
        return [f"-I{cand}"]
    return []


def _dwconv_causal(x, w, bias, K, tag):
    """x [C,T] f32, w [C,1,K] f32, bias [C] f32 -> [C,T] f32 (bf16 device round-trip, same 3.1e-08 vs
    3.1e-08-class precision verify_dwconv_causal.py already measured for this exact reparameterization,
    P=K-1). T is a template argument (dwconv1d_same_scalar<T,...>), so it is fixed per call -- exactly
    the whole-sequence T for one convnext stage, no further T-windowing (T*2 bytes is a few hundred
    bytes/channel, nowhere near the 64 KB budget; see quantizer_shapes.py's dwconv section)."""
    C, T = x.shape
    P = K - 1
    KW = K + 1  # taps[0..K-1] + bias[K]
    tiles = np.zeros((C, T + KW), np.float32)
    tiles[:, :T] = x
    tiles[:, T:T + K] = w[:, 0, :]
    tiles[:, T + K] = bias
    tiles_bf = tiles.astype(_bf16)

    sym = f"qd_dwconv_{_safe(tag)}"
    shim = (f"#include <stdint.h>\n"
            f'#include "{DWCONV_CC}"\n'
            f'extern "C" void {sym}(bfloat16 *tile, bfloat16 *out) {{\n'
            f"  bfloat16 *x = tile;\n  bfloat16 *w = tile + {T};\n"
            f"  dwconv1d_same_scalar<{T}, {K}, {P}, true>(x, w, out);\n}}\n")
    got = _run(f"dwconv_{tag}", shim, sym, tiles_bf, T, None,
              in_dt=_bf16, out_dt=_bf16, flags=_aie_kernels_aie2p_include())
    return got.reshape(C, T).astype(np.float32)


# =====================================================================================================
# 5. ROWWISE NORM (rmsnorm/layernorm) and ELEMENTWISE (swiglu, gelu-erf). All f32. Rowwise norm needs
# [T,C] layout (a free host-side transpose, see module docstring); swiglu/gelu are pure elementwise so
# the [C,T] row-major flatten needs no transpose at all -- chunk boundaries just don't align with
# channel boundaries, which does not matter for an op with no cross-element term.
#
# CUSTOM EPS. rmsnorm_f32/layernorm_ln_affine_f32's own extern "C" wrappers hardcode eps (1e-6f /
# 1e-5f respectively); this model needs 1e-5 (RVQ_NORM_EPS) and 1e-6 (CONVNEXT_LN_EPS) respectively --
# the OPPOSITE of each wrapper's own default -- so both shims below call the templated core directly
# with a literal eps, the same pattern bricks/_verify/verify_f1.py's do_rmsnorm/do_layernorm already
# use for the identical reason.
# =====================================================================================================

def _rmsnorm(x, gamma, eps, tag):
    """x [C,T] f32, gamma [C] f32 (pass ones(C) for a folded/normalize-only call) -> [C,T] f32. ONE
    dispatch covers every row of T (n_tiles=T, device-side loop) -- gamma is tiny (C f32) and does not
    scale with T, so L1 never forces a T-split at this segment's C=1024 (see quantizer_shapes.py)."""
    C, T = x.shape
    x_tok = np.ascontiguousarray(x.T)
    sym = f"qd_rmsnorm_{_safe(tag)}"
    shim = (f"#include <stdint.h>\n"
            f'#include "{RMSNORM_CC}"\n'
            f'extern "C" void {sym}(float *x, float *g, float *o) {{\n'
            f"  rmsnorm_f32<16>(x, g, nullptr, o, {C}, {eps}f);\n}}\n")
    got = _run(f"rmsnorm_{tag}", shim, sym, x_tok, C, gamma,
              in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32, resident_depth=2)
    return np.ascontiguousarray(got.reshape(T, C).T)


def _layernorm(x, gamma, beta, eps, tag):
    """x [C,T] f32, gamma/beta [C] f32 -> [C,T] f32. Same rowwise/ONE-dispatch shape as _rmsnorm."""
    C, T = x.shape
    x_tok = np.ascontiguousarray(x.T)
    sym = f"qd_layernorm_{_safe(tag)}"
    shim = (f"#include <stdint.h>\n"
            f'#include "{LAYERNORM_CC}"\n'
            f'extern "C" void {sym}(float *x, float *gb, float *o) {{\n'
            f"  route_b_bricks::layernorm_core<16, true, true>(x, o, {C}, {eps}f, gb, gb + {C});\n}}\n")
    resident = np.concatenate([gamma, beta]).astype(np.float32)
    got = _run(f"layernorm_{tag}", shim, sym, x_tok, C, resident,
              in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32, resident_depth=2)
    return np.ascontiguousarray(got.reshape(T, C).T)


def _swiglu(gate, up, tag):
    """gate,up [C,T] f32 -> [C,T] f32. swiglu_f32_f32(gate_ptr,up_ptr,out_ptr) takes 3 pointer params,
    which does not fit bricklib's resident=None 2-buffer contract -- so, like F1's own do_swiglu shim,
    ONE streamed tile packs [gate_chunk|up_chunk] and a 1-line wrapper splits the pointer in two.
    Batched at SWIGLU_TILES_PER_DISPATCH (TDR-time-derived, see module docstring) rather than one
    dispatch for the whole [C,T] -- at C=3072 that would exceed the watchdog for large T."""
    C, T = gate.shape
    total = C * T
    chunk = SWIGLU_ROW
    n_tiles_total = -(-total // chunk)
    pad = n_tiles_total * chunk
    g_flat = np.zeros(pad, np.float32); g_flat[:total] = gate.reshape(-1)
    u_flat = np.zeros(pad, np.float32); u_flat[:total] = up.reshape(-1)
    g_tiles = g_flat.reshape(n_tiles_total, chunk)
    u_tiles = u_flat.reshape(n_tiles_total, chunk)

    sym = f"qd_swiglu_{_safe(tag)}"
    shim = (f"#include <stdint.h>\n"
            f'#include "{SWIGLU_CC}"\n'
            f'extern "C" void {sym}(float *x, float *o) {{\n'
            f"  swiglu_f32_f32(x, x + {chunk}, o);\n}}\n")
    out_flat = np.zeros(pad, np.float32)
    for b0 in range(0, n_tiles_total, SWIGLU_TILES_PER_DISPATCH):
        nb = min(SWIGLU_TILES_PER_DISPATCH, n_tiles_total - b0)
        tiles = np.concatenate([g_tiles[b0:b0 + nb], u_tiles[b0:b0 + nb]], axis=1)
        got = _run(f"swiglu_{tag}_{b0}", shim, sym, tiles, chunk, None,
                  flags=[f"-DSWIGLU_M=1", f"-DSWIGLU_N={chunk}"])
        out_flat[b0 * chunk:(b0 + nb) * chunk] = got.reshape(-1)
    return out_flat[:total].reshape(C, T)


def _gelu(x, tag):
    """x [C,T] f32 -> [C,T] f32. gelu_erf_f32(input,output) processes EXACTLY 16 elements, no cols
    param, no internal loop -- bound directly, no wrapper shim needed (2-buffer arity already matches
    resident=None). Volume is supplied by n_tiles per dispatch, batched at GELU_TILES_PER_DISPATCH
    (conservative, unmeasured -- see module docstring)."""
    C, T = x.shape
    total = C * T
    n_tiles_total = -(-total // 16)
    pad = n_tiles_total * 16
    flat = np.zeros(pad, np.float32); flat[:total] = x.reshape(-1)
    out_flat = np.zeros(pad, np.float32)
    shim = f'#include <stdint.h>\n#include "{GELU_CC}"\n'
    for b0 in range(0, n_tiles_total, GELU_TILES_PER_DISPATCH):
        nb = min(GELU_TILES_PER_DISPATCH, n_tiles_total - b0)
        tiles = flat[b0 * 16:(b0 + nb) * 16].reshape(nb, 16)
        got = _run(f"gelu_{tag}_{b0}", shim, "gelu_erf_f32", tiles, 16, None)
        out_flat[b0 * 16:(b0 + nb) * 16] = got.reshape(-1)
    return out_flat[:total].reshape(C, T)


# =====================================================================================================
# UPSAMPLE CONV_TRANSPOSE, ctx=0 (see module docstring #2). Mirrors window_driver.conv/_conv_chunk's
# OWN ci_chunk-accumulate-on-host structure, with the window step and crop logic replaced: step==t
# (no overlap needed), no crop (out_len==t*stride already the answer, no trailing samples to drop).
# =====================================================================================================

def _conv_transpose_chunk(x, w, bias, k, stride, t, tag):
    c_in, L = x.shape
    c_out = w.shape[1]
    out = np.zeros((c_out, L * stride), np.float32)
    tile_w = c_in * k
    tiles = np.zeros((c_out, tile_w + 1), np.float32)
    for co in range(c_out):
        tiles[co, :tile_w] = w[:, co, :].reshape(-1)   # [c_in,c_out,k] -> this channel's [c_in,k]
        tiles[co, tile_w] = bias[co]

    sym = f"qd_ct_{_safe(tag)}"
    shim = (f"#include <stdint.h>\n"
            f'#include "{wd.CT_CHAN_CC}"\n'
            f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
            f"  route_b_bricks::conv_transpose_channel_core(resident, tile, tile[{tile_w}],\n"
            f"                                              out, {c_in}, {k}, {t}, {stride});\n}}\n")
    for o in range(0, L, t):
        n = min(t, L - o)
        win = np.zeros((c_in, t), np.float32)
        win[:, :n] = x[:, o:o + n]
        got = _run(f"ct_{tag}", shim, sym, tiles, t * stride, win, resident_depth=1)
        got = got.reshape(c_out, t * stride)
        out[:, o * stride:(o + n) * stride] = got[:, :n * stride]  # ctx=0: no crop, no offset
    return out


def _conv_transpose_ctx0(x, w, bias, k, stride, ci_chunk, t_window, tag):
    """x [c_in,L], w [c_in,c_out,k] (ggml_conv_transpose_1d layout), bias [c_out] -> [c_out,L*stride].
    ci_chunk splits c_in and accumulates on host, exactly window_driver.conv's own ci_chunk lever
    (bias applied on the first chunk only)."""
    c_in, L = x.shape
    c_out = w.shape[1]
    cic = ci_chunk or c_in
    out = np.zeros((c_out, L * stride), np.float32)
    for c0 in range(0, c_in, cic):
        cs = min(cic, c_in - c0)
        first = (c0 == 0)
        out += _conv_transpose_chunk(x[c0:c0 + cs], w[c0:c0 + cs],
                                     bias if first else np.zeros(c_out, np.float32),
                                     k, stride, t_window, f"{tag}_k{cic}")
    return out


# =====================================================================================================
# WEIGHT FOLDING -- see module docstring for the algebra and the pwconv2-bias correction.
# =====================================================================================================

def fold_weights(W):
    """Returns FW, a dict of pre-folded weight arrays keyed by GGUF-tensor-name-derived strings.
    Every fold is a pure per-channel scale with no nonlinearity between it and the adjacent linear
    op -- see module docstring for which axis (rows=output-side, columns=input-side) each one hits."""
    FW = {}
    for layer in range(N_LAYERS):
        stem = f"{QPREFIX}.post_module.layers.{layer}"

        attn_norm = W(f"{stem}.attention_norm.weight").reshape(-1)
        wqkv = W(f"{stem}.attention.wqkv.weight")                      # [3*DIM,DIM]
        FW[f"{stem}.wqkv"] = np.ascontiguousarray(
            (wqkv * attn_norm[None, :])[:, :, None]).astype(np.float32)  # column (input) scale

        ffn_norm = W(f"{stem}.ffn_norm.weight").reshape(-1)
        w1 = W(f"{stem}.feed_forward.w1.weight")
        w3 = W(f"{stem}.feed_forward.w3.weight")
        FW[f"{stem}.w1"] = np.ascontiguousarray((w1 * ffn_norm[None, :])[:, :, None]).astype(np.float32)
        FW[f"{stem}.w3"] = np.ascontiguousarray((w3 * ffn_norm[None, :])[:, :, None]).astype(np.float32)

        attn_gamma = W(f"{stem}.attention_layer_scale.gamma").reshape(-1)
        wo = W(f"{stem}.attention.wo.weight")                          # [DIM,DIM], no bias
        FW[f"{stem}.wo"] = np.ascontiguousarray(
            (wo * attn_gamma[:, None])[:, :, None]).astype(np.float32)   # row (output) scale

        ffn_gamma = W(f"{stem}.ffn_layer_scale.gamma").reshape(-1)
        w2 = W(f"{stem}.feed_forward.w2.weight")                       # [DIM,FFN_DIM], no bias
        FW[f"{stem}.w2"] = np.ascontiguousarray(
            (w2 * ffn_gamma[:, None])[:, :, None]).astype(np.float32)

    for stage in range(2):
        p = f"{QPREFIX}.upsample.{stage}.1"
        gamma = W(f"{p}.gamma").reshape(-1)
        pw2 = W(f"{p}.pwconv2.weight")                                  # [DIM,CONVNEXT_HIDDEN]
        pw2_b = W(f"{p}.pwconv2.bias").reshape(-1)
        # pwconv2 HAS a bias, unlike every projection above -- the fold's additive term needs the
        # SAME gamma scale, or the result is wrong at exactly the gate boundary (see module docstring).
        FW[f"{p}.pwconv2_w"] = np.ascontiguousarray(
            (pw2 * gamma[:, None])[:, :, None]).astype(np.float32)
        FW[f"{p}.pwconv2_b"] = (gamma * pw2_b).astype(np.float32)

    # post_module.norm's gamma is algebraically foldable into upsample[0]'s conv_transpose (same
    # rule as attention_norm/ffn_norm above), but deliberately NOT folded -- see module docstring:
    # it would make post_module()'s return value a different tensor than cq.rvq_transformer's,
    # breaking per-stage gating. Both conv_transpose weights are used UNFOLDED, straight from the
    # GGUF; the real gamma is applied inside post_module()'s own final rmsnorm call instead.
    FW["upsample.0.conv_w"] = W(f"{QPREFIX}.upsample.0.0.conv.weight").astype(np.float32)
    FW["upsample.0.conv_b"] = W(f"{QPREFIX}.upsample.0.0.conv.bias").reshape(-1).astype(np.float32)
    FW["upsample.1.conv_w"] = W(f"{QPREFIX}.upsample.1.0.conv.weight").astype(np.float32)
    FW["upsample.1.conv_b"] = W(f"{QPREFIX}.upsample.1.0.conv.bias").reshape(-1).astype(np.float32)
    return FW


def _selftest_weight_folds(W, seed=0, T=20):
    """Host-side, device-free: every fold above must reproduce the UNFOLDED computation to float32
    rounding. Run once per driver session (called from verify_quantizer_segment.py's main(), matching
    this codebase's own `_selftest_*` convention, e.g. prefill-attn's golden.py) -- not on every
    fold_weights() call, which would defeat the point of folding."""
    rng = np.random.default_rng(seed)

    def rel_l2(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    stem = f"{QPREFIX}.post_module.layers.0"
    x = rng.standard_normal((DIM, T)).astype(np.float32)

    attn_norm = W(f"{stem}.attention_norm.weight").reshape(-1)
    wqkv = W(f"{stem}.attention.wqkv.weight")
    ms = np.mean(x.astype(np.float64) ** 2, axis=0, keepdims=True)
    normed = (x.astype(np.float64) / np.sqrt(ms + qs.RVQ_NORM_EPS))
    unfolded = wqkv.astype(np.float64) @ (normed * attn_norm[:, None].astype(np.float64))
    folded_w = (wqkv * attn_norm[None, :]).astype(np.float64)
    folded = folded_w @ normed
    e1 = rel_l2(folded, unfolded)
    assert e1 < 1e-4, f"attn_norm->wqkv fold: rel_l2={e1:.3e}"

    attn_gamma = W(f"{stem}.attention_layer_scale.gamma").reshape(-1, 1)
    wo = W(f"{stem}.attention.wo.weight")
    a2d = rng.standard_normal((DIM, T)).astype(np.float32)
    unfolded2 = (wo.astype(np.float64) @ a2d.astype(np.float64)) * attn_gamma.astype(np.float64)
    folded2 = (wo * attn_gamma).astype(np.float64) @ a2d.astype(np.float64)
    e2 = rel_l2(folded2, unfolded2)
    assert e2 < 1e-4, f"attn_gamma->wo fold: rel_l2={e2:.3e}"

    p = f"{QPREFIX}.upsample.0.1"
    gamma = W(f"{p}.gamma").reshape(-1, 1)
    pw2 = W(f"{p}.pwconv2.weight")
    pw2_b = W(f"{p}.pwconv2.bias").reshape(-1)
    y = rng.standard_normal((CONVNEXT_HIDDEN, T)).astype(np.float32)
    unfolded5 = (pw2.astype(np.float64) @ y.astype(np.float64)
                + pw2_b.reshape(-1, 1).astype(np.float64)) * gamma.astype(np.float64)
    folded5 = (pw2 * gamma).astype(np.float64) @ y.astype(np.float64) \
        + (gamma.reshape(-1) * pw2_b).reshape(-1, 1).astype(np.float64)
    e5 = rel_l2(folded5, unfolded5)
    assert e5 < 1e-4, f"convnext gamma->pwconv2 (+bias) fold: rel_l2={e5:.3e}"

    print(f"[weight-fold self-test] attn_norm->wqkv {e1:.3e}  attn_gamma->wo {e2:.3e}  "
          f"convnext_gamma->pwconv2(+bias) {e5:.3e}  -- all < 1e-4, PASS")


# =====================================================================================================
# TRANSFORMER (post_module).
# =====================================================================================================

def post_module_layer(x, W, FW, layer, tag):
    """[DIM,T] -> [DIM,T]. rmsnorm(gamma=1, folded) -> wqkv -> RoPE -> flash attn -> wo(+gamma,+resid)
    -> rmsnorm(gamma=1) -> w1/w3 -> swiglu -> w2(+gamma,+resid)."""
    stem = f"{QPREFIX}.post_module.layers.{layer}"
    T = x.shape[1]
    ones = np.ones(DIM, np.float32)

    attn_in = _rmsnorm(x, ones, qs.RVQ_NORM_EPS, f"{tag}an")
    wqkv_plan = qs.proj_plan(3 * DIM, DIM, name=f"{stem}.wqkv")
    qkv = wd.conv(attn_in, FW[f"{stem}.wqkv"], np.zeros(3 * DIM, np.float32), 1, 1, f"{tag}wqkv",
                 ci_chunk=wqkv_plan["ci_chunk"], resident_depth=wqkv_plan["resident_depth"])
    q3 = qkv[0:DIM].reshape(N_HEAD, HD, T)
    k3 = qkv[DIM:2 * DIM].reshape(N_HEAD, HD, T)
    v3 = qkv[2 * DIM:3 * DIM].reshape(N_HEAD, HD, T)

    positions = np.arange(T, dtype=np.float64)
    mask = _causal_window_mask(T, qs.RVQ_WINDOW_SIZE)

    ctx_heads = np.zeros((N_HEAD, HD, T), np.float32)
    for h in range(N_HEAD):
        qh = _rope_head(q3[h], positions, f"{tag}h{h}q")
        kh = _rope_head(k3[h], positions, f"{tag}h{h}k")
        ctx = _prefill_attn_head(qh, kh, v3[h], mask)          # [T,HD]
        ctx_heads[h] = ctx.T
    attn2d = ctx_heads.reshape(DIM, T)

    wo_plan = qs.proj_plan(DIM, DIM, name=f"{stem}.wo")
    h_ = wd.conv(attn2d, FW[f"{stem}.wo"], np.zeros(DIM, np.float32), 1, 1, f"{tag}wo", add=x,
                ci_chunk=wo_plan["ci_chunk"], resident_depth=wo_plan["resident_depth"])

    ff_in = _rmsnorm(h_, ones, qs.RVQ_NORM_EPS, f"{tag}fn")
    ff_plan = qs.proj_plan(FFN_DIM, DIM, name=f"{stem}.w1w3")
    gate = wd.conv(ff_in, FW[f"{stem}.w1"], np.zeros(FFN_DIM, np.float32), 1, 1, f"{tag}w1",
                  ci_chunk=ff_plan["ci_chunk"], resident_depth=ff_plan["resident_depth"])
    up = wd.conv(ff_in, FW[f"{stem}.w3"], np.zeros(FFN_DIM, np.float32), 1, 1, f"{tag}w3",
                ci_chunk=ff_plan["ci_chunk"], resident_depth=ff_plan["resident_depth"])
    ff_h = _swiglu(gate, up, f"{tag}sg")

    w2_plan = qs.proj_plan(DIM, FFN_DIM, name=f"{stem}.w2")
    x_new = wd.conv(ff_h, FW[f"{stem}.w2"], np.zeros(DIM, np.float32), 1, 1, f"{tag}w2", add=h_,
                    ci_chunk=w2_plan["ci_chunk"], resident_depth=w2_plan["resident_depth"])
    return x_new


def post_module(x, W, FW, tag="pm"):
    """[DIM,T] -> [DIM,T]. 8 layers + final rmsnorm with its REAL gamma (unlike attn_norm/ffn_norm
    inside each layer, this norm's gamma is NOT folded downstream -- see module docstring and
    fold_weights: folding it would make this function's return value incomparable to
    cq.rvq_transformer's, breaking per-stage gating)."""
    for layer in range(N_LAYERS):
        x = post_module_layer(x, W, FW, layer, f"{tag}l{layer}")
    final_gamma = W(f"{QPREFIX}.post_module.norm.weight").reshape(-1)
    return _rmsnorm(x, final_gamma, qs.RVQ_NORM_EPS, f"{tag}fin")


# =====================================================================================================
# UPSAMPLE (conv_transpose + ConvNeXt), x2 stages.
# =====================================================================================================

def convnext_block(x, W, FW, prefix, tag):
    """[DIM,T] -> [DIM,T]. dwconv-causal -> LayerNorm(affine, eps=1e-6) -> pwconv1 -> GELU(erf) ->
    pwconv2(+gamma,+bias fold, +residual)."""
    dw_w = W(f"{prefix}.dwconv.conv.weight")
    dw_b = W(f"{prefix}.dwconv.conv.bias").reshape(-1)
    y = _dwconv_causal(x, dw_w, dw_b, dw_w.shape[2], f"{tag}dw")

    ln_g = W(f"{prefix}.norm.weight").reshape(-1)
    ln_b = W(f"{prefix}.norm.bias").reshape(-1)
    y = _layernorm(y, ln_g, ln_b, qs.CONVNEXT_LN_EPS, f"{tag}ln")

    pw1_w = W(f"{prefix}.pwconv1.weight")[:, :, None]   # [4096,1024] -> [4096,1024,1]
    pw1_b = W(f"{prefix}.pwconv1.bias").reshape(-1)
    pw1_plan = qs.proj_plan(CONVNEXT_HIDDEN, DIM, name=f"{prefix}.pwconv1")
    y = wd.conv(y, np.ascontiguousarray(pw1_w), pw1_b, 1, 1, f"{tag}pw1",
               ci_chunk=pw1_plan["ci_chunk"], resident_depth=pw1_plan["resident_depth"])

    y = _gelu(y, f"{tag}gelu")

    pw2_plan = qs.proj_plan(DIM, CONVNEXT_HIDDEN, name=f"{prefix}.pwconv2")
    y = wd.conv(y, FW[f"{prefix}.pwconv2_w"], FW[f"{prefix}.pwconv2_b"], 1, 1, f"{tag}pw2", add=x,
               ci_chunk=pw2_plan["ci_chunk"], resident_depth=pw2_plan["resident_depth"])
    return y


def quantizer_upsample(x, W, FW, stage, tag):
    """[DIM,L] -> [DIM,L*2]. causal_conv_transpose_1d(k=stride=2, ctx=0, no crop) + convnext_block."""
    c_in, c_out, k, stride = qs.upsample_shape(stage)
    w = FW[f"upsample.{stage}.conv_w"]
    b = FW[f"upsample.{stage}.conv_b"]
    plan = qs.ct_plan(c_in, c_out, k, stride, t=qs.RES_T, name=f"upsample.{stage}")
    y = _conv_transpose_ctx0(x, w, b, k, stride, plan["ci_chunk"], qs.RES_T, f"{tag}ct")
    y = convnext_block(y, W, FW, f"{QPREFIX}.upsample.{stage}.1", f"{tag}cx")
    return y


def decode(codes, W, verbose=False):
    def note(msg):
        if verbose:
            print(f"    {msg}", flush=True)

    reset_stats()
    FW = fold_weights(W)
    note(f"folded weights ({len(FW)} tensors)")

    stage = rvq_lookup(codes, W)
    note(f"RVQ lookup ({codes.shape[0]} codebooks) -> {stage.shape}  "
        f"[{stats()['dispatches']} dispatches]")

    x = post_module(stage, W, FW)
    note(f"post_module transformer -> {x.shape}  [{stats()['dispatches']} dispatches]")

    for i in range(len(qs.DOWNSAMPLE_FACTORS)):
        x = quantizer_upsample(x, W, FW, i, f"up{i}")
        note(f"upsample {i} -> {x.shape}  [{stats()['dispatches']} dispatches]")

    return x


if __name__ == "__main__":
    import gguf_extract as gx
    import codec_quantizer_ref as cq

    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dump-dir>")
    dump_dir = Path(sys.argv[1])

    _cache = {}
    def g(name):
        if name not in _cache:
            _cache[name] = gx.load(qs.GGUF, name).astype(np.float32)
        return _cache[name]

    _selftest_weight_folds(g)

    codes = cq.load_codes(dump_dir)
    print(f"codes {codes.shape}")
    t0 = time.time()
    latent = decode(codes, g, verbose=True)
    print(f"latent {latent.shape}  total dispatches {stats()['dispatches']}  "
         f"wall {time.time() - t0:.1f}s")
