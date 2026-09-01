#!/usr/bin/env python3
"""Window-stitching driver: run codec ops on device over streams longer than a core tile holds.

Every kernel here is capped at t=64 by L1 (the resident activation is the dominant term), while the
decoder's streams are 1760 to 112640 samples long. This module walks an op across a stream in
overlapping windows, carrying the causal left context each window needs, and stitches the valid
parts. The per-window results are already gated against the true unblocked stream
(verify_residual_multipass.py); what this adds is the stitching.

OP-MAJOR, NOT WINDOW-MAJOR, and that is forced rather than chosen. A window cannot be pipelined
through a residual stack: each dilated conv consumes its context, so 64 -> 58 (dil 1) -> 40 (dil 3)
-> -14 (dil 9). The window is exhausted before the third unit produces anything. So each op is run
to completion over the whole segment before the next op starts, which means every intermediate
materialises in host memory. That is the DDR traffic the fused design was trying to avoid, and at
these window lengths it is unavoidable -- see kb/upsampling-decoder-defeats-whole-tensor-residency.

LENGTH CONVENTION: every op returns only the outputs it can compute honestly. A causal conv with
reach `ctx` maps [C, L] -> [C, L - ctx], and output j corresponds to input position ctx + j. Callers
supply lead-in rather than the driver inventing zero padding mid-stream, so a stitched result is
bit-comparable to the unblocked one. snake is pointwise and preserves length.
"""
import contextlib
import io
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "route_b_kernels" / "bricks" / "_verify"))
import bricklib  # noqa: E402

UPSAMPLE_CC = (HERE / "upsample_stage.cc").resolve()
CT_CHAN_CC = (HERE / "conv_transpose_channel.cc").resolve()
SNAKE_CC = (ROOT / "route_b_kernels" / "bricks" / "snake" / "snake.cc").resolve()
CONV_CC = (ROOT / "route_b_kernels" / "bricks" / "conv-1d" / "conv_1d.cc").resolve()

T = 64          # rail max at c=96; the resident activation is what caps it
CONV_VEC = int(__import__("os").environ.get("CONV_VEC", "1"))  # vector conv core; 0 = scalar
_CB = int(time.time() * 1000) % 10**9

_stats = {"dispatches": 0, "useful": 0, "computed": 0}


def stats():
    """Recompute ratio, so the cost of the carry is reported rather than assumed."""
    s = dict(_stats)
    s["overhead"] = (1.0 - s["useful"] / s["computed"]) if s["computed"] else 0.0
    return s


def reset_stats():
    for k in _stats:
        _stats[k] = 0


def _run(name, shim_text, symbol, tiles, out_numel, resident, flags=None, resident_depth=2):
    """resident_depth is the objectFIFO depth of the RESIDENT operand only (default 2, the IRON
    default -- unchanged behaviour for every existing caller). Every op here acquires its resident
    operand ONCE before the streamed-tile loop and releases it once after (bricklib's
    `_build_streamed`), so resident_depth=1 is always numerically safe; it exists as a knob because
    the front stages' resident activation window is large enough that depth 2's doubling is what
    pushes the design over a 64 KiB core tile. See stage_shapes.py for the arithmetic."""
    shim = bricklib.GEN / f"wd_{name}_shim.cc"
    shim.write_text(shim_text)
    # bricklib prints a PASS/FAIL line per call, and every call here is an UNGATED intermediate
    # (golden zeros, gate inf) -- so those lines read as dozens of passing gates when nothing has
    # been gated. Only the stitched stage output is a gate. Swallow them.
    with contextlib.redirect_stdout(io.StringIO()):
        r = bricklib.verify_streamed(
            name=name, shim=shim, symbol=symbol, in_tiles=tiles, out_tile_numel=out_numel,
            resident=(None if resident is None else resident.reshape(-1).astype(np.float32)),
            unpack=lambda d: np.asarray(d), golden=np.zeros((tiles.shape[0], out_numel)),
            gate=np.inf,      # intermediates are not gated; the stitched stage output is
            in_dt=np.float32, out_dt=np.float32, compile_flags=flags,
            resident_dt=(None if resident is None else np.float32), resident_depth=resident_depth)
    _stats["dispatches"] += 1
    return np.asarray(r["got"], np.float32)


def snake(x, alpha, tag):
    """[C, L] -> [C, L]. Pointwise, so no context and no window overlap. alpha rides in the tile:
    the pure-buffer shim ABI carries no scalars, and one shim must serve every channel."""
    C, L = x.shape
    out = np.zeros((C, L), np.float32)
    shim_text = (f"// window_driver snake {tag} cb {_CB}\n#include <stdint.h>\n"
                 f'#include "{SNAKE_CC}"\n'
                 f'extern "C" void wd_snake_{tag}(float *tile, float *out) {{\n'
                 f"  snake_f32(tile, out, {T}, tile[{T}]);\n}}\n")
    for o in range(0, L, T):
        n = min(T, L - o)
        tiles = np.zeros((C, T + 1), np.float32)
        tiles[:, :n] = x[:, o:o + n]
        tiles[:, T] = alpha
        got = _run(f"snake_{tag}_{o}", shim_text, f"wd_snake_{tag}", tiles, T, None)
        out[:, o:o + n] = got.reshape(C, T)[:, :n]
        _stats["computed"] += C * T
        _stats["useful"] += C * n
    return out


def conv(x, w, bias, k, dilation, tag, add=None, ci_chunk=None, resident_depth=2):
    """[c_in, L] -> [c_out, L - (k-1)*dilation]. Output j corresponds to input position ctx + j.

    `add` supplies the residual connection, already aligned to the OUTPUT. It rides in the tile
    because it is a single row per output channel -- which is what lets this pass keep its one
    resident operand for the activation.

    ci_chunk splits the INPUT channels and accumulates partials ON THE HOST. Only stage 4 fits
    unchunked: the resident [c, t=64] operand is 24 KB at c=96 but 192 KB at c=768. Accumulating on
    the host rather than in a device-side buffer is deliberate -- an on-device accumulator would be
    exactly the static L1 state that made the fused kernels unreproducible
    (kb/static-l1-state-makes-kernels-unreproducible). Bias and the residual add are applied on the
    FIRST chunk only, so the partials sum to the right answer.

    The last chunk is zero-padded to ci_chunk rather than being its own size: zero weights times zero
    activations contribute nothing, and it keeps every chunk on ONE compiled design instead of
    forcing a second build for the remainder.

    resident_depth: forwarded to `_run` (default 2, unchanged for every caller that does not pass
    it). See stage_shapes.py for when 1 is required rather than merely smaller.
    """
    c_in, L = x.shape
    c_out = w.shape[0]
    ctx = (k - 1) * dilation
    M = L - ctx
    assert M > 0, f"segment {L} shorter than context {ctx}"
    if ci_chunk is None or ci_chunk > c_in:
        ci_chunk = c_in
    out = np.zeros((c_out, M), np.float32)
    for c0 in range(0, c_in, ci_chunk):
        cs = min(ci_chunk, c_in - c0)
        xc = np.zeros((ci_chunk, L), np.float32)
        xc[:cs] = x[c0:c0 + cs]
        wc = np.zeros((c_out, ci_chunk, k), np.float32)
        wc[:, :cs] = w[:, c0:c0 + cs, :]
        first = (c0 == 0)
        out += _conv_chunk(xc, wc, bias if first else np.zeros(c_out, np.float32),
                           k, dilation, f"{tag}_k{ci_chunk}",
                           add=(add if first else None), resident_depth=resident_depth)
    return out


def _conv_chunk(x, w, bias, k, dilation, tag, add=None, resident_depth=2):
    """One input-channel chunk of `conv`. Same windowing; the tag is shared across chunks so they
    all reuse a single compiled design."""
    c_in, L = x.shape
    c_out = w.shape[0]
    ctx = (k - 1) * dilation
    M = L - ctx
    step = T - ctx
    out = np.zeros((c_out, M), np.float32)

    wide = c_in * k + 1 + (T if add is not None else 0)
    # VECTOR core by default. Measured on device at identical tile geometry (probe_vec_device_ms.py):
    # scalar 1.123 ms/KiB, vector 0.049 -- 23x, and correctness is bit-comparable to the scalar
    # reference (verify_conv_1d.py: 4.483e-07 / 4.253e-07 / 2.860e-07 at dilation 1/3/9 against the
    # scalar's 4.475e-07 / 4.252e-07 / 2.863e-07, and 1.867e-07 at k=1). That settled a real open
    # question -- a scalar FMA loop should not have cost 1.1 ms/KiB, so the time might have been
    # per-tile overhead rather than compute, in which case this swap would have bought nothing.
    # It bought 23x, so it was compute. CONV_VEC=0 restores the scalar core for A/B.
    core = ("conv_1d_causal_core_vec<16>" if CONV_VEC else "conv_1d_causal_core")
    assert not CONV_VEC or T % 16 == 0, f"vector core needs T % 16 == 0, got T={T}"
    body = (f"  route_b_bricks::{core}(resident, tile, tile[{c_in * k}], out,\n"
            f"                                      {c_in}, {k}, {T}, {dilation});\n")
    if add is not None:
        body += f"  for (int p = 0; p < {T}; p++) out[p] += tile[{c_in * k + 1} + p];\n"
    sym = f"wd_conv_{tag}" + ("_a" if add is not None else "")
    shim_text = (f"// window_driver conv {tag} add={add is not None} cb {_CB}\n"
                 f"#include <stdint.h>\n"
                 f'#include "{CONV_CC}"\n'
                 f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
                 f"{body}}}\n")

    for o in range(0, M, step):
        n = min(step, M - o)
        win = np.zeros((c_in, T), np.float32)
        take = min(T, L - o)
        win[:, :take] = x[:, o:o + take]
        tiles = np.zeros((c_out, wide), np.float32)
        for co in range(c_out):
            tiles[co, :c_in * k] = w[co].reshape(-1)
            tiles[co, c_in * k] = bias[co]
            if add is not None:
                seg = add[co, o:o + n]
                tiles[co, c_in * k + 1 + ctx:c_in * k + 1 + ctx + len(seg)] = seg
        got = _run(f"conv_{tag}_{o}", shim_text, sym, tiles, T, win, resident_depth=resident_depth)
        out[:, o:o + n] = got.reshape(c_out, T)[:, ctx:ctx + n]
        _stats["computed"] += c_out * T
        _stats["useful"] += c_out * n
    return out


def residual_unit(x, wts, dilation, tag, ci_chunk=None, resident_depth=2):
    """[C, L] -> [C, L - 6*dilation]. snake -> conv_dilated -> snake -> conv_1x1 + residual."""
    a0, w1, b1, a2, w3, b3 = wts
    ctx = 6 * dilation
    s1 = snake(x, a0, f"{tag}a")
    h = conv(s1, w1, b1, 7, dilation, f"{tag}dil", ci_chunk=ci_chunk, resident_depth=resident_depth)
    s2 = snake(h, a2, f"{tag}b")
    # The residual reads x at the OUTPUT's positions, i.e. shifted by this unit's own context.
    return conv(s2, w3, b3, 1, 1, f"{tag}1x1", add=x[:, ctx:], ci_chunk=ci_chunk,
               resident_depth=resident_depth)


# The fused upsample stage keeps a [c_in, t] static scratch for snake's output, so its t is capped
# far below the convs' 64 -- 16 at c_in=192. That is fine: it is the one op whose fusion still pays,
# because snake there is consumed by a conv that reads every input channel, so materialising it once
# saves c_out-fold recomputation.
UPSAMPLE_T = 16


def upsample(x, alpha, w, bias, stride, tag, t=UPSAMPLE_T):
    """[c_in, L] -> [c_out, (L - ctx) * stride], fused snake -> conv_transpose_1d.

    A transposed conv with k = 2*stride and crop_right = stride maps t inputs to exactly t*stride
    outputs, and output p draws on inputs in [(p-k+1)/stride, p/stride] -- so it reaches back
    ceil((k-1)/stride) = 2 inputs. Two samples of lead-in per window, and the first ctx*stride
    outputs of each window are dropped.
    """
    c_in, L = x.shape
    k = 2 * stride
    ctx = -(-(k - 1) // stride)              # 2 for every codec rate
    c_out = w.shape[1]
    M = L - ctx
    step = t - ctx
    out = np.zeros((c_out, M * stride), np.float32)

    tile_w = c_in * k
    tiles = np.zeros((c_out, tile_w + 2), np.float32)
    for co in range(c_out):
        tiles[co, :tile_w] = w[:, co, :].reshape(-1)   # conv_transpose layout: [c_in, c_out, k]
        tiles[co, tile_w] = bias[co]
    tiles[0, tile_w + 1] = 1.0                          # compute snake once per stream

    shim_text = (f"// window_driver upsample {tag} cb {_CB}\n#include <stdint.h>\n"
                 f'#include "{UPSAMPLE_CC}"\n'
                 f'extern "C" void wd_up_{tag}(float *wtile, float *resident, float *out) {{\n'
                 f"  route_b_bricks::upsample_stage_core(wtile, resident, out);\n}}\n")

    for o in range(0, M, step):
        n = min(step, M - o)
        win = np.zeros((c_in, t), np.float32)
        take = min(t, L - o)
        win[:, :take] = x[:, o:o + take]
        resident = np.concatenate([alpha, win.reshape(-1)]).astype(np.float32)
        got = _run(f"up_{tag}_{o}", shim_text, f"wd_up_{tag}", tiles, t * stride, resident,
                   flags=[f"-DSTAGE_C_IN={c_in}", f"-DSTAGE_T={t}",
                          f"-DSTAGE_K={k}", f"-DSTAGE_STRIDE={stride}"])
        got = got.reshape(c_out, t * stride)
        out[:, o * stride:(o + n) * stride] = got[:, ctx * stride:(ctx + n) * stride]
        _stats["computed"] += c_out * t * stride
        _stats["useful"] += c_out * n * stride
    return out


def conv_transpose(x, w, bias, stride, tag, t=UPSAMPLE_T, ci_chunk=None, resident_depth=2):
    """[c_in, L] -> [c_out, (L - ctx) * stride]. No static state anywhere in the pass.

    Replaces the fused upsample for the driver. The fused kernel was measured green twice and then
    began returning NaN with its source unchanged; every kernel in this codec that keeps a static L1
    scratch has been fragile and every one that keeps none has been stable, so the driver buys
    stability by paying for snake as a separate pass. See conv_transpose_channel.cc.

    ci_chunk splits the INPUT channels and accumulates partials ON THE HOST, exactly like conv()'s
    ci_chunk (same reasoning: an on-device accumulator would be the static L1 state that made the
    fused kernels unreproducible). This is the lever the front stages actually need: the streamed
    weight row is c_in*k+1 floats REGARDLESS of the window length t, so a stage whose c_in*k alone
    exceeds a 64 KiB core tile (stage 1: 1536*16 = 24576 floats = 96 KiB, before the resident window
    or output tile are even counted) cannot be fixed by shrinking t -- only by shrinking c_in. See
    stage_shapes.py for the per-stage arithmetic and the chosen chunk sizes.

    resident_depth: forwarded to `_run` (default 2, unchanged for every caller that does not pass
    it -- verify_stage4_block.py's stage-4 call is bit-for-bit the same design as before this param
    existed).
    """
    c_in, L = x.shape
    k = 2 * stride
    if ci_chunk is None or ci_chunk > c_in:
        ci_chunk = c_in
    c_out = w.shape[1]
    ctx = -(-(k - 1) // stride)
    M = L - ctx
    out = np.zeros((c_out, M * stride), np.float32)
    for c0 in range(0, c_in, ci_chunk):
        cs = min(ci_chunk, c_in - c0)
        xc = np.zeros((ci_chunk, L), np.float32)
        xc[:cs] = x[c0:c0 + cs]
        wc = np.zeros((ci_chunk, c_out, k), np.float32)
        wc[:cs] = w[c0:c0 + cs]
        first = (c0 == 0)
        out += _conv_transpose_chunk(xc, wc, bias if first else np.zeros(c_out, np.float32),
                                     stride, f"{tag}_k{ci_chunk}", t, resident_depth)
    return out


def _conv_transpose_chunk(x, w, bias, stride, tag, t, resident_depth):
    """One input-channel chunk of `conv_transpose`. Same windowing conv_transpose always used; the
    tag carries ci_chunk so every chunk of the same width reuses one compiled design."""
    c_in, L = x.shape
    k = 2 * stride
    ctx = -(-(k - 1) // stride)
    c_out = w.shape[1]
    M = L - ctx
    step = t - ctx
    out = np.zeros((c_out, M * stride), np.float32)

    tile_w = c_in * k
    tiles = np.zeros((c_out, tile_w + 1), np.float32)
    for co in range(c_out):
        tiles[co, :tile_w] = w[:, co, :].reshape(-1)   # [c_in, c_out, k] -> this channel's [c_in,k]
        tiles[co, tile_w] = bias[co]

    shim_text = (f"// window_driver ct {tag} cb {_CB}\n#include <stdint.h>\n"
                 f'#include "{CT_CHAN_CC}"\n'
                 f'extern "C" void wd_ct_{tag}(float *tile, float *resident, float *out) {{\n'
                 f"  route_b_bricks::conv_transpose_channel_core(resident, tile, tile[{tile_w}],\n"
                 f"                                              out, {c_in}, {k}, {t}, {stride});\n}}\n")

    for o in range(0, M, step):
        n = min(step, M - o)
        win = np.zeros((c_in, t), np.float32)
        take = min(t, L - o)
        win[:, :take] = x[:, o:o + take]
        got = _run(f"ct_{tag}_{o}", shim_text, f"wd_ct_{tag}", tiles, t * stride, win,
                   resident_depth=resident_depth)
        got = got.reshape(c_out, t * stride)
        out[:, o * stride:(o + n) * stride] = got[:, ctx * stride:(ctx + n) * stride]
        _stats["computed"] += c_out * t * stride
        _stats["useful"] += c_out * n * stride
    return out


def upsample_unfused(x, alpha, w, bias, stride, tag, t=UPSAMPLE_T, ci_chunk=None, resident_depth=2):
    """snake then conv_transpose_1d as TWO passes, neither holding static state."""
    return conv_transpose(snake(x, alpha, f"{tag}sn"), w, bias, stride, f"{tag}ct", t,
                          ci_chunk=ci_chunk, resident_depth=resident_depth)
