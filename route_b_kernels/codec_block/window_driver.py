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
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "route_b_kernels" / "bricks" / "_verify"))
import bricklib  # noqa: E402

SNAKE_CC = (ROOT / "route_b_kernels" / "bricks" / "snake" / "snake.cc").resolve()
CONV_CC = (ROOT / "route_b_kernels" / "bricks" / "conv-1d" / "conv_1d.cc").resolve()

T = 64          # rail max at c=96; the resident activation is what caps it
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


def _run(name, shim_text, symbol, tiles, out_numel, resident):
    shim = bricklib.GEN / f"wd_{name}_shim.cc"
    shim.write_text(shim_text)
    r = bricklib.verify_streamed(
        name=name, shim=shim, symbol=symbol, in_tiles=tiles, out_tile_numel=out_numel,
        resident=(None if resident is None else resident.reshape(-1).astype(np.float32)),
        unpack=lambda d: np.asarray(d), golden=np.zeros((tiles.shape[0], out_numel)),
        gate=np.inf,      # intermediates are not gated; the stitched stage output is
        in_dt=np.float32, out_dt=np.float32,
        resident_dt=(None if resident is None else np.float32))
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


def conv(x, w, bias, k, dilation, tag, add=None):
    """[C, L] -> [C, L - (k-1)*dilation]. Output j corresponds to input position ctx + j.

    `add` supplies the residual connection, already aligned to the OUTPUT (i.e. x[:, ctx:]). It
    rides in the tile because it is a single row per output channel -- which is what lets this pass
    keep its one resident operand for the activation."""
    C, L = x.shape
    ctx = (k - 1) * dilation
    M = L - ctx
    assert M > 0, f"segment {L} shorter than context {ctx}"
    step = T - ctx
    out = np.zeros((C, M), np.float32)

    wide = C * k + 1 + (T if add is not None else 0)
    body = (f"  route_b_bricks::conv_1d_causal_core(resident, tile, tile[{C * k}], out,\n"
            f"                                      {C}, {k}, {T}, {dilation});\n")
    if add is not None:
        body += f"  for (int p = 0; p < {T}; p++) out[p] += tile[{C * k + 1} + p];\n"
    shim_text = (f"// window_driver conv {tag} cb {_CB}\n#include <stdint.h>\n"
                 f'#include "{CONV_CC}"\n'
                 f'extern "C" void wd_conv_{tag}(float *tile, float *resident, float *out) {{\n'
                 f"{body}}}\n")

    for o in range(0, M, step):
        n = min(step, M - o)
        win = np.zeros((C, T), np.float32)
        take = min(T, L - o)
        win[:, :take] = x[:, o:o + take]
        tiles = np.zeros((C, wide), np.float32)
        for co in range(C):
            tiles[co, :C * k] = w[co].reshape(-1)
            tiles[co, C * k] = bias[co]
            if add is not None:
                seg = add[co, o:o + n]
                tiles[co, C * k + 1 + ctx:C * k + 1 + ctx + len(seg)] = seg
        got = _run(f"conv_{tag}_{o}", shim_text, f"wd_conv_{tag}", tiles, T, win)
        out[:, o:o + n] = got.reshape(C, T)[:, ctx:ctx + n]
        _stats["computed"] += C * T
        _stats["useful"] += C * n
    return out


def residual_unit(x, wts, dilation, tag):
    """[C, L] -> [C, L - 6*dilation]. snake -> conv_dilated -> snake -> conv_1x1 + residual."""
    a0, w1, b1, a2, w3, b3 = wts
    ctx = 6 * dilation
    s1 = snake(x, a0, f"{tag}a")
    h = conv(s1, w1, b1, 7, dilation, f"{tag}dil")
    s2 = snake(h, a2, f"{tag}b")
    # The residual reads x at the OUTPUT's positions, i.e. shifted by this unit's own context.
    return conv(s2, w3, b3, 1, 1, f"{tag}1x1", add=x[:, ctx:])
