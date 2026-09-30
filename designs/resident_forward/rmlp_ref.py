"""Bit-level model of the M2 block (rmlp_design.py + rf_norm.cc): pre-FFN RMSNorm, gate/up with
GELU*up, down, post-FFN RMSNorm, residual, layer scalar. Also the weight stream with its gains."""
import numpy as np
from chain_ref import bf16, bf16_bits, acc_add
from mlp_ref import (load_layer0, gemm_chain, h_from_gateup, gemm_down_chain, gateup_stream,
                     down_stream, D, ELEM, NCOL, NROW)

EPS = np.float32(1e-6)
W = D // NCOL


def own_sums(xs):
    """rf_norm own_sums: [rows, w] bf16-valued f32 -> per-row f32, in the kernel's order. `w`
    (a column's slice width) comes from xs.shape, not the module's 12B W=480."""
    w = xs.shape[1]
    sq = (xs * xs).astype(np.float32)                     # exact: bf16 products
    a = sq[:, 0:32]
    for j in range(1, w // 32):
        a = acc_add(a, sq[:, 32 * j:32 * j + 32])
    h = acc_add(a[:, :16], a[:, 16:])
    t = h[:, 0]
    for k in range(1, 16):
        t = acc_add(t, h[:, k])                           # scalar adds run on the vector adder
    return t


def rstd_rows(x, d_model=D, ncol=NCOL):
    """Row-0 cascade over `ncol` column slices, then 1 / sqrt(sum / d_model + eps) in IEEE f32.
    Defaults reproduce the 12B cascade (d_model=3840, ncol=8) unchanged."""
    w = d_model // ncol
    tot = own_sums(x[:, :w])
    for c in range(1, ncol):
        tot = acc_add(tot, own_sums(x[:, c * w:(c + 1) * w]))
    m = acc_add((tot / np.float32(d_model)).astype(np.float32), np.full_like(tot, EPS))
    return (np.float32(1) / np.sqrt(m)).astype(np.float32)


def rmsnorm(x, g, d_model=D, ncol=NCOL):
    """bf16(bf16(x * bf16(rstd)) * g), all operands bf16-valued."""
    s = bf16(rstd_rows(x, d_model, ncol))[:, None]
    return bf16(bf16(x * s) * g)


def block(x, Wt):
    """x bf16-valued f32 [rows, 3840] -> the block output (bf16-valued f32), plus the pieces."""
    g_pre, g_post, ls = bf16(Wt["g_pre"]), bf16(Wt["g_post"]), bf16(np.float32(Wt["ls"]))
    xn = rmsnorm(x, g_pre)
    hv = h_from_gateup(gemm_chain(xn, Wt["gate_q"], Wt["gate_s"]), gemm_chain(xn, Wt["up_q"], Wt["up_s"]))
    y = bf16(gemm_down_chain(hv, Wt["down_q"], Wt["down_s"]))
    yn = rmsnorm(y, g_post)
    xo = bf16(acc_add(x.astype(np.float32), yn))
    return bf16(xo * ls), dict(xn=xn, hv=hv, y=y, yn=yn)


def gain_elements(g, ls, c):
    """8 elements, core r gets the r-th then the (r + 4)-th: the gain slice + layer scalar, then zeros."""
    e = np.zeros((2 * NROW, ELEM), np.uint8)
    body = np.concatenate([bf16_bits(g[c * W:(c + 1) * W]), bf16_bits(np.float32(ls))[None]])
    e[:NROW, :body.size * 2] = body.view(np.uint8)
    return e.reshape(-1)


def stream(Wt, gu_cols):
    """Per column: pre gains, gate/up (m1_data's stream), down, post gains."""
    out = []
    for c in range(NCOL):
        out += [gain_elements(Wt["g_pre"], Wt["ls"], c), gu_cols[c], down_stream(Wt, c),
                gain_elements(Wt["g_post"], Wt["ls"], c)]
    return np.concatenate(out)
