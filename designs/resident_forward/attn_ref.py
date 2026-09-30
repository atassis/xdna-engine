"""numpy model of FusedAttnX2 (IRON fused_attn.cc, stage 3b-1), bit for bit.

32 query rows, keys in 64-blocks, head_dim 256 or 512, per-row widths (keys at or past a row's
width are masked when mask != 0). Stages, in natural layout ([keys, rows] for S^T and P):
  qT       q * bf16(log2 e), RNE to bf16
  scores   S^T = K . qT on the emulated bf16 mmul: chain_ref's bfp16 blocks and mac_block
  softmax  bf16 row max, exp2 on the SFU (EXP2_TABLE), accfloat sums, f32 state (fa_smT_block)
  x V      O += P . V per block after O *= corr (fa_o_rescale), then O * (1 / l) to bf16
Float ops other than the mmul follow device-fitted rules (attn_fit.py):
  acc_add  every accfloat add/sub, and vector float add (chain_ref.acc_add)
  fmul     aie::mul of float vectors: Peano's mul_elem_*_accuracy_safe, three bf16 parts each
  inv      1.0f / x through __divsf3, IEEE RNE
"""
import os
import numpy as np
from bfp16_model import f32_to_bfp16
from chain_ref import bf16, mac_block, unpack_blocks, _rne_grid, _expo

HERE = os.path.dirname(os.path.abspath(__file__))
EXP2_TABLE = os.path.join(HERE, "attn_exp2_table.npz")
# ROWS used to be a module global that callers (wrongly) monkeypatched for row counts other than
# 32 -- see g3_270m_ref.py's old `attn_ref.ROWS = rows`. softmax_blocks/pv now read the row count
# off their own arrays; ROWS survives only as the historical default query-row count and KEYS as
# the softmax block size (a hardware SFU granularity, not a model dim), both overridable per call.
ROWS, KEYS = 32, 64
LOG2E = np.float32(1.4453125)
LOWEST = np.uint32(0xFF7F0000).view(np.float32)


def acc_add(a, b):
    """chain_ref.acc_add (accfloat add: both onto 2^(max exponent - 23) RNE, add, RNE to f32),
    defined where both are zero and passing non-finite lanes through."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    a, b = np.broadcast_arrays(a, b)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    fin = np.isfinite(a64) & np.isfinite(b64)
    def ex(v):
        ok = fin & (v != 0)
        return np.where(ok, _expo(np.where(ok, v, 1.0)), -2000)
    L = np.maximum(np.maximum(ex(a64), ex(b64)) - 23, -1000)
    with np.errstate(invalid="ignore"):
        r = (_rne_grid(np.where(fin, a64, 0), L) + _rne_grid(np.where(fin, b64, 0), L)).astype(np.float32)
        return np.where(fin, r, (a64 + b64).astype(np.float32))


def bf16_trunc(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


# ---- exp2 (VEXP2): a function of the bf16-truncated argument; the table is the device's output
# for every bf16 bit pattern (attn_probe.py prim)
_TAB = None


def exp2(x, quant=bf16_trunc):
    global _TAB
    if _TAB is None:
        _TAB = np.load(EXP2_TABLE)["out_bits"].astype(np.uint32)
    k = quant(np.asarray(x, np.float32)).view(np.uint32) >> 16
    return (_TAB[k] << 16).view(np.float32)


def exp2_bits_closed_form(k):
    """The table explained, for bf16 argument bits k: Mitchell's 2^x ~ bits((x + 127) * 2^23), in
    fixed point, one LSB lower when the sign bit is set (-0 included), truncated to bf16. It
    misses only the gradual-underflow outputs of x in [-133, -126.5]; the table holds those."""
    k = np.asarray(k, np.int64)
    x = (k.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    fin = np.isfinite(x)
    v = np.floor((np.clip(np.where(fin, x, 0), -200, 200) + 127) * 2 ** 23).astype(np.int64) - (k >> 15)
    return np.clip(v, 0, 0x7F80 << 16) >> 16


# ---- float multiply: mul_elem_32_accuracy_safe (aie2p_vmult.h). Each operand splits into three
# bf16 parts; the split's msc steps are exact. The nine bf16 products are exact and summed in the
# header's order with the accfloat add. Subnormal operands flush to zero.
_TINY = np.float32(2.0 ** -126)


def _split3(v):
    a = bf16(v)
    r = (v.astype(np.float64) - a).astype(np.float32)
    b = bf16(r)
    c = bf16((r.astype(np.float64) - b).astype(np.float32))
    return a, b, c


def fmul(x, y):
    x = np.asarray(x, np.float32); y = np.asarray(y, np.float32)
    x = np.where(np.abs(x) < _TINY, np.float32(0) * x, x)
    y = np.where(np.abs(y) < _TINY, np.float32(0) * y, y)
    a, b, c = _split3(x)
    d, e, f = _split3(y)
    terms = [(c, f), (c, e), (b, f), (a, f), (b, e), (d, c), (b, d), (a, e), (a, d)]
    t = (terms[0][0].astype(np.float64) * terms[0][1]).astype(np.float32)
    for p, q in terms[1:]:
        t = acc_add(t, (p.astype(np.float64) * q).astype(np.float32))
    return t


def inv(x):
    return (np.float32(1.0) / np.asarray(x, np.float32)).astype(np.float32)


# ---- stages
def qt(q):
    """q [ROWS, hd] bf16 values -> the kernel's scaled Q (natural [rows, hd])."""
    return bf16(np.asarray(q, np.float64) * np.float64(LOG2E))


def _blocks(x):
    m, e = unpack_blocks(f32_to_bfp16(np.asarray(x, np.float32).reshape(-1)))
    return m.reshape(x.shape[0], x.shape[1] // 8, 8), e.reshape(x.shape[0], x.shape[1] // 8)


def mmul_nt(a, b, acc=None):
    """acc [M, N] f32 += a [M, K] . b [N, K]^T on the emulated bf16 mmul, K in 8-blocks ascending
    (bfp16 blocks run along K on both operands)."""
    am, ae = _blocks(a)
    bm, be = _blocks(b)
    if acc is None:
        acc = np.zeros((a.shape[0], b.shape[0]), np.float32)
    for kb in range(am.shape[1]):
        S = am[:, kb] @ bm[:, kb].T
        s = ae[:, kb][:, None] + be[:, kb][None, :] - 254 - 12
        acc = mac_block(acc, S, s)
    return acc


def scores(qs, k):
    """S^T [keys, ROWS] f32 for scaled Q qs [ROWS, hd] and K [keys, hd]."""
    return mmul_nt(k, qs)


def softmax_blocks(sT, widths, mask=1, lows=None, keys=KEYS):
    """fa_smT_block over every `keys`-key block of sT [keys_total, rows]. Returns P [keys_total,
    rows] (bf16 values) and cl [n_blocks, 2, rows] (correction, running sum) as the kernel hands
    them on. `rows` is read off sT's own shape (a caller used to monkeypatch the module's ROWS
    global instead -- see attention()'s docstring); `keys` is the softmax block size, a hardware
    SFU granularity default of 64, not a model dim, so it is a parameter rather than a global. With
    lows, fa_smT_block_x2_lo: row r sees lows[r] <= key < widths[r], and a row may start with fully
    masked blocks (its max floors at the lowest bf16)."""
    rows = sT.shape[1]
    widths = np.asarray(widths, np.int64)
    has_lo = lows is not None
    lows = np.zeros(rows, np.int64) if lows is None else np.asarray(lows, np.int64)
    nb = sT.shape[0] // keys
    lo, hi = widths.min(), widths.max()
    m = np.full(rows, -np.inf, np.float32)
    l = np.zeros(rows, np.float32)
    P = np.zeros(sT.shape, np.float32)
    cl = np.zeros((nb, 2, rows), np.float32)
    kk = np.arange(keys)[:, None]
    for blk in range(nb):
        base = blk * keys
        if mask and base >= hi:
            cl[blk, 0], cl[blk, 1] = 1.0, l
            continue
        partial = bool(mask) and (base + keys > lo or base < lows.max())
        s = sT[base:base + keys].astype(np.float32)
        vis = ((base + kk) < widths[None, :]) & ((base + kk) >= lows[None, :])
        sm = np.where(vis, s, -np.inf).astype(np.float32) if partial else s
        bmax = bf16(sm).max(axis=0)
        if has_lo:                  # fa_smT_block_x2_lo floors the block max at the lowest bf16
            bmax = np.maximum(bmax, LOWEST)
        if blk > 0:
            m_new = np.maximum(bf16(m), bmax)
            corr = exp2(acc_add(m, -m_new))
        else:
            m_new, corr = bmax, np.ones(rows, np.float32)
        with np.errstate(invalid="ignore"):
            e = exp2(acc_add(s, -np.broadcast_to(m_new, s.shape)))
        if partial:
            e = np.where(vis, e, 0).astype(np.float32)
        acc = np.zeros((8, rows), np.float32)                 # lane (key in tile, row)
        for z in range(8):
            acc = acc_add(acc, e[z * 8:(z + 1) * 8])
        a = acc_add(acc[0:4], acc[4:8])
        b = acc_add(a[0:2], a[2:4])
        bsum = acc_add(b[0], b[1])
        l = acc_add(fmul(l, corr), bsum) if blk > 0 else bsum
        m = m_new.astype(np.float32)
        P[base:base + keys] = e
        cl[blk, 0], cl[blk, 1] = corr, l
    return P, cl


PV_NATIVE = False     # x V on the native bf16 mmul: per output, acc = o, then + p*v key by key
PV_EDGE = False       # native only for a row's partially visible blocks, bfp16 elsewhere (K055)


def _ex(v):
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.abs(v)))
    return np.where(v == 0, -10000, e)


def _rne(v, L):
    return np.round(v / np.exp2(L)) * np.exp2(L)


def vmac_f(acc, a, b):
    """One vmac.f lane: acc (f32) + a * b (bf16), fitted on device (vmac_run.py, 0 of 122880).
    The product carries exponent ea + eb + 1 (its [1, 4) significand's top); with E the larger of
    that and acc's exponent, acc is RNE'd onto 2^(E-26), the product onto 2^(E-23), the exact sum
    RNE'd to f32."""
    acc, a, b = (np.asarray(v, np.float64) for v in (acc, a, b))
    p = a * b
    E = np.maximum(np.maximum(_ex(acc), np.where(p == 0, -10000, _ex(a) + _ex(b) + 1)), -1000)
    s = _rne(acc, E - 26) + _rne(p, E - 23)
    return np.where(s == 0, 0, _rne(s, _ex(s) - 23)).astype(np.float32)


def mmul_native(p, vb, acc):
    """acc [ROWS, n] += p [ROWS, keys] . vb [keys, n] on the native bf16 mmul: one vmac.f per key,
    keys ascending."""
    for k in range(p.shape[1]):
        acc = vmac_f(acc, p[:, k, None], vb[None, k, :])
    return acc


def pv(P, cl, v, part=None, keys=KEYS):
    """One x V worker: P [keys_total, rows], cl as softmax_blocks returns, v [keys_total, hdw]
    (its half of the head). Returns the raw O accumulator [rows, hdw] f32 and the finished O (bf16
    values). `rows` is read off P's own shape; the multi-block correction path below still assumes
    rows % 8 == 0 (the device's lane grouping), same as the original ROWS=32 default."""
    rows = P.shape[1]
    nb = P.shape[0] // keys
    o = np.zeros((rows, v.shape[1]), np.float32)
    for blk in range(nb):
        corr = cl[blk, 0]
        if blk > 0:
            unit = (corr.view(np.uint32) == 0x3F800000).reshape(rows // 8, 8).all(axis=1)
            f = np.repeat(~unit, 8)
            if f.any():
                o[f] = fmul(o[f], np.broadcast_to(corr[f, None], o[f].shape))
        p = P[blk * keys:(blk + 1) * keys].T                  # [rows, keys]
        vb = v[blk * keys:(blk + 1) * keys]
        for sl in range(v.shape[1] // 64):
            if PV_EDGE and part is not None and part[blk].any():
                pe = np.where(part[blk][:, None], p, 0).astype(np.float32)
                pv_ = np.where(part[blk][:, None], 0, p).astype(np.float32)
                ob = mmul_nt(pv_, vb[:, sl * 64:(sl + 1) * 64].T, o[:, sl * 64:(sl + 1) * 64])
                o[:, sl * 64:(sl + 1) * 64] = mmul_native(pe, vb[:, sl * 64:(sl + 1) * 64], ob)
            elif PV_NATIVE:
                o[:, sl * 64:(sl + 1) * 64] = mmul_native(p, vb[:, sl * 64:(sl + 1) * 64], o[:, sl * 64:(sl + 1) * 64])
            else:
                o[:, sl * 64:(sl + 1) * 64] = mmul_nt(p, vb[:, sl * 64:(sl + 1) * 64].T,
                                                      o[:, sl * 64:(sl + 1) * 64])
    iv = inv(cl[-1, 1])
    out = bf16(fmul(o, np.broadcast_to(iv[:, None], o.shape)))
    return o, out


def attention(q, k, v, widths, mask=1, lows=None, keys=KEYS):
    """FusedAttnX2's output [rows, hd] (bf16 values) for q [rows, hd], k, v [keys_total, hd].
    `rows` is q.shape[0]; a caller no longer needs to set attn_ref.ROWS (or KEYS) before calling
    this for a row/key count other than the historical defaults 32/64 -- pass `keys` explicitly
    and every row-shaped array below is sized off q/sT/P themselves."""
    rows, hd = q.shape
    sT = scores(qt(q), k)
    P, cl = softmax_blocks(sT, widths, mask, lows, keys=keys)
    part = None
    if PV_EDGE:
        lo = np.zeros(rows, np.int64) if lows is None else np.asarray(lows, np.int64)
        hi = np.asarray(widths, np.int64)
        base = np.arange(P.shape[0] // keys)[:, None] * keys
        vis = np.clip(np.minimum(hi[None, :], base + keys) - np.maximum(lo[None, :], base), 0, None)
        part = (vis > 0) & (vis < keys)                       # [block][row]
    halves = [pv(P, cl, v[:, h * hd // 2:(h + 1) * hd // 2], part, keys=keys)[1] for h in range(2)]
    return np.concatenate(halves, axis=1)
