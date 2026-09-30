"""numpy model of the chain GEMM kernel (chain_mm_bfp16.cc), bit for bit.

Layouts are the kernel's: bfp16 8x8 subtiles of 72 B, each 8 blocks of [exp][8 int8 mantissas];
A [K/8][RB] (block = one row, 8 K); W [NB/16][K/8][2] (block = one n column, 8 K); int4 g32
sub-blocks [SUB_K/32][NB] bf16 scales then [NB/8][SUB_K/8] 32-byte subtiles, nibble i = (n=i/8,
k=i%8), low nibble first.
"""
import numpy as np
from bfp16_model import f32_to_bfp16

BLK = 72


def bf16_bits(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_val(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    return bf16_val(bf16_bits(x))


def sub_bytes(sub_k, nb):
    return sub_k // 32 * nb * 2 + sub_k * nb // 2


def decode_sub(src, sub_k, nb):
    """int4 g32 sub-block -> f32 [sub_k, nb] (exact: int4 x bf16 scale)."""
    src = np.asarray(src, np.uint8)
    sc = bf16_val(src[:sub_k // 32 * nb * 2].view(np.uint16)).reshape(sub_k // 32, nb)
    q = src[sub_k // 32 * nb * 2:].reshape(nb // 8, sub_k // 8, 32)
    lo = (q & 15).astype(np.int8); hi = (q >> 4).astype(np.int8)
    nib = np.stack([lo, hi], -1).reshape(nb // 8, sub_k // 8, 64)       # lane i = (n=i/8, k=i%8)
    nib = np.where(nib > 7, nib - 16, nib).astype(np.float32)
    v = nib.reshape(nb // 8, sub_k // 8, 8, 8)                           # [ns][kb][n][k]
    w = v.transpose(1, 3, 0, 2).reshape(sub_k, nb)                       # [k][n]
    return w * np.repeat(sc, 32, axis=0)


def w_block_from_f32(w):
    """f32 [K, NB] -> the kernel's converted W bytes [NB/16][K/8][2] subtiles."""
    K, NB = w.shape
    t = w.reshape(K // 8, 8, NB // 16, 2, 8)             # [kb][k][p][j][n]
    t = t.transpose(2, 0, 3, 4, 1)                        # [p][kb][j][n][k]
    return f32_to_bfp16(t.reshape(-1))


def a_bytes_from_f32(x, rb):
    """f32 [8*rb, K] -> A bytes [K/8][rb] subtiles (block = row, 8 K)."""
    M, K = x.shape
    t = x.reshape(rb, 8, K // 8, 8).transpose(2, 0, 1, 3)   # [kb][r][row][k]
    return f32_to_bfp16(t.reshape(-1))


def unpack_blocks(b):
    """bfp16 bytes -> (mant int64 [..., 8], exp int64 [...])"""
    b = np.asarray(b, np.uint8).reshape(-1, 9)
    return b[:, 1:].view(np.int8).astype(np.int64), b[:, 0].astype(np.int64)


def a_mant(a, rb, K):
    m, e = unpack_blocks(a)
    return m.reshape(K // 8, rb, 8, 8), e.reshape(K // 8, rb, 8)       # [kb][r][row][k], [kb][r][row]


def w_mant(w, K, NB):
    m, e = unpack_blocks(w)
    return m.reshape(NB // 16, K // 8, 2, 8, 8), e.reshape(NB // 16, K // 8, 2, 8)   # [p][kb][j][n][k]


def _rne_grid(v, L):
    """RNE of f64-exact values v onto the grid 2^L (elementwise)."""
    u = np.exp2(L.astype(np.float64))
    m = v / u
    fl = np.floor(m)
    fr = m - fl
    up = (fr > 0.5) | ((fr == 0.5) & (np.mod(fl, 2) == 1))
    return (fl + up) * u


def _expo(v):
    """unbiased binary exponent of nonzero f32/f64 values (floor(log2|v|))."""
    return np.frexp(v)[1].astype(np.int64) - 1


def mac_block(acc, S, s):
    """One bfp16 mac step, fitted on device (m1_probe3, 3072/3072 isolated adds): align at
    E = max(exp(acc), s + 12) (s + 12 = the two block exponents, unbiased), round acc and the exact
    block dot S * 2^s RNE onto 2^(E - 23), add exactly, round RNE to f32."""
    a64 = acc.astype(np.float64)
    ea = np.where(acc != 0, _expo(np.where(acc != 0, a64, 1.0)), -10**6)
    E = np.maximum(ea, s + 12)
    L = E - 23
    x = S.astype(np.float64) * np.exp2(s.astype(np.float64))
    t = _rne_grid(a64, L) + _rne_grid(x, L)
    return t.astype(np.float32)


def acc_add(a, b):
    """aie::add of two accfloat vectors, fitted on device (M2a, 34560/34560): the same alignment as
    mac_block, not IEEE: round both onto 2^(max exponent - 23) RNE, add, round RNE to f32."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    ea = np.where(a != 0, _expo(np.where(a != 0, a64, 1.0)), -10**6)
    eb = np.where(b != 0, _expo(np.where(b != 0, b64, 1.0)), -10**6)
    L = np.maximum(np.maximum(ea, eb) - 23, -1000)                         # both zero: 0, not 0/0
    return (_rne_grid(a64, L) + _rne_grid(b64, L)).astype(np.float32)


def mmul_acc(acc, am, ae, wm, we):
    """acc [8, NB] f32 += one K-slice: am [kb][8][8], ae [kb][8]; wm [kb][NB][8], we [kb][NB]."""
    for kb in range(am.shape[0]):
        S = am[kb] @ wm[kb].T                                              # exact int [8, NB]
        s = ae[kb][:, None] + we[kb][None, :] - 254 - 12
        acc = mac_block(acc, S, s)
    return acc


def chain_f32(A_slices, W_slices, rb):
    """A_slices[c]: A bytes of column c; W_slices[c]: converted W bytes; -> f32 [8*rb, NB] per the
    kernel's accumulation order (columns in order, k blocks in order), zero seed."""
    K = len(A_slices[0]) // BLK * 64 // (8 * rb)
    NB = len(W_slices[0]) // BLK * 64 // K
    acc = np.zeros((rb, 8, NB), np.float32)
    for a, w in zip(A_slices, W_slices):
        am, ae = a_mant(a, rb, K)
        wm, we = w_mant(w, K, NB)
        wmn = wm.transpose(1, 0, 2, 3, 4).reshape(K // 8, NB, 8)          # [kb][n][k]
        wen = we.transpose(1, 0, 2, 3).reshape(K // 8, NB)
        for r in range(rb):
            acc[r] = mmul_acc(acc[r], am[:, r], ae[:, r], wmn, wen)
    return acc.reshape(8 * rb, NB)


def out_f32_layout(y, rb):
    """[8*rb, NB] -> the ToF32 sink layout [NB/8][rb][8][8]."""
    NB = y.shape[1]
    return y.reshape(rb, 8, NB // 8, 8).transpose(2, 0, 1, 3).reshape(-1)


def convert_w(sub_blocks, sub_k, nb):
    """int4 sub-blocks of one K slice -> converted W bytes (the kernel's chain_convert_w)."""
    w = np.concatenate([decode_sub(s, sub_k, nb) for s in sub_blocks], 0)
    return w_block_from_f32(w)
