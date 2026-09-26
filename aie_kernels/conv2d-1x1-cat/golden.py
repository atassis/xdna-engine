"""Integer golden for conv1x1_cat.cc and its params packing. Rows use conv2d-3x3-u8's layout."""
import numpy as np


def conv1x1_cat_ref(xs, w, b, shift, valid_lo=0, valid_hi=None, pre_shift=0, mult=None):
    """xs: list of [CSRC,H,W] int8; w [COUT, NSRC*CSRC] int8; b [COUT] int32 -> [COUT,H,W] int8."""
    x = np.concatenate([a.astype(np.int64) for a in xs], 0)
    c, h, wd = x.shape
    acc = (w.astype(np.int64) @ x.reshape(c, -1)).reshape(-1, h, wd) + b.astype(np.int64)[:, None, None]
    rs = lambda a, s: (a + (1 << (s - 1))) >> s if s > 0 else a
    v = np.clip(rs(acc, pre_shift), -32768, 32767)
    m = np.ones(w.shape[0], np.int64) if mult is None else np.asarray(mult, np.int64)
    out = np.clip(rs(v * m[:, None, None], shift), -128, 127).astype(np.int8)
    hi = wd if valid_hi is None else valid_hi
    out[:, :, :valid_lo] = 0
    out[:, :, hi:] = 0
    return out


def pack_params(w, b, nsrc, mult=None):
    """w [COUT, NSRC*CSRC] -> [COUT/8][NSRC][CSRC/8][8 ic][8 oc] int8, then bias [COUT/8][8][8] int32
    and mult [COUT/8][8][8] int16."""
    cout, k = w.shape
    csrc = k // nsrc
    wb = w.reshape(cout // 8, 8, nsrc, csrc // 8, 8)          # ob, n, s, cb, k
    wb = wb.transpose(0, 2, 3, 4, 1)                          # ob, s, cb, k, n
    bias = np.repeat(b.astype(np.int32).reshape(cout // 8, 1, 8), 8, axis=1)
    mult = np.ones(cout, np.int64) if mult is None else np.asarray(mult)
    mul = np.repeat(mult.astype(np.int16).reshape(cout // 8, 1, 8), 8, axis=1)
    return np.concatenate([np.ascontiguousarray(wb).reshape(-1).view(np.int8),
                           bias.reshape(-1).view(np.int8), mul.reshape(-1).view(np.int8)])
