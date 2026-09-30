"""numpy model of AIE2P f32 -> bfp16ebs8 under conv_even, memory layout per 8-value block:
[shared exponent byte][8 two's-complement int8 mantissas]; value = m * 2^(E - 127) / 64."""
import numpy as np

def _rne_div_pow2(v, sh):
    """round-half-even of integer v / 2^sh (sh >= 0), exact integer arithmetic."""
    v = v.astype(np.int64); sh = sh.astype(np.int64)
    q = v >> sh
    rem = v - (q << sh)
    half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    return q + up

def f32_to_bfp16(x):
    """x: f32 array, length multiple of 8 -> uint8 array of len*9/8."""
    x = np.asarray(x, np.float32).reshape(-1, 8)
    u = x.view(np.uint32).astype(np.int64)
    sign = (u >> 31) & 1
    ex = (u >> 23) & 0xFF
    man = (u & 0x7FFFFF) | np.where(ex > 0, 1 << 23, 0)        # 24-bit significand
    exn = np.maximum(ex, 1)                                    # subnormals scale like ex = 1
    E = np.where(man.any(axis=1, keepdims=True), np.where(man > 0, exn, 0).max(axis=1, keepdims=True), 0)
    out = np.zeros((x.shape[0], 9), np.uint8)
    for bump in (0, 1):
        Eb = E + bump
        # value = man * 2^(exn - 150); target m = value * 64 / 2^(Eb - 127) = man * 2^(exn - Eb - 17)
        sh = (Eb - exn + 17)
        mag = _rne_div_pow2(man, np.clip(sh, 0, 62))
        mag = np.where(sh > 62, 0, mag)
        m = np.where(sign == 1, -mag, mag)
        ok = (m <= 127) & (m >= -128)
        if bump == 0:
            need = ~ok.all(axis=1, keepdims=True)
            m0, E0 = m, Eb
        else:
            m = np.where(need, m, m0); Eb = np.where(need, Eb, E0)
    out[:, 0] = Eb[:, 0].astype(np.uint8)
    out[:, 1:] = m.astype(np.int8).view(np.uint8)
    return out.reshape(-1)

def bfp16_to_f32(b):
    b = np.asarray(b, np.uint8).reshape(-1, 9)
    E = b[:, :1].astype(np.int64)
    m = b[:, 1:].view(np.int8).astype(np.float64)
    return (m * np.exp2(E - 127.0) / 64).astype(np.float32).reshape(-1)
