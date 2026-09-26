"""Integer golden for conv3x3_u8.cc and the host-side packing it expects."""
import numpy as np


def conv3x3_u8_ref(x, w, b, shift, valid_lo=0, valid_hi=None):
    """x [CIN,H,W] uint8, w [COUT,CIN,3,3] int8, b [COUT] int32 -> [COUT,H,W] uint8.

    Zero-padded 'same' conv; (acc + 2^(shift-1)) >> shift, clamped to [0, 255];
    columns outside [valid_lo, valid_hi) forced to zero.
    """
    cin, h, wd = x.shape
    xp = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, dy:dy + h, dx:dx + wd] for dy in range(3) for dx in range(3)], 1)
    acc = w.astype(np.int64).reshape(w.shape[0], -1) @ cols.reshape(cin * 9, h * wd)
    acc = acc.reshape(-1, h, wd) + b.astype(np.int64)[:, None, None]
    if shift > 0:
        acc = (acc + (1 << (shift - 1))) >> shift
    out = np.clip(acc, 0, 255).astype(np.uint8)
    hi = wd if valid_hi is None else valid_hi
    out[:, :, :valid_lo] = 0
    out[:, :, hi:] = 0
    return out


PAD = 8  # zero pixels on each side of every row


def pack_rows(x):
    """[C,H,W] -> [H][C/8][W+16][8]: channel-blocked rows with an 8-pixel zero margin."""
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (PAD, PAD)))
    return np.ascontiguousarray(xp.reshape(c // 8, 8, h, wd + 2 * PAD).transpose(2, 0, 3, 1))


def unpack_rows(flat, c, h, wd, margins=False):
    """Inverse of pack_rows; margins=True keeps the padded columns."""
    x = flat.reshape(h, c // 8, wd + 2 * PAD, 8).transpose(1, 3, 0, 2).reshape(c, h, wd + 2 * PAD)
    return x if margins else x[:, :, PAD:-PAD]


def pack_params(w, b):
    """w [COUT,CIN,3,3] int8, b [COUT] int32 -> the kernel's params blob (int8 bytes)."""
    cout, cin = w.shape[:2]
    wb = w.reshape(cout // 8, 8, cin // 8, 8, 3, 3)          # ob, n, icb, k, ky, kx
    wb = wb.transpose(0, 4, 2, 5, 3, 1)                       # ob, ky, icb, kx, k, n
    bias = np.repeat(b.astype(np.int32).reshape(cout // 8, 1, 8), 8, axis=1)  # ob, px, n
    return np.concatenate([np.ascontiguousarray(wb).reshape(-1).view(np.int8),
                           bias.reshape(-1).view(np.int8)])
