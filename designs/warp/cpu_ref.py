"""numpy reference for designs/warp: RIFE backward warp (bilinear, align_corners=True,
border padding). Two functions:

- `warp_bilinear_border`: exact RIFE semantics, clamps taps to the TRUE image border.
  Matches torch grid_sample bit-for-bit modulo float order (see warp_ref_check.py).
- `warp_tile_halo`: what the kernel actually computes -- clamps taps to a TILE+HALO
  window's edge, not the image edge. Diverges from the exact op only for pixels whose
  flow magnitude exceeds HALO (measured fraction: see README).
"""
import numpy as np


def warp_bilinear_border(img, fx, fy):
    H, W, C = img.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    sx = xx + fx.astype(np.float64)
    sy = yy + fy.astype(np.float64)
    return _bilinear_gather(img, sx, sy, 0, H - 1, 0, W - 1)


def warp_tile_halo(in_padded, fx, fy, halo):
    """in_padded: [tile_h+2h, tile_w+2h, C]. fx/fy: [tile_h, tile_w], flow relative to each
    output pixel. Output pixel (y,x) samples in_padded at (halo+y+fy, halo+x+fx), clamped
    to the padded buffer's own edge (the kernel's fallback, not the true image border)."""
    th, tw = fx.shape
    ph, pw = in_padded.shape[0], in_padded.shape[1]
    yy, xx = np.meshgrid(np.arange(th, dtype=np.float64), np.arange(tw, dtype=np.float64), indexing="ij")
    sx = xx + halo + fx.astype(np.float64)
    sy = yy + halo + fy.astype(np.float64)
    return _bilinear_gather(in_padded, sx, sy, 0, ph - 1, 0, pw - 1)


def _bilinear_gather(img, sx, sy, ylo, yhi, xlo, xhi):
    x0 = np.floor(sx)
    y0 = np.floor(sy)
    wx = sx - x0
    wy = sy - y0
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    x0c = np.clip(x0, xlo, xhi)
    x1c = np.clip(x0 + 1, xlo, xhi)
    y0c = np.clip(y0, ylo, yhi)
    y1c = np.clip(y0 + 1, ylo, yhi)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    wx = wx[..., None]
    wy = wy[..., None]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy
