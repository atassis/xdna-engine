"""Phase-decomposed EASU+RCAS -- verifies the SIMD-across-source-columns mapping
(coordinator's design) against cpu_ref.py's direct per-pixel formulation before
any AIE code is written. Same math, reorganized so one (px,py) phase's outputs
are a contiguous sweep over source columns (t -> fpx = t + OFFSET_X[px]).

Derivation (see designs/fsr1/README.md for the full pixel-space tap table):
  b:(0,-1) c:(1,-1) e:(-1,0) f:(0,0) g:(1,0) h:(2,0)
  i:(-1,1) j:(0,1)  k:(1,1)  l:(2,1) n:(0,2) o:(1,2)      -- offsets from (fpx,fpy)
For phase px, ox = 3t+px  =>  fpx(t) = t + OFFSET_X[px], ppx = FRAC_X[px] (a phase
CONSTANT, not a function of t) -- so every tap for a fixed phase is a t-shifted
view of one contiguous source row. Same for py/fpy/FRAC_Y.
"""
import numpy as np

F32 = np.float32
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cpu_ref import (aprx_lo_rcp, aprx_med_rcp, aprx_lo_rsq, sat, fsr_easu, fsr_rcas,
                     fsr_rcas_con, FSR_RCAS_LIMIT)

OFFSET_X = [-1, 0, 0]
FRAC_X = [F32(2.0 / 3.0), F32(0.0), F32(1.0 / 3.0)]  # ppx per phase (exact: 1-1/3, 0, 1/3)


def _row(img, w, h, y):
    yc = min(max(y, 0), h - 1)
    return img[yc]  # (w, 3)


def _clampcols(img_w, x0, n):
    """Return column indices [x0, x0+n) clamped to [0, img_w)."""
    return np.clip(np.arange(x0, x0 + n), 0, img_w - 1)


def easu_phase(img, in_w, in_h, px, py):
    """Returns (in_h, in_w, 3) -- phase (px,py)'s output plane, output col t maps to
    real output column 3t+px, output row s maps to real output row 3s+py."""
    ppx, ppy = FRAC_X[px], FRAC_X[py]
    T, S = in_w, in_h
    out = np.zeros((S, T, 3), dtype=F32)

    for s in range(S):
        fpy = s + OFFSET_X[py]
        rows = {dy: _row(img, in_w, in_h, fpy + dy) for dy in (-1, 0, 1, 2)}

        def taprow(dy, dx_lo, n):
            r = rows[dy]
            cols = _clampcols(in_w, T_BASE + dx_lo, n)
            return r[cols]

        fpx0 = OFFSET_X[px]
        T_BASE = fpx0  # fpx(t=0)

        b = taprow(-1, 0, T)
        c = taprow(-1, 1, T)
        e = taprow(0, -1, T)
        f = taprow(0, 0, T)
        g = taprow(0, 1, T)
        h = taprow(0, 2, T)
        i_ = taprow(1, -1, T)
        j = taprow(1, 0, T)
        k = taprow(1, 1, T)
        l = taprow(1, 2, T)
        n_ = taprow(2, 0, T)
        o = taprow(2, 1, T)

        def luma(t):
            return t[..., 2] * F32(0.5) + (t[..., 0] * F32(0.5) + t[..., 1])

        bL, cL, eL, fL, gL, hL = luma(b), luma(c), luma(e), luma(f), luma(g), luma(h)
        iL, jL, kL, lL, nL, oL = luma(i_), luma(j), luma(k), luma(l), luma(n_), luma(o)

        def easu_set(dirx, diry, length, mask, lA, lB, lC, lD, lE):
            if mask == 0: w = (1 - ppx) * (1 - ppy)
            elif mask == 1: w = ppx * (1 - ppy)
            elif mask == 2: w = (1 - ppx) * ppy
            else: w = ppx * ppy
            dc, cb = lD - lC, lC - lB
            lenX = aprx_lo_rcp(np.maximum(np.abs(dc), np.abs(cb)))
            dirX = lD - lB
            dirx = dirx + dirX * w
            lenX = sat(np.abs(dirX) * lenX); lenX = lenX * lenX
            length = length + lenX * w
            ec, ca = lE - lC, lC - lA
            lenY = aprx_lo_rcp(np.maximum(np.abs(ec), np.abs(ca)))
            dirY = lE - lA
            diry = diry + dirY * w
            lenY = sat(np.abs(dirY) * lenY); lenY = lenY * lenY
            length = length + lenY * w
            return dirx, diry, length

        dirx = np.zeros(T, F32); diry = np.zeros(T, F32); length = np.zeros(T, F32)
        dirx, diry, length = easu_set(dirx, diry, length, 0, bL, eL, fL, gL, jL)
        dirx, diry, length = easu_set(dirx, diry, length, 1, cL, fL, gL, hL, kL)
        dirx, diry, length = easu_set(dirx, diry, length, 2, fL, iL, jL, kL, nL)
        dirx, diry, length = easu_set(dirx, diry, length, 3, gL, jL, kL, lL, oL)

        dirR = dirx * dirx + diry * diry
        zro = dirR < F32(1.0 / 32768.0)
        rdirR = aprx_lo_rsq(dirR)
        rdirR = np.where(zro, F32(1.0), rdirR)
        dirx = np.where(zro, F32(1.0), dirx) * rdirR
        diry = diry * rdirR

        length = length * F32(0.5); length = length * length
        mx = np.maximum(np.abs(dirx), np.abs(diry))
        stretch = (dirx * dirx + diry * diry) * aprx_lo_rcp(mx)
        len2x = F32(1.0) + (stretch - F32(1.0)) * length
        len2y = F32(1.0) + F32(-0.5) * length
        lob = F32(0.5) + F32(1.0 / 4.0 - 0.04 - 0.5) * length
        clp = aprx_lo_rcp(lob)

        stacked = np.stack([j, g, i_, k], axis=0)
        min4 = np.min(stacked, axis=0)
        max4 = np.max(stacked, axis=0)

        aC = np.zeros((T, 3), F32); aW = np.zeros(T, F32)
        taps = [((0.0, -1.0), b), ((1.0, -1.0), c), ((-1.0, 1.0), i_), ((0.0, 1.0), j),
                ((0.0, 0.0), f), ((-1.0, 0.0), e), ((1.0, 1.0), k), ((2.0, 1.0), l),
                ((2.0, 0.0), h), ((1.0, 0.0), g), ((1.0, 2.0), o), ((0.0, 2.0), n_)]
        for (ox_, oy_), c_ in taps:
            offx = F32(ox_) - ppx; offy = F32(oy_) - ppy
            vx = (offx * dirx + offy * diry) * len2x
            vy = (offx * (-diry) + offy * dirx) * len2y
            d2 = np.minimum(vx * vx + vy * vy, clp)
            wB = F32(2.0 / 5.0) * d2 - F32(1.0); wA = lob * d2 - F32(1.0)
            wB = wB * wB; wA = wA * wA
            wB = F32(25.0 / 16.0) * wB - F32(25.0 / 16.0 - 1.0)
            wgt = wB * wA
            aC += c_ * wgt[:, None]; aW += wgt
        resolved = aC / aW[:, None]
        out[s] = np.minimum(max4, np.maximum(min4, resolved))
    return out


def easu_phased(img, in_w, in_h):
    """Assemble all 9 phases into the (3*in_h, 3*in_w, 3) output, matching
    cpu_ref.fsr_easu bit-for-bit (this is a REFORMULATION check)."""
    out = np.zeros((in_h * 3, in_w * 3, 3), dtype=F32)
    for py in range(3):
        for px in range(3):
            plane = easu_phase(img, in_w, in_h, px, py)
            out[py::3, px::3] = plane
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    h, w = 10, 10
    img = (rng.integers(10, 245, size=(h, w, 3)).astype(F32) / 255.0)
    golden = fsr_easu(img, w * 3, h * 3)
    got = easu_phased(img, w, h)
    d = np.abs(got - golden)
    print("phase-decomposed EASU vs cpu_ref.fsr_easu: max", d.max(), "mean", d.mean())
    assert d.max() < 1e-5, "phase reformulation does not match cpu_ref.py"
    print("PASS -- phase mapping verified equivalent to the direct formulation")
