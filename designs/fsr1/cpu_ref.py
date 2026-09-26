"""CPU reference for AMD FSR1 EASU+RCAS, fixed integer scale x3, RGB8.

Transliterated from ffx_fsr1.h / ffx_a.h (MIT, AMD FidelityFX), matching gamescope's
call sites: cs_easu.comp (FsrEasuCon/FsrEasuF) and cs_composite_rcas.comp
(FsrRcasCon/FsrRcasF), sharpness = g_upscaleFilterSharpness/10 (default 2 -> 0.2).

All math in float32, including the bit-trick fast rcp/rsqrt approximations
(APrxLoRcpF1/APrxMedRcpF1/APrxLoRsqF1) -- these are NOT ordinary reciprocals, they
are part of the algorithm's output and change the sharpening/lobe shape.

Edge handling: EASU's textureGather and RCAS's texelFetch both read outside [0,W)x[0,H)
at the border; gamescope's sampler state is not modelled here, so both clamp to the
input edge (clamp-to-edge), the common choice and the FSR1 reference's own suggestion.
"""
import numpy as np

F32 = np.float32


def _bits(a):
    return a.astype(np.float32).view(np.uint32)


def _from_bits(u):
    return u.astype(np.uint32).view(np.float32)


def aprx_lo_rcp(a):
    return _from_bits(np.uint32(0x7EF07EBB) - _bits(a))


def aprx_med_rcp(a):
    b = _from_bits(np.uint32(0x7EF19FFF) - _bits(a))
    return b * (-b * a + F32(2.0))


def aprx_lo_rsq(a):
    return _from_bits(np.uint32(0x5F347D74) - (_bits(a) >> np.uint32(1)))


def sat(x):
    return np.clip(x, F32(0.0), F32(1.0))


# ---------------------------------------------------------------- EASU ----

def fsr_easu_con(input_w, input_h, output_w, output_h):
    """Returns (con0, con1, con2, con3), each a 4-tuple of float32 (kept as
    float, not the packed AU4 the GPU uses -- no precision loss for our path)."""
    iw, ih, ow, oh = F32(input_w), F32(input_h), F32(output_w), F32(output_h)
    con0 = (
        iw / ow,
        ih / oh,
        F32(0.5) * iw / ow - F32(0.5),
        F32(0.5) * ih / oh - F32(0.5),
    )
    con1 = (F32(1.0) / iw, F32(1.0) / ih, F32(1.0) / iw, F32(-1.0) / ih)
    con2 = (F32(-1.0) / iw, F32(2.0) / ih, F32(1.0) / iw, F32(2.0) / ih)
    con3 = (F32(0.0), F32(4.0) / ih, F32(0.0), F32(0.0))
    return con0, con1, con2, con3


def _gather_rgb(img, gx, gy):
    """textureGather-equivalent: 2x2 taps at (gx-0.5,gy-0.5)..(gx+0.5,gy+0.5) in the
    GL_GATHER offset order b(top-left) a(top-right) / r(bot-left) g(bot-right) --
    FSR1's own comment calls this "a b / r g" reading top-to-bottom, so we return
    (topleft, topright, botleft, botright) as (w,x,y,z) matching con1/con2/con3 use.
    """
    h, w = img.shape[:2]
    x0 = np.floor(gx - 0.5).astype(np.int64)
    y0 = np.floor(gy - 0.5).astype(np.int64)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    tl = img[y0c, x0c]
    tr = img[y0c, x1c]
    bl = img[y1c, x0c]
    br = img[y1c, x1c]
    # GL textureGather component order for a 2x2 footprint is (x,y)=(0,1),(1,1),(1,0),(0,0)
    # i.e. .x=bottom-left .y=bottom-right .z=top-right .w=top-left. FSR1's own
    # ASCII picture (b c / e f g h ...) matches this: verified against the shader's
    # "a b / r g" comment and its usage of .x=i,.y=j,.z=f,.w=e for the p1 gather.
    return bl, br, tr, tl  # x, y, z, w


def _easu_tap(aC, aW, off_x, off_y, dirx, diry, len2x, len2y, lob, clp, c):
    vx = off_x * dirx + off_y * diry
    vy = off_x * (-diry) + off_y * dirx
    vx = vx * len2x
    vy = vy * len2y
    d2 = vx * vx + vy * vy
    d2 = np.minimum(d2, clp)
    wB = F32(2.0 / 5.0) * d2 + F32(-1.0)
    wA = lob * d2 + F32(-1.0)
    wB = wB * wB
    wA = wA * wA
    wB = F32(25.0 / 16.0) * wB + F32(-(25.0 / 16.0 - 1.0))
    w = wB * wA
    aC[..., 0] += c[..., 0] * w
    aC[..., 1] += c[..., 1] * w
    aC[..., 2] += c[..., 2] * w
    aW += w


def _easu_set(dirx, diry, length, ppx, ppy, mask, lA, lB, lC, lD, lE):
    w = np.where(mask == 0, (F32(1.0) - ppx) * (F32(1.0) - ppy),
        np.where(mask == 1, ppx * (F32(1.0) - ppy),
        np.where(mask == 2, (F32(1.0) - ppx) * ppy, ppx * ppy)))
    dc = lD - lC
    cb = lC - lB
    lenX = np.maximum(np.abs(dc), np.abs(cb))
    lenX = aprx_lo_rcp(lenX)
    dirX = lD - lB
    dirx += dirX * w
    lenX = sat(np.abs(dirX) * lenX)
    lenX = lenX * lenX
    length += lenX * w

    ec = lE - lC
    ca = lC - lA
    lenY = np.maximum(np.abs(ec), np.abs(ca))
    lenY = aprx_lo_rcp(lenY)
    dirY = lE - lA
    diry += dirY * w
    lenY = sat(np.abs(dirY) * lenY)
    lenY = lenY * lenY
    length += lenY * w
    return dirx, diry, length


def fsr_easu(img_f32, out_w, out_h):
    """img_f32: HxWx3 float32 in [0,1]. Returns out_h x out_w x 3 float32."""
    in_h, in_w = img_f32.shape[:2]
    con0, con1, con2, con3 = fsr_easu_con(in_w, in_h, out_w, out_h)

    oy, ox = np.meshgrid(np.arange(out_h, dtype=F32), np.arange(out_w, dtype=F32), indexing="ij")
    ppx = ox * con0[0] + con0[2]
    ppy = oy * con0[1] + con0[3]
    fpx = np.floor(ppx)
    fpy = np.floor(ppy)
    ppx = ppx - fpx
    ppy = ppy - fpy

    p0x = fpx * con1[0] + con1[2]
    p0y = fpy * con1[1] + con1[3]
    p1x = p0x + con2[0]
    p1y = p0y + con2[1]
    p2x = p0x + con2[2]
    p2y = p0y + con2[3]
    p3x = p0x + con3[0]
    p3y = p0y + con3[1]

    # convert normalized-space gather centers back to pixel-space for our gather helper
    def to_px(gx, gy):
        return gx * in_w, gy * in_h

    gx0, gy0 = to_px(p0x, p0y)
    gx1, gy1 = to_px(p1x, p1y)
    gx2, gy2 = to_px(p2x, p2y)
    gx3, gy3 = to_px(p3x, p3y)

    bczz = _gather_rgb(img_f32, gx0, gy0)  # (x=b?,y=c?,z=?,w=?) -- see below rename
    ijfe = _gather_rgb(img_f32, gx1, gy1)
    klhg = _gather_rgb(img_f32, gx2, gy2)
    zzon = _gather_rgb(img_f32, gx3, gy3)

    # Shader renames: bczzR.x=b bczzR.y=c ; ijfeR.x=i .y=j .z=f .w=e
    # klhgR.x=k .y=l .z=h .w=g ; zzonR.z=o .w=n
    b_, c_, _, _ = bczz
    i_, j_, f_, e_ = ijfe
    k_, l_, h_, g_ = klhg
    _, _, o_, n_ = zzon

    def luma(t):
        return t[..., 2] * F32(0.5) + (t[..., 0] * F32(0.5) + t[..., 1])

    bL, cL, iL, jL, fL, eL, kL, lL, hL, gL, oL, nL = (
        luma(b_), luma(c_), luma(i_), luma(j_), luma(f_), luma(e_),
        luma(k_), luma(l_), luma(h_), luma(g_), luma(o_), luma(n_),
    )

    dirx = np.zeros_like(ppx)
    diry = np.zeros_like(ppx)
    length = np.zeros_like(ppx)
    dirx, diry, length = _easu_set(dirx, diry, length, ppx, ppy, 0, bL, eL, fL, gL, jL)
    dirx, diry, length = _easu_set(dirx, diry, length, ppx, ppy, 1, cL, fL, gL, hL, kL)
    dirx, diry, length = _easu_set(dirx, diry, length, ppx, ppy, 2, fL, iL, jL, kL, nL)
    dirx, diry, length = _easu_set(dirx, diry, length, ppx, ppy, 3, gL, jL, kL, lL, oL)

    dir2x = dirx * dirx
    dir2y = diry * diry
    dirR = dir2x + dir2y
    zro = dirR < F32(1.0 / 32768.0)
    dirR = aprx_lo_rsq(dirR)
    dirR = np.where(zro, F32(1.0), dirR)
    dirx = np.where(zro, F32(1.0), dirx)
    dirx = dirx * dirR
    diry = diry * dirR

    length = length * F32(0.5)
    length = length * length
    stretch = (dirx * dirx + diry * diry) * aprx_lo_rcp(np.maximum(np.abs(dirx), np.abs(diry)))
    len2x = F32(1.0) + (stretch - F32(1.0)) * length
    len2y = F32(1.0) + F32(-0.5) * length
    lob = F32(0.5) + F32((1.0 / 4.0 - 0.04) - 0.5) * length
    clp = aprx_lo_rcp(lob)

    # min4/max4 over {j(ijfe.y), g(klhg.w), i(ijfe.x), k(klhg.x)}
    stacked = np.stack([j_, g_, i_, k_], axis=0)
    min4 = np.min(stacked, axis=0)
    max4 = np.max(stacked, axis=0)

    aC = np.zeros(ppx.shape + (3,), dtype=F32)
    aW = np.zeros_like(ppx)
    taps = [
        ((0.0, -1.0), b_), ((1.0, -1.0), c_), ((-1.0, 1.0), i_), ((0.0, 1.0), j_),
        ((0.0, 0.0), f_), ((-1.0, 0.0), e_), ((1.0, 1.0), k_), ((2.0, 1.0), l_),
        ((2.0, 0.0), h_), ((1.0, 0.0), g_), ((1.0, 2.0), o_), ((0.0, 2.0), n_),
    ]
    for (ox_, oy_), c in taps:
        off_x = F32(ox_) - ppx
        off_y = F32(oy_) - ppy
        _easu_tap(aC, aW, off_x, off_y, dirx, diry, len2x, len2y, lob, clp, c)

    # FSR1 uses the exact ARcpF1 (1/x) here, not an approximate rcp.
    resolved = aC / aW[..., None]
    out = np.minimum(max4, np.maximum(min4, resolved))
    return out


# ---------------------------------------------------------------- RCAS ----

FSR_RCAS_LIMIT = F32(0.25 - 1.0 / 16.0)


def fsr_rcas_con(sharpness):
    return F32(2.0) ** F32(-sharpness)


def fsr_rcas(img_f32, sharpness):
    h, w = img_f32.shape[:2]
    con = fsr_rcas_con(sharpness)

    def load(dy, dx):
        yy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
        xx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
        return img_f32[yy, xx]

    b = load(-1, 0)
    d = load(0, -1)
    e = img_f32
    f = load(0, 1)
    hh = load(1, 0)

    def luma(t):
        return t[..., 2] * F32(0.5) + (t[..., 0] * F32(0.5) + t[..., 1])

    bL, dL, eL, fL, hL = luma(b), luma(d), luma(e), luma(f), luma(hh)
    del bL, dL, eL, fL, hL  # nz/noise path unused (FSR_RCAS_DENOISE off, matches gamescope)

    mn4 = np.minimum(np.minimum(np.minimum(b, d), f), hh)
    mx4 = np.maximum(np.maximum(np.maximum(b, d), f), hh)

    peak_x, peak_y = F32(1.0), F32(-4.0)
    hitMin = np.minimum(mn4, e) / (F32(4.0) * mx4)
    hitMax = (peak_x - np.maximum(mx4, e)) / (F32(4.0) * mn4 + peak_y)
    lobe_c = np.maximum(-hitMin, hitMax)
    lobe = np.max(lobe_c, axis=-1, keepdims=True)
    lobe = np.maximum(-FSR_RCAS_LIMIT, np.minimum(lobe, F32(0.0))) * con

    rcpL = aprx_med_rcp(F32(4.0) * lobe + F32(1.0))
    out = (lobe * b + lobe * d + lobe * hh + lobe * f + e) * rcpL
    return out


def fsr1_x3(img_u8_rgb):
    """img_u8_rgb: HxWx3 uint8. Returns 3H x 3W x 3 uint8, EASU then RCAS
    (sharpness = g_upscaleFilterSharpness/10, default int 2 -> 0.2)."""
    h, w = img_u8_rgb.shape[:2]
    img_f = img_u8_rgb.astype(F32) / F32(255.0)
    easu_out = fsr_easu(img_f, w * 3, h * 3)
    rcas_out = fsr_rcas(easu_out, F32(0.2))
    out_u8 = np.clip(np.round(rcas_out * F32(255.0)), 0, 255).astype(np.uint8)
    return out_u8
