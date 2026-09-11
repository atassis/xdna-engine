"""Weight-format simulator: quantize -> dequantize a weight matrix, host-side.

Every scheme returns the EFFECTIVE weight the device would multiply by, so a host forward
pass with these weights sees exactly the information the format destroyed. It does NOT
reproduce the kernel's accumulation order -- that is a different question, and the
`kernel_round` flag below is the only place kernel rounding is emulated.

Two families:

  sym    w = q*s,           q in [-qmax, qmax] signed,  s = amax/qmax per (row, group).
         This is what we ship (iron/common/quant.py).

  affine w = q*s + m,       q signed, s and m free per (row, group).
         GGUF Q4_1 / FastFlowLM Q4NX shape. Read mlir-air-q4nx proj_qmm_pack.py:6-9 and
         kernels/q4_k.h:228-230: they store bf16 scale + bf16 min per (row, 32-col group)
         and fold the min into the accumulator as min*sum(B), never per element.
         Their q is UNSIGNED 0..15; signed [-8,7] is the same 16 levels re-centred by m,
         so the choice is free and signed keeps our existing nibble unpack.
"""
import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16
_QMAX_SYM = {4: 7, 8: 127}          # symmetric: [-qmax, qmax], one level unused
_QLO_AFF = {4: -8, 8: -128}          # affine: full signed range, all levels used
_QHI_AFF = {4: 7, 8: 127}


def _round_to(x, dtype):
    if dtype == "f32":
        return np.asarray(x, np.float32)
    if dtype == "bf16":
        return np.asarray(x, np.float32).astype(BF16).astype(np.float32)
    raise ValueError(dtype)


def bits_per_element(nbits, group, scheme, scale_dtype="bf16", min_dtype="bf16"):
    sb = {"f32": 32, "bf16": 16}[scale_dtype]
    hdr = sb if scheme == "sym" else sb + {"f32": 32, "bf16": 16}[min_dtype]
    return nbits + hdr / group


def quantize_sym(W, group, nbits=4, scale_dtype="f32", kernel_round=True):
    """Our shipped scheme. scale_dtype='f32' is the default we ship; the kernel casts it
    to bf16 before the MAC anyway (mv_quant.cc:88), which kernel_round reproduces."""
    M, K = W.shape
    qmax = _QMAX_SYM[nbits]
    Wg = np.asarray(W, np.float32).reshape(M, K // group, group)
    amax = np.abs(Wg).max(axis=2)
    s = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    s = _round_to(s, scale_dtype)
    q = np.clip(np.round(Wg / s[:, :, None]), -qmax, qmax)
    s_eff = _round_to(s, "bf16") if kernel_round else s
    out = q * s_eff[:, :, None]
    if kernel_round:                        # mv_quant.cc narrows the product to bf16
        out = _round_to(out, "bf16")
    return out.reshape(M, K).astype(np.float32)


def quantize_affine(W, group, nbits=4, scale_dtype="bf16", min_dtype="bf16",
                    kernel_round=True, zero_on_grid=False):
    """q*s + m, with q signed. s and m are stored at `scale_dtype`/`min_dtype` width and
    the quantization is done AGAINST the stored (rounded) values, so the rounding is
    inside the fit rather than applied after it."""
    M, K = W.shape
    lo, hi = _QLO_AFF[nbits], _QHI_AFF[nbits]
    Wg = np.asarray(W, np.float32).reshape(M, K // group, group)
    wmin, wmax = Wg.min(axis=2), Wg.max(axis=2)
    s = ((wmax - wmin) / (hi - lo)).astype(np.float32)
    s = np.where(s > 0, s, 1.0)
    s = _round_to(s, scale_dtype)
    if zero_on_grid:
        # Constrain m to -z*s with INTEGER z, so the reconstruction grid contains exact 0 the way
        # the symmetric grid always does. Costs a fraction of a step of range; the question is
        # whether preserving zero is worth more than that fraction. Same wire bytes either way --
        # m is still stored as a bf16 -- so this is a packer choice, not a format change.
        # w = (q - z)*s, so m = -z*s. q = lo must land on wmin: (lo - z)*s = wmin
        # => z = lo + round(-wmin/s).
        z = lo + np.round(-wmin / s)
        m = _round_to(-z * s, min_dtype)
    else:
        m = _round_to(wmin - lo * s, min_dtype)  # w = q*s + m, q = lo at w = wmin
    q = np.clip(np.round((Wg - m[:, :, None]) / s[:, :, None]), lo, hi)
    out = q * s[:, :, None] + m[:, :, None]
    if kernel_round:
        # q4_k.h folds the min as min*sum(B) at the accumulator, so only the q*s product
        # is narrowed to bf16 per element; the min term keeps accumulator width.
        out = _round_to(q * s[:, :, None], "bf16") + m[:, :, None]
    return out.reshape(M, K).astype(np.float32)


SCHEMES = {"sym": quantize_sym, "affine": quantize_affine}


def apply(W, spec):
    """spec: dict(scheme=, nbits=, group=, scale_dtype=, min_dtype=, kernel_round=)."""
    if spec["scheme"] == "bf16":
        return _round_to(W, "bf16")
    fn = SCHEMES[spec["scheme"]]
    kw = dict(group=spec["group"], nbits=spec.get("nbits", 4),
              scale_dtype=spec.get("scale_dtype", "bf16"),
              kernel_round=spec.get("kernel_round", True))
    if spec["scheme"] == "affine":
        kw["min_dtype"] = spec.get("min_dtype", "bf16")
        kw["zero_on_grid"] = spec.get("zero_on_grid", False)
    return fn(np.asarray(W, np.float32), **kw)


def rel_l2(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))
