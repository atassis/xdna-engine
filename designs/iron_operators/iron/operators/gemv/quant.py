# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side packer for the GEMV `weight_dtype` axis (int4 / int8 group-quantized A).

Byte-for-byte contract with aie_kernels/generic/mv_quant.cc: one row of the MxK weight matrix
packs as ``[n_groups x scale][payload]``, payload = K/2 nibble-packed bytes (int4, low nibble
= even column) or K int8 bytes (int8, one byte per element). Scale defaults to f32 (4B/group,
matching today's kernel) and is selectable to bf16 (2B/group, `scale_dtype="bf16"`) -- see
`scale_dtype` below and mv_quant.cc's `SCALE_BF16` build macro. This lives beside the operator
(not in a model generator) because it IS the kernel's on-wire format, not a model-specific concern
-- any caller of GEMV(weight_dtype=...) packs its weight the same way.

Default call shape (`quantize_weight(W, group_size, weight_dtype)`,
`dequantize_weight(packed, M, K, group_size, weight_dtype)`) is BYTE-IDENTICAL to before this file
grew the `clip_search` / `scale_dtype` / `emulate_kernel_scale_cast` axes below: every new knob is
keyword-only and defaults to the old behavior (naive symmetric round-to-nearest, per-row-per-group
absmax/qmax scale, f32 header). Flip a knob to opt into an axis; nothing here changes size or
values for an existing caller that doesn't.
"""

import numpy as np

_QMAX = {"int4": 7, "int8": 127}
_SCALE_BYTES = {"f32": 4, "bf16": 2}


def _scale_header_bytes(n_groups: int, scale_dtype: str) -> int:
    if scale_dtype not in _SCALE_BYTES:
        raise ValueError(f"unknown scale_dtype {scale_dtype!r} (expected 'f32' or 'bf16')")
    return _SCALE_BYTES[scale_dtype] * n_groups


def row_stride_bytes(K: int, group_size: int, weight_dtype: str, scale_dtype: str = "f32") -> int:
    if weight_dtype not in _QMAX:
        raise ValueError(f"unknown weight_dtype {weight_dtype!r} (expected 'int4' or 'int8')")
    if K % group_size != 0:
        raise ValueError(f"K={K} must be a whole number of groups (group_size={group_size})")
    n_groups = K // group_size
    payload = K // 2 if weight_dtype == "int4" else K
    header = _scale_header_bytes(n_groups, scale_dtype)
    stride = header + payload
    if stride % 4 != 0:
        # mv_quant.cc reads the per-row scale via a scalar `reinterpret_cast<const T*>(rowp)`
        # (T = float for scale_dtype="f32", bfloat16 for "bf16"); row 0 is aligned by the buffer
        # allocator, but every later row starts at row*stride, so a stride that is not a multiple
        # of 4 puts that scalar load at a misaligned address on every row past the first, AND
        # still misaligns the shared bf16-granule arena downstream of this buffer
        # (iron/common/sequence.py) regardless of scale width -- that arena's 4-byte requirement is
        # what %4 actually enforces here, not the header's own element size (2B for bf16, which by
        # itself would only need 2-byte alignment). For f32 (4B/group), `header` is already a
        # multiple of 4, so this reduces to payload % 4 (K % 8 for int4, K % 4 for int8) -- the
        # original check. For bf16 (2B/group), `header` is a multiple of 4 only when n_groups is
        # even, so an ODD n_groups (K/group_size) now also needs payload%4 == 2 to compensate, or
        # the row does not pack.
        raise ValueError(f"row stride {stride}B is not 4-byte aligned for K={K} "
                         f"group_size={group_size} {weight_dtype} scale_dtype={scale_dtype} "
                         f"(header={header}B, payload={payload}B) -- the per-row scale read would "
                         f"be misaligned on every row past the first")
    return stride


def _mse_optimal_scale(Wg: np.ndarray, qmax: int, n_candidates: int,
                        clip_lo: float, clip_hi: float) -> np.ndarray:
    """Per-group MSE-optimal symmetric scale: grid-search c in [clip_lo, clip_hi] and pick
    scale = c*amax/qmax minimising the group's reconstruction MSE. Same clip-and-round quantizer
    as the c=1.0 (today's) case -- only the scale differs.

    Wg: [M, n_groups, group_size]. Returns scale [M, n_groups] float32.
    """
    amax = np.max(np.abs(Wg), axis=2)  # [M, n_groups]
    cs = np.linspace(clip_lo, clip_hi, n_candidates)
    best_scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)  # c=1.0 fallback
    best_mse = np.full(amax.shape, np.inf, dtype=np.float64)
    for c in cs:
        scale_c = np.where(amax > 0, c * amax / qmax, 1.0).astype(np.float32)
        q = np.clip(np.round(Wg / scale_c[:, :, None]), -qmax, qmax)
        recon = q * scale_c[:, :, None]
        mse = np.mean((Wg - recon) ** 2, axis=2, dtype=np.float64)
        better = mse < best_mse
        best_mse = np.where(better, mse, best_mse)
        best_scale = np.where(better, scale_c, best_scale)
    return best_scale


def quantize_weight(W: np.ndarray, group_size: int, weight_dtype: str, *,
                     clip_search: bool = False, n_clip_candidates: int = 61,
                     clip_range: tuple = (0.4, 1.0), scale_dtype: str = "f32") -> np.ndarray:
    """W: [M, K] float-ish array. Returns a flat np.int8 array of M * row_stride_bytes bytes,
    the exact device-side wire format mv_quant.cc reads (bit-for-bit; dtype is int8 purely so the
    generated MLIR types this buffer's shim BDs as ``i8``, matching decode_ddr_bytes.py's parser --
    the values themselves are opaque packed bytes, not signed quantities).

    clip_search: if True, replace the default scale = amax/qmax with a per-group MSE-optimal
    clip ratio (grid search over `n_clip_candidates` points in `clip_range`). Same wire format,
    same kernel, same bytes-per-row -- only which scale value gets written changes. Default False
    reproduces the exact byte-for-byte legacy scale.

    scale_dtype: "f32" (default, unchanged) or "bf16" -- narrows the stored per-group scale to 2
    bytes, shrinking row_stride_bytes accordingly. mv_quant.cc must be built with the matching
    `SCALE_BF16` setting to read this format; the two sides move together (see file docstring).
    """
    if weight_dtype not in _QMAX:
        raise ValueError(f"unknown weight_dtype {weight_dtype!r} (expected 'int4' or 'int8')")
    M, K = W.shape
    stride = row_stride_bytes(K, group_size, weight_dtype, scale_dtype)
    n_groups = K // group_size
    qmax = _QMAX[weight_dtype]

    Wf = np.asarray(W, dtype=np.float32)
    Wg = Wf.reshape(M, n_groups, group_size)
    if not clip_search:
        amax = np.max(np.abs(Wg), axis=2)                          # [M, n_groups]
        scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    else:
        lo, hi = clip_range
        scale = _mse_optimal_scale(Wg, qmax, n_clip_candidates, lo, hi)
    q = np.clip(np.round(Wg / scale[:, :, None]), -qmax, qmax).astype(np.int32).reshape(M, K)

    out = np.zeros((M, stride), dtype=np.uint8)
    header_bytes = _scale_header_bytes(n_groups, scale_dtype)
    if scale_dtype == "f32":
        out[:, :header_bytes] = scale.reshape(M, n_groups).view(np.uint8).reshape(M, header_bytes)
    else:
        import ml_dtypes  # lazy: only needed on the bf16-scale path
        scale_bf16 = scale.astype(ml_dtypes.bfloat16)
        out[:, :header_bytes] = scale_bf16.reshape(M, n_groups).view(np.uint8).reshape(
            M, header_bytes)
    if weight_dtype == "int4":
        # LOW nibble = even column, HIGH nibble = odd column -- matches
        # dequant_int4_group.cc's contract and mv_quant.cc's unpack.
        q_nibble = (q.astype(np.int16) & 0xF).astype(np.uint8)
        packed = (q_nibble[:, 0::2] | (q_nibble[:, 1::2] << 4)).astype(np.uint8)
    else:
        packed = q.astype(np.int8).view(np.uint8)
    out[:, header_bytes:] = packed
    return out.reshape(-1).view(np.int8)


def dequantize_weight(packed: np.ndarray, M: int, K: int, group_size: int, weight_dtype: str, *,
                       scale_dtype: str = "f32", emulate_kernel_scale_cast: bool = False
                       ) -> np.ndarray:
    """Inverse of quantize_weight, for a CPU-side golden reference. Returns [M, K] float32.

    emulate_kernel_scale_cast: mv_quant.cc:71-72 casts the loaded scale to bfloat16
    (`(bfloat16)scale[...]`) before the MAC, REGARDLESS of the stored scale width -- so even the
    shipped f32-scale path throws away precision on-device that this golden used to keep at full
    f32. Default False preserves that old (kernel-unfaithful) golden byte-for-byte; True narrows
    the read-back scale through bf16 the way the kernel does, which is the fidelity fix. It is a
    no-op in effect (but not literally skipped) when scale_dtype="bf16", since the header is
    already bf16-narrow by construction.
    """
    stride = row_stride_bytes(K, group_size, weight_dtype, scale_dtype)
    n_groups = K // group_size
    header_bytes = _scale_header_bytes(n_groups, scale_dtype)
    rows = np.asarray(packed).view(np.uint8).reshape(M, stride)
    if scale_dtype == "f32":
        scale = rows[:, :header_bytes].reshape(M, header_bytes).view(np.float32).reshape(
            M, n_groups)
    else:
        import ml_dtypes  # lazy: only needed on the bf16-scale path
        scale = rows[:, :header_bytes].reshape(M, header_bytes).view(ml_dtypes.bfloat16).reshape(
            M, n_groups).astype(np.float32)
    if emulate_kernel_scale_cast:
        import ml_dtypes
        scale = scale.astype(ml_dtypes.bfloat16).astype(np.float32)
    payload = rows[:, header_bytes:]
    if weight_dtype == "int4":
        lo = (payload & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = ((payload >> 4) & 0x0F).astype(np.int8)
        hi = np.where(hi >= 8, hi - 16, hi)
        q = np.empty((M, K), dtype=np.int8)
        q[:, 0::2] = lo
        q[:, 1::2] = hi
    else:
        q = payload.view(np.int8)
    q = q.reshape(M, n_groups, group_size).astype(np.float32)
    W = q * scale[:, :, None]
    return W.reshape(M, K)


def quantize_weight_asymmetric_research(W: np.ndarray, group_size: int, *, nbits: int = 4,
                                         clip_search: bool = False, n_clip_candidates: int = 61,
                                         clip_range: tuple = (0.4, 1.0)):
    """RESEARCH-ONLY host-side asymmetric (per-group uint zero point + scale) quantizer.

    NOT wired to mv_quant.cc or any wire format, and must never be called from quantize_weight or
    any device-facing path: mv_quant.cc has no zero-point subtract, and adopting one needs a
    tile-planar [Q][S][Z] layout (mlir-air's convention) instead of our row-interleaved
    [header][payload], which is a bigger change to gemv/design.py's TAP arithmetic than the kernel
    edit. Measured host-side on this model's own gate/up/down tensors, the asymmetric form is
    about 15% better in rel-L2 at slightly FEWER bytes per row than the symmetric one at the same
    group size -- enough to be worth measuring, not enough to adopt without the layout work. This
    function exists ONLY to measure rel-L2 against that scheme; it emits no on-wire bytes.

    dequant = (q - z) * s, q,z in [0, 2**nbits - 1] (uint), s per group (f32, full precision --
    the byte-narrowing question is orthogonal and already answered for the symmetric case).
    Returns (q, scale, zp), each [M, n_groups] or [M, n_groups, group_size] as documented below.
    """
    qmax = (1 << nbits) - 1
    M, K = W.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} must be a whole number of groups (group_size={group_size})")
    n_groups = K // group_size
    Wf = np.asarray(W, dtype=np.float32)
    Wg = Wf.reshape(M, n_groups, group_size)

    if not clip_search:
        wmin = Wg.min(axis=2)
        wmax = Wg.max(axis=2)
        rng = np.where(wmax > wmin, wmax - wmin, 1.0)
        scale = (rng / qmax).astype(np.float32)
        zp = np.clip(np.round(-wmin / scale), 0, qmax).astype(np.float32)
    else:
        lo, hi = clip_range
        wmin = Wg.min(axis=2)
        wmax = Wg.max(axis=2)
        center = (wmin + wmax) / 2.0
        half_rng = np.maximum((wmax - wmin) / 2.0, 1e-12)
        cs = np.linspace(lo, hi, n_clip_candidates)
        best_scale = ((wmax - wmin) / qmax).astype(np.float32)
        best_zp = np.clip(np.round(-wmin / np.where(best_scale == 0, 1.0, best_scale)),
                           0, qmax).astype(np.float32)
        best_mse = np.full(wmin.shape, np.inf, dtype=np.float64)
        for c in cs:
            lo_c = center - c * half_rng
            hi_c = center + c * half_rng
            rng_c = np.where(hi_c > lo_c, hi_c - lo_c, 1.0)
            scale_c = (rng_c / qmax).astype(np.float32)
            zp_c = np.clip(np.round(-lo_c / scale_c), 0, qmax).astype(np.float32)
            q_c = np.clip(np.round(Wg / scale_c[:, :, None] + zp_c[:, :, None]), 0, qmax)
            recon = (q_c - zp_c[:, :, None]) * scale_c[:, :, None]
            mse = np.mean((Wg - recon) ** 2, axis=2, dtype=np.float64)
            better = mse < best_mse
            best_mse = np.where(better, mse, best_mse)
            best_scale = np.where(better, scale_c, best_scale)
            best_zp = np.where(better, zp_c, best_zp)
        scale, zp = best_scale, best_zp

    q = np.clip(np.round(Wg / scale[:, :, None] + zp[:, :, None]), 0, qmax)
    return q, scale, zp


def dequantize_weight_asymmetric_research(q: np.ndarray, scale: np.ndarray, zp: np.ndarray,
                                           M: int, K: int, group_size: int) -> np.ndarray:
    """Inverse of quantize_weight_asymmetric_research. RESEARCH-ONLY, see that function's
    docstring."""
    n_groups = K // group_size
    W = (q - zp[:, :, None]) * scale[:, :, None]
    return W.reshape(M, K)
