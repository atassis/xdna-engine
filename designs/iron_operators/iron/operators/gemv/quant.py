# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side packer for the GEMV `weight_dtype` axis (group-quantized A).

Four dtypes, two families. Byte-for-byte contract with aie_kernels/generic/mv_quant.cc:

  SYMMETRIC ("int4", "int8")   w = q*s,      q signed, one scale per (row, group).
      row = ``[n_groups x scale][payload]``
  AFFINE    ("int4a", "int8a") w = q*s + m,  q signed, a bf16 scale AND a bf16 min per
      (row, group).  row = ``[n_groups x bf16 scale][n_groups x bf16 min][payload]``

payload = K/2 nibble-packed bytes (int4, low nibble = even column) or K int8 bytes.
At equal BYTES the affine family is strictly better on this model: at K=1024 the symmetric
f32-scale g=128 row and the affine g=128 row are both 544 B, and mean dequant rel-L2 over the
MLP tensors is 0.1230 against 0.1034. The extra expressiveness is worth more than a finer group:
affine g=256 (528 B) also beats symmetric g=128 (544 B), cheaper AND closer. Scale defaults to f32 (4B/group,
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

# AFFINE ("a"-suffixed) weight dtypes: dequant = q*s + m, with a bf16 scale AND a bf16 min per
# (row, group). This is GGUF Q4_1's shape and is what FastFlowLM's shipped codec stores -- see
# mlir-air-q4nx programming_examples/fused_decode/proj_qmm_pack.py:6-9 (bf16 scales, then bf16
# mins, then the nibbles) and kernels/q4_k.h:228-230, which folds the min into the accumulator
# as min*sum(B) rather than subtracting it per element.
#
# Two deliberate divergences from that reference, both free:
#   * q is SIGNED here, [-8, 7], where theirs is unsigned [0, 15]. Sixteen levels either way --
#     m re-centres them -- so signed keeps mv_quant.cc's existing `vector_cast<int4>` unpack and
#     avoids a signed/unsigned unpack switch, which is exactly the ambient-representation seam
#     this tree keeps paying for.
#   * the header is scale AND min at bf16, so it is 4 B/group: ALWAYS 4-byte aligned and 16-byte
#     aligned whenever n_groups % 4 == 0. A uint4/uint8 zero point instead would make the header
#     3 B/group, which is what put the payload at a 24 B offset and made an int4 vector load
#     misaligned -- the constraint that was read as forcing a tile-planar [Q][S][Z] layout.
_AFFINE = {"int4a": 4, "int8a": 8}
_AFFINE_HEADER_BYTES = 4          # bf16 scale + bf16 min, per group


def _affine_levels(weight_dtype):
    n = _AFFINE[weight_dtype]
    lo = -(1 << (n - 1))
    return lo, (1 << (n - 1)) - 1


def is_affine(weight_dtype):
    return weight_dtype in _AFFINE


def _scale_header_bytes(n_groups: int, scale_dtype: str) -> int:
    if scale_dtype not in _SCALE_BYTES:
        raise ValueError(f"unknown scale_dtype {scale_dtype!r} (expected 'f32' or 'bf16')")
    return _SCALE_BYTES[scale_dtype] * n_groups


def row_stride_bytes(K: int, group_size: int, weight_dtype: str, scale_dtype: str = "f32") -> int:
    if weight_dtype not in _QMAX and weight_dtype not in _AFFINE:
        raise ValueError(f"unknown weight_dtype {weight_dtype!r} "
                         f"(expected 'int4', 'int8', 'int4a' or 'int8a')")
    if K % group_size != 0:
        raise ValueError(f"K={K} must be a whole number of groups (group_size={group_size})")
    n_groups = K // group_size
    if is_affine(weight_dtype):
        payload = K // 2 if weight_dtype == "int4a" else K
        header = _AFFINE_HEADER_BYTES * n_groups
        return header + payload      # 4 B/group: always %4, no alignment case to check
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


def _pack_affine(W: np.ndarray, group_size: int, weight_dtype: str) -> np.ndarray:
    """Affine pack: row = [n_groups x bf16 scale][n_groups x bf16 min][payload].

    The scale and min are rounded to bf16 BEFORE q is solved, so the fit is against the values
    the kernel will actually read rather than against f32 values that are then narrowed. That is
    the same reason the f32 scale buys nothing in the symmetric path -- the rounding cancels when
    it is inside the fit and does not when it is applied after.
    """
    import ml_dtypes
    M, K = W.shape
    lo, hi = _affine_levels(weight_dtype)
    n_groups = K // group_size
    Wg = np.asarray(W, dtype=np.float32).reshape(M, n_groups, group_size)
    wmin, wmax = Wg.min(axis=2), Wg.max(axis=2)
    s = ((wmax - wmin) / (hi - lo)).astype(np.float32)
    s = np.where(s > 0, s, 1.0).astype(ml_dtypes.bfloat16).astype(np.float32)
    m = (wmin - lo * s).astype(ml_dtypes.bfloat16).astype(np.float32)
    q = np.clip(np.round((Wg - m[:, :, None]) / s[:, :, None]), lo, hi).astype(np.int32)
    q = q.reshape(M, K)

    stride = row_stride_bytes(K, group_size, weight_dtype)
    out = np.zeros((M, stride), dtype=np.uint8)
    hdr = 2 * n_groups
    out[:, :hdr] = s.astype(ml_dtypes.bfloat16).view(np.uint8).reshape(M, hdr)
    out[:, hdr:2 * hdr] = m.astype(ml_dtypes.bfloat16).view(np.uint8).reshape(M, hdr)
    if weight_dtype == "int4a":
        nib = (q.astype(np.int16) & 0xF).astype(np.uint8)
        out[:, 2 * hdr:] = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    else:
        out[:, 2 * hdr:] = q.astype(np.int8).view(np.uint8)
    return out.reshape(-1).view(np.int8)


def _unpack_affine(packed: np.ndarray, M: int, K: int, group_size: int,
                    weight_dtype: str) -> np.ndarray:
    import ml_dtypes
    stride = row_stride_bytes(K, group_size, weight_dtype)
    n_groups = K // group_size
    hdr = 2 * n_groups
    rows = np.asarray(packed).view(np.uint8).reshape(M, stride)
    s = rows[:, :hdr].view(ml_dtypes.bfloat16).reshape(M, n_groups).astype(np.float32)
    m = rows[:, hdr:2 * hdr].view(ml_dtypes.bfloat16).reshape(M, n_groups).astype(np.float32)
    payload = rows[:, 2 * hdr:]
    if weight_dtype == "int4a":
        lo_n = (payload & 0x0F).astype(np.int8)
        lo_n = np.where(lo_n >= 8, lo_n - 16, lo_n)
        hi_n = ((payload >> 4) & 0x0F).astype(np.int8)
        hi_n = np.where(hi_n >= 8, hi_n - 16, hi_n)
        q = np.empty((M, K), dtype=np.int8)
        q[:, 0::2] = lo_n
        q[:, 1::2] = hi_n
    else:
        q = payload.view(np.int8)
    q = q.reshape(M, n_groups, group_size).astype(np.float32)
    return (q * s[:, :, None] + m[:, :, None]).reshape(M, K)


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
    if is_affine(weight_dtype):
        if clip_search:
            raise ValueError("clip_search is a symmetric-scale search and does not apply to the "
                             "affine dtypes; the affine fit already uses the group's full range")
        return _pack_affine(W, group_size, weight_dtype)
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
    if is_affine(weight_dtype):
        return _unpack_affine(packed, M, K, group_size, weight_dtype)
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
