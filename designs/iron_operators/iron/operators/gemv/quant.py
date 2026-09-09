# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side packer for the GEMV `weight_dtype` axis (int4 / int8 group-quantized A).

Byte-for-byte contract with aie_kernels/generic/mv_quant.cc: one row of the MxK weight matrix
packs as ``[n_groups x f32 scale][payload]``, payload = K/2 nibble-packed bytes (int4, low nibble
= even column) or K int8 bytes (int8, one byte per element). This lives beside the operator (not
in a model generator) because it IS the kernel's on-wire format, not a model-specific concern --
any caller of GEMV(weight_dtype=...) packs its weight the same way.
"""

import numpy as np

_QMAX = {"int4": 7, "int8": 127}


def row_stride_bytes(K: int, group_size: int, weight_dtype: str) -> int:
    if weight_dtype not in _QMAX:
        raise ValueError(f"unknown weight_dtype {weight_dtype!r} (expected 'int4' or 'int8')")
    if K % group_size != 0:
        raise ValueError(f"K={K} must be a whole number of groups (group_size={group_size})")
    n_groups = K // group_size
    payload = K // 2 if weight_dtype == "int4" else K
    stride = 4 * n_groups + payload
    if stride % 4 != 0:
        # mv_quant.cc reads the per-row scale via `reinterpret_cast<const float*>(rowp)`; row 0
        # is aligned by the buffer allocator, but every later row starts at row*stride, so a
        # stride that is not a multiple of 4 puts a scalar f32 load at a misaligned address on
        # every row past the first (4 * n_groups is already a multiple of 4, so this reduces to
        # payload % 4 -- K % 8 for int4, K % 4 for int8). A stride that is even but not a
        # multiple of 4 would also still misalign the shared bf16-granule arena downstream of
        # this buffer (iron/common/sequence.py), which %4 subsumes.
        raise ValueError(f"row stride {stride}B is not 4-byte aligned for K={K} "
                         f"group_size={group_size} {weight_dtype} -- the per-row f32 scale read "
                         f"would be misaligned on every row past the first")
    return stride


def quantize_weight(W: np.ndarray, group_size: int, weight_dtype: str) -> np.ndarray:
    """W: [M, K] float-ish array. Returns a flat np.int8 array of M * row_stride_bytes bytes,
    the exact device-side wire format mv_quant.cc reads (bit-for-bit; dtype is int8 purely so the
    generated MLIR types this buffer's shim BDs as ``i8``, matching decode_ddr_bytes.py's parser --
    the values themselves are opaque packed bytes, not signed quantities).
    """
    if weight_dtype not in _QMAX:
        raise ValueError(f"unknown weight_dtype {weight_dtype!r} (expected 'int4' or 'int8')")
    M, K = W.shape
    stride = row_stride_bytes(K, group_size, weight_dtype)
    n_groups = K // group_size
    qmax = _QMAX[weight_dtype]

    Wf = np.asarray(W, dtype=np.float32)
    Wg = Wf.reshape(M, n_groups, group_size)
    amax = np.max(np.abs(Wg), axis=2)                          # [M, n_groups]
    scale = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
    q = np.clip(np.round(Wg / scale[:, :, None]), -qmax, qmax).astype(np.int32).reshape(M, K)

    out = np.zeros((M, stride), dtype=np.uint8)
    out[:, : n_groups * 4] = scale.reshape(M, n_groups).view(np.uint8).reshape(M, n_groups * 4)
    if weight_dtype == "int4":
        # LOW nibble = even column, HIGH nibble = odd column -- matches
        # dequant_int4_group.cc's contract and mv_quant.cc's unpack.
        q_nibble = (q.astype(np.int16) & 0xF).astype(np.uint8)
        packed = (q_nibble[:, 0::2] | (q_nibble[:, 1::2] << 4)).astype(np.uint8)
    else:
        packed = q.astype(np.int8).view(np.uint8)
    out[:, n_groups * 4 :] = packed
    return out.reshape(-1).view(np.int8)


def dequantize_weight(packed: np.ndarray, M: int, K: int, group_size: int,
                       weight_dtype: str) -> np.ndarray:
    """Inverse of quantize_weight, for a CPU-side golden reference. Returns [M, K] float32."""
    stride = row_stride_bytes(K, group_size, weight_dtype)
    n_groups = K // group_size
    rows = np.asarray(packed).view(np.uint8).reshape(M, stride)
    scale = rows[:, : n_groups * 4].reshape(M, n_groups * 4).view(np.float32).reshape(M, n_groups)
    payload = rows[:, n_groups * 4 :]
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
