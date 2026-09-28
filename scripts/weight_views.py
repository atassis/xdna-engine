# SPDX-License-Identifier: Apache-2.0
"""Read decode's `row_group_planar` int4 arena weight buffers back as [N, K] rows.

Findings (Task 1 Step 1, from `designs/decode_fused/gen_llm_decode.py` on
xdna-engine main, 2026-09-25) -- all defaults below are Gemma-4-12B's shipped plan
(int4, group_size=32, scale_dtype=bf16); pass different kwargs for another model.

(a) G per K. `_pack()` (gen_llm_decode.py:262-274) derives the row-group block size as::

        derive_row_group([K], spec.group_size, spec.dtype,
                          vec_size=widest_chunk(spec.group_size, spec.dtype),
                          scale_dtype=_BUILD_STATE["scale_dtype"])

    `widest_chunk(32, "int4")` is 64 (iron/common/quant.py:139-145: `min(cap, 2*group_size)`
    for a non-affine dtype). Measured on the served decode artifact
    (`decode_s6912_l48_are_wgm_pwm_rr_mvq2_aw_bsingle_main`): G=2 at K=3840 (Wqkv, Wg, Wu),
    G=1 at K=15360 (Wd) and K=8192 (Wo, both global and sliding geometries) -- confirmed by
    `.bin` file sizes matching `N * row_stride_bytes(K, 32, "int4", "bf16")` exactly (e.g.
    L0_Wd.bin = 3840 * 8640 = 33,177,600 B).

(b) K-chunked dump -> one arena buffer. The dump chunks `down_proj` (always, 4-way) and
    global-layer `o_proj` (2-way, layers where `layer_idx % sw_pattern == sw_pattern - 1`,
    i.e. 5, 11, 17, ...) -- each chunk is its OWN independently-quantized
    `row_group_planar` block (own scales, own grid). The arena buffer is NOT those planar
    chunks concatenated. gen_llm_decode.py's `_dequant_wd_from_kchunks` (L2959-2984) and
    `_dequant_from_probed_kchunks` (L2987-3016), called at the per-layer weight loop
    (L3160-3166), instead: DEQUANTIZE each chunk to float32 via
    `dequantize_weight_chunked` (chunk's own row_group_planar layout, chunk's own K),
    CONCATENATE the float32 chunks along K into one [N, K_total] matrix, then RE-QUANTIZE
    the whole row with `_pack()` (== `quantize_weight(..., layout="row_group_planar")`)
    using the DUMP's OWN grid -- `_repack_grid = (_qmf.get("full_range", False),
    _qmf.get("clip_search", False))` where `_qmf` is the decode dump's `quant.json`
    (L1850-1851), never the live precision plan's grid. Confirmed against
    `weights_int4g32sbf16_planar_qat_rg/quant.json`: `full_range: true, clip_search: true`.
    Net effect: dequant-then-requant over the full K, not a byte-level splice -- the arena
    `.bin` for a chunked tensor is bit-identical to `quantize_weight` on the float
    reconstruction, not to any rearrangement of the chunk bytes. On this build
    (GEMV_B_SINGLE / `bsingle`) every arena buffer under `buffers/` is already this single
    merged file (`L*_Wd.bin`, `L*_Wo.bin`, no `k0`/`k1` suffix) -- `arena_weight_rows` below
    reads it directly and does not re-run the merge.

(c) Wqkv row order. `dims.wqkv_head_major` is `false` for the shipped artifacts (meta.json),
    which takes the plain path at gen_llm_decode.py:3251:
    `weights[p + "Wqkv"] = np.concatenate(qkv_parts)`, where `qkv_parts` is built by
    appending `_pack(w, "qkv")` for Wq, then Wk, then Wv in that loop order (`mixer_tensors`
    at L3095-3096; Wv skipped when `not g.has_v`, true on global layers). So Wqkv is three
    independently-packed `row_group_planar` blocks (same G, since all three share K=D)
    concatenated row-major: q rows [0, qd), k rows [qd, qd+kvd), v rows [qd+kvd, qd+2kvd)
    (v only on sliding layers). Confirmed by size: L0_Wqkv.bin (sliding, has_v) =
    (16*256 + 8*256*2) rows * 2160 B/row = 17,694,720 B; L5_Wqkv.bin (global, no v) =
    (16*512 + 1*512) rows * 2160 B/row = 18,800,640 B -- both match the files on disk.
"""
import os

import numpy as np

from iron.common.quant import (
    derive_row_group,
    payload_bytes,
    row_stride_bytes,
    widest_chunk,
    _planar_to_rows,
)

GROUP_SIZE = 32
WEIGHT_DTYPE = "int4"
SCALE_DTYPE = "bf16"


def row_group_for(K, group_size=GROUP_SIZE, weight_dtype=WEIGHT_DTYPE, scale_dtype=SCALE_DTYPE):
    """G for a K-wide row under the shipped plan -- see module docstring (a)."""
    vec_size = widest_chunk(group_size, weight_dtype)
    return derive_row_group([K], group_size, weight_dtype, vec_size=vec_size,
                            scale_dtype=scale_dtype)


def planar_to_rows(blob, N, K, G, group_size=GROUP_SIZE, weight_dtype=WEIGHT_DTYPE,
                    scale_dtype=SCALE_DTYPE):
    """Flat `row_group_planar` bytes -> [N, stride] `[header][payload]` rows (uint8)."""
    stride = row_stride_bytes(K, group_size, weight_dtype, scale_dtype)
    return _planar_to_rows(np.asarray(blob), N, K, stride, weight_dtype, G)


def codes_and_scales(rows, N, K, scale_dtype=SCALE_DTYPE, group_size=GROUP_SIZE,
                      weight_dtype=WEIGHT_DTYPE):
    """[N, stride] header-first rows -> (codes [N, K] signed int4-in-int8, scales [N, n_groups] f32).

    Low nibble = even column, high nibble = odd column (quant.py's `quantize_weight`,
    matching `dequant_int4_group.cc`/`mv_quant.cc`'s contract).
    """
    if weight_dtype != "int4":
        raise NotImplementedError(f"codes_and_scales: only int4 is implemented, got {weight_dtype!r}")
    stride = row_stride_bytes(K, group_size, weight_dtype, scale_dtype)
    rows = np.asarray(rows).view(np.uint8).reshape(N, stride)
    n_groups = K // group_size
    scale_elem_bytes = {"f32": 4, "bf16": 2}[scale_dtype]
    scale_bytes = scale_elem_bytes * n_groups
    header = rows[:, :scale_bytes]
    if scale_dtype == "f32":
        scales = header.reshape(N, n_groups, 4).copy().view(np.float32).reshape(N, n_groups)
    else:
        import ml_dtypes
        scales = header.reshape(N, n_groups, 2).copy().view(ml_dtypes.bfloat16).reshape(
            N, n_groups).astype(np.float32)
    payload = rows[:, scale_bytes:scale_bytes + payload_bytes(K, weight_dtype)]
    lo = (payload & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = ((payload >> 4) & 0x0F).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty((N, K), dtype=np.int8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    return codes, scales


def arena_weight_rows(decode_dir, name, N, K, group_size=GROUP_SIZE, weight_dtype=WEIGHT_DTYPE,
                       scale_dtype=SCALE_DTYPE):
    """Read `decode_dir/buffers/<name>.bin` and return it as [N, stride] header-first rows.

    On a GEMV_B_SINGLE build (the served artifacts this task targets) every weight buffer is
    already the single merged file the build-time dequant-then-requant produced -- see module
    docstring (b). A `<name>k0.bin`/`<name>k1.bin`-chunked arena (a different build) is not
    read here: the merge rule needs the dump's own quant.json grid (full_range/clip_search),
    which is not recoverable from the arena directory alone, so it is refused rather than
    guessed.
    """
    path = os.path.join(decode_dir, "buffers", f"{name}.bin")
    if not os.path.isfile(path):
        chunk0 = os.path.join(decode_dir, "buffers", f"{name}k0.bin")
        if os.path.isfile(chunk0):
            raise NotImplementedError(
                f"{path}: not found, but {chunk0} exists -- this arena buffer is K-chunked. "
                f"arena_weight_rows only reads GEMV_B_SINGLE's single-buffer form; the "
                f"dequant-then-requant merge (module docstring (b)) needs the dump's own "
                f"quant.json grid, not available from the arena directory alone.")
        raise FileNotFoundError(path)
    blob = np.fromfile(path, dtype=np.uint8)
    G = row_group_for(K, group_size, weight_dtype, scale_dtype)
    return planar_to_rows(blob, N, K, G, group_size, weight_dtype, scale_dtype)


def qkv_rows(rows, qd, kvd, has_v):
    """[total_rows, stride] Wqkv rows -> {"q": ..., "k": ..., "v": ...} row slices.

    Row order is q, then k, then v (module docstring (c)); "v" is absent when `has_v` is
    False (global layers).
    """
    out = {"q": rows[:qd], "k": rows[qd:qd + kvd]}
    if has_v:
        out["v"] = rows[qd + kvd:qd + 2 * kvd]
    return out
