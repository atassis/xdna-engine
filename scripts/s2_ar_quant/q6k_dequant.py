#!/usr/bin/env python3
"""Dequantize GGUF Q6_K blocks to f32 -- a from-scratch numpy port of ggml's
dequantize_row_q6_K (s2.cpp/ggml/src/ggml-quants.c), needed because gguf_extract.py
(the existing GGUF reader in this tree) only handles f32/f16/bf16 and the S2 AR
transformer's weights (`layers.*`, `fast_layers.*`, `embeddings.weight`) are Q6_K,
not f16 like the codec sub-tree -- confirmed via gguf_shapes.py against the real
s2-pro-q6_k.gguf: 204 Q6_K tensors, 609 f16, zero Q4/Q5/Q8.

Layout (QK_K=256 elements/superblock, ggml-common.h block_q6_K, 210 bytes):
    ql[128]   quants, low 4 bits, 2 values/byte
    qh[64]    quants, high 2 bits, 4 values/byte
    scales[16] int8, one per 16-element sub-block
    d         f16 superblock scale
    value = d * scales[sub_block] * (6-bit quant - 32)

Ported by reading ggml-quants.c directly rather than re-deriving the bit layout
by hand (kb: hanging numbers are bugs until proven otherwise) -- see
tests/test_q6k_dequant.py, which compiles that same upstream C function via a
tiny standalone harness and checks this port against its output bit-for-bit.

    python3 q6k_dequant.py <file.gguf> <tensor-name>
"""
import struct
import sys

import numpy as np

QK_K = 256
BLOCK_BYTES = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2  # ql + qh + scales + f16 d = 210


def dequantize_q6k(raw: bytes, n_elements: int) -> np.ndarray:
    """raw: tightly packed block_q6_K array, n_elements a multiple of QK_K. Vectorized
    transliteration of dequantize_row_q6_K's four-lane inner loop (it writes lanes
    l+0/32/64/96 per iteration, scaled by sub-block index is=l//16, is+0/2/4/6)."""
    assert n_elements % QK_K == 0, f"{n_elements} not a multiple of QK_K={QK_K}"
    nb = n_elements // QK_K
    assert len(raw) == nb * BLOCK_BYTES, (len(raw), nb, BLOCK_BYTES)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, BLOCK_BYTES)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208].view(np.int8)
    d = blocks[:, 208:210].view(np.float16).astype(np.float32).reshape(nb, 1)

    out = np.empty((nb, QK_K), dtype=np.float32)
    for n in range(0, QK_K, 128):
        ql_n = ql[:, n // 2:n // 2 + 64]      # [nb,64] -- ql advances 64B/128-elem half
        qh_n = qh[:, n // 4:n // 4 + 32]      # [nb,32] -- qh advances 32B/128-elem half
        sc_n = sc[:, n // 16:n // 16 + 8]     # [nb,8]  -- scales advance 8/128-elem half
        l = np.arange(32)
        is_ = l // 16                          # [32] sub-block index within this half

        q1 = ((ql_n[:, l] & 0x0F) | (((qh_n[:, l] >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql_n[:, l + 32] & 0x0F) | (((qh_n[:, l] >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql_n[:, l] >> 4) | (((qh_n[:, l] >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql_n[:, l + 32] >> 4) | (((qh_n[:, l] >> 6) & 3) << 4)).astype(np.int16) - 32

        out[:, n + l] = d * sc_n[:, is_ + 0] * q1
        out[:, n + l + 32] = d * sc_n[:, is_ + 2] * q2
        out[:, n + l + 64] = d * sc_n[:, is_ + 4] * q3
        out[:, n + l + 96] = d * sc_n[:, is_ + 6] * q4

    return out.reshape(-1)


def read_gguf_index(path):
    """Parse a GGUF header (magic, KV metadata, tensor table) without touching tensor
    data. Returns (open file, data-section start offset, {name: (dims, type, offset)})
    -- reuse this across many load_q6k_from_index calls to parse the header once
    rather than once per tensor (price_int4_requant.py scans dozens of tensors)."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rd_str():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz:
            f.read(sz[t])
        elif t == 8:
            rd_str()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_val(et)
        else:
            raise ValueError(f"unknown kv type {t}")

    alignment = 32
    for _ in range(n_kv):
        key = rd_str()
        t, = struct.unpack("<I", f.read(4))
        if key == "general.alignment" and t == 4:
            alignment, = struct.unpack("<I", f.read(4))
        else:
            skip_val(t)

    infos = {}
    for _ in range(n_tensors):
        tname = rd_str()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        infos[tname] = (dims, ty, off)

    pos = f.tell()
    data_start = pos + (-pos) % alignment
    return f, data_start, infos


def load_q6k_from_index(f, data_start, infos, name):
    """Dequantize one Q6_K tensor given an already-parsed index (see read_gguf_index).
    Returns (flat f32 array, element count) -- the flat form is what price_int4_requant.py
    quantizes; load_q6k below reshapes it to ggml's ne-reversed numpy layout."""
    dims, ty, off = infos[name]
    assert ty == 14, f"{name} is GGUF type {ty}, not Q6_K (14)"
    n = 1
    for dd in dims:
        n *= dd
    nb = n // QK_K
    f.seek(data_start + off)
    raw = f.read(nb * BLOCK_BYTES)
    return dequantize_q6k(raw, n), n


def load_q6k(path, name):
    """Read one Q6_K tensor from a GGUF by name, dequantized to f32, shaped like ggml's
    ne reversed (matches gguf_extract.load's convention for f32/f16/bf16 tensors)."""
    f, data_start, infos = read_gguf_index(path)
    dims = infos[name][0]
    flat, _ = load_q6k_from_index(f, data_start, infos, name)
    return flat.reshape(tuple(reversed(dims))).copy()


if __name__ == "__main__":
    arr = load_q6k(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}\n  numpy shape {arr.shape} dtype {arr.dtype}")
    print(f"  finite={bool(np.isfinite(arr).all())} rms={float(np.sqrt((arr.astype(np.float64)**2).mean())):.5f}"
          f" min={float(arr.min()):+.4f} max={float(arr.max()):+.4f}")
