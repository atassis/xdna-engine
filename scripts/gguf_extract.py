#!/usr/bin/env python3
"""Extract one tensor from a GGUF as a numpy array (f32/f16/bf16 only, no dequantization).

    python3 scripts/gguf_extract.py <file.gguf> <tensor-name>

Companion to gguf_shapes.py. Returns the array in numpy C-order with ggml's ne reversed, i.e. a
ggml ne=[k, c_out, c_in] weight comes back shaped [c_in, c_out, k] -- which is the layout the
conv-transpose-1d brick's golden expects, so extraction and reference agree by construction.
"""
import struct
import sys

import numpy as np

DT = {0: np.float32, 1: np.float16, 30: None}   # 30 = bf16, handled via uint16 view


def _read(path, want=None):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
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
        name = rd_str()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        infos[name] = (dims, ty, off)

    pos = f.tell()
    data_start = pos + (-pos) % alignment
    if want is None:
        return f, data_start, infos
    dims, ty, off = infos[want]
    n = 1
    for d in dims:
        n *= d
    f.seek(data_start + off)
    if ty == 0:
        a = np.frombuffer(f.read(n * 4), dtype=np.float32)
    elif ty == 1:
        a = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
    elif ty == 30:
        u = np.frombuffer(f.read(n * 2), dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    else:
        raise NotImplementedError(f"{want} is type id {ty}; only f32/f16/bf16 supported")
    # ggml ne is fastest-first; numpy wants slowest-first
    return a.reshape(tuple(reversed(dims))).copy()


def load(path, name):
    return _read(path, name)


if __name__ == "__main__":
    arr = load(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}\n  numpy shape {arr.shape} dtype {arr.dtype}")
    print(f"  finite={bool(np.isfinite(arr).all())} rms={float(np.sqrt((arr.astype(np.float64)**2).mean())):.5f}"
          f" min={float(arr.min()):+.4f} max={float(arr.max()):+.4f}")
