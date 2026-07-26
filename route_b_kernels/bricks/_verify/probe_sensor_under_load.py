#!/usr/bin/env python3
"""Is DRM_AMDXDNA_QUERY_SENSORS a usable NPU energy instrument? Sample it under SUSTAINED load.

The question this answers is narrow and load-bearing: package RAPL cannot resolve NPU per-token
decode energy, so every energy claim in the roadmap rests on an instrument that cannot see the
signal. The driver already exposes a power sensor through the UAPI. Does it read anything?

The trap to avoid is measuring the wrong window. A brick verify spends most of its wall-clock in
Python and JIT, and the device is busy for a small fraction of it, so sampling "during a verify
run" mostly samples an idle NPU and makes a working sensor look dead. Here the design is built
ONCE and then dispatched back-to-back for a fixed duration, so the NPU is busy nearly the whole
sampling window, and the sampler runs in a thread alongside it.
"""
import sys
import threading
import time

import ml_dtypes
import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
import aie.iron as iron  # noqa: E402
import bricklib  # noqa: E402
from npu_sensors import read_sensors  # noqa: E402
from verify_f2 import tile_pack  # noqa: E402
from verify_f2b import GEN, golden_mod, pack_int4, pack_int4_blocks  # noqa: E402

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
M, K, N, G, Nt = 4, 1024, 4096, 128, 16


def build():
    _, cc = golden_mod("gemm-int8xint4-dequant", "gemm_int8xint4_dequant.cc")
    rng = np.random.default_rng(29)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, (K // G, N)).astype(np.float32)
    tiles = []
    for t in range(N // Nt):
        sl = slice(t * Nt, (t + 1) * Nt)
        tiles.append(np.concatenate([
            pack_int4_blocks(b[:, sl], K, Nt, pack_int4),
            np.ascontiguousarray(scale[:, sl]).reshape(-1).view(np.int8)]))
    b_tile_bytes = int(pack_int4_blocks(b[:, :Nt], K, Nt, pack_int4).size)
    sym = f"gi4dq_load_{M}x{K}x{N}_g{G}_nt{Nt}"
    (GEN / f"{sym}_shim.cc").write_text(
        '#include <stdint.h>\n'
        f'#include "{cc}"\n'
        f'extern "C" void {sym}(const int8_t*wtile,const int8_t*a,bfloat16*c){{'
        'const int4*b=(const int4*)wtile;'
        f'const float*s=(const float*)(wtile+{b_tile_bytes});'
        f'gemm_int8xint4_dequant_tile<{M},{K},{Nt},{G}>(a,b,s,c);}}')
    in_tiles = np.stack(tiles)
    design = bricklib._build_streamed(sym, GEN / f"{sym}_shim.cc", in_tiles.shape[0],
                                      in_tiles.shape[1], M * Nt, M * K, [],
                                      np.int8, ml_dtypes.bfloat16, np.int8)
    in_t = iron.tensor(np.ascontiguousarray(in_tiles.reshape(-1)), dtype=np.int8, device="npu")
    r_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    out_t = iron.zeros((in_tiles.shape[0] * M * Nt,), dtype=ml_dtypes.bfloat16, device="npu")
    return design, in_t, r_t, out_t


design, in_t, r_t, out_t = build()
design(in_t, r_t, out_t)  # warm: first call carries any lazy setup
print(f"[load] design built and warmed; dispatching back-to-back for {SECONDS:.0f}s", flush=True)

samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append((time.monotonic(), {s["label"]: s["value"] for s in read_sensors()}))
        time.sleep(0.02)


th = threading.Thread(target=sampler, daemon=True)
th.start()
time.sleep(3.0)
n_idle = len(samples)

t0 = time.monotonic()
n_dispatch = 0
while time.monotonic() - t0 < SECONDS:
    design(in_t, r_t, out_t)
    n_dispatch += 1
t1 = time.monotonic()
stop.set()
th.join(timeout=2)

busy = t1 - t0
print(f"[load] {n_dispatch} dispatches in {busy:.1f}s "
      f"({busy / max(n_dispatch, 1) * 1e3:.2f} ms/dispatch, each 256 tiles x 8704 B)", flush=True)

labels = sorted({k for _, d in samples for k in d})
idle_s = [d for ts, d in samples[:n_idle]]
load_s = [d for ts, d in samples if t0 <= ts <= t1]
print(f"\nsamples: {len(idle_s)} idle, {len(load_s)} under load")
print(f"{'sensor':26s} {'idle max':>10s} {'LOAD max':>10s} {'LOAD mean':>11s} {'nonzero':>12s}")
for lb in labels:
    iv = [d.get(lb, 0.0) for d in idle_s]
    lv = [d.get(lb, 0.0) for d in load_s]
    if not lv:
        continue
    nz = sum(1 for v in lv if v > 0)
    print(f"{lb:26s} {max(iv) if iv else 0:10.4f} {max(lv):10.4f} "
          f"{sum(lv) / len(lv):11.5f} {nz:5d}/{len(lv):<6d}")
import os
os._exit(0)
