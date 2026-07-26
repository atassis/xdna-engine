#!/usr/bin/env python3
"""What does a per-token HOST ROUND-TRIP actually cost?

The on-device decode-feedback work (argmax -> gather -> on-chip loop) is justified by a cost
that had been argued but never measured. This puts a number on it by DECOMPOSITION rather
than by timing a whole decoder, so the figure is attributable:

  A  dispatch only          -- submit the design, never touch the output from the host
  B  A + device->host read  -- add the output readback (forces the sync the loop needs)
  C  B + host feedback      -- add the host-side argmax and the write of the next input

Then:
  readback  = B - A     the cost of getting the result to the host at all
  feedback  = C - B     the cost of the host deciding and handing the token back
  round-trip = C - A    what a fully on-device loop would delete per token

All three run the SAME design on the SAME prebuilt tensors, so the difference is the host
path and nothing else. Gated on the iron.tensor rail, not the XRTTensor/XRTSubBuffer callable
path, which has a cache-line-granular CLFLUSH read race.

Energy is reported as a RELATIVE mean sensor reading only -- the NPU power sensor is monotone
in load but its absolute scale is not credible, so there is no mJ figure here on purpose.
"""
import statistics
import sys
import threading
import time
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
import aie.iron as iron  # noqa: E402
import bricklib  # noqa: E402
from npu_sensors import read_sensors  # noqa: E402
from verify_f2 import tile_pack  # noqa: E402
from verify_f2b import GEN, golden_mod, pack_int4, pack_int4_blocks  # noqa: E402

# A decode-shaped FFN projection: M=4 is the native-tile rounding of M=1 decode.
# SHAPE=small selects a cheap dispatch on purpose: the round-trip delta is a few ms against
# a ~156 ms tall-K dispatch whose own spread is ~30 ms, so the tall-K run establishes the
# ORDER but cannot resolve the split. Shrinking the dispatch raises the signal-to-noise on
# the very thing being measured, without changing the host path at all.
import os as _os
if _os.environ.get("SHAPE", "tallk") == "small":
    M, K, N, G, Nt = 64, 128, 128, 64, 32
else:
    M, K, N, G, Nt = 4, 1024, 4096, 128, 16
ITERS = int(_os.environ.get("ITERS", "60"))


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
    bb = int(pack_int4_blocks(b[:, :Nt], K, Nt, pack_int4).size)
    sym = f"gi4dq_rt_{M}x{K}x{N}_g{G}_nt{Nt}"
    (GEN / f"{sym}_shim.cc").write_text(
        '#include <stdint.h>\n'
        f'#include "{cc}"\n'
        f'extern "C" void {sym}(const int8_t*wtile,const int8_t*a,bfloat16*c){{'
        'const int4*b=(const int4*)wtile;'
        f'const float*s=(const float*)(wtile+{bb});'
        f'gemm_int8xint4_dequant_tile<{M},{K},{Nt},{G}>(a,b,s,c);}}')
    in_tiles = np.stack(tiles)
    design = bricklib._build_streamed(sym, GEN / f"{sym}_shim.cc", in_tiles.shape[0],
                                      in_tiles.shape[1], M * Nt, M * K, [],
                                      np.int8, ml_dtypes.bfloat16, np.int8)
    in_t = iron.tensor(np.ascontiguousarray(in_tiles.reshape(-1)), dtype=np.int8, device="npu")
    r_host = np.ascontiguousarray(tile_pack(a, 4, 16))
    r_t = iron.tensor(r_host, dtype=np.int8, device="npu")
    out_t = iron.zeros((in_tiles.shape[0] * M * Nt,), dtype=ml_dtypes.bfloat16, device="npu")
    return design, in_t, r_t, out_t, r_host


# CACHE=1 measures the DEVICE, not the harness. bricklib builds with use_cache=False as a
# correctness workaround (the JIT external-kernel cache keys on the shim, not on what it
# #includes, so a kernel edit does not invalidate it -- jit-external-kernel-cache-staleness).
# That workaround costs ~150 ms of recompilation PER CALL, which is ~800x the actual dispatch
# and buries every host-path delta this probe exists to measure. Enabling the cache is safe
# HERE and only here: nothing edits a kernel between calls, the design is built once up front.
if _os.environ.get("CACHE") == "1":
    _orig_jit = iron.jit

    def _cached_jit(fn=None, **kw):
        kw["use_cache"] = True
        return _orig_jit(fn, **kw) if fn is not None else _orig_jit(**kw)

    iron.jit = _cached_jit
    bricklib.iron.jit = _cached_jit

design, in_t, r_t, out_t, r_host = build()
if _os.environ.get("CACHE") == "1":
    iron.jit = _orig_jit
    bricklib.iron.jit = _orig_jit
design(in_t, r_t, out_t)
print(f"[rt] design built + warmed; {ITERS} iters per mode, shape {M}x{K}x{N}_g{G} Nt={Nt}",
      flush=True)

power, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        power.append((time.monotonic(), read_sensors()[0]["raw_input"]))
        time.sleep(0.02)


th = threading.Thread(target=sampler, daemon=True)
th.start()


def timeit(fn):
    per = []
    t_start = time.monotonic()
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        per.append((time.perf_counter() - t0) * 1e3)
    t_end = time.monotonic()
    p = [v for ts, v in power if t_start <= ts <= t_end]
    return per, (sum(p) / len(p) if p else float("nan"))


def mode_a():
    design(in_t, r_t, out_t)


def mode_b():
    design(in_t, r_t, out_t)
    _ = out_t.numpy()


def mode_c1():
    """B + the host DECISION only (argmax), no new device buffer."""
    design(in_t, r_t, out_t)
    dev = out_t.numpy()
    tok = int(np.argmax(dev.astype(np.float32)))
    r_host[tok % r_host.size] = np.int8((tok % 251) - 125)


def mode_c():
    """C1 + handing the next input back as a fresh device tensor.

    Split from C1 on purpose: a real decode writes into a buffer it already owns, so the
    allocation here is an artifact of the harness and belongs in its own line rather than
    silently inflating the round-trip figure.
    """
    design(in_t, r_t, out_t)
    dev = out_t.numpy()
    tok = int(np.argmax(dev.astype(np.float32)))
    r_host[tok % r_host.size] = np.int8((tok % 251) - 125)
    return iron.tensor(r_host, dtype=np.int8, device="npu")


res = {}
for name, fn in (("A dispatch only", mode_a), ("B +readback", mode_b),
                 ("C1 +host argmax", mode_c1), ("C +tensor handback", mode_c)):
    per, pw = timeit(fn)
    res[name] = (statistics.median(per), min(per), statistics.mean(per), pw)
    print(f"  {name:20s} median {res[name][0]:8.3f} ms   min {res[name][1]:8.3f}   "
          f"mean {res[name][2]:8.3f}   mean_pwr_raw {pw:.3f}", flush=True)

stop.set()
th.join(timeout=2)

a_med = res["A dispatch only"][0]
b_med = res["B +readback"][0]
c1_med = res["C1 +host argmax"][0]
c_med = res["C +tensor handback"][0]
print("\n==== per-token host round-trip decomposition (medians) ====")
print(f"  dispatch only              {a_med:8.3f} ms")
print(f"  readback     (B - A)       {b_med - a_med:8.3f} ms")
print(f"  host argmax  (C1 - B)      {c1_med - b_med:8.3f} ms")
print(f"  tensor handback (C - C1)   {c_med - c1_med:8.3f} ms   (harness artifact: a real")
print(f"                                        decode reuses its buffer)")
print(f"  ROUND-TRIP, real  (C1 - A) {c1_med - a_med:8.3f} ms")
print(f"  ROUND-TRIP, w/alloc (C - A){c_med - a_med:8.3f} ms")
print(f"\n  NOTE: the RATIO to this harness's {a_med:.0f} ms dispatch is NOT transferable -- the")
print(f"  streamed verify rail walks {N // Nt} sequential tile acquires, which is far more")
print(f"  dispatch cost than a production decode op. Quote the ABSOLUTE ms, not the %.")
import os  # noqa: E402
os._exit(0)
