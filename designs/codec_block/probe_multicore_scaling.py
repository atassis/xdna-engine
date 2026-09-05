#!/usr/bin/env python3
"""Does adding WORKERS add performance? The falsification test for "compute-bound".

Established: at constant streamed weight bytes, device time scales with the window t, i.e. with the
MAC count -- so a dispatch is work-limited, not delivery-limited. And it runs at ~0.185 MAC/cycle,
~1.2% of one core's f32 vector peak, on exactly ONE core (every bricklib design is
`workers=[worker]`; the built MLIR has one `aie.core`).

I previously argued from that 1.2% that parallelising would just give idle cores. THAT WAS WRONG,
and this probe exists because of it: per-core inefficiency does not prevent parallel throughput. The
output channels are independent, so W workers each doing n_tiles/W of them should take ~1/W the
time, whatever each core's own efficiency is.

  time ~ 1/W   -> compute-bound confirmed, and core count is a real lever
  time flat    -> the conclusion is wrong; something shared (DMA, shim, memtile) is the limit

Each worker gets its own in/resident/out objectFIFO and a disjoint contiguous slice of the tiles,
taken from group_tiler's per-group taps rather than by slicing on the host.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aie.utils.hostruntime.xrtruntime import hostruntime as hr
_dev = {"ms": 0.0}
_orig = hr.CachedXRTRuntime.run
def _timed(self, *a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(self, *a, **k)
    finally:
        _dev["ms"] += (time.perf_counter() - t0) * 1e3
hr.CachedXRTRuntime.run = _timed

import window_driver as wd
import bricklib
import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

C_IN, K, T, C_OUT = 128, 7, 64, 384
ROW = C_IN * K + 1
RES = C_IN * T
rng = np.random.default_rng(53)

sym = "mc_conv"
shim = bricklib.GEN / "mc_conv_shim.cc"
shim.write_text(
    f"#include <stdint.h>\n"
    f'#include "{wd.CONV_CC}"\n'
    f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
    f"  route_b_bricks::conv_1d_causal_core_vec<16>(resident, tile, tile[{C_IN*K}],\n"
    f"      out, {C_IN}, {K}, {T}, 1);\n}}\n")

def build(n_workers):
    tpw = C_OUT // n_workers                      # tiles per worker
    # annotations are load-bearing: iron.jit reads them to build the runtime signature,
    # and without them the generator is called with no args (bricklib:208 does the same).
    def design(inp: In, cst: In, out: Out):
        in_row  = bricklib._npty((ROW,), np.float32)
        out_row = bricklib._npty((T,), np.float32)
        cst_ty  = bricklib._npty((RES,), np.float32)
        in_full = bricklib._npty((C_OUT * ROW,), np.float32)
        out_full= bricklib._npty((C_OUT * T,), np.float32)
        kern = ExternalFunction(sym, source_file=str(shim),
                                arg_types=[in_row, cst_ty, out_row], compile_flags=None)
        infs = [ObjectFifo(in_row,  name=f"inf{w}") for w in range(n_workers)]
        cfs  = [ObjectFifo(cst_ty,  name=f"cf{w}", depth=1) for w in range(n_workers)]
        ofs  = [ObjectFifo(out_row, name=f"of{w}") for w in range(n_workers)]

        def core(inf, cf, of, kern):
            ec = cf.acquire(1)                    # resident acquired once for the whole slice
            for _ in range_(tpw):
                ei = inf.acquire(1); eo = of.acquire(1)
                kern(ei, ec, eo)
                of.release(1); inf.release(1)
            cf.release(1)

        workers = [Worker(core, fn_args=[infs[w].cons(), cfs[w].cons(), ofs[w].prod(), kern])
                   for w in range(n_workers)]
        # one tap PER GROUP of tpw rows -> tap w is worker w's disjoint slice
        in_taps  = TensorTiler2D.group_tiler((C_OUT, ROW), (1, ROW), (tpw, 1))
        out_taps = TensorTiler2D.group_tiler((C_OUT, T),   (1, T),   (tpw, 1))
        cst_tap  = TensorTiler2D.group_tiler((1, RES), (1, RES), (1, 1))[0]

        def sequence(a, c, o, *handles):
            n = n_workers
            in_hs, cst_hs, out_hs = handles[:n], handles[n:2*n], handles[2*n:]
            for w in range(n):
                in_hs[w].fill(a, in_taps[w])
                cst_hs[w].fill(c, cst_tap)
            for w in range(n):
                out_hs[w].drain(o, out_taps[w], wait=True)

        rt = Runtime(sequence,
                     [in_full, cst_ty, out_full]
                     + [f.prod() for f in infs] + [f.prod() for f in cfs]
                     + [f.cons() for f in ofs])
        return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
    design.__name__ = design.__qualname__ = f"mc_conv_w{n_workers}"
    return iron.jit(design, use_cache=False)

print(f"c_in={C_IN} k={K} T={T} c_out={C_OUT}  (work split across workers by output channel)",
      flush=True)
print(f"{'workers':>8} {'tiles/w':>8} {'device_ms':>10} {'speedup':>8} {'MAC/cyc':>8}", flush=True)
tiles = (rng.standard_normal((C_OUT, ROW)) * 0.02).astype(np.float32)
win = rng.standard_normal((C_IN, T)).astype(np.float32)
base = None
for W in (16,):
    try:
        d = build(W)
        in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
        c_t = bricklib.iron.tensor(win.reshape(-1), dtype=np.float32, device="npu")
        out_t = bricklib.iron.zeros((C_OUT * T,), dtype=np.float32, device="npu")
        d(in_t, c_t, out_t); _dev["ms"] = 0.0
        d(in_t, c_t, out_t)
        got = out_t.numpy().copy()
        ms = _dev["ms"]; base = base or ms
        macs = C_OUT * C_IN * K * T
        print(f"{W:8d} {C_OUT//W:8d} {ms:10.1f} {base/ms:8.2f}x {macs/(ms/1e3)/1.8e9:8.4f}"
              f"   nz={np.abs(got).sum():.4e}", flush=True)
    except Exception as e:
        print(f"{W:8d} FAILED:\n{str(e)[:1400]}", flush=True)
