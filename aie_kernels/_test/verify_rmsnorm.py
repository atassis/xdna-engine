#!/usr/bin/env python3
"""Device rel-L2 verify for the rmsnorm brick (brick-wave1-device-verify anchor).

Proves the generic single-kernel device-verify rail: a thin pure-buffer verify-shim
(cols/eps baked via -D) #include's the brick .cc; an @iron.jit design streams the
golden's exact inputs through the kernel on aie2p and reads the output back; rel-L2 vs
the numpy golden gates it. Read-twice (run-twice) self-check guards the CLFLUSH read race.

Gate: rel_l2 <= 3e-2 (brick threshold). Run under npu_lock.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).parent
BRICK = HERE.parent / "rmsnorm"
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)

M, COLS = 8, 768
GATE = 3e-2

# --- generate the pure-buffer verify shim (cols/eps baked via -D) ---
SHIM = GEN / "rmsnorm_shim.cc"
SHIM.write_text(f'''// AUTO-GENERATED verify shim for the rmsnorm brick. Pure-buffer entry so IRON's
// ExternalFunction needs no scalar-ABI; cols/eps baked via -D at compile time.
// Resident affine params gamma|beta packed into ONE const buffer (gb) so the design
// stays within a core tile's 2-input DMA channels (x stream + gb const).
#include <stdint.h>
#ifndef VCOLS
#define VCOLS {COLS}
#endif
#ifndef VEPS
#define VEPS 1e-6f
#endif
#include "{BRICK / 'rmsnorm.cc'}"
extern "C" void rmsnorm_verify(float *x, float *gb, float *out) {{
  rmsnorm_f32<16>(x, gb, gb + VCOLS, out, VCOLS, VEPS);
}}
''')

# --- import the brick's numpy golden ---
spec = importlib.util.spec_from_file_location("rmsnorm_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


@iron.jit
def rmsnorm_design(
    inp: In,
    gb: In,
    out: Out,
    *,
    m: CompileTime[int] = M,
    cols: CompileTime[int] = COLS,
):
    row_ty = np.ndarray[(cols,), np.dtype[np.float32]]
    gb_ty = np.ndarray[(2 * cols,), np.dtype[np.float32]]
    in_ty = np.ndarray[(m * cols,), np.dtype[np.float32]]

    kern = ExternalFunction(
        "rmsnorm_verify",
        source_file=str(SHIM),
        arg_types=[row_ty, gb_ty, row_ty],
        compile_flags=[f"-DVCOLS={cols}", "-DVEPS=1e-6f"],
    )

    in_fifo = ObjectFifo(row_ty, name="in_fifo")
    gb_fifo = ObjectFifo(gb_ty, name="gb_fifo")
    out_fifo = ObjectFifo(row_ty, name="out_fifo")

    def core(inf, gbf, of, kern):
        egb = gbf.acquire(1)
        for _ in range_(m):
            ei = inf.acquire(1)
            eo = of.acquire(1)
            kern(ei, egb, eo)
            of.release(1)
            inf.release(1)
        gbf.release(1)

    worker = Worker(
        core,
        fn_args=[in_fifo.cons(), gb_fifo.cons(), out_fifo.prod(), kern],
    )

    in_tap = TensorTiler2D.group_tiler((m, cols), (1, cols), (m, 1))[0]
    gb_tap = TensorTiler2D.group_tiler((1, 2 * cols), (1, 2 * cols), (1, 1))[0]

    def sequence(a, g, c, in_h, gb_h, out_h):
        in_h.fill(a, in_tap)
        gb_h.fill(g, gb_tap)
        out_h.drain(c, in_tap, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, gb_ty, in_ty, in_fifo.prod(), gb_fifo.prod(), out_fifo.cons()],
    )

    return Program(
        iron.get_current_device(), rt, workers=[worker]
    ).resolve_program()


def run_once(x, gamma, beta):
    gb = np.concatenate([gamma, beta]).astype(np.float32)
    x_t = iron.tensor(x.reshape(-1).copy(), dtype=np.float32, device="npu")
    gb_t = iron.tensor(gb.copy(), dtype=np.float32, device="npu")
    out_t = iron.zeros((M * COLS,), dtype=np.float32, device="npu")
    rmsnorm_design(x_t, gb_t, out_t, m=M, cols=COLS)
    return out_t.numpy().reshape(M, COLS).copy()


def main():
    # exact golden inputs (golden.py __main__ seed 0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((M, COLS)).astype(np.float32)
    gamma = (rng.standard_normal((COLS,)).astype(np.float32) * 0.1 + 1.0)
    beta = (rng.standard_normal((COLS,)).astype(np.float32) * 0.01)

    expected = golden.rmsnorm_ref(x, gamma, beta, eps=1e-6)

    dev1 = run_once(x, gamma, beta)
    dev2 = run_once(x, gamma, beta)  # read/run-twice self-check (CLFLUSH race guard)

    nz1 = float(np.abs(dev1).sum())
    determ = float(np.linalg.norm((dev1 - dev2).ravel()))
    rl2 = golden.rel_l2(dev1, expected)

    print(f"[rmsnorm] device nonzero-sum={nz1:.3e}  run-to-run L2={determ:.3e}  rel_l2={rl2:.3e}  gate={GATE:.1e}")
    print(f"  dev1[0,:4]={dev1[0,:4]}  exp[0,:4]={expected[0,:4]}")
    if nz1 == 0.0:
        print("FAIL: device output all-zero (cold-read race or dead kernel)")
        return 2
    if determ > 1e-6:
        print(f"WARN: run-to-run not deterministic (L2={determ:.3e}) -- possible read race")
    if rl2 <= GATE:
        print(f"PASS rmsnorm rel_l2={rl2:.3e} <= {GATE:.1e}")
        return 0
    print(f"FAIL rmsnorm rel_l2={rl2:.3e} > {GATE:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
