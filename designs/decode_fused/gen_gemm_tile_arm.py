#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ONE GEMM at ONE tiling, as a fused full ELF -- the unit the tile sweep builds and times.

`sweep_gemm_tiles.py` enumerates candidate tilings for a prefill shape, filters them against the
kernel's real rules (`llm_decode_spec.gemm_tiling_rejection`), and runs this file once per survivor.
Each run emits the `meta.json` + `buffers/` layout `rust/npu-probes/src/bin/fused_elf_probe.rs`
already consumes, so timing an arm needs no new host code:

    fused_elf_probe <arm-dir> --warmup 20 --iters 200

The arm is a single `C[M,N] = A[M,K] @ B[K,N]` with A the activation tile (an input), B the weight
(resident scratch, written once) and C the output -- the prefill projection stripped of everything
that is not the GEMM, because the sweep is comparing tilings of the same arithmetic and any norm or
activation around it is common-mode noise.

=== The name is the cache key, and a collision measures the wrong binary ===

IRON keys its build artifacts by FILENAME and mtime (`iron/common/compilation/base.py:
is_available_in_filesystem` compares `Path(self.filename).exists()` and dependency mtimes -- not
content). Worse, `GEMM.name` -- which is the per-operator MLIR artifact's filename -- is built from
the dataclass fields with `repr=True`, and `emulate_bf16_mmul_with_bfp16`, `prio_accuracy` and
`round_conv_even` are all declared `repr=False`. So two arms differing only in those flags produce
the same `.mlir` filename, and the second silently links the first one's design.

Two defences, both used here: every knob goes into the `OperatorSequence` name (which IS the fused
ELF's filename), and the sweep gives every arm its own work directory. `sweep_gemm_tiles.py` then
md5s every ELF and fails if two arms collide, because "the ELFs are identical" is the only symptom
this failure has.

Run inside the fork IRON env. AIE_DEVICE=npu2 pins the target so the build never opens /dev/accel.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import gemm_l1_bytes, gemm_memtile_bytes, gemm_tiling_rejection  # noqa: E402

import newstack_compat  # noqa: F401,E402 -- MUST precede iron imports
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence, load_elf  # noqa: E402
from iron.operators.gemm.op import GEMM  # noqa: E402

BF16 = ml_dtypes.bfloat16


def bf16(a):
    return np.asarray(a).astype(BF16)


def arm_name(M, K, N, tile_m, tile_k, tile_n, cols, b_col_maj, emulate, prio_acc, round_even,
             dtype_in="bf16", dtype_out="bf16"):
    """Every knob that changes the graph, in the sequence name. See the module docstring."""
    return (f"gemmtile_m{M}k{K}n{N}_tm{tile_m}tk{tile_k}tn{tile_n}_c{cols}"
            f"_bcm{int(b_col_maj)}_bfp{int(emulate)}_acc{int(prio_acc)}_re{int(round_even)}"
            f"_{dtype_in}2{dtype_out}")


def arm_dirname(tile_m, tile_k, tile_n, cols):
    return f"tm{tile_m}_tk{tile_k}_tn{tile_n}_c{cols}"


def build_arm(out_dir, M, K, N, tile_m, tile_k, tile_n, cols, *, b_col_maj=True, emulate=True,
              prio_accuracy=False, round_conv_even=True, label=None, seed=17):
    """Build one arm into `out_dir`. Returns the meta dict it wrote."""
    # K007: the shape is picked HERE, so the rules are checked HERE -- and against the same
    # function the sweep filters on, so a candidate can never be "legal to the sweep, illegal to
    # the build".
    rej = gemm_tiling_rejection(M, K, N, tile_m, tile_k, tile_n, cols, bfp16=emulate,
                                prio_accuracy=prio_accuracy)
    if rej is not None:
        raise ValueError(f"[{rej.code}] {rej.detail}")

    if os.environ.get("AIE_DEVICE"):
        import aie.utils as _aie_utils
        from aie.iron.device import from_name as _from_name
        _aie_utils.set_current_device(_from_name(os.environ["AIE_DEVICE"], n_cols=None))

    ctx = AIEContext()
    gemm = GEMM(M=M, K=K, N=N, tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
                num_aie_columns=cols, b_col_maj=b_col_maj,
                emulate_bf16_mmul_with_bfp16=emulate, prio_accuracy=prio_accuracy,
                round_conv_even=round_conv_even, context=ctx)
    name = arm_name(M, K, N, tile_m, tile_k, tile_n, cols, b_col_maj, emulate, prio_accuracy,
                    round_conv_even)
    fused = OperatorSequence(name, [(gemm, "A", "B", "C")], input_args=["A"], output_args=["C"],
                             context=ctx)
    fused.compile()

    rng = np.random.default_rng(seed)
    A = bf16(rng.standard_normal((M, K)).astype(np.float32) * (K ** -0.5))
    # B is stored the way the prefill weights are stored: [Nout, K] read `b_col_maj`, which is
    # decode's GEMV layout and the reason prefill can share decode's weight arena byte for byte.
    B = bf16(rng.standard_normal((N, K) if b_col_maj else (K, N)).astype(np.float32))
    C = bf16(np.asarray(A, np.float32) @ (np.asarray(B, np.float32).T if b_col_maj
                                          else np.asarray(B, np.float32)))

    os.makedirs(os.path.join(out_dir, "buffers"), exist_ok=True)
    bdir = os.path.join(out_dir, "buffers")
    for nm, arr in (("A", A), ("B", B), ("C", C)):
        open(os.path.join(bdir, f"{nm}.bin"), "wb").write(np.asarray(arr, BF16).reshape(-1).tobytes())

    elf = load_elf(fused).view(np.uint8).tobytes()
    elf_path = os.path.join(out_dir, "gemm_arm.elf")
    open(elf_path, "wb").write(elf)
    in_sz, out_sz, scr = fused.buffer_sizes
    lay = {n: fused.get_layout_for_buffer(n) for n in ("A", "B", "C")}

    meta = {
        "elf": "gemm_arm.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz), "output_size": int(out_sz), "scratch_size": int(scr),
        "layout": {n: {"type": v[0], "offset": int(v[1]), "len": int(v[2])}
                   for n, v in lay.items()},
        "inputs": ["A"], "weights": ["B"], "output": "C",
        "sequence_name": fused.name,
        "elf_md5": hashlib.md5(elf).hexdigest(),
        "elf_bytes": len(elf),
        "label": label,
        "dims": {"M": M, "K": K, "N": N,
                 "tile": [tile_m, tile_k, tile_n], "cols": cols,
                 "b_col_maj": bool(b_col_maj), "emulate": bool(emulate),
                 "prio_accuracy": bool(prio_accuracy), "round_conv_even": bool(round_conv_even),
                 "dtype": "bf16>bf16"},
        # DDR bytes the arm moves, so a device timing converts to GB/s without the caller
        # re-deriving it. One pass per operand: a FLOOR, since it ignores the broadcast of A to
        # every column and any MemTile reuse, in both directions.
        "bytes": {"a": M * K * 2, "b": K * N * 2, "c": M * N * 2,
                  "total": (M * K + K * N + M * N) * 2},
        "macs": int(M) * int(K) * int(N),
        # The two capacity budgets nothing in the toolchain checks for GEMM, recorded so a sweep
        # result can be read against occupancy rather than only against the clock.
        "budget": {
            "l1_bytes": gemm_l1_bytes(tile_m, tile_k, tile_n, prio_accuracy=prio_accuracy),
            "l1_limit": 65536,
            "memtile_bytes": gemm_memtile_bytes(tile_m, tile_k, tile_n, cols),
            "memtile_limit": 0x80000,
            "k_tiles": K // tile_k,
            "c_narrowings": 0 if prio_accuracy else K // tile_k,
        },
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--tile-m", type=int, required=True)
    ap.add_argument("--tile-k", type=int, required=True)
    ap.add_argument("--tile-n", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-b-col-maj", action="store_true",
                    help="B stored [K, N] (the ctx GEMM, which reads the V cache as [K=S, N=HD])")
    ap.add_argument("--no-emulate", action="store_true",
                    help="turn OFF GEMM's default bfp16 emulation; changes (r,s,t) to (4,8,8)")
    ap.add_argument("--prio-accuracy", action="store_true",
                    help="f32 L1 accumulator: one bf16 narrowing per GEMM instead of K/tile_k")
    ap.add_argument("--floor-rounding", action="store_true",
                    help="round bf16 conversions FLOOR instead of nearest-even (mm.cc's default)")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    meta = build_arm(a.out, a.M, a.K, a.N, a.tile_m, a.tile_k, a.tile_n, a.cols,
                     b_col_maj=not a.no_b_col_maj, emulate=not a.no_emulate,
                     prio_accuracy=a.prio_accuracy, round_conv_even=not a.floor_rounding,
                     label=a.label, seed=a.seed)
    b = meta["budget"]
    print(f"[ok] {meta['sequence_name']}  elf={meta['elf_bytes']}B md5={meta['elf_md5']}  "
          f"L1={b['l1_bytes']}B ({100 * b['l1_bytes'] / b['l1_limit']:.0f}%) "
          f"MemTile={b['memtile_bytes']}B ({100 * b['memtile_bytes'] / b['memtile_limit']:.0f}%) "
          f"k_tiles={b['k_tiles']} -> {a.out}")


if __name__ == "__main__":
    main()
