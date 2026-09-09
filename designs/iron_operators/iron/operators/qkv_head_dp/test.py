#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the data-parallel QKV head: does aiecc PLACE one `aie.device` running every
stage on N cores' own head slice, and does each core's program fit the 16 KB region?

Placement is the question that shaped the previous attempt at this group: an AIE2P compute tile has
2 input and 2 output DMA channels, and fuse/qkv-head bought its placement by collapsing SIX input
streams onto one D-wide channel, which forced its matvec tile from 4 rows to 1. So this gate prints
`tile_size_input` next to the result -- a build that succeeds at tsi=1 is not the same result as
one that succeeds at 4, and nothing in aiecc's output distinguishes them.

Placement alone is NOT the gate. This design's first device run hung with
ERT_CMD_STATE_TIMEOUT off a runtime-sequence deadlock that no build-only check can see, so
`--device` additionally runs the operator against its CPU reference on /dev/accel. Run that before
spending a full-graph build on it.

`tile_size_input` is sys.argv[1] (default 4); add `--device` for the on-device correctness run.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from iron.common import AIEContext
from iron.operators.qkv_head_dp.op import QKVHeadDataParallel

PROGRAM_MEM_BYTES = 0x4000  # AIETargetModel.h getProgramMemorySize() for AIE2/AIE2P.


def main():
    D, HD, Hq, Hkv, N = 1024, 128, 16, 8, 8      # Qwen3-0.6B decode shapes
    argv = [a for a in sys.argv[1:] if a != "--device"]
    on_device = "--device" in sys.argv
    tsi = int(argv[0]) if argv else 4
    # The device arm drops the kv_off ScratchpadParameter -- run_test has no way to write one --
    # and appends at a static offset 0. That leaves BD patching ungated HERE; it is gated by the
    # full-graph token-parity run, and it is StridedCopy's own long-standing path. Placement is
    # checked WITH the parameter, since that is the configuration that ships.
    S = 512 if on_device else 2048
    kvpar = None if on_device else "kv_off"

    build_dir = Path(__file__).resolve().parents[4] / "build" / f"qkv_head_dp_tsi{tsi}_S{S}"
    op = QKVHeadDataParallel(D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=N,
                             tile_size_input=tsi, kv_offset_parameter=kvpar,
                             context=AIEContext(build_dir=build_dir))
    print(f"operator: {op.name}")
    op.compile()   # raises on any failed command, aiecc placement included

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    text = mlir_path.read_text()
    n_devices = len(re.findall(r"\baie\.device\b", text))
    n_cores = len(re.findall(r"\baie\.core\(", text))
    assert n_devices == 1, f"expected exactly one aie.device, found {n_devices}"
    assert n_cores == N, f"expected {N} aie.core ops (one per column), found {n_cores}"

    build_subdir = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    tiles = sorted(tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
                   for d in build_subdir.glob("elfs_main_core_*") if d.is_dir())
    print(f"placed compute tiles (col,row): {tiles}")

    max_pct = 0.0
    if shutil.which("llvm-size"):
        for col, row in tiles:
            elf = build_subdir / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            out = subprocess.run(["llvm-size", "-A", str(elf)], capture_output=True, text=True)
            tb = next((int(ln.split()[1]) for ln in out.stdout.splitlines()
                       if ln.startswith(".text")), None)
            if tb is not None:
                max_pct = max(max_pct, 100.0 * tb / PROGRAM_MEM_BYTES)
                print(f"  ({col},{row}): .text={tb} B ({100.0*tb/PROGRAM_MEM_BYTES:.1f}%)")
    else:
        print("llvm-size not on PATH -- skipping the per-core .text measurement")

    print(f"PLACES: one aie.device, {n_cores} cores over {len(tiles)} tiles, "
          f"tile_size_input={tsi}, max .text {max_pct:.1f}% of {PROGRAM_MEM_BYTES} B.")
    if on_device:
        run_on_device(op, D, HD, Hq, Hkv, S)
    print("PASS")


def run_on_device(op, D, HD, Hq, Hkv, S):
    import torch
    from iron.common.test_utils import run_test
    from iron.operators.qkv_head_dp.reference import reference

    torch.manual_seed(0)
    QD, KVD = Hq * HD, Hkv * HD

    def rnd(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16)

    cur, n_in = rnd(D), rnd(D)
    wqkv = rnd(QD + 2 * KVD, D) * 0.05
    n_qn, n_kn, ang = rnd(HD), rnd(HD), rnd(HD)
    golden = reference(cur, n_in, wqkv.reshape(-1), n_qn, n_kn, ang, D, HD, Hq, Hkv, op.epsilon)

    # k and v are APPENDED to the caches at [head][pos][HD]; at kv_off=0 that is row 0 of each
    # head, so the expected cache is zeros with one HD-wide row written per head.
    kc = torch.zeros(Hkv * S * HD, dtype=torch.bfloat16)
    vc = torch.zeros(Hkv * S * HD, dtype=torch.bfloat16)
    for h in range(Hkv):
        kc[h * S * HD: h * S * HD + HD] = golden[QD + h * HD: QD + (h + 1) * HD]
        vc[h * S * HD: h * S * HD + HD] = golden[QD + KVD + h * HD: QD + KVD + (h + 1) * HD]

    # kc/vc are `inout`: run_test takes their INITIAL contents from the input dict (zeros, the
    # cache before this token) and compares the SAME buffer against the output dict afterwards.
    errors, latency_us, _ = run_test(
        op,
        {"cur": cur, "n_in": n_in, "wqkv": wqkv.reshape(-1),
         "n_qn": n_qn, "n_kn": n_kn, "ang": ang,
         "kc": torch.zeros_like(kc), "vc": torch.zeros_like(vc)},
        {"q": golden[:QD], "kc": kc, "vc": vc},
        rel_tol=0.05, abs_tol=0.5,
    )
    # rel-L2 alongside the elementwise check: this is a NOTE, not the gate (error-metrics-are-
    # notes-not-gates); the gate is the graph's own token parity.
    print(f"device latency: {latency_us:.1f} us")
    assert not errors, f"device mismatch: {errors}"


if __name__ == "__main__":
    sys.exit(main())
