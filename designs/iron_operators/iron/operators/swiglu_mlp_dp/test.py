#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the data-parallel SwiGLU MLP block: does aiecc PLACE and BUILD one
`aie.device` running every stage on N cores' own 1/N slice? Deliberately does NOT touch
/dev/accel -- CPU-only aiecc. `N` is passed as sys.argv[1] (default 8); FUSE_O=1 in the
environment builds the op_o-folded arm instead (see design.py's FUSE_O module docstring).
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from iron.common import AIEContext
from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel
from iron.operators.swiglu_mlp_dp.reference import (
    generate_golden_reference,
    generate_golden_reference_fused_o,
)

PROGRAM_MEM_BYTES = 0x4000  # AIETargetModel.h getProgramMemorySize() for AIE2/AIE2P.


def main():
    D, FF, QD = 1024, 3072, 2048  # Qwen3-0.6B decode shapes (d_model, ffn, attn context width)
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    fuse_o = os.environ.get("FUSE_O", "0") == "1"
    # N<=8: one core per column (n_aie_rows=1, plain ObjectFifos -- placement PROVED this at N=8).
    # N>8: n_aie_cols=8, n_aie_rows=N/8 (MemTile split/join -- see design.py's module docstring).
    n_aie_cols = min(N, 8)
    n_aie_rows = N // n_aie_cols
    assert n_aie_cols * n_aie_rows == N
    if fuse_o and n_aie_rows != 1:
        raise NotImplementedError("FUSE_O is only derived for n_aie_rows=1 (N<=8) -- see design.py")

    build_dir = Path(__file__).resolve().parents[4] / "build" / f"swiglu_mlp_dp_n{N}{'_fo' if fuse_o else ''}"
    ctx = AIEContext(build_dir=build_dir)
    op = SwiGLUMLPDataParallel(
        D=D, FF=FF, num_aie_columns=n_aie_cols, num_aie_rows=n_aie_rows,
        QD=QD if fuse_o else None, fuse_o=fuse_o, context=ctx,
    )
    print(f"operator: {op.name}")
    print(f"build dir: {build_dir}")

    op.compile()  # raises RuntimeError on any failed compilation command (incl. aiecc placement)

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    mlir_text = mlir_path.read_text()
    n_devices = len(re.findall(r"\baie\.device\b", mlir_text))
    n_cores = len(re.findall(r"\baie\.core\(", mlir_text))
    print(f"aie.device count: {n_devices}")
    print(f"aie.core count:   {n_cores}")
    assert n_devices == 1, f"expected exactly one aie.device, found {n_devices}"
    assert n_cores == N, f"expected {N} aie.core ops (one per column), found {n_cores}"

    build_subdir = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    placed_tiles = sorted(
        tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
        for d in build_subdir.glob("elfs_main_core_*") if d.is_dir()
    )
    cols_used = sorted({c for c, r in placed_tiles})
    rows_used = sorted({r for c, r in placed_tiles})
    print(f"placed compute tiles (col,row): {placed_tiles}")
    print(f"columns used: {cols_used}  rows used: {rows_used}")

    llvm_size = shutil.which("llvm-size")
    max_pct = 0.0
    if llvm_size:
        print(f"program memory per core: {PROGRAM_MEM_BYTES} B (0x{PROGRAM_MEM_BYTES:x})")
        for col, row in placed_tiles:
            elf = build_subdir / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            out = subprocess.run([llvm_size, "-A", str(elf)], capture_output=True, text=True)
            text_bytes = next(
                (int(line.split()[1]) for line in out.stdout.splitlines() if line.startswith(".text")),
                None,
            )
            if text_bytes is not None:
                pct = 100.0 * text_bytes / PROGRAM_MEM_BYTES
                max_pct = max(max_pct, pct)
                print(f"  ({col},{row}): .text={text_bytes} B ({pct:.1f}% of {PROGRAM_MEM_BYTES})")
    else:
        print("llvm-size not found on PATH -- skipping per-core .text measurement")

    xclbin = Path(op.xclbin_artifact.filename)
    insts = Path(op.insts_artifact.filename)
    print(f"xclbin: {xclbin} ({xclbin.stat().st_size} bytes)")
    print(f"insts:  {insts} ({insts.stat().st_size} bytes)")

    if fuse_o:
        golden = generate_golden_reference_fused_o(D, FF, QD, wo_rows_padded=op._wo_rows_padded)
    else:
        golden = generate_golden_reference(D, FF)
    print(f"host reference nxt[:8] = {golden['nxt'][:8]}")

    print(f"PASS: one aie.device, aiecc placed and built it ({n_cores} aie.core across "
          f"{len(placed_tiles)} tiles, columns {cols_used}, rows {rows_used}, "
          f"max .text {max_pct:.1f}%{', fuse_o=1' if fuse_o else ''}).")


if __name__ == "__main__":
    sys.exit(main())
