#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FE1 Task-0 spike generator: a minimal 2-GEMV *fused full ELF* via IRON's FusedMLIROperator.

This is the artifact side of the dispatch-shim crux ([[decode-fused-elf-handoff]] /
2026-06-15-iron-fused-elf-whisper-porting.md step 1). It produces a fused ELF that chains
    y0 = W0 @ x        (GEMV, M=K=128)
    y1 = W1 @ y0       (GEMV, M=K=128)
into ONE program with the 3-arena (input/output/scratch) ABI, plus pre-assembled arena blobs and a
numpy golden, so the Rust `fused_elf_probe` can load the ELF through our NEW `shim_run_elf` path
(xrt::elf -> hw_context(device,elf) -> ext::kernel("main:sequence")) and prove it dispatches
correctly on device — WITHOUT any device access here (compile + golden are host-only).

Run inside the IRON env:
    cd ~/repositories/ns/atassis/xdna-engine-workspace/amd/IRON && source ironenv/bin/activate
    python <this>  --out <worktree>/artifacts/fused_spike

Outputs (in --out): spike2gemv.elf, input_arena.bin, scratch_arena.bin, golden_output.bin, meta.json
"""
import argparse
import json
import os

import numpy as np
import ml_dtypes

from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator, load_elf
from iron.operators.gemv.op import GEMV

BF16 = ml_dtypes.bfloat16
M = 128
K = 128  # GEMV requires K % kernel_vector_size(64) == 0; the smallest known-good config is 128.
GEMV_KW = dict(num_aie_columns=1, tile_size_input=32, tile_size_output=128)


def bf16(a):
    return np.asarray(a).astype(BF16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for the ELF + arena blobs + meta")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ctx = AIEContext()
    g0 = GEMV(M=M, K=K, context=ctx, **GEMV_KW)
    g1 = GEMV(M=M, K=K, context=ctx, **GEMV_KW)

    runlist = [
        (g0, "W0", "x", "y0"),   # arg spec order: (matrix[M,K], vector[K], out[M])
        (g1, "W1", "y0", "y1"),
    ]
    fused = FusedMLIROperator(
        "spike2gemv",
        runlist,
        input_args=["x"],
        output_args=["y1"],
        context=ctx,
    )
    fused.compile()

    elf_data = load_elf(fused)  # np.uint32 view of the full ELF
    elf_bytes = elf_data.view(np.uint8).tobytes()
    in_sz, out_sz, scratch_sz = fused.buffer_sizes

    def layout(name):
        buf_type, off, length = fused.get_layout_for_buffer(name)
        return buf_type, int(off), int(length)

    lay = {n: layout(n) for n in ("x", "y1", "W0", "W1", "y0")}
    print("buffer_sizes (in,out,scratch) =", fused.buffer_sizes)
    for n, v in lay.items():
        print(f"  {n}: type={v[0]} off={v[1]} len={v[2]}")

    # --- random bf16 inputs (what the hardware actually sees) ---
    rng = np.random.default_rng(args.seed)
    x = bf16(rng.standard_normal(K) * 0.5)
    W0 = bf16(rng.standard_normal((M, K)) * 0.1)
    W1 = bf16(rng.standard_normal((M, K)) * 0.1)

    # --- assemble arenas (bf16, byte-addressed at the layout offsets) ---
    input_arena = np.zeros(in_sz // 2, dtype=BF16)
    scratch_arena = np.zeros(scratch_sz // 2, dtype=BF16)

    def place(arena, name, vals):
        _, off, length = lay[name]
        flat = np.asarray(vals, dtype=BF16).reshape(-1)
        assert flat.nbytes == length, f"{name}: {flat.nbytes} != {length}"
        arena[off // 2 : off // 2 + flat.size] = flat

    place(input_arena, "x", x)
    place(scratch_arena, "W0", W0.reshape(-1))
    place(scratch_arena, "W1", W1.reshape(-1))

    # --- numpy golden, matching hw bf16 dataflow (round y0 to bf16 before gemv1) ---
    y0 = bf16(W0.astype(np.float32) @ x.astype(np.float32))
    y1 = bf16(W1.astype(np.float32) @ y0.astype(np.float32))
    golden_out = np.zeros(out_sz // 2, dtype=BF16)
    _, y1_off, y1_len = lay["y1"]
    golden_out[y1_off // 2 : y1_off // 2 + y1.size] = y1

    with open(os.path.join(args.out, "spike2gemv.elf"), "wb") as f:
        f.write(elf_bytes)
    # Pre-assembled arenas (for the simple blob-based probe path).
    with open(os.path.join(args.out, "input_arena.bin"), "wb") as f:
        f.write(input_arena.tobytes())
    with open(os.path.join(args.out, "scratch_arena.bin"), "wb") as f:
        f.write(scratch_arena.tobytes())
    with open(os.path.join(args.out, "golden_output.bin"), "wb") as f:
        f.write(golden_out.tobytes())
    # Raw per-buffer values (for the layout-driven probe: place each by NAME via meta offsets).
    bufs = os.path.join(args.out, "buffers")
    os.makedirs(bufs, exist_ok=True)
    for name, vals in (("x", x), ("W0", W0.reshape(-1)), ("W1", W1.reshape(-1)), ("y1", y1)):
        with open(os.path.join(bufs, f"{name}.bin"), "wb") as f:
            f.write(np.asarray(vals, dtype=BF16).tobytes())

    meta = {
        "elf": "spike2gemv.elf",
        "kernel_name": "main:sequence",
        "input_size": int(in_sz),
        "output_size": int(out_sz),
        "scratch_size": int(scratch_sz),
        "elf_nbytes": len(elf_bytes),
        "layout": {n: {"type": v[0], "offset": v[1], "len": v[2]} for n, v in lay.items()},
        "inputs": ["x"],
        "weights": ["W0", "W1"],
        "output": "y1",
        "dims": {"M": M, "K": K},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote ELF ({len(elf_bytes)} bytes) + arenas + golden + meta.json to {args.out}")


if __name__ == "__main__":
    main()
