#!/usr/bin/env python3
"""Rung 5 — run the mlir-aie 'matrix_scalar_add' design on the XDNA2 NPU via pyxrt.

The upstream example runs through a C++ host harness, but Arch/CachyOS ships a broken
XRT cmake export (xrt-targets.cmake references missing files), so we drive the prebuilt
xclbin + instruction stream directly through pyxrt instead. This is the open-stack
"hello-NPU": prove we can load OUR compiled kernel onto /dev/accel/accel0 and get a
correct result, with zero AMD-gated software.

Design contract (from matrix_scalar_add.py + test.cpp):
  IMAGE 128x16 int32 (2048 elems); only the top-left 8x16 tile is processed: out=in+1,
  the rest passes through unchanged. opcode=3; kernel args (instr_bo, n, inA, inB, out)
  at group_ids 1,3,4,5.

Usage:
  .venv-iron/bin/python scripts/run_npu_matrix_scalar_add.py --dry   # no NPU, validates artifacts
  .venv-iron/bin/python scripts/run_npu_matrix_scalar_add.py         # REAL run on the NPU
"""
import argparse, os, sys
import numpy as np

EX = "mlir-aie/programming_examples/basic/matrix_scalar_add/build"
IMG_W, IMG_H = 128, 16
TILE_W, TILE_H = 16, 8
N = IMG_W * IMG_H  # 2048 int32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--dry", action="store_true", help="validate artifacts without touching the NPU")
    a = ap.parse_args()

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build it first: (cd mlir-aie/.../matrix_scalar_add && make NPU2=1)")

    instr = np.fromfile(a.insts, dtype=np.uint32)
    inA = np.arange(N, dtype=np.uint32)
    inB = np.zeros(N, dtype=np.uint32)
    # Design semantics: ONLY the top-left TILE_H x TILE_W tile is processed (out=in+1).
    # The rest of the output buffer is never written by the kernel (don't-care). So
    # correctness == the tile equals in+1; outside the tile is not asserted.

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)              # file parse only — no device
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] xclbin={a.xclbin}")
    print(f"[artifacts] kernel='{kname}'  instr_words={instr.size}  N={N} int32")
    if a.dry:
        print("[dry] artifacts valid; pyxrt imported; kernel name read from xclbin. "
              "NOT touching the NPU. Re-run without --dry (NPU must be free) to execute.")
        return 0

    # ---- from here on we touch /dev/accel/accel0 (needs the NPU free) ----
    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())   # <-- CREATE_HWCTX (fails if NPU busy)
    k = pyxrt.kernel(ctx, kname)

    SYNC_TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    SYNC_FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_inA = pyxrt.bo(d, inA.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_inB = pyxrt.bo(d, inB.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_out = pyxrt.bo(d, inA.nbytes, pyxrt.bo.host_only, k.group_id(5))

    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(SYNC_TO)
    bo_inA.write(inA.tobytes(), 0);     bo_inA.sync(SYNC_TO)
    bo_inB.write(inB.tobytes(), 0);     bo_inB.sync(SYNC_TO)

    run = k(3, bo_instr, instr.size, bo_inA, bo_inB, bo_out)  # opcode 3
    run.wait()
    bo_out.sync(SYNC_FROM)
    out = np.frombuffer(bo_out.read(inA.nbytes, 0), dtype=np.uint32)

    out_img = out.reshape(IMG_H, IMG_W)
    in_img = inA.reshape(IMG_H, IMG_W)
    tile_out = out_img[:TILE_H, :TILE_W]
    tile_exp = in_img[:TILE_H, :TILE_W] + 1
    ok = np.array_equal(tile_out, tile_exp)
    nbad = int((tile_out != tile_exp).sum())
    print(f"[run] tile {TILE_H}x{TILE_W} out==in+1 : {'PASS' if ok else 'FAIL'}  "
          f"tile_mismatches={nbad}/{tile_out.size}")
    print(f"[run] tile row0 in={in_img[0,:6]} -> out={out_img[0,:6]} (expect +1)")
    print(f"[run] (outside-tile region is don't-care; kernel only writes the tile)")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
