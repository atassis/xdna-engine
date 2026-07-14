#!/usr/bin/env python3
"""PROTOTYPE: device-side BO hand-off across TWO xclbins (resident-rails feasibility).

Resolves the one open unknown for Task 3: can a BO written by kernel A (ctxLN, f32
out) be consumed by kernel B (cast, f32->bf16) WITHOUT a host round-trip
(sync_from_device + re-upload)? If yes, the resident LN->fc1 seam is buildable.

Chain:  x[f32] --(ctxln xclbin)--> bo_ln[f32, device-resident] --(cast xclbin)-->
        bo_bf16[bf16]   -- bo_ln is NEVER synced to host between the two dispatches.

Gate: device bf16 output within 1 bf16 ULP of host LN_2pass(x) (the LN reference),
proving the intermediate survived device-side and both hw-contexts co-resident.

Run (NPU free, from MAIN worktree):
  .venv-iron/bin/python ../xdna-engine-ln/scripts/prototype_ln_cast_resident.py --rows 512 --cols 1024
"""
import argparse, os, sys
import numpy as np
from ml_dtypes import bfloat16

EX = "mlir-aie/programming_examples/ml/layernorm/build"
EPS = 1e-5


def ln_2pass_ref(x):
    x = x.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    return (x - mean) * inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=1024)
    a = ap.parse_args()
    rows, cols = a.rows, a.cols
    ln_x = f"{EX}/final_ctxln_{rows}x{cols}.xclbin"
    ln_i = f"{EX}/insts_ctxln_{rows}x{cols}.txt"
    ca_x = f"{EX}/final_cast_{rows}x{cols}.xclbin"
    ca_i = f"{EX}/insts_cast_{rows}x{cols}.txt"
    for p in (ln_x, ln_i, ca_x, ca_i):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build ctxln + cast at {rows}x{cols} first")

    rng = np.random.RandomState(0)
    x = rng.uniform(-4.0, 8.0, size=(rows, cols)).astype(np.float32)
    ref = ln_2pass_ref(x)  # host LN (f32); bf16 truth is within 1 ULP of this
    X = np.ascontiguousarray(x).reshape(-1)
    instr_ln = np.fromfile(ln_i, dtype=np.uint32)
    instr_ca = np.fromfile(ca_i, dtype=np.uint32)

    import pyxrt
    d = pyxrt.device(0)
    xln = pyxrt.xclbin(ln_x); d.register_xclbin(xln)
    xca = pyxrt.xclbin(ca_x); d.register_xclbin(xca)
    ctx_ln = pyxrt.hw_context(d, xln.get_uuid())
    ctx_ca = pyxrt.hw_context(d, xca.get_uuid())
    k_ln = pyxrt.kernel(ctx_ln, xln.get_kernels()[0].get_name())
    k_ca = pyxrt.kernel(ctx_ca, xca.get_kernels()[0].get_name())
    print(f"[proto] both xclbins registered co-resident: ln='{xln.get_kernels()[0].get_name()}' cast='{xca.get_kernels()[0].get_name()}'")
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    HO = pyxrt.bo.host_only
    CA = pyxrt.bo.cacheable

    bo_instr_ln = pyxrt.bo(d, instr_ln.nbytes, CA, k_ln.group_id(1))
    bo_instr_ca = pyxrt.bo(d, instr_ca.nbytes, CA, k_ca.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, HO, k_ln.group_id(3))
    # THE INTERMEDIATE: ctxLN output (arg 4) reused as cast input (arg 3), device-resident.
    bo_ln = pyxrt.bo(d, X.nbytes, HO, k_ln.group_id(4))
    bo_bf16 = pyxrt.bo(d, rows * cols * 2, HO, k_ca.group_id(4))
    # dummies for each kernel's 2-buf-path placeholders
    dtmp_l = pyxrt.bo(d, 1, HO, k_ln.group_id(5)); dctl_l = pyxrt.bo(d, 8, HO, k_ln.group_id(6)); dtr_l = pyxrt.bo(d, 1, HO, k_ln.group_id(7))
    dtmp_c = pyxrt.bo(d, 1, HO, k_ca.group_id(5)); dctl_c = pyxrt.bo(d, 8, HO, k_ca.group_id(6)); dtr_c = pyxrt.bo(d, 1, HO, k_ca.group_id(7))

    bo_instr_ln.write(instr_ln.tobytes(), 0); bo_instr_ln.sync(TO)
    bo_instr_ca.write(instr_ca.tobytes(), 0); bo_instr_ca.sync(TO)
    bo_x.write(X.tobytes(), 0); bo_x.sync(TO)

    # (1) ctxLN: x -> bo_ln   -- do NOT sync bo_ln to host
    k_ln(3, bo_instr_ln, instr_ln.size, bo_x, bo_ln, dtmp_l, dctl_l, dtr_l).wait()
    # (2) cast: bo_ln -> bo_bf16   -- bo_ln consumed device-side, no host round-trip
    k_ca(3, bo_instr_ca, instr_ca.size, bo_ln, bo_bf16, dtmp_c, dctl_c, dtr_c).wait()

    bo_bf16.sync(FROM)
    Y = np.frombuffer(bo_bf16.read(rows * cols * 2, 0), dtype=bfloat16).reshape(rows, cols).astype(np.float32)
    rel = np.abs(Y - ref) / np.maximum(np.abs(ref), 1e-3)
    ONE_ULP = 2.0 ** -7
    within = int((rel <= ONE_ULP * 1.001).sum()); N = rows * cols
    print(f"[proto] device-side LN->cast (no host round-trip on the intermediate):")
    print(f"[proto]   <=1 bf16 ULP of host LN: {within}/{N} ({100*within/N:.2f}%)  max_rel={rel.max():.3e}")
    print(f"[proto]   Y[0,:4]={Y[0,:4]}  hostLN={ref[0,:4]}")
    ok = (rel.max() <= ONE_ULP * 1.001)
    print(f"[proto] 2-xclbin device-side BO hand-off: {'FEASIBLE (PASS)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
