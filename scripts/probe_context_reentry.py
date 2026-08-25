"""A-B-A: is the k768 rail's per-brick entry cost a one-time warmup or a per-transition tax?

Both contexts are registered up front and stay alive. We dispatch A x5, B x5, then A x5 again.
If A's SECOND visit is cheap, entry is one-time warmup (amortizable, irrelevant to a steady rail).
If it costs the same as its first visit, entry is paid per transition -- the tax one-xclbin deletes.
"""
import sys, time
import numpy as np
from ml_dtypes import bfloat16
import pyxrt

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
LN = "mlir-aie/programming_examples/ml/layernorm/build"
M, KRES, DFF = 512, 768, 3072
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
dev = pyxrt.device(0)


def arm(xclbin, insts, ins, nout):
    xb = pyxrt.xclbin(xclbin); dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kk = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(insts, np.uint32)
    bi = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    bufs = []
    for i, a in enumerate(ins):
        raw = np.ascontiguousarray(a); raw = raw.view(np.uint16) if raw.dtype == bfloat16 else raw
        b = pyxrt.bo(dev, raw.nbytes, pyxrt.bo.host_only, kk.group_id(3 + i))
        b.write(raw.tobytes(), 0); b.sync(TO); bufs.append(b)
    bc = pyxrt.bo(dev, nout, pyxrt.bo.host_only, kk.group_id(3 + len(ins)))
    return ctx, kk, bi, instr.size, bufs, bc


def fire(h):
    _, kk, bi, n, bufs, bc = h
    t0 = time.perf_counter(); kk(3, bi, n, *bufs, bc).wait(); return (time.perf_counter() - t0) * 1e3


rng = np.random.default_rng(7)
x = np.asarray(rng.standard_normal((M, KRES)), np.float32)
Kaug = KRES + 32
A1 = np.zeros((M, Kaug), bfloat16); A1[:, :KRES] = x.astype(bfloat16); A1[:, KRES] = bfloat16(1.0)
B1 = np.asarray(rng.standard_normal((Kaug, DFF)) / 28, np.float32).astype(bfloat16)

A = arm(f"{LN}/final_cast_{M}x{KRES}.xclbin", f"{LN}/insts_cast_{M}x{KRES}.txt", [x], M * KRES * 2)
B = arm(f"{WA}/final_{M}x{Kaug}x{DFF}_64x32x128_8c_modalgelu.xclbin",
        f"{WA}/insts_{M}x{Kaug}x{DFF}_64x32x128_8c_modalgelu.txt", [A1, B1], M * DFF * 4)
print("armed: A=cast_512x768 (small program), B=fc1 modalgelu (large program). Both contexts alive.\n")

for label, h in [("A visit 1", A), ("B visit 1", B), ("A visit 2", A), ("B visit 2", B),
                 ("A visit 3", A), ("B visit 3", B)]:
    ts = [fire(h) for _ in range(6)]
    print(f"  {label}: first={ts[0]:6.3f} ms   rest={' '.join(f'{t:.3f}' for t in ts[1:])}")
