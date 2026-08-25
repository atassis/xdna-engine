"""Does the k768 rail's per-transition entry cost survive if we stop waiting between submits?

[[2026-08-25-eighty-one-percent-of-the-k768-rail-was-my-own-harness-changing-contexts]] measured a
~7.3 ms entry cost per pass through the 5-xclbin chain and closed by naming one untried lever:
"queueing commands instead of submit-then-wait was not tried". That matters before building the rail
as a fused block -- if the tax is host-side submit serialization it overlaps for free, and fusion is
buying much less than 7.3 ms.

Timing only, on INDEPENDENT bricks: XRT gives no cross-context ordering guarantee, so a queued chain
with real data dependencies is a separate (correctness) question this does not touch.

ARTIFACTS=<dir> (default artifacts/k768_gelu_rail), PAD_M=<n>, REPS=<n>.
"""
import os, statistics, time
import numpy as np
from ml_dtypes import bfloat16
import pyxrt

ART = os.environ.get("ARTIFACTS", "artifacts/k768_gelu_rail")
M = int(os.environ.get("PAD_M", 512))
REPS = int(os.environ.get("REPS", 9))
KRES, DFF = 768, 3072
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
dev = pyxrt.device(0)


def arm(stem, ins, nout):
    xb = pyxrt.xclbin(f"{ART}/final_{stem}.xclbin"); dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kk = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())
    instr = np.fromfile(f"{ART}/insts_{stem}.txt", np.uint32)
    bi = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    bufs = []
    for i, a in enumerate(ins):
        raw = np.ascontiguousarray(a); raw = raw.view(np.uint16) if raw.dtype == bfloat16 else raw
        b = pyxrt.bo(dev, raw.nbytes, pyxrt.bo.host_only, kk.group_id(3 + i))
        b.write(raw.tobytes(), 0); b.sync(TO); bufs.append(b)
    bc = pyxrt.bo(dev, nout, pyxrt.bo.host_only, kk.group_id(3 + len(ins)))
    # keep every handle alive: ctx must outlive the kernel or the context is torn down
    return (ctx, kk, bi, instr.size, bufs, bc)


def submit(h):
    _, kk, bi, n, bufs, bc = h
    return kk(3, bi, n, *bufs, bc)


def ms(f):
    t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) * 1e3


rng = np.random.default_rng(7)
x = np.asarray(rng.standard_normal((M, KRES)), np.float32)
Kaug = KRES + 32
A1 = np.zeros((M, Kaug), bfloat16); A1[:, :KRES] = x.astype(bfloat16); A1[:, KRES] = bfloat16(1.0)
W1 = np.asarray(rng.standard_normal((Kaug, DFF)) / 28, np.float32).astype(bfloat16)

A = arm(f"cast_{M}x{KRES}", [x], M * KRES * 2)                                  # small program
B = arm(f"{M}x{Kaug}x{DFF}_64x32x128_8c_modalgelu", [A1, W1], M * DFF * 4)      # large program
print(f"PAD_M={M} REPS={REPS}  A=cast_{M}x{KRES} (small)  B=fc1 modalgelu (large). Both contexts alive.\n")

for h, nm in ((A, "A"), (B, "B")):   # warm both so nothing below pays a first-touch
    submit(h).wait()


def serial_pair():
    submit(A).wait(); submit(B).wait()

def queued_pair():
    r1 = submit(A); r2 = submit(B); r1.wait(); r2.wait()

def serial_same():
    submit(A).wait(); submit(A).wait()

def queued_same():
    r1 = submit(A); r2 = submit(A); r1.wait(); r2.wait()

def solo(h):
    return lambda: submit(h).wait()


bench = [
    ("A alone            ", solo(A)),
    ("B alone            ", solo(B)),
    ("A;B  submit+wait   ", serial_pair),
    ("A,B  queued        ", queued_pair),
    ("A;A  submit+wait   ", serial_same),
    ("A,A  queued        ", queued_same),
]
res = {}
for name, fn in bench:
    ts = sorted(ms(fn) for _ in range(REPS))
    res[name.strip()] = statistics.median(ts)
    print(f"  {name} median {statistics.median(ts):7.3f} ms   min {ts[0]:7.3f}   max {ts[-1]:7.3f}")

a, b = res["A alone"], res["B alone"]
print(f"\n  cross-context: serial {res['A;B  submit+wait']:.3f} vs queued {res['A,B  queued']:.3f} ms"
      f"   (sum-of-solos {a+b:.3f}, max-of-solos {max(a,b):.3f})")
print(f"  same-context : serial {res['A;A  submit+wait']:.3f} vs queued {res['A,A  queued']:.3f} ms"
      f"   (2x solo {2*a:.3f})")
print("\n  queued ~ sum  -> the cost is device-serial; fusion is the lever."
      "\n  queued ~ max  -> the cost is host submit serialization; queueing buys it without fusion.")
