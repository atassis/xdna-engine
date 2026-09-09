#!/usr/bin/env python3
"""Verify + TIME the query-tiled attention conveyor (N_QT tiles streamed, k/V resident).
Usage: run_conveyor_attn.py [conv|mono]  (default conv). Verifies ctx vs numpy + reports per-dispatch ms."""
import os, sys, time, numpy as np, pyxrt
from ml_dtypes import bfloat16

EX = os.path.join(os.path.dirname(__file__), "build")
which = sys.argv[1] if len(sys.argv) > 1 else "conv"
TQ = int(os.environ.get("ATTN_TQ", 8))
T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 64))
N_QT = int(os.environ.get("ATTN_NQT", 16))
SCALE = float(os.environ.get("ATTN_SCALE", 1.0 / (DK ** 0.5)))
NQ = N_QT * TQ

RELPOS = os.environ.get("ATTN_RELPOS", "0") == "1"
H = int(os.environ.get("ATTN_HEADS", 1))   # data-parallel heads (one conveyor per column)
P = 2 * T - 1


def gen_head(seed):
    # independent per-head q/k/v (+ p for relpos); returns (query-belt, k, v, ctx_ref) for this head.
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((NQ, DK)).astype(bfloat16)
    k = rng.standard_normal((T, DK)).astype(bfloat16)
    v = rng.standard_normal((T, DK)).astype(bfloat16)
    if RELPOS:
        p = rng.standard_normal((P, DK)).astype(bfloat16)
        ac_raw = q.astype(np.float32) @ k.astype(np.float32).T
        BD = q.astype(np.float32) @ p.astype(np.float32).T
        BD_shifted = np.empty((NQ, T), np.float32)
        for i in range(NQ):
            base = (T - 1 - i) if i < T else 0
            BD_shifted[i] = BD[i, base:base + T]
        bd_bf = BD_shifted.astype(bfloat16)
        scores = (ac_raw + bd_bf.astype(np.float32)) * SCALE
        q_belt = np.concatenate([q.reshape(N_QT, TQ * DK), bd_bf.reshape(N_QT, TQ * T)], axis=1).reshape(-1)
    else:
        scores = SCALE * (q.astype(np.float32) @ k.astype(np.float32).T)
        q_belt = q.reshape(-1)
    scn = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scn); probs = e / e.sum(axis=1, keepdims=True)
    ctx = probs @ v.astype(np.float32)
    return q_belt.reshape(-1), k.reshape(-1), v.reshape(-1), ctx


heads = [gen_head(h) for h in range(H)]
GJ = 4                                       # heads per MemTile group (must match the generator)
QELEM = TQ * DK + (TQ * T if RELPOS else 0)
# q GROUP-MAJOR step-interleaved (the input split expects one [gsz*QELEM] object/step). k/v head-major.
_qh = [hd[0].reshape(N_QT, QELEM) for hd in heads]
_qparts = []
for _g in range(0, H, GJ):
    _gsz = min(GJ, H - _g)
    _qparts.append(np.stack([_qh[_g + i] for i in range(_gsz)], axis=1).reshape(-1))
q_belt = np.concatenate(_qparts)
kpack = np.concatenate([hd[1] for hd in heads])           # [H * KT] bf16 head-major
if os.environ.get("ATTN_MMUL", "0") == "1":
    # The mmul scores kernel consumes k PRE-TILED into t x s blocks, tiles row-major:
    #   ktiled[((j*colA + c)*t + tt)*s + ss] = k[(j*t + tt)*DK + (c*s + ss)]
    # This CANNOT be done by the shim on the H=8 grouped path: it needs 5 DMA dims (head, group,
    # c, tt, ss) against the shim's 4, and merging head+group gives 88 > the 6-bit (max 64)
    # iteration field. So the host does it -- free here, and free in npu.rs on the host-packed
    # rail. The device-in rail (k straight off a GEMM output BO) still needs a device-side answer.
    _s = _t8 = 8
    _kt = kpack.reshape(H, T, DK).reshape(H, T // _t8, _t8, DK // _s, _s)   # [H, j, tt, c, ss]
    # -> [H, j, c, ss, tt]: tile ORDER stays j-major (keeps a j-pair contiguous) but tile CONTENT is
    # [ss][tt], so the kernel needs no aie::transpose in its inner loop.
    kpack = _kt.transpose(0, 1, 3, 2, 4).reshape(-1)   # [H, j, c, tt, ss]
v_all = np.concatenate([hd[2] for hd in heads])           # [H * VT] bf16
ctx_ref = np.concatenate([hd[3] for hd in heads], axis=0)  # [H*NQ, DK] f32

instr = np.fromfile(f"{EX}/{which}.insts", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/{which}.xclbin")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
hw = pyxrt.hw_context(d, xclbin.get_uuid())
kern = pyxrt.kernel(hw, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

qb = q_belt.reshape(-1).view(np.uint16); kb = kpack.reshape(-1).view(np.uint16); vb = v_all.reshape(-1).view(np.uint16)
bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_q = pyxrt.bo(d, qb.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_k = pyxrt.bo(d, kb.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_v = pyxrt.bo(d, vb.nbytes, pyxrt.bo.host_only, kern.group_id(5))
bo_c = pyxrt.bo(d, H * NQ * DK * 2, pyxrt.bo.host_only, kern.group_id(6))
bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
bo_q.write(qb.tobytes(), 0); bo_q.sync(TO)
bo_k.write(kb.tobytes(), 0); bo_k.sync(TO)
bo_v.write(vb.tobytes(), 0); bo_v.sync(TO)

# Optional trace BO, appended at the TAIL of the runtime sequence (group_id 7) so the q/k/v/ctx
# group ids and the ABI above are untouched. ATTN_TRACE = trace buffer bytes; 0 = production.
TRACE = int(os.environ.get("ATTN_TRACE", 0))
bo_t = pyxrt.bo(d, TRACE, pyxrt.bo.host_only, kern.group_id(7)) if TRACE else None
if bo_t is not None:
    bo_t.write(bytearray(TRACE), 0); bo_t.sync(TO)

def once():
    if bo_t is not None:
        r = kern(3, bo_instr, instr.size, bo_q, bo_k, bo_v, bo_c, bo_t)
    else:
        r = kern(3, bo_instr, instr.size, bo_q, bo_k, bo_v, bo_c)
    r.wait()

once()
if bo_t is not None:
    bo_t.sync(FROM)
    open(os.environ.get("ATTN_TRACE_FILE", "trace.txt"), "w").write(
        "\n".join("0x%08X" % w for w in np.frombuffer(bo_t.read(TRACE, 0), dtype=np.uint32)))
    print(f"[trace] wrote {TRACE} bytes -> {os.environ.get('ATTN_TRACE_FILE','trace.txt')}")
bo_c.sync(FROM)
# joined ctx: heads grouped by GJ into MemTiles; each group drains contiguously as [N_QT, gsz, TQ, DK].
# de-interleave per group (H<->N_QT) then concat -> per-head [H*NQ, DK]. (GJ must match the generator.)
GJ = 4
_dev_raw = np.frombuffer(bo_c.read(H * NQ * DK * 2, 0), dtype=np.uint16).view(bfloat16).astype(np.float32)
_parts, _off = [], 0
for _g in range(0, H, GJ):
    _gsz = min(GJ, H - _g)
    _n = N_QT * _gsz * TQ * DK
    _parts.append(_dev_raw[_off:_off + _n].reshape(N_QT, _gsz, TQ, DK).transpose(1, 0, 2, 3).reshape(_gsz * NQ, DK))
    _off += _n
dev_ctx = np.concatenate(_parts, axis=0)   # [H*NQ, DK]
rel = np.linalg.norm(dev_ctx - ctx_ref) / np.linalg.norm(ctx_ref)
ph = [np.linalg.norm(dev_ctx[h*NQ:(h+1)*NQ]-ctx_ref[h*NQ:(h+1)*NQ])/max(np.linalg.norm(ctx_ref[h*NQ:(h+1)*NQ]),1e-9) for h in range(H)]
print(f"  per-head rel-err ({H} heads):", " ".join(f"{x:.3f}" for x in ph))

iters = 200
t0 = time.perf_counter()
for _ in range(iters):
    once()
dt = (time.perf_counter() - t0) / iters * 1e3

print(f"[{which}] kernel='{kname}'  rel-L2={rel:.4e}  {'PASS' if rel < 3e-2 else 'FAIL'}")
print(f"[{which}] {N_QT} query tiles/dispatch  ->  {dt:.4f} ms/dispatch  ({dt/N_QT*1e3:.2f} us/query-tile)")
