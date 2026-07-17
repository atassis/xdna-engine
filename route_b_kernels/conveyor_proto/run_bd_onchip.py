#!/usr/bin/env python3
"""H-head arithmetic gate for the BD-ON-CHIP 4th stage.
5-BO ABI: kern(3, instr, n, qpv_all, p_all, k_all, v_all, ctx_all). Per head: feeds qpv=q_pass||qv +
p/k/v (NO host BD precompute); the on-chip BD tile computes BD = rel_shift((q+bias_v)@p^T). Verifies ctx
vs a host relpos-MHA golden. Build first: BDON=1 ATTN_T=.. ATTN_NQT=.. ATTN_HEADS=H make ..."""
import os, numpy as np, pyxrt
from ml_dtypes import bfloat16

EX = os.path.join(os.path.dirname(__file__), "build")
TQ = int(os.environ.get("ATTN_TQ", 8)); T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 128)); N_QT = int(os.environ.get("ATTN_NQT", 1))
H = int(os.environ.get("ATTN_HEADS", 1))
SCALE = float(os.environ.get("ATTN_SCALE", 1.0 / (DK ** 0.5)))
BD_SPLIT = int(os.environ.get("BD_SPLIT", 0))
NQ = N_QT * TQ; P = 2 * T - 1

def bf(x): return x.astype(bfloat16).astype(np.float32)

def gen_head(seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((NQ, DK)).astype(np.float32)
    k = bf(rng.standard_normal((T, DK))); v = bf(rng.standard_normal((T, DK)))
    p = bf(rng.standard_normal((P, DK))); bias_v = bf(0.1 * rng.standard_normal((DK,)))
    q = bf(q / ((q @ k.T).std() + 1e-6))
    qv = bf(q + bias_v)
    AC = q @ k.T; BD = qv @ p.T
    BD_sh = np.stack([BD[i, (T - 1 - i):(T - 1 - i) + T] for i in range(NQ)])
    BD_hi = bf(BD_sh); BD_car = BD_hi + (bf(BD_sh - BD_hi) if BD_SPLIT else 0.0)
    scores = (AC + BD_car) * SCALE
    e = np.exp(scores - scores.max(1, keepdims=True))
    ctx = (e / e.sum(1, keepdims=True)) @ v
    qpv = np.concatenate([q.reshape(N_QT, TQ * DK), qv.reshape(N_QT, TQ * DK)], axis=1).reshape(-1)
    return qpv, p.reshape(-1), k.reshape(-1), v.reshape(-1), ctx

heads = [gen_head(h) for h in range(H)]
qpv_all = np.concatenate([h[0] for h in heads]).astype(bfloat16).view(np.uint16)
p_all = np.concatenate([h[1] for h in heads]).astype(bfloat16).view(np.uint16)
k_all = np.concatenate([h[2] for h in heads]).astype(bfloat16).view(np.uint16)
v_all = np.concatenate([h[3] for h in heads]).astype(bfloat16).view(np.uint16)
ctx_ref = np.concatenate([h[4] for h in heads], axis=0)   # [H*NQ, DK]

instr = np.fromfile(f"{EX}/insts.bin", dtype=np.uint32)
xclbin = pyxrt.xclbin(f"{EX}/final.xclbin")
kname = xclbin.get_kernels()[0].get_name()
d = pyxrt.device(0); d.register_xclbin(xclbin)
hw = pyxrt.hw_context(d, xclbin.get_uuid()); kern = pyxrt.kernel(hw, kname)
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_qpv = pyxrt.bo(d, qpv_all.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_p = pyxrt.bo(d, p_all.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_k = pyxrt.bo(d, k_all.nbytes, pyxrt.bo.host_only, kern.group_id(5))
bo_v = pyxrt.bo(d, v_all.nbytes, pyxrt.bo.host_only, kern.group_id(6))
bo_c = pyxrt.bo(d, H * NQ * DK * 2, pyxrt.bo.host_only, kern.group_id(7))
for bo, arr in ((bo_instr, instr), (bo_qpv, qpv_all), (bo_p, p_all), (bo_k, k_all), (bo_v, v_all)):
    bo.write(arr.tobytes(), 0); bo.sync(TO)

import time
r = kern(3, bo_instr, instr.size, bo_qpv, bo_p, bo_k, bo_v, bo_c); r.wait()
bo_c.sync(FROM)
# timing: mean ms/dispatch over 200 iters (device only; mask-independent)
_t0 = time.perf_counter()
for _ in range(200):
    r = kern(3, bo_instr, instr.size, bo_qpv, bo_p, bo_k, bo_v, bo_c); r.wait()
_dt = (time.perf_counter() - _t0) / 200 * 1e3
print(f"[bd_onchip] {N_QT} tiles/dispatch, H={H} -> {_dt:.4f} ms/dispatch")
ctx_dev = np.frombuffer(bo_c.read(H * NQ * DK * 2, 0), dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H * NQ, DK)

rel = np.linalg.norm(ctx_dev - ctx_ref) / (np.linalg.norm(ctx_ref) + 1e-12)
ph = [np.linalg.norm(ctx_dev[h*NQ:(h+1)*NQ] - ctx_ref[h*NQ:(h+1)*NQ]) /
      (np.linalg.norm(ctx_ref[h*NQ:(h+1)*NQ]) + 1e-12) for h in range(H)]
print(f"[bd_onchip] T={T} TQ={TQ} DK={DK} N_QT={N_QT} H={H} P={P} BD_SPLIT={BD_SPLIT} kernel='{kname}'")
print(f"[bd_onchip] per-head rel-L2: " + " ".join(f"{x:.3e}" for x in ph))
print(f"[bd_onchip] TOTAL rel-L2={rel:.5e}  gate<=5e-3  {'PASS' if rel <= 5e-3 else 'FAIL'}")
