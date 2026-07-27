#!/usr/bin/env python3
"""Traced BD-onchip conveyor: same 5-BO ABI as run_bd_onchip.py plus a 6th TRACE BO at the tail.
Dumps the raw trace buffer for python/utils/trace/parse.py. Correctness still gated (rel-L2) so a
traced build that silently computes garbage cannot be mistaken for a valid profile."""
import os, sys, numpy as np, pyxrt
from ml_dtypes import bfloat16
EX = sys.argv[1]; TRACE_SIZE = int(os.environ.get("TRACE_SIZE", 65536))
TQ=int(os.environ["ATTN_TQ"]); T=int(os.environ["ATTN_T"]); DK=int(os.environ["ATTN_DK"])
N_QT=int(os.environ["ATTN_NQT"]); H=int(os.environ["ATTN_HEADS"])
SCALE=float(os.environ.get("ATTN_SCALE", 1.0/(DK**0.5)))
NQ=N_QT*TQ; P=2*T-1
def bf(x): return x.astype(bfloat16).astype(np.float32)
def gen_head(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((NQ,DK)).astype(np.float32)
    k=bf(rng.standard_normal((T,DK))); v=bf(rng.standard_normal((T,DK)))
    p=bf(rng.standard_normal((P,DK))); bias_v=bf(0.1*rng.standard_normal((DK,)))
    q=bf(q/((q@k.T).std()+1e-6)); qv=bf(q+bias_v)
    AC=q@k.T; BD=qv@p.T
    BD_sh=np.stack([BD[i,(T-1-i):(T-1-i)+T] for i in range(NQ)])
    scores=(AC+bf(BD_sh))*SCALE
    e=np.exp(scores-scores.max(1,keepdims=True)); ctx=(e/e.sum(1,keepdims=True))@v
    qpv=np.concatenate([q.reshape(N_QT,TQ*DK),qv.reshape(N_QT,TQ*DK)],axis=1).reshape(-1)
    return qpv,p.reshape(-1),k.reshape(-1),v.reshape(-1),ctx
heads=[gen_head(h) for h in range(H)]
qpv_all=np.concatenate([h[0] for h in heads]).astype(bfloat16).view(np.uint16)
p_all=np.concatenate([h[1] for h in heads]).astype(bfloat16).view(np.uint16)
k_all=np.concatenate([h[2] for h in heads]).astype(bfloat16).view(np.uint16)
v_all=np.concatenate([h[3] for h in heads]).astype(bfloat16).view(np.uint16)
ctx_ref=np.concatenate([h[4] for h in heads],axis=0); CTX_ELEMS=H*NQ*DK
instr=np.fromfile(f"{EX}/insts_trace.bin",dtype=np.uint32)
xclbin=pyxrt.xclbin(f"{EX}/final_trace.xclbin"); kname=xclbin.get_kernels()[0].get_name()
d=pyxrt.device(0); d.register_xclbin(xclbin)
hw=pyxrt.hw_context(d,xclbin.get_uuid()); kern=pyxrt.kernel(hw,kname)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE; FROM=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bo_instr=pyxrt.bo(d,instr.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bo_qpv=pyxrt.bo(d,qpv_all.nbytes,pyxrt.bo.host_only,kern.group_id(3))
bo_p=pyxrt.bo(d,p_all.nbytes,pyxrt.bo.host_only,kern.group_id(4))
bo_k=pyxrt.bo(d,k_all.nbytes,pyxrt.bo.host_only,kern.group_id(5))
bo_v=pyxrt.bo(d,v_all.nbytes,pyxrt.bo.host_only,kern.group_id(6))
bo_c=pyxrt.bo(d,CTX_ELEMS*2,pyxrt.bo.host_only,kern.group_id(7))
bo_tr=pyxrt.bo(d,TRACE_SIZE,pyxrt.bo.host_only,kern.group_id(8))
for bo,arr in ((bo_instr,instr),(bo_qpv,qpv_all),(bo_p,p_all),(bo_k,k_all),(bo_v,v_all)):
    bo.write(arr.tobytes(),0); bo.sync(TO)
bo_tr.write(b"\x00"*TRACE_SIZE,0); bo_tr.sync(TO)
r=kern(3,bo_instr,instr.size,bo_qpv,bo_p,bo_k,bo_v,bo_c,bo_tr); r.wait()
bo_c.sync(FROM); bo_tr.sync(FROM)
ctx_dev=np.frombuffer(bo_c.read(CTX_ELEMS*2,0),dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H*NQ,DK)
rel=np.linalg.norm(ctx_dev-ctx_ref)/(np.linalg.norm(ctx_ref)+1e-12)
print(f"[trace] rel-L2={rel:.5e} gate<=5e-3 {'PASS' if rel<=5e-3 else 'FAIL'}  (a garbage build must not be profiled)")
raw=bo_tr.read(TRACE_SIZE,0)
open(sys.argv[2],"wb").write(raw)
nz=int(np.count_nonzero(np.frombuffer(raw,dtype=np.uint32)))
print(f"[trace] wrote {len(raw)} bytes, {nz} non-zero words")
