#!/usr/bin/env python3
"""P1 measurement: dispatch latency of REUSE (resident k, no LPDDR fetch) vs FILL (fetch k each dispatch)
on one resident context. Delta = the per-dispatch LPDDR-fetch saving from weight residency (176 KB of k:
4 heads x T*DK bf16). Times kernel dispatch + wait only (no host readback in the loop)."""
import os, time, numpy as np, pyxrt
from ml_dtypes import bfloat16
EX=os.path.join(os.path.dirname(__file__),"build")
TQ=8;T=176;DK=128;N_QT=22;H=4;NQ=N_QT*TQ
instr_fill=np.fromfile(f"{EX}/insts.broken.bin",dtype=np.uint32)
instr_reuse=np.fromfile(f"{EX}/insts_reuse.bin",dtype=np.uint32)
xb=pyxrt.xclbin(f"{EX}/reuse.xclbin");kn=xb.get_kernels()[0].get_name();d=pyxrt.device(0);d.register_xclbin(xb)
hw=pyxrt.hw_context(d,xb.get_uuid());kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
rng=np.random.default_rng(0)
def mkbo(nbytes,gid): return pyxrt.bo(d,nbytes,pyxrt.bo.host_only,gid)
bif=pyxrt.bo(d,instr_fill.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bir=pyxrt.bo(d,instr_reuse.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=mkbo(H*N_QT*2*TQ*DK*2,kern.group_id(3));bp=mkbo(H*(2*T-1)*DK*2,kern.group_id(4))
bk=mkbo(H*T*DK*2,kern.group_id(5));bv=mkbo(H*T*DK*2,kern.group_id(6));bc=mkbo(H*NQ*DK*2,kern.group_id(7))
for b in (bq,bp,bk,bv): b.write(rng.standard_normal(b.size()//2).astype(np.float32).astype(bfloat16).view(np.uint16).tobytes(),0); b.sync(TO)
bif.write(instr_fill.tobytes(),0);bif.sync(TO);bir.write(instr_reuse.tobytes(),0);bir.sync(TO)
def dispatch(bi,sz): kern(3,bi,sz,bq,bp,bk,bv,bc).wait()
def timed(bi,sz,N,warm=10):
    for _ in range(warm): dispatch(bi,sz)
    t0=time.perf_counter()
    for _ in range(N): dispatch(bi,sz)
    return (time.perf_counter()-t0)/N*1e3
N=50
dispatch(bif,instr_fill.size)                 # dispatch 1: fill k -> resident
t_reuse=timed(bir,instr_reuse.size,N)         # reuse: no k fetch
t_fill=timed(bif,instr_fill.size,N)           # fill: fetch k each dispatch
print(f"[P1] fill  (fetch k, {instr_fill.size} words) : {t_fill:.4f} ms/dispatch")
print(f"[P1] reuse (resident,{instr_reuse.size} words) : {t_reuse:.4f} ms/dispatch")
print(f"[P1] delta (k-fetch of 176KB saved)   : {t_fill-t_reuse:.4f} ms/dispatch  ({100*(t_fill-t_reuse)/t_fill:.1f}%)")
