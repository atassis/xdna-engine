#!/usr/bin/env python3
"""True-residency proof: dispatch 1 fills k and holds it resident (acquire-once); dispatches 2+ use a
runtime sequence with NO k fetch + a set_lock re-arm (cons_lock->1) so the consumer re-reads the SAME
resident k. Same input every dispatch, so correct d2/d3 == resident k reused with zero LPDDR re-fetch."""
import os, numpy as np, pyxrt
from ml_dtypes import bfloat16
EX=os.path.join(os.path.dirname(__file__),"build")
TQ=8;T=176;DK=128;N_QT=22;H=4;SCALE=0.08838835;NQ=N_QT*TQ;P=2*T-1
def bf(x): return x.astype(bfloat16).astype(np.float32)
def gen(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((NQ,DK)).astype(np.float32)
    k=bf(rng.standard_normal((T,DK)));v=bf(rng.standard_normal((T,DK)));p=bf(rng.standard_normal((P,DK)));bv=bf(0.1*rng.standard_normal((DK,)))
    q=bf(q/((q@k.T).std()+1e-6));qv=bf(q+bv);AC=q@k.T;BD=qv@p.T
    BDs=np.stack([BD[i,(T-1-i):(T-1-i)+T] for i in range(NQ)]);sc=(AC+bf(BDs))*SCALE
    e=np.exp(sc-sc.max(1,keepdims=True));ctx=(e/e.sum(1,keepdims=True))@v
    qpv=np.concatenate([q.reshape(N_QT,TQ*DK),qv.reshape(N_QT,TQ*DK)],axis=1).reshape(-1)
    return qpv,p.reshape(-1),k.reshape(-1),v.reshape(-1),ctx
def pack(seeds):
    hs=[gen(s) for s in seeds]
    return (np.concatenate([h[0] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[1] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[2] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[3] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[4] for h in hs],axis=0))
A=pack([0,1,2,3])
instr_fill=np.fromfile(f"{EX}/insts.broken.bin",dtype=np.uint32)    # d1: fills + holds resident k
instr_reuse=np.fromfile(f"{EX}/insts_reuse.bin",dtype=np.uint32)    # d2+: no k fetch + set_lock re-arm
xb=pyxrt.xclbin(f"{EX}/reuse.xclbin");kn=xb.get_kernels()[0].get_name();d=pyxrt.device(0);d.register_xclbin(xb)
hw=pyxrt.hw_context(d,xb.get_uuid());kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE;FR=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bif=pyxrt.bo(d,instr_fill.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bir=pyxrt.bo(d,instr_reuse.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=pyxrt.bo(d,A[0].nbytes,pyxrt.bo.host_only,kern.group_id(3));bp=pyxrt.bo(d,A[1].nbytes,pyxrt.bo.host_only,kern.group_id(4))
bk=pyxrt.bo(d,A[2].nbytes,pyxrt.bo.host_only,kern.group_id(5));bv=pyxrt.bo(d,A[3].nbytes,pyxrt.bo.host_only,kern.group_id(6))
bc=pyxrt.bo(d,H*NQ*DK*2,pyxrt.bo.host_only,kern.group_id(7))
bif.write(instr_fill.tobytes(),0);bif.sync(TO);bir.write(instr_reuse.tobytes(),0);bir.sync(TO)
for bo,a in ((bq,A[0]),(bp,A[1]),(bk,A[2]),(bv,A[3])):bo.write(a.tobytes(),0);bo.sync(TO)
def run(bi,sz):
    r=kern(3,bi,sz,bq,bp,bk,bv,bc);r.wait();bc.sync(FR)
    return np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H*NQ,DK).copy()
def rel(a,b):return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)
o1=run(bif,instr_fill.size); o2=run(bir,instr_reuse.size); o3=run(bir,instr_reuse.size)
print(f"[reuse] d1 (fill k)   vs goldenA: {rel(o1,A[4]):.4e}  (expect ~4e-3)")
print(f"[reuse] d2 (REUSE k)  vs goldenA: {rel(o2,A[4]):.4e}  (~4e-3 => resident k reused via set_lock, NO re-fetch)")
print(f"[reuse] d3 (REUSE k)  vs goldenA: {rel(o3,A[4]):.4e}  (~4e-3 => holds across dispatches)")
