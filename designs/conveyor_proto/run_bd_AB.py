#!/usr/bin/env python3
"""Dispatch head-set A, then a DIFFERENT head-set B, reusing the same kernel/BOs (mimics the Rust
H=4x2 group loop). If B is garbage vs its own golden -> the 2nd dispatch is broken. If B matches
A's golden on k/v/p-driven output -> resident weights from A are reused (not reloaded)."""
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
A=pack([0,1,2,3]); B=pack([10,11,12,13])
instr=np.fromfile(f"{EX}/insts.bin",dtype=np.uint32)
xb=pyxrt.xclbin(f"{EX}/final.xclbin");kn=xb.get_kernels()[0].get_name();d=pyxrt.device(0);d.register_xclbin(xb)
hw=pyxrt.hw_context(d,xb.get_uuid());kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE;FR=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bi=pyxrt.bo(d,instr.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=pyxrt.bo(d,A[0].nbytes,pyxrt.bo.host_only,kern.group_id(3));bp=pyxrt.bo(d,A[1].nbytes,pyxrt.bo.host_only,kern.group_id(4))
bk=pyxrt.bo(d,A[2].nbytes,pyxrt.bo.host_only,kern.group_id(5));bv=pyxrt.bo(d,A[3].nbytes,pyxrt.bo.host_only,kern.group_id(6))
bc=pyxrt.bo(d,H*NQ*DK*2,pyxrt.bo.host_only,kern.group_id(7))
bi.write(instr.tobytes(),0);bi.sync(TO)
def load(S):
    for bo,a in ((bq,S[0]),(bp,S[1]),(bk,S[2]),(bv,S[3])):bo.write(a.tobytes(),0);bo.sync(TO)
def run():
    r=kern(3,bi,instr.size,bq,bp,bk,bv,bc);r.wait();bc.sync(FR)
    return np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H*NQ,DK).copy()
def rel(a,b):return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)
load(A);oA=run()
load(B);oB=run()
print(f"[AB] dispatch1 A vs goldenA: {rel(oA,A[4]):.4e}  (expect ~4e-3)")
print(f"[AB] dispatch2 B vs goldenB: {rel(oB,B[4]):.4e}  (garbage => 2nd dispatch broken)")
print(f"[AB] dispatch2 B vs goldenA: {rel(oB,A[4]):.4e}  (~small => reused A's resident weights)")
# also: reload A after B to see if it recovers
load(A);oA2=run()
print(f"[AB] dispatch3 A vs goldenA: {rel(oA2,A[4]):.4e}  (recovers? => order-dependent state)")
