#!/usr/bin/env python3
"""R2 stability: dispatch 1 fills+holds resident k, then N reuse dispatches (no k fetch, both-lock re-arm
prod->0/cons->1). Confirms the fixed point holds INDEFINITELY -- every reuse output BIT-IDENTICAL to d1,
no drift, no lock overflow, no hang. This is the linchpin 'proven ok' criterion."""
import os, sys, numpy as np, pyxrt
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
    return tuple(np.concatenate([h[i] for h in hs]).astype(bfloat16).view(np.uint16) for i in range(4))+(np.concatenate([h[4] for h in hs],axis=0),)
A=pack([0,1,2,3])
XB=os.environ.get("STAB_XCLBIN","reuse.xclbin")         # green control: STAB_XCLBIN=final.fixed.xclbin
FILL=os.environ.get("STAB_FILL","insts.broken.bin")     #   STAB_FILL=insts.fixed.bin STAB_REUSE=insts.fixed.bin
REUSE=os.environ.get("STAB_REUSE","insts_reuse.bin")
instr_fill=np.fromfile(f"{EX}/{FILL}",dtype=np.uint32)
instr_reuse=np.fromfile(f"{EX}/{REUSE}",dtype=np.uint32)
xb=pyxrt.xclbin(f"{EX}/{XB}");kn=xb.get_kernels()[0].get_name();d=pyxrt.device(0);d.register_xclbin(xb)
hw=pyxrt.hw_context(d,xb.get_uuid());kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE;FR=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bif=pyxrt.bo(d,instr_fill.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bir=pyxrt.bo(d,instr_reuse.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=pyxrt.bo(d,A[0].nbytes,pyxrt.bo.host_only,kern.group_id(3));bp=pyxrt.bo(d,A[1].nbytes,pyxrt.bo.host_only,kern.group_id(4))
bk=pyxrt.bo(d,A[2].nbytes,pyxrt.bo.host_only,kern.group_id(5));bv=pyxrt.bo(d,A[3].nbytes,pyxrt.bo.host_only,kern.group_id(6))
bc=pyxrt.bo(d,H*NQ*DK*2,pyxrt.bo.host_only,kern.group_id(7))
bif.write(instr_fill.tobytes(),0);bif.sync(TO);bir.write(instr_reuse.tobytes(),0);bir.sync(TO)
for b,a in ((bq,A[0]),(bp,A[1]),(bk,A[2]),(bv,A[3])):b.write(a.tobytes(),0);b.sync(TO)
def run(bi,sz):
    kern(3,bi,sz,bq,bp,bk,bv,bc).wait();bc.sync(FR)
    return np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).copy()
def rel(u): return float(np.linalg.norm(u.view(bfloat16).astype(np.float32).reshape(H*NQ,DK)-A[4])/(np.linalg.norm(A[4])+1e-12))
N=int(sys.argv[1]) if len(sys.argv)>1 else 5000
ref=run(bif,instr_fill.size)               # d1: fill + hold resident k
print(f"[stab] d1 fill rel-L2 = {rel(ref):.4e}")
mism=0; first_mism=None
for i in range(1,N+1):
    o=run(bir,instr_reuse.size)            # reuse: no fetch + both-lock re-arm
    if not np.array_equal(o,ref):
        mism+=1
        if first_mism is None: first_mism=i
    if i%1000==0: print(f"[stab] dispatch {i:5d}: rel-L2 {rel(o):.4e}  bit-identical-to-d1 {'YES' if np.array_equal(o,ref) else 'NO'}  mismatches-so-far {mism}")
print(f"[stab] DONE {N} reuse dispatches. bit-identical-to-d1: {N-mism}/{N}  first-mismatch: {first_mism}  VERDICT: {'STABLE' if mism==0 else 'DRIFT/UNSTABLE'}")
