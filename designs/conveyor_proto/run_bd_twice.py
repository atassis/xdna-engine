#!/usr/bin/env python3
"""Dispatch the SAME H=4 inputs TWICE back-to-back; compare run2 vs run1 and both vs golden.
If run2 != run1 -> kernel carries state across dispatches (the multi-group bug)."""
import os, numpy as np, pyxrt
from ml_dtypes import bfloat16
EX=os.path.join(os.path.dirname(__file__),"build")
TQ=8;T=int(os.environ.get("ATTN_T",176));DK=128;N_QT=int(os.environ.get("ATTN_NQT",22));H=int(os.environ.get("ATTN_HEADS",4))
SCALE=float(os.environ.get("ATTN_SCALE",0.08838835));NQ=N_QT*TQ;P=2*T-1
def bf(x): return x.astype(bfloat16).astype(np.float32)
def gen_head(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((NQ,DK)).astype(np.float32)
    k=bf(rng.standard_normal((T,DK)));v=bf(rng.standard_normal((T,DK)));p=bf(rng.standard_normal((P,DK)));bv=bf(0.1*rng.standard_normal((DK,)))
    q=bf(q/((q@k.T).std()+1e-6));qv=bf(q+bv);AC=q@k.T;BD=qv@p.T
    BDs=np.stack([BD[i,(T-1-i):(T-1-i)+T] for i in range(NQ)]);sc=(AC+bf(BDs))*SCALE
    e=np.exp(sc-sc.max(1,keepdims=True));ctx=(e/e.sum(1,keepdims=True))@v
    qpv=np.concatenate([q.reshape(N_QT,TQ*DK),qv.reshape(N_QT,TQ*DK)],axis=1).reshape(-1)
    return qpv,p.reshape(-1),k.reshape(-1),v.reshape(-1),ctx
hs=[gen_head(h) for h in range(H)]
qpv=np.concatenate([h[0] for h in hs]).astype(bfloat16).view(np.uint16);pp=np.concatenate([h[1] for h in hs]).astype(bfloat16).view(np.uint16)
kk=np.concatenate([h[2] for h in hs]).astype(bfloat16).view(np.uint16);vv=np.concatenate([h[3] for h in hs]).astype(bfloat16).view(np.uint16)
ref=np.concatenate([h[4] for h in hs],axis=0)
instr=np.fromfile(f"{EX}/insts.bin",dtype=np.uint32)
xb=pyxrt.xclbin(f"{EX}/final.xclbin");kn=xb.get_kernels()[0].get_name();d=pyxrt.device(0);d.register_xclbin(xb)
hw=pyxrt.hw_context(d,xb.get_uuid());kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE;FR=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bi=pyxrt.bo(d,instr.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=pyxrt.bo(d,qpv.nbytes,pyxrt.bo.host_only,kern.group_id(3));bp=pyxrt.bo(d,pp.nbytes,pyxrt.bo.host_only,kern.group_id(4))
bk=pyxrt.bo(d,kk.nbytes,pyxrt.bo.host_only,kern.group_id(5));bv=pyxrt.bo(d,vv.nbytes,pyxrt.bo.host_only,kern.group_id(6))
bc=pyxrt.bo(d,H*NQ*DK*2,pyxrt.bo.host_only,kern.group_id(7))
for bo,a in ((bi,instr),(bq,qpv),(bp,pp),(bk,kk),(bv,vv)):bo.write(a.tobytes(),0);bo.sync(TO)
def run():
    r=kern(3,bi,instr.size,bq,bp,bk,bv,bc);r.wait();bc.sync(FR)
    return np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H*NQ,DK).copy()
o1=run();o2=run();o3=run()
def rel(a,b):return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)
print(f"[twice] run1 vs golden: {rel(o1,ref):.4e}")
print(f"[twice] run2 vs golden: {rel(o2,ref):.4e}")
print(f"[twice] run3 vs golden: {rel(o3,ref):.4e}")
print(f"[twice] run2 vs run1  : {rel(o2,o1):.4e}   (0 => stateless; >0 => carries state)")
print(f"[twice] run3 vs run2  : {rel(o3,o2):.4e}")
