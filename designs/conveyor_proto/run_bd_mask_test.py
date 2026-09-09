#!/usr/bin/env python3
"""Isolate the BD-onchip KERNEL masking at short t_active (< BUILT_T), independent of Rust.
Patches the discovered t_active RTP words in insts.bin to TA, builds a masked golden at length TA
(pad keys j>=TA excluded; rel_shift base = TA-1-i; p = real [2*TA-1] table zero-padded to P=2*T-1),
and compares device ctx (first TA query rows) vs golden. If PASS -> kernel masking correct at short T,
residual WER is Rust-side; if FAIL -> kernel bug."""
import os, numpy as np, pyxrt
from ml_dtypes import bfloat16
EX = os.path.join(os.path.dirname(__file__), "build")
TQ=int(os.environ.get("ATTN_TQ",8)); T=int(os.environ.get("ATTN_T",176))
DK=int(os.environ.get("ATTN_DK",128)); N_QT=int(os.environ.get("ATTN_NQT",22))
H=int(os.environ.get("ATTN_HEADS",4)); SCALE=float(os.environ.get("ATTN_SCALE",0.08838835))
TA=int(os.environ.get("TA",100))                 # active length to test (<= T)
WORDS=[int(x) for x in os.environ.get("TACT_WORDS","8,20,32,44,56,68,80,92").split(",")]
NQ=N_QT*TQ; P=2*T-1
def bf(x): return x.astype(bfloat16).astype(np.float32)
def gen_head(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((NQ,DK)).astype(np.float32)
    # only first TA keys are real; pad rows zero (mirrors the Rust push_pad_rows)
    k=np.zeros((T,DK),np.float32); v=np.zeros((T,DK),np.float32)
    k[:TA]=bf(rng.standard_normal((TA,DK))); v[:TA]=bf(rng.standard_normal((TA,DK)))
    # p = real [2*TA-1] rel-pos table zero-padded to P
    p=np.zeros((P,DK),np.float32); p[:2*TA-1]=bf(rng.standard_normal((2*TA-1,DK)))
    bias_v=bf(0.1*rng.standard_normal((DK,)))
    q=bf(q/((q@k.T).std()+1e-6)); qv=bf(q+bias_v)
    AC=q@k.T; BD=qv@p.T                            # [NQ,P]
    # masked golden: rel_shift window base = TA-1-i, width TA; scores over first TA keys only
    ctx=np.zeros((NQ,DK),np.float32)
    for i in range(NQ):
        base=TA-1-i if i<TA else 0
        bd=BD[i, base:base+TA] if 0<=base<=P-TA else np.zeros(TA)
        sc=(AC[i,:TA]+bd)*SCALE
        e=np.exp(sc-sc.max()); w=e/e.sum()
        ctx[i]=w@v[:TA]
    qpv=np.concatenate([q.reshape(N_QT,TQ*DK),qv.reshape(N_QT,TQ*DK)],axis=1).reshape(-1)
    return qpv,p.reshape(-1),k.reshape(-1),v.reshape(-1),ctx
heads=[gen_head(h) for h in range(H)]
qpv=np.concatenate([h[0] for h in heads]).astype(bfloat16).view(np.uint16)
pp =np.concatenate([h[1] for h in heads]).astype(bfloat16).view(np.uint16)
kk =np.concatenate([h[2] for h in heads]).astype(bfloat16).view(np.uint16)
vv =np.concatenate([h[3] for h in heads]).astype(bfloat16).view(np.uint16)
ref=np.concatenate([h[4] for h in heads],axis=0)
instr=np.fromfile(f"{EX}/insts.bin",dtype=np.uint32).copy()
for w in WORDS: instr[w]=TA                        # patch t_active RTP sites
xb=pyxrt.xclbin(f"{EX}/final.xclbin"); kn=xb.get_kernels()[0].get_name()
d=pyxrt.device(0); d.register_xclbin(xb); hw=pyxrt.hw_context(d,xb.get_uuid()); kern=pyxrt.kernel(hw,kn)
TO=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE; FR=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bi=pyxrt.bo(d,instr.nbytes,pyxrt.bo.cacheable,kern.group_id(1))
bq=pyxrt.bo(d,qpv.nbytes,pyxrt.bo.host_only,kern.group_id(3)); bp=pyxrt.bo(d,pp.nbytes,pyxrt.bo.host_only,kern.group_id(4))
bk=pyxrt.bo(d,kk.nbytes,pyxrt.bo.host_only,kern.group_id(5)); bv=pyxrt.bo(d,vv.nbytes,pyxrt.bo.host_only,kern.group_id(6))
bc=pyxrt.bo(d,H*NQ*DK*2,pyxrt.bo.host_only,kern.group_id(7))
for bo,a in ((bi,instr),(bq,qpv),(bp,pp),(bk,kk),(bv,vv)): bo.write(a.tobytes(),0); bo.sync(TO)
r=kern(3,bi,instr.size,bq,bp,bk,bv,bc); r.wait(); bc.sync(FR)
dev=np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).view(bfloat16).astype(np.float32).reshape(H*NQ,DK)
# compare only the first TA query rows per head (rows >= TA are pad/garbage, discarded by Rust)
ph=[]
for h in range(H):
    dv=dev[h*NQ:h*NQ+TA]; rf=ref[h*NQ:h*NQ+TA]
    ph.append(np.linalg.norm(dv-rf)/(np.linalg.norm(rf)+1e-12))
tot=np.mean(ph)
print(f"[mask_test] TA={TA} T={T} H={H} per-head rel-L2(first {TA} rows): "+" ".join(f"{x:.3e}" for x in ph))
print(f"[mask_test] MEAN rel-L2={tot:.5e} gate<=5e-3 {'PASS' if tot<=5e-3 else 'FAIL'}")
