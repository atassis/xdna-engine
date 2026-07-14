import numpy as np
from ml_dtypes import bfloat16
DK=128
def bf(x): return np.ascontiguousarray(x,np.float32).astype(bfloat16).reshape(-1)
def ceildiv(x,y): return (x+y-1)//y
def pad_rows(x,n):
    r=x.shape[0]
    if r<n: x=np.concatenate([x,np.zeros((n-r,DK),np.float32)],0)
    return x

def check(T,TQ=8,KB=43,seed=1,sig=0.3):
    P=2*T-1; r=np.random.default_rng(seed)
    # synth data (build_head synth path)
    qu=r.standard_normal((T,DK)).astype(np.float32)*sig
    qv=r.standard_normal((T,DK)).astype(np.float32)*sig
    kh=r.standard_normal((T,DK)).astype(np.float32)*sig
    ph=r.standard_normal((P,DK)).astype(np.float32)*sig
    Vh=r.standard_normal((T,DK)).astype(np.float32)*sig
    # rescale like the runner (qu_d/qv_d used by BOTH ref and packing)
    qu_d,qv_d=qu,qv   # (rescale is a scalar; irrelevant to layout consistency)

    # ---- RUNNER --stream packing (verbatim) ----
    n_qt=ceildiv(T,TQ); n_kb=ceildiv(T,KB); n_pb=ceildiv(P,KB)
    Tp,Pp=n_kb*KB,n_pb*KB
    quv_tiles=[]
    for q in range(n_qt):
        q0=q*TQ
        quv_tiles.append(pad_rows(qu_d[q0:q0+TQ],TQ))
        quv_tiles.append(pad_rows(qv_d[q0:q0+TQ],TQ))
    QUV=bf(np.concatenate(quv_tiles,0))                       # flat bf16
    KPV=np.concatenate([bf(pad_rows(kh,Tp)),bf(pad_rows(ph,Pp)),bf(pad_rows(Vh,Tp))])

    # ---- DE-PACK per the KERNEL's expected layout ----
    QUVr=np.frombuffer(QUV.tobytes(),dtype=bfloat16).reshape(2*n_qt,TQ,DK)  # per-tile blocks
    KPVr=np.frombuffer(KPV.tobytes(),dtype=bfloat16).reshape(-1,DK)   # [Tp+Pp+Tp, DK]
    # quv: block 2q = qu_tile_q, block 2q+1 = qv_tile_q ; take tq real rows
    qu_dp=np.zeros((T,DK),np.float32); qv_dp=np.zeros((T,DK),np.float32)
    for q in range(n_qt):
        q0=q*TQ; tq=min(TQ,T-q0)
        qu_dp[q0:q0+tq]=QUVr[2*q][:tq].astype(np.float32)
        qv_dp[q0:q0+tq]=QUVr[2*q+1][:tq].astype(np.float32)
    # kpv: sections k(Tp)|p(Pp)|V(Tp), strip padding to T/P/T
    k_dp=KPVr[0:Tp][:T].astype(np.float32)
    p_dp=KPVr[Tp:Tp+Pp][:P].astype(np.float32)
    V_dp=KPVr[Tp+Pp:Tp+Pp+Tp][:T].astype(np.float32)

    # reference data as the device SHOULD see it (bf16-rounded originals)
    def b(x): return np.frombuffer(bf(x).tobytes(),dtype=bfloat16).reshape(-1,DK).astype(np.float32)
    def eq(a,ref,name):
        d=np.abs(a-ref).max(); print(f"  {name}: max|dp-ref|={d:.2e} {'OK' if d==0 else 'MISMATCH'}")
        return d==0
    print(f"T={T} n_kb={n_kb} n_pb={n_pb} n_qt={n_qt} Tp={Tp} Pp={Pp}")
    ok=True
    ok&=eq(qu_dp,b(qu_d),"qu"); ok&=eq(qv_dp,b(qv_d),"qv")
    ok&=eq(k_dp,b(kh),"k"); ok&=eq(p_dp,b(ph),"p"); ok&=eq(V_dp,b(Vh),"V")
    print(f"  => {'ALL MATCH (packing consistent with kernel layout)' if ok else 'PACKING BUG FOUND'}")

check(32); print(); check(172)
