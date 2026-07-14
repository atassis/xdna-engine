import numpy as np
from ml_dtypes import bfloat16 as BF16
DK=128; LOG2E=np.float32(1.4426950408889634)
def bf16(x): return np.asarray(x,np.float32).astype(BF16).astype(np.float32)
def mm(a,b): return (bf16(a).astype(np.float32)@bf16(b).astype(np.float32)).astype(np.float32)
def smax(s):
    m=s.max(); e=np.exp2(((s-m)*LOG2E).astype(np.float32)).astype(BF16).astype(np.float32)
    return bf16(e*bf16(np.float32(1.0)/e.sum(dtype=np.float32)))
def ceildiv(a,b): return (a+b-1)//b
def pad(x,n): return np.concatenate([x,np.zeros((n-x.shape[0],DK),np.float32)],0) if x.shape[0]<n else x

# --- runner's f32 oracle (host_probs_ctx + rel_shift_host) ---
def rel_shift_host(bd):
    Hh,T,P=bd.shape
    x=np.pad(bd,((0,0),(0,0),(1,0))); x=x.reshape(Hh,P+1,T); x=x[:,1:].reshape(Hh,T,P)
    return x[:,:,:T]
def host_ctx(qu,qv,k,p,V,scale):
    ac=qu@k.T; bd=qv@p.T; bd_sh=rel_shift_host(bd[None])[0]
    s=(ac+bd_sh)*scale; s=s-s.max(-1,keepdims=True); a=np.exp(s); a/=a.sum(-1,keepdims=True)
    return (a@V).astype(np.float32)

# --- device-faithful block-decomposed model (bf16, tiled, --stream packed) ---
def block_stream(qu,qv,k,p,V,TQ,KB,isc):
    T,P=qu.shape[0],p.shape[0]
    Tp=ceildiv(T,KB)*KB; Pp=ceildiv(P,KB)*KB
    kpad,ppad,Vpad=pad(k,Tp),pad(p,Pp),pad(V,Tp)
    Tkf=(T//KB)*KB; krag=T-Tkf; Ppf=(P//KB)*KB; prag=P-Ppf; Tqf=(T//TQ)*TQ; qrag=T-Tqf
    ctx=np.zeros((T,DK),np.float32)
    def db(A,B,out,tq,kb,j0,nc):
        for il in range(tq):
            for jj in range(kb): out[il,j0+jj]=np.dot(bf16(A[il]).astype(np.float32),bf16(B[jj]).astype(np.float32))
    def cb(pr,Vb,cf,tq,kb,j0):
        for il in range(tq):
            a=np.zeros(DK,np.float32)
            for jj in range(kb): a+=bf16(pr[il,j0+jj]).astype(np.float32)*bf16(Vb[jj]).astype(np.float32)
            cf[il]+=a
    def tile(tq,q0):
        qut=pad(qu[q0:q0+tq],TQ); qvt=pad(qv[q0:q0+tq],TQ)
        AC=np.zeros((TQ,T),np.float32); BD=np.zeros((TQ,P),np.float32)
        for j0 in range(0,Tkf,KB): db(qut,kpad[j0:j0+KB],AC,tq,KB,j0,T)
        if krag: db(qut,kpad[Tkf:Tkf+KB],AC,tq,krag,Tkf,T)
        for j0 in range(0,Ppf,KB): db(qvt,ppad[j0:j0+KB],BD,tq,KB,j0,P)
        if prag: db(qvt,ppad[Ppf:Ppf+KB],BD,tq,prag,Ppf,P)
        pr=np.zeros((TQ,T),np.float32)
        for il in range(tq):
            b=(T-1)-(q0+il); pr[il]=smax((AC[il]+BD[il,b:b+T])*isc)
        cf=np.zeros((TQ,DK),np.float32)
        for j0 in range(0,Tkf,KB): cb(pr,Vpad[j0:j0+KB],cf,tq,KB,j0)
        if krag: cb(pr,Vpad[Tkf:Tkf+KB],cf,tq,krag,Tkf)
        ctx[q0:q0+tq]=bf16(cf[:tq])
    for q0 in range(0,Tqf,TQ): tile(TQ,q0)
    if qrag: tile(qrag,Tqf)
    return ctx

def run(T,TQ=8,KB=43,seed=1,sig=0.3):
    r=np.random.default_rng(seed); P=2*T-1
    qu=r.standard_normal((T,DK)).astype(np.float32)*sig; qv=r.standard_normal((T,DK)).astype(np.float32)*sig
    k=r.standard_normal((T,DK)).astype(np.float32)*sig; p=r.standard_normal((P,DK)).astype(np.float32)*sig
    V=r.standard_normal((T,DK)).astype(np.float32)*sig; isc=np.float32(1/np.sqrt(DK))
    # non-degenerate rescale like the runner default (fold 1/std into qu/qv)
    ac=qu@k.T; bd=qv@p.T; bdsh=rel_shift_host(bd[None])[0]
    std=float(((ac+bdsh)*isc).std())+1e-6
    qud,qvd=qu/std,qv/std
    ref=host_ctx(qud,qvd,k,p,V,isc)          # runner's f32 oracle
    dev=block_stream(qud,qvd,k,p,V,TQ,KB,isc)  # device-faithful bf16 block model
    rel=float(np.linalg.norm(dev-ref)/(np.linalg.norm(ref)+1e-12))
    corr=float(np.corrcoef(dev.ravel(),ref.ravel())[0,1])
    print(f"T={T:3d} n_kb={ceildiv(T,KB)} n_pb={ceildiv(P,KB)}: block-model vs runner-oracle "
          f"rel-L2={rel:.4e} corr={corr:.6f}  {'OK' if rel<0.08 else 'DIVERGE!!'}")
    return rel

run(32); run(64); run(86); run(129); run(172)
