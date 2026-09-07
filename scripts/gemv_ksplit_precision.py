import numpy as np, ml_dtypes
BF=ml_dtypes.bfloat16
rng=np.random.default_rng(7)
M,K=3840,15360
A=np.asarray(rng.standard_normal((M,K),dtype=np.float32)/np.sqrt(K),BF)
x=np.asarray(rng.standard_normal(K,dtype=np.float32),BF)
truth=A.astype(np.float64)@x.astype(np.float64)
uns=np.asarray(A.astype(np.float32)@x.astype(np.float32),BF).astype(np.float64)
def rel(v): return float(np.linalg.norm(v-truth)/np.linalg.norm(truth))
base=rel(uns)
print(f"  unsplit  (1 rounding, mv.cc:63)            rel-L2 {base:.4e}")
for n in (2,4,8):
    ks=K//n
    parts=[np.asarray(A[:,i*ks:(i+1)*ks].astype(np.float32)@x[i*ks:(i+1)*ks].astype(np.float32),BF) for i in range(n)]
    acc=parts[0]
    for p in parts[1:]:
        acc=np.asarray(acc.astype(np.float32)+p.astype(np.float32),BF)
    r=rel(acc.astype(np.float64))
    print(f"  split {n} ({n} roundings + {n-1} bf16 adds)      rel-L2 {r:.4e}   {r/base:.2f}x unsplit")
