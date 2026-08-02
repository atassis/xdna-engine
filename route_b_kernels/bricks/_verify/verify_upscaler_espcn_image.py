#!/usr/bin/env python3
"""Produce a REAL image upscaled BY THE NPU: run the whole ESPCN (bf16) on a 32x32 Y
patch (M-tiled + K-tiled over the 64x64x64 gemm-bfp16-ebs8 brick), save the NPU SR Y,
and gate vs the CPU reference. The tangible whole-net-on-silicon result.

Run under the NPU lock:  ./run.sh verify_upscaler_espcn_image.py
"""
import numpy as np, ml_dtypes
from pathlib import Path
from bricklib import _build_oneshot, iron, GEN
from verify_f2 import tile_pack, tile_unpack, golden_mod

# Anchored at the repo root: run.sh cds into _verify/, so a relative "artifacts/..." never resolved.
# Regenerate the fixtures with scripts/make_espcn_gate_fixtures.py.
WD=Path(__file__).resolve().parents[3]/"artifacts/upscaler/wholenet"
def L(n): return np.load(WD/n)
T=64; BF=ml_dtypes.bfloat16
def bf16(x): return x.astype(BF).astype(np.float32)
_,cc=golden_mod("gemm-bfp16-ebs8","gemm_bfp16_ebs8.cc")
(GEN/"img_shim.cc").write_text(f'#include <stdint.h>\n#include "{cc}"\n')
design=_build_oneshot("gemm_bfp16_ebs8_64x64x64",GEN/"img_shim.cc",[T*T,T*T],T*T,[BF,BF],BF,[])
def pbt(B): return B.reshape(T//8,8,T//8,8).transpose(0,2,3,1).reshape(-1)
def dev(A,B):
    ta=iron.tensor(np.ascontiguousarray(tile_pack(A.astype(BF),8,8)),dtype=BF,device="npu")
    tb=iron.tensor(np.ascontiguousarray(pbt(B.astype(BF))),dtype=BF,device="npu")
    tc=iron.zeros((T*T,),dtype=BF,device="npu"); design(ta,tb,tc)
    return tile_unpack(np.array(tc.numpy(),copy=True),T,T,8,8).astype(np.float32)
def im2col(x,k,pad):
    Cin,H,W=x.shape; xp=np.pad(x,((0,0),(pad,pad),(pad,pad))); Ho,Wo=H+2*pad-k+1,W+2*pad-k+1
    C=np.zeros((Ho*Wo,Cin*k*k),np.float32); i=0
    for oy in range(Ho):
        for ox in range(Wo): C[i]=xp[:,oy:oy+k,ox:ox+k].reshape(-1); i+=1
    return C,Ho,Wo
def conv(x,w,b,k,pad,relu):
    Cin,H,W=x.shape; Cout=w.shape[0]
    A,Ho,Wo=im2col(bf16(x),k,pad); M,Kf=A.shape
    Bm=w.reshape(Cout,Kf).T; Kp=((Kf+T-1)//T)*T
    Bp=np.zeros((Kp,T),np.float32); Bp[:Kf,:Cout]=Bm
    out=np.zeros((M,Cout),np.float32)
    for mb in range((M+T-1)//T):                       # M-tile
        rows=slice(mb*T,min((mb+1)*T,M)); nrow=rows.stop-rows.start
        Ab=np.zeros((T,Kp),np.float32); Ab[:nrow,:Kf]=A[rows]
        acc=np.zeros((T,T),np.float32)
        for kb in range(Kp//T): acc+=dev(Ab[:,kb*T:(kb+1)*T],Bp[kb*T:(kb+1)*T,:])  # K-tile
        o=acc[:nrow,:Cout]+b
        out[rows]=np.maximum(o,0) if relu else o
    return out.reshape(Ho,Wo,Cout).transpose(2,0,1)
def pshuf(x,r):
    Cr2,H,W=x.shape; C=Cr2//(r*r)
    return x.reshape(C,r,r,H,W).transpose(0,3,1,4,2).reshape(C,H*r,W*r)

x=L("patch_y.npy")[0]
for i,(wn,k,pad,rl) in enumerate([("conv1",5,2,1),("conv2",3,1,1),("conv3",3,1,1),("conv4",3,1,0)]):
    print(f"{wn}..",end="",flush=True); x=conv(x,L(f"{wn}_w.npy"),L(f"{wn}_b.npy"),k,pad,rl)
sr=pshuf(x,3)[None]
np.save(WD/"patch_npu_sr_y.npy", sr)
ref=L("patch_cpu_sr_y.npy")
rl2=float(np.linalg.norm((sr-ref).ravel())/(np.linalg.norm(ref.ravel())+1e-12))
mse=np.mean((sr-ref)**2); ps=10*np.log10((ref.max()-ref.min())**2/mse)
print(f"\nNPU image {sr.shape}: bf16 NPU vs fp32 CPU  rel-L2={rl2:.3e}  PSNR={ps:.1f}dB  "
      f"-> {'PASS' if rl2<=0.06 else 'FAIL'}. NPU SR saved.")
# Without this the gate printed FAIL and still exited 0, so no drain could ever fail on it.
assert rl2<=0.06, f"ESPCN image gate failed: rel-L2={rl2:.3e} > 0.06"
