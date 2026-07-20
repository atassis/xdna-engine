#!/usr/bin/env python3
"""WHOLE-NET: run the full ESPCN (4-conv, x3) end-to-end on the NPU in bf16, gate vs the
CPU reference. The M1 culmination -- every op runs THROUGH the NPU:
  conv1(5x5,1->64,relu) -> conv2(3x3,64->64,relu) -> conv3(3x3,64->32,relu)
  -> conv4(3x3,32->9) -> pixel_shuffle(r=3)  ==  CPU ESPCN, within bf16 tolerance.

Each conv = host im2col(bf16) -> K-tiled 64x64x64 gemm-bfp16-ebs8 (B block-transposed)
-> accumulate -> +bias -> relu. 8x8 input (same-pad) => M=64 (one M-tile) => 24 GEMMs.
Activations rounded to bf16 between layers (the resident bf16 stream). Weights/tile/ref
are the REAL trained ESPCN, prepped in the session scratchpad.

Run under the NPU lock:  ./run.sh verify_upscaler_espcn_wholenet.py
"""
import numpy as np, ml_dtypes
from pathlib import Path
from bricklib import _build_oneshot, iron, GEN
from verify_f2 import tile_pack, tile_unpack, golden_mod

WD = Path("artifacts/upscaler/wholenet")
def L(n): return np.load(WD/n)
T = 64
BF = ml_dtypes.bfloat16
def bf16(x): return x.astype(BF).astype(np.float32)   # round-trip through bf16

# build the bf16 gemm design once
_, cc = golden_mod("gemm-bfp16-ebs8", "gemm_bfp16_ebs8.cc")
(GEN/"wn_shim.cc").write_text(f'#include <stdint.h>\n#include "{cc}"\n')
design = _build_oneshot("gemm_bfp16_ebs8_64x64x64", GEN/"wn_shim.cc", [T*T, T*T], T*T,
                        [BF, BF], BF, [])
def pack_B_blockT(B): return B.reshape(T//8,8,T//8,8).transpose(0,2,3,1).reshape(-1)
def dev_gemm(A_kb, B_kb):
    ta=iron.tensor(np.ascontiguousarray(tile_pack(A_kb.astype(BF),8,8)),dtype=BF,device="npu")
    tb=iron.tensor(np.ascontiguousarray(pack_B_blockT(B_kb.astype(BF))),dtype=BF,device="npu")
    tc=iron.zeros((T*T,),dtype=BF,device="npu"); design(ta,tb,tc)
    return tile_unpack(np.array(tc.numpy(),copy=True),T,T,8,8).astype(np.float32)

def im2col(x,k,pad):
    Cin,H,W=x.shape; xp=np.pad(x,((0,0),(pad,pad),(pad,pad)))
    Ho,Wo=H+2*pad-k+1, W+2*pad-k+1
    cols=np.zeros((Ho*Wo,Cin*k*k),np.float32); i=0
    for oy in range(Ho):
        for ox in range(Wo):
            cols[i]=xp[:,oy:oy+k,ox:ox+k].reshape(-1); i+=1
    return cols,Ho,Wo

def conv_bf16(x,w,b,k,pad,relu):
    Cin,H,W=x.shape; Cout=w.shape[0]
    A,Ho,Wo=im2col(bf16(x),k,pad); M,Kf=A.shape
    Bm=w.reshape(Cout,Kf).T
    Kp=((Kf+T-1)//T)*T
    Ap=np.zeros((M,Kp),np.float32); Ap[:,:Kf]=A
    Bp=np.zeros((Kp,T),np.float32); Bp[:Kf,:Cout]=Bm
    acc=np.zeros((M,T),np.float32)
    for kb in range(Kp//T):
        acc+=dev_gemm(Ap[:,kb*T:(kb+1)*T], Bp[kb*T:(kb+1)*T,:])
    out=acc[:,:Cout]+b
    if relu: out=np.maximum(out,0)
    return out.reshape(Ho,Wo,Cout).transpose(2,0,1)   # [Cout,Ho,Wo]

def pixel_shuffle(x,r):   # [C*r^2,H,W]->[C,H*r,W*r], CRD
    Cr2,H,W=x.shape; C=Cr2//(r*r)
    return x.reshape(C,r,r,H,W).transpose(0,3,1,4,2).reshape(C,H*r,W*r)

# ---- run the whole net on the NPU ----
x=L("input_tile.npy")[0]                                  # [1,8,8]
print("conv1..", end="",flush=True); x=conv_bf16(x,L("conv1_w.npy"),L("conv1_b.npy"),5,2,True)
print("conv2..", end="",flush=True); x=conv_bf16(x,L("conv2_w.npy"),L("conv2_b.npy"),3,1,True)
print("conv3..", end="",flush=True); x=conv_bf16(x,L("conv3_w.npy"),L("conv3_b.npy"),3,1,True)
print("conv4..", end="",flush=True); x=conv_bf16(x,L("conv4_w.npy"),L("conv4_b.npy"),3,1,False)
sr=pixel_shuffle(x,3)[None]                               # [1,1,24,24]

ref=L("cpu_ref_sr.npy")
def rl2(a,b): a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64); return np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12)
def psnr(a,b): m=np.mean((a-b)**2); return 99.0 if m==0 else 10*np.log10((ref.max()-ref.min())**2/m)
r=rl2(sr,ref); p=psnr(sr.astype(np.float64),ref.astype(np.float64))
ok=(r<=0.06)
print(f"\nWHOLE-NET ESPCN on NPU (bf16) vs CPU: SR {sr.shape} rel-L2={r:.3e} PSNR={p:.1f}dB "
      f"-> {'PASS' if ok else 'FAIL'} (gate rel-L2<=0.06)")
