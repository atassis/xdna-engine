#!/usr/bin/env python3
"""A1 MHA precision sim — WHICH bf16 quantity dominates the encoder-MHA error (no device).

Four independent precision knobs in the flash (online-softmax) attention:
  score = the QK output A (the scores tile)         prob  = the softmax output P (probs / PV input)
  stat  = m / l / rescale (scale_buffer bookkeeping) Oacc  = the O output accumulator
Conclusions (2026-06-20, the saved score-vs-prob isolation; device-validated against rel-L2):
  - scale_buffer / stats-f32 ALONE is NEGLIGIBLE (0.0102 vs 0.0113) — confirmed on-device (0.0244->0.0237).
  - The DOMINANT lever is f32 SCORES (the QK matmul output A): 0.0071 alone (1.6x). mm.cc already has
    bf16->f32 matmul variants, so it's an instantiation, not a new kernel.
  - f32 PROBS add NOTHING (avoids the hard mixed-type PV matmul).
  - Full WER-recovery (~6x) needs score + stat + O-accumulator f32. See A1-mha-precision-kernel-spec.md."""
import numpy as np
from ml_dtypes import bfloat16
np.random.seed(42)
S, d, BK = 1536, 64, 64
Qf=(np.random.rand(S,d)*4).astype(np.float32); Kf=(np.random.rand(S,d)*4).astype(np.float32); Vf=(np.random.rand(S,d)*4).astype(np.float32)
bf=lambda x:x.astype(bfloat16).astype(np.float32); scale=1/np.sqrt(d)
def golden(Q,K,V):
    s=(Q@K.T)*scale; s-=s.max(1,keepdims=True); e=np.exp(s); return (e/e.sum(1,keepdims=True))@V
def flash(Q,K,V,score,prob,stat,Oacc):
    Q,K,V=bf(Q),bf(K),bf(V); O=np.zeros((S,d),Oacc); m=np.full((S,1),-np.inf,stat); l=np.zeros((S,1),stat)
    for j0 in range(0,S,BK):
        Kb,Vb=K[j0:j0+BK],V[j0:j0+BK]
        sc=((Q.astype(np.float32)@Kb.T.astype(np.float32))*scale).astype(score)        # QK output A
        mn=np.maximum(m,sc.max(1,keepdims=True).astype(stat))
        p=np.exp((sc.astype(np.float32)-mn).astype(np.float32)).astype(prob)            # probs P
        a=np.exp((m-mn).astype(np.float32)).astype(stat)
        l=(a*l+p.astype(np.float32).sum(1,keepdims=True).astype(stat)).astype(stat)
        pv=(p.astype(np.float32)@Vb) if prob==np.float32 else (bf(p.astype(np.float32))@Vb)
        O=(a.astype(Oacc)*O+pv.astype(Oacc)).astype(Oacc); m=mn
    return O.astype(np.float32)/l.astype(np.float32)
g=golden(Qf,Kf,Vf); rl=lambda x:float(np.linalg.norm(x-g)/np.linalg.norm(g)); B,F=bfloat16,np.float32
print(f"score bf16 | prob bf16 | stat bf16 | O bf16   (current kernel)            {rl(flash(Qf,Kf,Vf,B,B,B,B)):.5f}")
print(f"score bf16 | prob bf16 | stat F32  | O bf16   (scale_buffer fix = DUD)     {rl(flash(Qf,Kf,Vf,B,B,F,B)):.5f}")
print(f"score F32  | prob bf16 | stat bf16 | O bf16   (QK->f32 SCORES, the lever)  {rl(flash(Qf,Kf,Vf,F,B,B,B)):.5f}")
print(f"score F32  | prob bf16 | stat F32  | O bf16   (scores + f32 stats)         {rl(flash(Qf,Kf,Vf,F,B,F,B)):.5f}")
print(f"score F32  | prob F32  | stat F32  | O bf16   (+probs: adds NOTHING)       {rl(flash(Qf,Kf,Vf,F,F,F,B)):.5f}")
print(f"score F32  | prob bf16 | stat F32  | O F32    (scores+stats+O = full ~6x)  {rl(flash(Qf,Kf,Vf,F,B,F,F)):.5f}")
