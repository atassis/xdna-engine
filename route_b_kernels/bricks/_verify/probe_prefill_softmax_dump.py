#!/usr/bin/env python3
"""What does softmax_core actually RETURN inside prefill_attn?

The stage bisect localised the first wrong stage to softmax_core (rel_l2 1.042 at stop=3) with the
output fully FINITE -- so the kernel's NaN is downstream of an already-wrong softmax, and the
softmax itself is not producing the NaN. This dumps the actual values so the defect has a shape.

Run:  ./run.sh probe_prefill_softmax_dump.py
"""
from pathlib import Path

import ml_dtypes
import numpy as np

import bricklib
import verify_prefill_attn as V

golden = V.golden
BF = ml_dtypes.bfloat16
HD, M, HEAD = golden.HD, golden.M, 0
SPAD = 16
KERNEL = Path(__file__).resolve().parents[1] / "prefill-attn" / "prefill_attn.cc"

_ar, q_full, k_full, v_full = golden.build_qkv(seed=0, m_tokens=M)
in_tiles, kv_resident, h_kv = golden.pack_head_rows(q_full, k_full, v_full, HEAD)

tiles = np.asarray(in_tiles, np.float32)
q_rows = tiles[:, :HD]
mask_rows = tiles[:, HD:HD + SPAD]
kvf = np.asarray(kv_resident, np.float32)
K = kvf[:M * HD].reshape(M, HD)
scores = np.zeros((len(tiles), SPAD), np.float32)
scores[:, :M] = (q_rows @ K.T) / np.sqrt(float(HD))
masked = scores + mask_rows
e = np.exp(masked - masked.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)

exp = np.zeros((len(tiles), HD), np.float32)
exp[:, :SPAD] = probs

res = bricklib.verify_streamed(
    name="prefill_sm_dump", shim=KERNEL, symbol="prefill_attn_row",
    in_tiles=in_tiles, out_tile_numel=HD, resident=kv_resident,
    unpack=lambda d: d, golden=exp, gate=V.GATE,
    in_dt=BF, out_dt=np.float32, resident_dt=BF, resident_depth=1,
    compile_flags=[f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}", "-DPREFILL_STOP_AFTER=3"],
)
got = np.asarray(res.get("got"), np.float32)[:, :SPAD]

np.set_printoptions(precision=4, suppress=True, linewidth=200)
print(f"status={res.get('status')} rel_l2={res.get('rel_l2'):.3e}", flush=True)
for r in (0, 1, M - 1):
    print(f"\n--- query row {r} (mask visible lanes: {int((mask_rows[r] > -1e8).sum())}) ---")
    print("  masked scores :", masked[r])
    print("  got  (device) :", got[r])
    print("  exp  (numpy)  :", probs[r])
print("\nper-row sums  device:", got.sum(axis=1))
print("per-row sums  numpy :", probs.sum(axis=1))
print("device row max      :", got.max(axis=1))
print("device argmax       :", got.argmax(axis=1))
print("numpy  argmax       :", probs.argmax(axis=1))
