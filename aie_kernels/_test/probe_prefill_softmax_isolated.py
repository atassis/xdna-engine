#!/usr/bin/env python3
"""Is softmax_core wrong ON ITS OWN, or only when prefill_attn feeds it?

The stage bisect put the first wrong stage at softmax_core (stop=3, rel_l2 1.042, output FINITE,
lane 0 erratic -- sometimes all the mass, sometimes exactly zero). Two very different causes fit:

  A. softmax_core is defective at this shape (SPAD=VL=16, heavily -1e9-masked input).
  B. softmax_core is fine, and the defect is the INTERACTION in prefill_attn_row_impl: pass 1b
     does aie::store_v(row_scores, ...) and then hands `row_scores` to softmax_core as a pointer
     PARAMETER. That is the rope-lut trigger shape from the open load-above-store hunt
     (store, then a dependent read across a call on a restrict pointer param), as opposed to the
     dwconv1d shape (same function, local array) which does NOT trigger.

This feeds the ALREADY-MASKED scores in as the streamed operand and calls softmax_core directly,
so there is no preceding store in the same function.

  A-clean => softmax_core is exonerated; the fault is the in-function store/call ordering.
  A-dirty => softmax_core itself is broken at this shape and prefill_attn is only its first victim.

Verify in ISOLATION before letting a defect claim justify a workaround.

Run:  ./run.sh probe_prefill_softmax_isolated.py
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
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

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

# Streamed input = the masked scores, bf16, padded to the HD+SPAD tile width the rail expects.
sm_tiles = np.zeros((len(tiles), HD + SPAD), np.float32)
sm_tiles[:, :SPAD] = masked
sm_tiles = sm_tiles.astype(BF)

exp = np.zeros((len(tiles), HD), np.float32)
exp[:, :SPAD] = probs

body = (
    '#include <stdint.h>\n'
    '#include <aie_api/aie.hpp>\n'
    '#include "../../softmax/softmax.cc"\n'
    'static constexpr int VL = 16;\n'
    'static constexpr int SPAD = 16;\n'
    'extern "C" void prefill_attn_row(bfloat16 *qm_row, bfloat16 *kv, float *ctx_row) {\n'
    '  (void)kv;\n'
    '  alignas(64) float s[SPAD];\n'
    '  alignas(64) float p[SPAD];\n'
    '  for (int j = 0; j < SPAD; j++) s[j] = (float)qm_row[j];\n'
    '  route_b_bricks::softmax_core<VL>(s, p, SPAD);\n'
    '  for (int vi = 0; vi < (int)PREFILL_HD / VL; vi++)\n'
    '    aie::store_v(ctx_row + vi * VL, aie::zeros<float, VL>());\n'
    '  for (int j = 0; j < SPAD; j++) ctx_row[j] = p[j];\n'
    '}\n')
cc = GEN / "prefill_sm_isolated.cc"
cc.write_text(body)

res = bricklib.verify_streamed(
    name="prefill_sm_isolated", shim=cc, symbol="prefill_attn_row",
    in_tiles=sm_tiles, out_tile_numel=HD, resident=kv_resident,
    unpack=lambda d: d, golden=exp, gate=V.GATE,
    in_dt=BF, out_dt=np.float32, resident_dt=BF, resident_depth=1,
    compile_flags=[f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}"],
)
got = np.asarray(res.get("got"), np.float32)[:, :SPAD]

np.set_printoptions(precision=4, suppress=True, linewidth=200)
rel = res.get("rel_l2")
print(f"\nISOLATED softmax_core: status={res.get('status')} rel_l2={rel:.3e} "
      f"non-finite {int((~np.isfinite(got)).sum())}/{got.size}", flush=True)
for r in (0, 1, M - 1):
    print(f"\n--- row {r} ---")
    print("  got :", got[r])
    print("  exp :", probs[r])
print("\ndevice argmax:", got.argmax(axis=1))
print("numpy  argmax:", probs.argmax(axis=1))
print("device lane0 :", got[:, 0])
print("numpy  lane0 :", probs[:, 0])
