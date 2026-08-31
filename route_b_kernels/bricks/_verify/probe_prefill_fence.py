#!/usr/bin/env python3
"""Does a compiler memory barrier before softmax_core fix prefill-attn?

softmax_core is EXONERATED in isolation (rel_l2 2.825e-04, every argmax matching numpy), yet the
same code is wrong when prefill_attn_row_impl feeds it. The remaining difference is the ordering:
pass 1b does aie::store_v(row_scores, ...) and then hands `row_scores` to softmax_core as a pointer
parameter -- the rope-lut trigger shape from the open load-above-store hunt.

Four arms, at both the softmax stage and the full kernel, so the barrier's effect is attributable:

  stop=3 fence=0 : softmax stage, as shipped        -- known FAIL (rel_l2 1.042)
  stop=3 fence=1 : softmax stage, barrier inserted
  stop=0 fence=0 : full kernel, as shipped          -- known NaN
  stop=0 fence=1 : full kernel, barrier inserted

Barrier turns them green => the defect is store/load ORDERING across the call, and this is a far
smaller reproducer than rope-lut for the upstream hunt.
Barrier changes nothing => ordering is refuted here too; the difference is something else about
the call site, and this becomes the sixth refuted mechanism rather than a cause.

Run:  ./run.sh probe_prefill_fence.py
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
Vm = kvf[M * HD:2 * M * HD].reshape(M, HD)
scores = np.zeros((len(tiles), SPAD), np.float32)
scores[:, :M] = (q_rows @ K.T) / np.sqrt(float(HD))
masked = scores + mask_rows
e = np.exp(masked - masked.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)

exp_sm = np.zeros((len(tiles), HD), np.float32)
exp_sm[:, :SPAD] = probs
exp_full = (probs[:, :M] @ Vm).astype(np.float32)   # ctx_row = sum_j probs[j] * V[j]

print(f"===== prefill-attn fence probe, head {HEAD} (HD={HD} M={M}, gate {V.GATE:.1e}) =====",
      flush=True)

for stop, exp, what in ((3, exp_sm, "softmax stage"), (0, exp_full, "full kernel ")):
    for fence in (0, 1):
        flags = [f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}", f"-DPREFILL_SCALAR_MASK={fence}"]
        if stop:
            flags.append(f"-DPREFILL_STOP_AFTER={stop}")
        try:
            res = bricklib.verify_streamed(
                name=f"prefill_fence_s{stop}_f{fence}", shim=KERNEL, symbol="prefill_attn_row",
                in_tiles=in_tiles, out_tile_numel=HD, resident=kv_resident,
                unpack=lambda d: d, golden=exp, gate=V.GATE,
                in_dt=BF, out_dt=np.float32, resident_dt=BF, resident_depth=1,
                compile_flags=flags,
            )
            got = np.asarray(res.get("got"), np.float32)
            nf = int((~np.isfinite(got)).sum())
            rel = res.get("rel_l2")
            rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
            print(f"{what} scalar_mask={fence}  status={res.get('status')} rel_l2={rel_s} "
                  f"non-finite {nf}/{got.size}", flush=True)
        except Exception as ex:
            print(f"{what} scalar_mask={fence}  ERROR {type(ex).__name__}: {str(ex)[:130]}", flush=True)
