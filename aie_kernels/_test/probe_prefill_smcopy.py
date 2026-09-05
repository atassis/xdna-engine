#!/usr/bin/env python3
"""Is prefill-attn's fault the softmax CALLEE, or that buffer's live range across the call?

Established earlier: the stage bisect puts the first WRONG stage at softmax_core (finite, rel-L2
1.042), softmax_core is CLEAN in isolation on the same masked scores (2.825e-04), and `emit_stage`
reads row_scores correctly at the same program point. So the callee is exonerated and what remains is
something about `row_scores` itself across that call.

PREFILL_SM_COPY hands softmax_core a copy in a distinct buffer, leaving row_scores untouched:

  copy CLEAN  => the defect is tied to that buffer's live range, not to softmax, and prefill-attn has
                 a one-line fix rather than needing the from-mha_decode rewrite.
  copy DIRTY  => the buffer is not the discriminator either; the next cut is inline-vs-noinline.

Both the softmax stage (stop=3) and the FULL kernel are gated, because a stage fix that does not
carry to the full kernel is not a fix.

Run:  ./run.sh probe_prefill_smcopy.py
"""
from pathlib import Path

import numpy as np

import bricklib
import verify_prefill_attn as V

golden = V.golden
import ml_dtypes  # noqa: E402

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
exp_full = (probs[:, :M] @ Vm).astype(np.float32)

print(f"===== prefill-attn: pass softmax_core a COPY? (HD={HD} M={M}, gate {V.GATE:.1e}) =====",
      flush=True)
for stop, exp, what in ((3, exp_sm, "softmax stage"), (0, exp_full, "FULL kernel  ")):
    for cp in (0, 1):
        flags = [f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}", f"-DPREFILL_SM_COPY={cp}"]
        if stop:
            flags.append(f"-DPREFILL_STOP_AFTER={stop}")
        try:
            res = bricklib.verify_streamed(
                name=f"prefill_cp{cp}_s{stop}", shim=KERNEL, symbol="prefill_attn_row",
                in_tiles=in_tiles, out_tile_numel=HD, resident=kv_resident,
                unpack=lambda d: d, golden=exp, gate=V.GATE,
                in_dt=BF, out_dt=np.float32, resident_dt=BF, resident_depth=1,
                compile_flags=flags,
            )
            got = np.asarray(res.get("got"), np.float32)
            nf = int((~np.isfinite(got)).sum())
            rel = res.get("rel_l2")
            rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
            lbl = "COPY" if cp else "as-shipped"
            print(f"  {what} {lbl:11s} status={res.get('status')} rel_l2={rel_s} "
                  f"non-finite {nf}/{got.size}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  {what} cp={cp} ERROR {type(ex).__name__}: {str(ex)[:110]}", flush=True)
