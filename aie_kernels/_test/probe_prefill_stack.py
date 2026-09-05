#!/usr/bin/env python3
"""Is prefill_attn's failure the same stack-frame overrun, at a bigger frame?

probe_stack_reservation.py confirmed the mechanism to the byte on a synthetic arm: fill steps
0.5 -> 1.0 exactly between reservations 0x700 and 0x740, and the zero-frame control passes at every
reservation. prefill_attn is the other kernel on this thread, and the code-volume correlation that
led here was found on it.

But its symptom is NOT the same. The synthetic arm loses half its output TILES and the survivors
are bit-exact; prefill_attn returns wrong VALUES in completed tiles (stage 3 wrong-but-finite,
full kernel NaN at 1408/1408). So this is a real test, not a formality: same cause would mean a
frame merely larger than the synthetic one, and raising the reservation far enough must turn it
green. Anything else -- green at no reservation, or a partial improvement -- says prefill_attn is a
different defect that happened to share a code-size correlation.

Head 0 only, M=11, one build per reservation. The full 32-head gate is only worth running if a
reservation here goes green.

Run:  ./run.sh probe_prefill_stack.py
"""
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

import bricklib

HERE = Path(__file__).parent.resolve()
BRICK_DIR = (HERE.parent / "prefill-attn").resolve()
BRICK_CC = BRICK_DIR / "prefill_attn.cc"
sys.path.insert(0, str(BRICK_DIR))
import golden  # noqa: E402

GATE = 3e-2
_bf16 = ml_dtypes.bfloat16
HEAD = 0

# 0x400 is the default (the failing condition). The synthetic arm needed 0x740; prefill_attn holds
# row_scores[16] + row_probs[16] + acc_row[128] as alignas(64) locals, so if spills scale with that
# the frame could be several KB. 0x4000 is 16 KB -- far past any plausible frame, and the point of
# including it is that if even THAT is red, frame size is not the variable at all.
STACKS = [0x400, 0x800, 0xD00, 0x1800, 0x4000]

ar_ref, q_full, k_full, v_full = golden.build_qkv(seed=0, m_tokens=golden.M)
in_tiles, kv_resident, h_kv = golden.pack_head_rows(q_full, k_full, v_full, HEAD, mask=None)
exp = golden.reference_direct(q_full, k_full, v_full, HEAD)

print(f"prefill_attn head {HEAD}, M={golden.M}, HD={golden.HD}, gate {GATE:.1e}")
print(f"{'stack':>8} {'rel-L2':>12} {'fill':>8} {'non-finite':>12} {'run2run':>10}  verdict")
rows = {}
for st in STACKS:
    try:
        res = bricklib.verify_streamed(
            name=f"prefill_stack_h{HEAD}_{st:x}",
            shim=BRICK_CC, symbol="prefill_attn_row",
            in_tiles=in_tiles, out_tile_numel=golden.HD, resident=kv_resident,
            unpack=lambda d: d, golden=exp, gate=GATE,
            in_dt=_bf16, out_dt=np.float32, resident_dt=_bf16,
            resident_depth=1,
            compile_flags=[f"-DPREFILL_HD={golden.HD}", f"-DPREFILL_M={golden.M}"],
            stack_size=st)
        got = np.asarray(res["got"], np.float64)
        fill = float((got != 0).mean())
        nf = int((~np.isfinite(got)).sum())
        rows[st] = (res["rel_l2"], fill, nf, res["ok"])
        print(f"{st:>#8x} {res['rel_l2']:12.3e} {fill:8.4f} {nf:>7}/{got.size} "
              f"{res['run2run']:10.2e}  {'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{st:>#8x} ERROR {type(ex).__name__}: {str(ex)[:110]}")

print()
green = [s for s in STACKS if rows.get(s, (1, 0, 0, False))[3]]
if green:
    print(f"SAME MECHANISM: prefill_attn goes green at stack_size {min(green):#x}. Its frame is "
          f"simply\n         larger than the synthetic arm's 0x740. Re-gate all 32 heads there.")
else:
    base = rows.get(0x400)
    big = rows.get(max(STACKS))
    if base and big and big[0] < base[0] * 0.5:
        print("PARTIAL: no reservation is green, but the error dropped substantially at the largest."
              "\n         The frame is A factor, not THE factor -- do not close this on it.")
    else:
        print("DIFFERENT DEFECT: raising the reservation to 16 KB does not move prefill_attn."
              "\n         It shares the code-size correlation with the synthetic arm but not the"
              "\n         cause. The stack finding does NOT explain prefill-attn; keep it open.")
