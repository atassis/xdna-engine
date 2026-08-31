#!/usr/bin/env python3
"""Does prefill-attn's device NaN depend on the number of streamed query rows?

WHY. verify_prefill_attn's main gate is NaN on ALL 32 heads (rel_l2=nan, nz=nan, run2run=nan --
the device output is entirely non-finite, not merely wrong), while every CPU self-check passes
WITH TEETH (causal-vs-bidirectional 1.492, GQA repeat-interleave-vs-tile 1.377). So the golden and
the test data are sound and the fault is device-side, uniform across heads.

The gate asserts on the M=11 case before its own M=1 degenerate check ever reaches the device, so
"is M=1 clean?" is unanswered -- and it is the discriminating question. This brick's own header
records its Revision 1 internal-loop failure as "green at one iteration, red at two", and Revision 3
moved the volume into bricklib.verify_streamed's WORKER loop specifically to escape that. If NaN
now tracks the row count again, the escape did not work and the defect follows volume wherever it
is expressed. If M=1 is ALSO NaN, the row count is irrelevant and the fault is in the kernel body
or the resident-KV plumbing -- a different hunt entirely.

One head only (head 0): this asks about M, not about GQA, and each M is a full build.

Run:  ./run.sh probe_prefill_attn_m_sweep.py
"""
import numpy as np

import verify_prefill_attn as V

golden = V.golden

M_VALUES = [1, 2, 3, 11]
HEAD = 0

rows = []
for m in M_VALUES:
    try:
        _ar, q, k, v = golden.build_qkv(seed=0, m_tokens=m)
        res = V._verify_head(HEAD, q, k, v, m, name_suffix=f"_sweep{m}")
        rl2 = res.get("rel_l2")
        nz = res.get("nonzero", res.get("nz"))
        rows.append((m, res.get("status"), rl2, nz))
    except Exception as e:
        msg = str(e)
        kind = "BUILD-FAIL" if "exceeded available memory" in msg or "aiecc" in msg else "ERROR"
        rows.append((m, kind, None, None))
        print(f"  M={m}: {kind}: {msg[:160]}", flush=True)

print(f"\n===== prefill-attn M sweep, head {HEAD} (gate {V.GATE:.1e}) =====", flush=True)
for m, status, rl2, nz in rows:
    s = f"{rl2:.3e}" if isinstance(rl2, float) else "--"
    n = f"{nz:.3e}" if isinstance(nz, float) else "--"
    print(f"  M={m:3d}  {str(status):12s} rel_l2={s}  nz={n}", flush=True)

ran = [r for r in rows if isinstance(r[2], float)]
finite = [r for r in ran if np.isfinite(r[2])]
if len(ran) < 2:
    print("\nVERDICT: INCONCLUSIVE -- fewer than two arms reached the device")
elif not finite:
    print("\nVERDICT: NaN at EVERY M including the smallest -- the row count is NOT the axis; "
          "look at the kernel body or the resident-KV plumbing, not the worker loop")
elif len(finite) == len(ran):
    print("\nVERDICT: every arm finite -- the NaN does not reproduce in this configuration; "
          "re-check what differs between this probe and the failing gate before concluding")
else:
    worst = max(finite, key=lambda r: r[2])
    print(f"\nVERDICT: finite up to M={worst[0]} and NaN above -- the fault TRACKS ROW COUNT, so "
          "Revision 3's move into the worker loop did not escape it")
