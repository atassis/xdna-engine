#!/usr/bin/env python3
"""Does rope-lut's unwritten hole follow the FIRST LOOP ITERATION or the FIRST 64 OUTPUT LANES?

WHY. Under the PINNED aie_api, probe_rope_identity at pos=0 returns 1984/2048 bit-exact with every
missing element in ROW 0, COLUMNS 0..63, value exactly 0.0 -- never written. Under the WHEEL it is
2048/2048. The apply loop stores TWICE per (m, i):

    ::aie::store_v(row + i,             out1);   // columns 0..kRotHalf-1   <- this one is missing
    ::aie::store_v(row + kRotHalf + i,  out2);   // columns kRotHalf..D-1   <- this one lands

So for m=0 the FIRST store does not land and the SECOND does, and for m>=1 both land. Two readings
fit that, and they imply different bugs:

  A. LOOP-PROLOGUE: the first iteration of the m loop is special (peeled, pipelined differently),
     and its first store is dropped. Predicts the hole stays on row 0 at EVERY ROPE_M, including
     ROPE_M=1 where the loop has a single iteration.
  B. FIRST-LANES: the first 64 output lanes of the buffer are the casualty regardless of loop
     structure (an alignment or first-store lowering issue). Also predicts row 0 -- but it should
     then also appear at ROPE_M=1.

Both predict a hole at ROPE_M=1, so ROPE_M=1 does NOT discriminate them. What discriminates is
whether the hole scales: if it is loop-prologue, the damaged region stays exactly one row's first
half no matter how many rows there are. This sweep records the damaged (row, half) set per M so the
shape is on the record either way, and settles the prior question the kernel header raises -- its own
history recorded the INVERSE pattern (rows 1+ unwritten, row 0 fine, ROPE_M=1 bit-exact).

Run under a PIN arm (the wheel is clean, so a wheel run is the control):
    arm3.sh pin probe_rope_identity_m_sweep.py
"""
import numpy as np
import ml_dtypes

import bricklib
import verify_rope_lut as V

D = V.D
ROT = V.ROT
HALF = ROT // 2


def run_m(m):
    g, cc = V.load_golden()
    rng = np.random.default_rng(0)
    qk_f32 = rng.standard_normal((m, D)).astype(np.float32)
    qk_bf16 = qk_f32.astype(ml_dtypes.bfloat16)
    qk_in = qk_bf16.astype(np.float32)
    inv_freq = g.build_inv_freq(ROT).astype(np.float32)
    pos = np.zeros(m, dtype=np.int32)
    cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)

    flags = [f"-DROPE_D={D}", f"-DROPE_ROT={ROT}", f"-DROPE_M={m}"] + \
            [f for f in V.COMPILE_FLAGS if not f.startswith(("-DROPE_D", "-DROPE_ROT", "-DROPE_M"))]
    res = bricklib.verify_oneshot(
        f"rope-identity-m{m}", cc, V.SHIM_BODY, "rope_lut_verify",
        inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cbuf, np.int32)],
        out_numel=m * D, out_shape=(m, D),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(m, D),
        golden=qk_in, gate=1e9, out_dt=ml_dtypes.bfloat16, compile_flags=flags)
    got = res.get("got")
    if got is None:
        return None
    got = np.asarray(got, np.float32).reshape(m, D)
    bad = got != qk_in
    # A lane is "unwritten" when it is exactly 0.0 while the input there was not.
    unwritten = (got == 0.0) & (qk_in != 0.0)
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    halves = []
    for r in rows:
        lo = bool(bad[r, :HALF].any())
        hi = bool(bad[r, HALF:].any())
        halves.append(f"r{r}:{'first' if lo else ''}{'+' if lo and hi else ''}{'second' if hi else ''}")
    return dict(m=m, exact=int((got == qk_in).sum()), total=got.size,
                rows=rows, halves=halves, unwritten=int(unwritten.sum()))


for m in (1, 2, 4, 16):
    try:
        r = run_m(m)
    except Exception as e:
        print(f"M={m:3d}  ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        continue
    if r is None:
        print(f"M={m:3d}  no device output", flush=True)
        continue
    print(f"M={r['m']:3d}  exact {r['exact']:5d}/{r['total']:5d}  unwritten-lanes {r['unwritten']:4d}  "
          f"damaged rows {r['rows']}  {' '.join(r['halves'])}", flush=True)
