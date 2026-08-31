#!/usr/bin/env python3
"""Is row_scores correct only when read SCALAR-wise? (VEC_EMIT=1 makes emit read it as softmax does.)

The stage bisect's green stages 1/2 prove row_scores is right when emit_stage copies it with a
SCALAR loop. softmax_core reads the same buffer with ONE aie::load_v<16>. So "stage 2 green, stage 3
red" does NOT isolate the callee -- it also changes scalar-read to vector-read. This runs the SAME
stages with emit reading via load_v. If stage 2 now FAILS, the buffer is bad for vector loads and
softmax_core is not the variable.

Original docstring follows.

Which STAGE of prefill_attn.cc first goes non-finite on device?

WHY. The rail is exonerated -- two copy-only kernels with prefill_attn_row's exact 3-buffer
signature, through the SAME verify_streamed call the failing gate uses, both return 0.000e+00.
Combined with the M sweep (M=1 hangs,
M>=2 all NaN, so not row-count) the fault is inside the kernel body. This finds WHICH stage.

RESOLVED 2026-08-31: prefill-attn is device-green -- 32/32 heads, worst rel-L2 8.540e-08 --
once the aie.core stack reservation was raised past its 0x400 default. This probe is kept as
the stage-bisect instrument, not as a description of current state.

HOW. Compiles the REAL kernel with -DPREFILL_STOP_AFTER=N, which short-circuits after stage N and
writes that stage's SPAD-wide intermediate into ctx_row. Instrumenting the shipped source (rather
than re-implementing the stages in a probe shim) is deliberate: during the rope-lut hunt a
reimplementation worked while the kernel did not, so a probe that rewrites the code under test can
exonerate a stage that is actually broken.

  1 = after the bf16 dot products over resident K
  2 = after the additive mask
  3 = after softmax_core
  4 = full kernel (V-weighted accumulate) -- the shipped path, expected NaN

Read the FIRST stage with non-finite output; everything downstream of it is a consequence.

Run:  ./run.sh probe_prefill_stage_bisect.py
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
import os
VEC = int(os.environ.get("VEC_EMIT", "0"))
DIRECT = int(os.environ.get("SM_OUT_DIRECT", "0"))
INSTR = int(os.environ.get("SM_IN_STREAMED", "0"))
ROUND = int(os.environ.get("SET_ROUNDING", "0"))
WARM = int(os.environ.get("SM_WARM", "0"))
KERNEL = Path(__file__).resolve().parents[1] / "prefill-attn" / "prefill_attn.cc"

_ar, q_full, k_full, v_full = golden.build_qkv(seed=0, m_tokens=M)
in_tiles, kv_resident, h_kv = golden.pack_head_rows(q_full, k_full, v_full, HEAD)

# CPU reference for each stage, from the same packed tiles the device gets.
tiles = np.asarray(in_tiles, np.float32)          # [rows, HD+SPAD]
q_rows = tiles[:, :HD]
mask_rows = tiles[:, HD:HD + SPAD]
kvf = np.asarray(kv_resident, np.float32)
K = kvf[:M * HD].reshape(M, HD)
scale = 1.0 / np.sqrt(float(HD))

scores = np.zeros((len(tiles), SPAD), np.float32)
scores[:, :M] = (q_rows @ K.T) * scale
masked = scores + mask_rows
mx = masked.max(axis=1, keepdims=True)
e = np.exp(masked - mx)
probs = e / e.sum(axis=1, keepdims=True)

STAGES = {
    1: ("after dot products ", scores),
    2: ("after additive mask", masked),
    3: ("after softmax_core ", probs),
}

print(f"===== prefill-attn stage bisect, head {HEAD} (HD={HD} M={M} SPAD={SPAD}, gate {V.GATE:.1e}) "
      f"=====", flush=True)

for stop, (label, ref) in STAGES.items():
    exp = np.zeros((len(tiles), HD), np.float32)
    exp[:, :SPAD] = ref
    try:
        res = bricklib.verify_streamed(
            name=f"prefill_stage{stop}",
            shim=KERNEL,
            symbol="prefill_attn_row",
            in_tiles=in_tiles,
            out_tile_numel=HD,
            resident=kv_resident,
            unpack=lambda d: d,
            golden=exp,
            gate=V.GATE,
            in_dt=BF, out_dt=np.float32, resident_dt=BF,
            resident_depth=1,
            compile_flags=[f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}",
                           f"-DPREFILL_STOP_AFTER={stop}", f"-DPREFILL_VEC_EMIT={VEC}", f"-DPREFILL_SM_OUT_DIRECT={DIRECT}", f"-DPREFILL_SM_IN_STREAMED={INSTR}", f"-DPREFILL_SET_ROUNDING={ROUND}", f"-DPREFILL_SM_WARM={WARM}"],
        )
        got = np.asarray(res.get("got"), np.float32)
        nf = int((~np.isfinite(got)).sum())
        rel = res.get("rel_l2")
        rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
        # only the first SPAD lanes carry the intermediate; the tail is a zero-fill control
        head_nf = int((~np.isfinite(got[:, :SPAD])).sum())
        tail_nz = int((np.abs(got[:, SPAD:]) > 0).sum())
        print(f"stop={stop} {label} status={res.get('status')} rel_l2={rel_s} "
              f"non-finite {nf}/{got.size} (payload {head_nf}/{got[:, :SPAD].size}) "
              f"tail-nonzero {tail_nz}", flush=True)
    except Exception as e:
        print(f"stop={stop} {label} ERROR {type(e).__name__}: {str(e)[:140]}", flush=True)
