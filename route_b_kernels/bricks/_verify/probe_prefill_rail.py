#!/usr/bin/env python3
"""Is prefill-attn's NaN the verify_streamed RAIL, or the kernel body?

WHY. prefill-attn is NaN on all 32 heads and at every M>=2 (M=1 hangs), while its CPU self-checks
all pass with teeth, so the fault is device-side and not row-count-dependent.
Before bisecting a large attention
kernel, clear or implicate the DATA PATH it rides on -- the same move that exonerated the rail
during the rope-lut hunt and saved chasing the kernel for nothing.

Two copy-only kernels with prefill_attn_row's EXACT three-buffer signature (2 in + 1 out, no scalar,
the arity=3 resident-operand contract), run through the SAME bricklib.verify_streamed call the real
gate uses -- streamed [q_row|mask_row] tiles in, resident KV held at depth=1, streamed ctx_row out:

  R1 streamed->out : ctx_row[i] = qm_row[i]   -- tests the streamed input and the output stream
  R2 resident->out : ctx_row[i] = kv[i]       -- tests that the RESIDENT operand is delivered

Both clean  => the rail moves data correctly and the fault is inside prefill_attn.cc's body.
R2 dirty    => the resident KV operand is the problem, and the kernel body is not implicated at all.

Run:  ./run.sh probe_prefill_rail.py
"""
from pathlib import Path

import ml_dtypes
import numpy as np

import bricklib
import verify_prefill_attn as V

golden = V.golden
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)
BF = ml_dtypes.bfloat16
HD, M, HEAD = golden.HD, golden.M, 0

_ar, q_full, k_full, v_full = golden.build_qkv(seed=0, m_tokens=M)
in_tiles, kv_resident, h_kv = golden.pack_head_rows(q_full, k_full, v_full, HEAD)

ARMS = {
    "R1 streamed->out": (
        'extern "C" void prefill_attn_row(bfloat16 *qm_row, bfloat16 *kv, float *ctx_row) {\n'
        '  (void)kv;\n'
        '  for (unsigned i = 0; i < (unsigned)PREFILL_HD; ++i) ctx_row[i] = (float)qm_row[i];\n'
        '}\n',
        # one output row per streamed tile: each tile's first HD entries are that row's q
        np.asarray(in_tiles, np.float32)[:, :HD]),
    "R2 resident->out": (
        'extern "C" void prefill_attn_row(bfloat16 *qm_row, bfloat16 *kv, float *ctx_row) {\n'
        '  (void)qm_row;\n'
        '  for (unsigned i = 0; i < (unsigned)PREFILL_HD; ++i) ctx_row[i] = (float)kv[i];\n'
        '}\n',
        # the resident operand is the same for every call: K row 0 of this head's h_kv
        np.repeat(np.asarray(kv_resident, np.float32)[:HD][None], M, axis=0)),
}

for label, (body, exp) in ARMS.items():
    cc = GEN / f"prefill_rail_{label[:2]}.cc"
    cc.write_text('#include <stdint.h>\n#include <aie_api/aie.hpp>\n' + body)
    try:
        res = bricklib.verify_streamed(
            name=f"prefill_rail_{label[:2]}",
            shim=cc,
            symbol="prefill_attn_row",
            in_tiles=in_tiles,
            out_tile_numel=HD,
            resident=kv_resident,
            unpack=lambda d: d,
            golden=exp,
            gate=V.GATE,
            in_dt=BF, out_dt=np.float32, resident_dt=BF,
            resident_depth=1,
            compile_flags=[f"-DPREFILL_HD={HD}", f"-DPREFILL_M={M}"],
        )
        got = np.asarray(res.get("got"), np.float32)
        nf = int((~np.isfinite(got)).sum())
        print(f"{label:20s} status={res.get('status')} rel_l2={res.get('rel_l2'):.3e} "
              f"non-finite {nf}/{got.size}", flush=True)
    except Exception as e:
        print(f"{label:20s} ERROR {type(e).__name__}: {str(e)[:130]}", flush=True)
