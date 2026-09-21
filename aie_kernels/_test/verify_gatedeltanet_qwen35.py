#!/usr/bin/env python3
"""gatedeltanet at Qwen3.5-4B head dims, with the recurrent state carried across two dispatches.

DK=128 is the model's key dim and the axis the internal-loop pressure hypothesis names
(probe_gatedeltanet_dim_sweep.py), so it runs full width. DV is a 16-wide slice of the 128-wide
value head: the one-shot harness double-buffers every fifo, and a 32-wide slice with s_out read back
does not fit L1. Inputs follow the model: k and q L2-normalised, q scaled by DK^-0.5, alpha over
the decay range exp(-exp(A_log) * softplus(.)) actually reaches.

The output buffer is [o bf16 T*DV | s_out f32 DK*DV]. Dispatch 2 takes dispatch 1's DEVICE s_out as
its s_in, so the second gate covers the state hand-off, not just one call.
"""
import importlib.util
import sys
from pathlib import Path

import ml_dtypes
import numpy as np

BRICK_DIR = Path(__file__).parent.parent / "gatedeltanet"
BRICK_CC = str(BRICK_DIR / "gatedeltanet.cc")
DK, DV = 128, 16
# The pre-2026-09-21 recurrence (error read from the undecayed state) sits 2.1e-2..4.1e-2 from this
# one on these inputs, under the 3e-2 the other bricks use; 1e-3 separates them, the state gate too.
OUT_GATE, STATE_GATE = 1e-3, 1e-4
# aiecc measures the core frame and refuses to build below it; 2560 is its measurement at these dims.
STACK = 2560


def _golden():
    spec = importlib.util.spec_from_file_location("gdn_golden", BRICK_DIR / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def model_inputs(rng, T):
    k = rng.standard_normal((T, DK)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q = rng.standard_normal((T, DK)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= DK ** -0.5
    v = rng.standard_normal((T, DV)).astype(np.float32)
    alpha = rng.uniform(0.3, 0.999, size=T).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, size=T).astype(np.float32)
    return k, v, q, np.stack([alpha, beta], -1)


def dispatch(bricklib, T, k, v, q, gates, s_in):
    bf = ml_dtypes.bfloat16
    packed = np.concatenate([np.ascontiguousarray(a).astype(bf).reshape(-1).view(np.int8)
                             for a in (k, v, q)] + [gates.astype(np.float32).reshape(-1).view(np.int8)])
    k_off, v_off = 0, T * DK * 2
    q_off = v_off + T * DV * 2
    g_off = q_off + T * DK * 2
    s_off = T * DV * 2
    shim = (
        'extern "C" void gdn_q35(int8_t* packed, float* s_in, int8_t* out) {\n'
        f'  gatedeltanet_step((bfloat16*)(packed + {k_off}), (bfloat16*)(packed + {v_off}),\n'
        f'                    (bfloat16*)(packed + {q_off}), (float*)(packed + {g_off}),\n'
        f'                    s_in, (bfloat16*)out, (float*)(out + {s_off}));\n'
        '}\n'
    )

    def unpack(flat):
        b = np.asarray(flat, np.int8)
        o = b[:s_off].view(bf).astype(np.float32)
        s = b[s_off:].view(np.float32)
        return np.concatenate([o, s])

    g = _golden()
    o_ref, s_ref = g.kernel_model(k, v, q, gates, DK, DV, S0=s_in)
    r = bricklib.verify_oneshot(
        name=f"gdn_q35_T{T}", brick_cc=BRICK_CC, shim_body=shim, symbol="gdn_q35",
        inputs=[(packed, np.int8), (s_in.reshape(-1).astype(np.float32), np.float32)],
        out_numel=s_off + DK * DV * 4, out_shape=None, unpack=unpack,
        golden=np.concatenate([o_ref.reshape(-1), s_ref.reshape(-1)]), gate=OUT_GATE,
        compile_flags=[f"-DGDN_DK={DK}", f"-DGDN_DV={DV}", f"-DGDN_T={T}"],
        out_dt=np.int8, stack_size=STACK)
    got = r["got"]
    o_dev, s_dev = got[:T * DV].reshape(T, DV), got[T * DV:].reshape(DK, DV)
    return r, o_dev, s_dev.astype(np.float32), o_ref, s_ref


def run(T):
    import bricklib
    g = _golden()
    rng = np.random.default_rng(T)
    k, v, q, gates = model_inputs(rng, 2 * T)
    h = slice(0, T), slice(T, 2 * T)
    ok = True
    s_in = np.zeros((DK, DV), np.float32)
    for n, sl in enumerate(h, 1):
        r, o_dev, s_dev, o_ref, s_ref = dispatch(bricklib, T, k[sl], v[sl], q[sl], gates[sl], s_in)
        ro, rs = g.rel_l2(o_dev, o_ref), g.rel_l2(s_dev, s_ref)
        good = r["ok"] and ro <= OUT_GATE and rs <= STATE_GATE and np.isfinite(s_dev).all()
        print(f"[gdn-q35 T={T} dispatch {n}] out rel-L2 {ro:.3e} (<= {OUT_GATE:.0e})  "
              f"state rel-L2 {rs:.3e} (<= {STATE_GATE:.0e})  run2run {r['run2run']:.1e}  "
              f"-> {'PASS' if good else 'FAIL'}")
        ok &= good
        s_in = s_dev
    return ok


def main():
    Ts = [int(a) for a in sys.argv[1:]] or [4, 16]
    return 0 if all([run(T) for T in Ts]) else 1


if __name__ == "__main__":
    sys.exit(main())
