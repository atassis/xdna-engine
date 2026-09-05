#!/usr/bin/env python3
"""Device-gate rope-lut at PARTIAL rotary (ROPE_ROT < ROPE_D).

verify_rope_lut gates ROT = D = 128, so it measures full rotary only. partial_dim is in the
brick's scope and is the axis Llama/Gemma use, so it needs its own numeric gate rather than
inheriting the full-rotary verdict. Each width is gated against the same quantized golden,
which leaves dims [ROT, D) untouched -- so a kernel that ignored ROT and rotated everything
would fail here while passing verify_rope_lut.

ROT candidates are what the kernel's `kRotHalf % kVec` static_assert admits (ROT % 32 == 0).
ROT=64 additionally exercises Xilinx/llvm-aie#1155, the reg-offset fallback for non-encodable
VST_dmx_sts_x_spill frame offsets, carried in the pinned Peano. A build failure is caught and
reported per-shape instead of ending the sweep.
"""
import sys

import numpy as np
import ml_dtypes

import bricklib
import verify_rope_lut as V

ROTS = (128, 96, 64, 32)


def run_one(rot):
    """Build + run rope-lut at this rotary width; return (rel_l2, nonzero) or raise."""
    g, cc = V.load_golden()
    rng = np.random.default_rng(0)
    qk_bf16 = rng.standard_normal((V.M, V.D)).astype(np.float32).astype(ml_dtypes.bfloat16)
    pos = np.arange(V.M, dtype=np.int32)
    inv_freq = g.build_inv_freq(rot).astype(np.float32)
    cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)

    exp = g.rope_reference_quantized(qk_bf16.astype(np.float32), pos, rot)

    return bricklib.verify_oneshot(
        f"rope-lut-rot{rot}", cc, V.SHIM_BODY, "rope_lut_verify",
        inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cbuf, np.int32)],
        out_numel=V.M * V.D, out_shape=(V.M, V.D),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(V.M, V.D),
        golden=exp, gate=V.GATE, out_dt=ml_dtypes.bfloat16,
        compile_flags=[f"-DROPE_D={V.D}", f"-DROPE_ROT={rot}", f"-DROPE_M={V.M}",
                       f"-DROPE_SCALE_INV={V.SCALE_INV:.1f}f"])


def main():
    verdicts = {}
    bad = []
    for rot in ROTS:
        try:
            res = run_one(rot)
            verdicts[rot] = f"rel_l2={res['rel_l2']:.3e} -> {'PASS' if res['ok'] else 'FAIL'}"
            if not res["ok"]:
                bad.append(rot)
        except Exception as e:  # a backend crash is a result here, not a run-ender
            verdicts[rot] = f"BUILD/RUN ERROR: {type(e).__name__}: {str(e)[:160]}"
            bad.append(rot)
        print(f"ROT={rot:4d}  {verdicts[rot]}", flush=True)

    print("\n===== partial-rotary summary (gate %.1e) =====" % V.GATE)
    for rot, v in verdicts.items():
        print(f"  ROT={rot:4d}  D={V.D}  {'FULL  ' if rot == V.D else 'PARTIAL'}  {v}")

    # Without this the sweep printed FAIL and still exited 0, so no drain could fail on it.
    # A build/run error counts as a failure: an un-buildable width is not a passing width.
    if bad:
        print(f"FAIL partial-rotary: {len(bad)} of {len(ROTS)} widths bad -> {bad}")
        return 1
    print(f"PASS partial-rotary: all {len(ROTS)} widths within {V.GATE:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
