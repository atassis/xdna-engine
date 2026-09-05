#!/usr/bin/env python3
"""F3 special device-verify: gatedeltanet (op-TYPE `recurrence{delta,gated}`).

THE HARDEST brick of the wave: a STATE recurrence (not a sliding window). Output
step t needs the full accumulated state S_{t-1}, so there is no time-parallel axis
-- only D_V vectorizes within one step; T is a sequential outer loop. See the
gatedeltanet.cc header for the math.

WHY THIS IS DEVICE-GATED (author best-effort; do NOT run on CPU/import):
  * MANY buffers -- k,v,q (bf16), gates (f32), s_in (f32) IN; o (bf16) + s_out
    (f32) OUT. The core tile has 2-in/2-out DMA channels, and this rail
    (bricklib._build_oneshot) is device-proven at <=2 inputs / 1 output. So we
    PACK: input#0 = [k | v | q | gates] byte-concatenated (split by byte offset in
    the shim, exactly like verify_f2b packs [B | scale] into one weight buffer);
    input#1 = s_in (recurrent state seed, f32). The SINGLE output buffer is the
    bf16 readout `o`. The updated state `s_out` is written to a shim-local scratch
    buffer and is NOT read back here -- verifying it needs a second probe (see
    "DEVICE-GATED RISKS" at the bottom).

WHICH GOLDEN IS THE GATE: the on-chip datapath loads k/v/q as bf16, upcasts to
f32, carries S and the gates in f32 (accfloat, un-rounded across steps), and does
ONE bf16 round on the per-step output store. That is EXACTLY golden.kernel_model
(NOT host_reference, which is full-fp32 k/v/q). So the DEVICE gate compares the
device readout against kernel_model's [T,DV] output. The CPU cross-check in
__main__ additionally asserts kernel_model-vs-host_reference stays within the
golden's own T=64 gate (3e-2), i.e. the model we gate against is itself faithful.
"""
import importlib.util
from pathlib import Path

import numpy as np
import ml_dtypes

BRICKS = Path(__file__).parent.parent
BRICK = "gatedeltanet"
BRICK_DIR = BRICKS / BRICK
BRICK_CC = str(BRICK_DIR / "gatedeltanet.cc")

# Tile dims -- MUST match the GDN_DK/GDN_DV/GDN_T #ifndef defaults in
# gatedeltanet.cc (32/32/64) AND golden.py's main() default, so no -D flags are
# needed and the packed byte offsets below line up with the kernel's loads.
GDN_DK, GDN_DV, GDN_T = 32, 32, 64


def _load_golden():
    p = BRICK_DIR / "golden.py"
    spec = importlib.util.spec_from_file_location("gatedeltanet_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def do_gatedeltanet():
    """DEVICE-run verify config (do NOT call at import/CPU time -- triggers an
    iron.jit device build). Packs k|v|q|gates into one input, s_in into a second,
    runs the recurrence on aie2p, gates the bf16 readout vs kernel_model."""
    import bricklib  # lazy: importing aie.iron is device-toolchain-only

    g = _load_golden()
    T, DK, DV = GDN_T, GDN_DK, GDN_DV
    rng = np.random.default_rng(0)
    k, v, q, gates = g.make_inputs(rng, T, DK, DV)

    # Golden = the model the on-chip datapath actually implements (bf16 io,
    # f32 state, one bf16 round on store). s_in seed is all-zero, matching
    # kernel_model's S=zeros for a fresh sequence.
    ker, _s_ker = g.kernel_model(k, v, q, gates, DK, DV)  # ([T,DV] f32, [DK,DV] f32)

    # Round k/v/q to bf16 so the device's bf16 loads reproduce kernel_model's
    # internal bf16() rounding bit-for-bit, then byte-view for the packed buffer.
    kb = np.ascontiguousarray(k).astype(ml_dtypes.bfloat16)
    vb = np.ascontiguousarray(v).astype(ml_dtypes.bfloat16)
    qb = np.ascontiguousarray(q).astype(ml_dtypes.bfloat16)
    gf = np.ascontiguousarray(gates).astype(np.float32)
    s_in = np.zeros((DK, DV), dtype=np.float32)

    # Byte layout of input#0 (all offsets 64-byte aligned; each kernel row-load
    # &k[t*DK] etc. lands on a 64-byte boundary -> aligned aie::load_v):
    k_off = 0                       # k: T*DK*2 bytes
    v_off = k_off + T * DK * 2      # v: T*DV*2 bytes
    q_off = v_off + T * DV * 2      # q: T*DK*2 bytes
    g_off = q_off + T * DK * 2      # gates: T*2*4 bytes
    packed = np.concatenate([
        kb.reshape(-1).view(np.int8),
        vb.reshape(-1).view(np.int8),
        qb.reshape(-1).view(np.int8),
        gf.reshape(-1).view(np.int8),
    ])

    # Pure-buffer verify-shim: split the packed input by byte offset, hand a
    # shim-local static scratch for the (unverified-here) updated state s_out,
    # and write the bf16 readout to the single output buffer.
    shim = (
        'extern "C" void gdn_verify(int8_t* packed, float* s_in, bfloat16* o) {\n'
        f'  bfloat16* k = (bfloat16*)(packed + {k_off});\n'
        f'  bfloat16* v = (bfloat16*)(packed + {v_off});\n'
        f'  bfloat16* q = (bfloat16*)(packed + {q_off});\n'
        f'  float*    g = (float*)(packed + {g_off});\n'
        '  static float s_out_scratch[GDN_DK * GDN_DV];\n'
        '  gatedeltanet_step(k, v, q, g, s_in, o, s_out_scratch);\n'
        '}\n'
    )

    return bricklib.verify_oneshot(
        name="gatedeltanet",
        brick_cc=BRICK_CC,
        shim_body=shim,
        symbol="gdn_verify",
        inputs=[(packed, np.int8), (s_in.reshape(-1), np.float32)],
        out_numel=T * DV,
        out_shape=(T, DV),
        unpack=lambda flat: np.asarray(flat).reshape(T, DV).astype(np.float32),
        golden=ker.astype(np.float32),
        gate=3e-2,
        compile_flags=[],           # GDN_DK/DV/T defaults (32/32/64) already match
        out_dt=ml_dtypes.bfloat16,
    )
do_gatedeltanet.brick_name = "gatedeltanet"


if __name__ == "__main__":
    # CPU-ONLY golden cross-check. Does NOT touch the device (no iron.jit / no
    # do_gatedeltanet()): recompute both reference models in numpy, assert the
    # readout is well-formed, and confirm the model we gate the DEVICE against
    # (kernel_model) is itself within the golden's own T=64 gate of the fp32
    # reference. This is the sanity net that the device gate is meaningful.
    g = _load_golden()
    T, DK, DV = GDN_T, GDN_DK, GDN_DV
    rng = np.random.default_rng(0)
    k, v, q, gates = g.make_inputs(rng, T, DK, DV)

    ref, s_ref = g.host_reference(k, v, q, gates, DK, DV)      # full fp32
    ker, s_ker = g.kernel_model(k, v, q, gates, DK, DV)        # bf16 io / f32 state

    assert ker.shape == (T, DV), f"readout shape {ker.shape} != {(T, DV)}"
    assert np.isfinite(ker).all(), "kernel_model readout has non-finite values"
    assert np.isfinite(ref).all(), "host_reference readout has non-finite values"
    assert np.isfinite(s_ker).all(), "kernel_model state has non-finite values"

    rl2 = g.rel_l2(ker, ref)
    s_rl2 = g.rel_l2(s_ker, s_ref)
    gate = 3e-2
    assert rl2 <= gate, f"kernel_model vs host_reference readout rel-L2 {rl2:.3e} > gate {gate:.1e}"

    print(f"[gatedeltanet cpu-xcheck] T={T} DK={DK} DV={DV}  "
          f"readout rel-L2(kernel_model,host_reference)={rl2:.3e}  "
          f"state rel-L2={s_rl2:.3e}  shape={ker.shape}  "
          f"-> PASS (gate={gate:.0e})")
