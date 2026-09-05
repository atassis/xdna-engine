#!/usr/bin/env python3
"""Device rel-L2 verify config for the rope-lut brick (brick-wave1-device-verify).

rope_lut_prologue rotates a resident [M,D] bf16 Q/K tile IN PLACE via a 256-entry
int8-keyed sin/cos gather-LUT (aie::parallel_lookup). It is a ONE-SHOT kernel: the
`for m` row loop lives inside the kernel, so the whole [M,D] buffer flows through a
single call -> we drive it through bricklib.verify_oneshot (not the per-row rail).

THREE kernel inputs (qk, pos, inv_freq) vs a core tile's 2 input DMA channels, so
pos(int32[M]) + inv_freq(f32[ROT/2]) are PACKED into ONE int32 const buffer and split
in the shim (pos first, inv_freq bit-reinterpreted after). qk is the streamed input;
because the kernel mutates qk in place and the oneshot rail hands us separate in/out
buffers, the shim copies qk_in -> qk_out then rotates qk_out in place -> qk_out is the
rotated result.

Scalars (ROPE_D/ROPE_ROT/ROPE_M/ROPE_SCALE_INV) are baked as compile-time LITERALS via
ExternalFunction compile_flags (-D...); they are macros the .cc consumes at include time,
and the shim body references those same literals. Gate rel-L2 vs rope_reference_quantized
(the bit-faithful device LUT model), NOT the exact np.sin/np.cos reference.

DEVICE-RUN ONLY: do_rope_lut() builds+runs on aie2p. The __main__ block is a CPU-ONLY
golden cross-check and MUST NOT invoke do_rope_lut.
"""
import importlib.util
from pathlib import Path

import numpy as np
import ml_dtypes

BRICKS = Path(__file__).parent.parent

# --- brick geometry (small: qk = M*D*2 = 4KB, comfortably inside 64KB L1) ---
# Full rotary. Partial rotary (ROT < D) is gated by probe_rope_partial_rotary.py,
# which sweeps ROT over the shapes the kernel's %32 static_assert admits.
D = 128       # ROPE_D   head_dim / feature width per Q/K row
ROT = 128     # ROPE_ROT rotary width (== D here: full rotary; ROT<D = partial)
M = 16        # ROPE_M   resident tile rows (tokens) processed this call
SCALE_INV = 1.0   # ROPE_SCALE_INV (1.0 == no NTK linear position scaling)
GATE = 3e-2       # brick threshold; loose vs bf16 eps (~4e-3) + 256-bin LUT

# NOTE on numeric fidelity of the quantized golden: pos is kept SMALL
# (arange(M), max M-1) on purpose. The device computes the wrap+quantize in f32
# while rope_reference_quantized does it in f64; for small theta the f32 error is
# ~1e-6 in radians, far below one 256-bin LUT step (pi/128 ~ 0.0245), so no int8
# key flips -> the two paths agree to bf16 rounding. Large decode positions would
# flip keys and need the exact ref instead.


def load_golden():
    """Import the brick's numpy golden module + return (module, abs .cc path)."""
    cc = BRICKS / "rope-lut" / "rope_lut.cc"
    p = BRICKS / "rope-lut" / "golden.py"
    spec = importlib.util.spec_from_file_location("rope_lut_golden", p)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g, str(cc)


def build_case():
    """Build (qk_bf16, cbuf, exp) shared by the device run and the CPU cross-check.

    qk_bf16 : [M,D] ml_dtypes.bfloat16   device input (rotated in place -> output)
    cbuf    : [M+ROT/2] int32            packed [pos(int32) | inv_freq(f32-bits)]
    exp     : [M,D] float64              rope_reference_quantized(qk, pos, ROT)
    """
    g, _ = load_golden()
    rng = np.random.default_rng(0)
    qk_f32 = rng.standard_normal((M, D)).astype(np.float32)
    qk_bf16 = qk_f32.astype(ml_dtypes.bfloat16)      # what the device actually sees
    qk_ref_in = qk_bf16.astype(np.float32)           # SAME values feed the golden

    pos = np.arange(M, dtype=np.int32)               # decode-style KV offsets
    inv_freq = g.build_inv_freq(ROT).astype(np.float32)   # [ROT/2] host-folded const

    # pack pos + inv_freq into ONE int32 buffer (inv_freq reinterpreted bit-for-bit);
    # the shim splits it back with pointer casts (pos = cbuf, inv_freq = cbuf+M).
    cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)

    exp = g.rope_reference_quantized(qk_ref_in, pos, ROT)  # gate against THIS (LUT model)
    return qk_bf16, cbuf, pos, inv_freq, exp


# --- pure-buffer verify shim: split the packed const, copy qk_in->qk_out, rotate in place ---
# ROTATE FIRST, COPY OUT SECOND -- the order matters and is not cosmetic. The old body copied
# qk_in into qk_out and then rotated qk_out IN PLACE, i.e. a scalar store loop immediately followed
# by a vector load loop over the SAME buffer. Under the pinned aie_api that is miscompiled: the
# rotate's first-iteration loads are scheduled ahead of the copy's stores, so row 0 read the
# buffer's initial zeros (128/256 at ROPE_M=2, bit-exact at ROPE_M=1, clean under the wheel).
# Rotating the INPUT buffer -- which the host has already filled, so no kernel store precedes the
# loads -- and copying out afterwards is the same work and is bit-exact under BOTH header sets.
SHIM_BODY = (
    'extern "C" void rope_lut_verify(bfloat16 *qk_in, int32_t *cbuf, bfloat16 *qk_out) {\n'
    '  const int32_t *pos = cbuf;                       // [ROPE_M] runtime positions\n'
    '  const float *inv_freq = (const float *)(cbuf + ROPE_M);  // [ROPE_ROT/2] host-folded\n'
    '  rope_lut_prologue(qk_in, pos, inv_freq);         // rotates qk_in IN PLACE\n'
    '  for (unsigned i = 0; i < (unsigned)(ROPE_M * ROPE_D); ++i) qk_out[i] = qk_in[i];\n'
    '}\n'
)

COMPILE_FLAGS = [f"-DROPE_D={D}", f"-DROPE_ROT={ROT}", f"-DROPE_M={M}",
                 f"-DROPE_SCALE_INV={SCALE_INV:.1f}f"]


def do_rope_lut():
    """DEVICE-RUN. Build + run rope-lut on aie2p, gate rel-L2 vs the quantized golden.
    Do NOT call at import/CPU time -- this triggers an iron.jit device build."""
    import bricklib  # lazy: keeps __main__ CPU-only (no aie.iron import)
    _, cc = load_golden()
    qk_bf16, cbuf, _pos, _invf, exp = build_case()
    return bricklib.verify_oneshot(
        "rope-lut", cc, SHIM_BODY, "rope_lut_verify",
        inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cbuf, np.int32)],
        out_numel=M * D, out_shape=(M, D),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(M, D),
        golden=exp, gate=GATE, out_dt=ml_dtypes.bfloat16,
        compile_flags=COMPILE_FLAGS)
do_rope_lut.brick_name = "rope-lut"


if __name__ == "__main__":
    # CPU-ONLY cross-check: recompute the quantized reference for the exact inputs the
    # device path uses, assert shape + finiteness + that rotation actually happened.
    # Deliberately does NOT touch the device / iron.jit / do_rope_lut.
    g, _ = load_golden()
    qk_bf16, cbuf, pos, inv_freq, exp = build_case()

    qk_in = qk_bf16.astype(np.float32)
    assert exp.shape == (M, D), f"ref shape {exp.shape} != {(M, D)}"
    assert np.all(np.isfinite(exp)), "ref has non-finite values"
    assert cbuf.dtype == np.int32 and cbuf.size == M + ROT // 2, "packed const malformed"
    # rotation must change the input (LUT-quantized RoPE is not identity here)
    assert not np.allclose(exp, qk_in, atol=1e-3), "rotation was a no-op"

    # cross-validate the packing split the shim will do: pos = cbuf[:M],
    # inv_freq = cbuf[M:].view(f32), reproduce the ref from the UNPACKED halves.
    pos_split = cbuf[:M]
    invf_split = cbuf[M:].view(np.float32)
    assert np.array_equal(pos_split, pos), "pos split mismatch"
    assert np.array_equal(invf_split, inv_freq), "inv_freq split mismatch"
    exp2 = g.rope_reference_quantized(qk_in, pos_split, ROT)
    assert np.array_equal(exp2, exp), "unpacked-half reference mismatch"

    # sanity: distance from exact RoPE is the LUT-quantization cost only (small)
    exact = g.rope_reference_exact(qk_in, pos, ROT)
    lut_cost = g.rel_l2(exp, exact)
    print(f"[rope-lut CPU cross-check] shape={exp.shape} pos<={int(pos.max())} "
          f"LUT-vs-exact rel_l2={lut_cost:.3e} gate={GATE:.1e}")
    print(f"  qk_in[0,:4]={qk_in[0,:4]}")
    print(f"  ref  [0,:4]={exp[0,:4].astype(np.float32)}")
    print("PASS: quantized reference finite, non-identity, packing split verified "
          "(CPU-only; device run = do_rope_lut()).")
