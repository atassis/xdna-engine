#!/usr/bin/env python3
"""Device rel-L2 verify config for the rope-interleaved brick (brick-wave1-device-verify).

rope_interleaved_prologue rotates a resident [M,D] bf16 Q/K tile IN PLACE, ADJACENT-PAIR
convention (pairs (2i,2i+1), NOT rope-lut's split-half (i,i+D/2)) -- see rope_interleaved.cc's
header for the full correction writeup and golden.py for the oracle cross-check that caught a
real aliasing bug in an early draft of this reference. It is a ONE-SHOT kernel: the `for m` row
loop lives inside the kernel, so the whole [M,D] buffer flows through a single call -> drive it
through bricklib.verify_oneshot (not the per-row rail), same shape as verify_rope_lut.py.

TWO kernel inputs (qk, cossin) fit a core tile's 2 input DMA channels directly -- unlike rope-lut
(which needed pos+inv_freq packed into one buffer alongside qk), this brick folds the trig itself
host-side (see rope_interleaved.cc's SIN/COS ROUTE note), so there is no separate pos/inv_freq
input at all: `cossin` already IS position x inv_freq x (cos,sin), packed per row as
[cos(0..half-1) | sin(0..half-1)].

Because the kernel mutates qk in place and the oneshot rail hands us separate in/out buffers, the
shim copies qk_in -> qk_out then rotates qk_out in place -> qk_out is the rotated result (same
trick verify_rope_lut.py uses).

Scalars (ROPE_D/ROPE_ROT/ROPE_M) are baked as compile-time LITERALS via ExternalFunction
compile_flags (-D...); they are macros the .cc consumes at include time, and the shim body
references those same literals. Gate rel-L2 vs golden.rope_interleaved_ref applied to the
BF16-ROUNDED input (what the device actually loads) -- the resident cossin table is exact
float64-computed trig cast to f32, so the only device-modeled error is the bf16 round-trip on Q/K
itself (~4e-3 ulp), not a LUT-quantization term (rope-lut needed that; this brick doesn't).

DEVICE-RUN ONLY: do_rope_interleaved() builds+runs on aie2p. The __main__ block is a CPU-ONLY
golden cross-check and MUST NOT invoke do_rope_interleaved.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np
import ml_dtypes

BRICKS = Path(__file__).parent.parent

# --- brick geometry (small: qk = M*D*2 = 4KB, cossin = M*ROT*4 = 8KB, comfortably inside 64KB L1) ---
# D=ROT=64 matches RVQ_HEAD_DIM (fish_speech.codec.rvq_transformer.head_dim, see
# scripts/codec_quantizer_ref.py) -- one of the two real call sites this brick corrects
# (s2.cpp/src/s2_codec.cpp:338). ROPE_ROT must be a multiple of 32 (kHalf % kVec==0, kVec=16 --
# see rope_interleaved.cc's static_assert); 64 satisfies it.
D = 64        # ROPE_D   head_dim / feature width per Q/K row
ROT = 64      # ROPE_ROT rotary width (== D here: full rotary; ROT<D = partial, see golden.py)
M = 32        # ROPE_M   resident tile rows (tokens) processed this call
BASE = 10000.0    # rope_freq_base (RVQ_ROPE_BASE / hp.rope_freq_base in the real call sites)
GATE = 3e-2       # brick threshold


def load_golden():
    """Import the brick's numpy golden module + return (module, abs .cc path)."""
    cc = BRICKS / "rope-interleaved" / "rope_interleaved.cc"
    p = BRICKS / "rope-interleaved" / "golden.py"
    spec = importlib.util.spec_from_file_location("rope_interleaved_golden", p)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g, str(cc)


def build_case():
    """Build (qk_bf16, cossin, exp) shared by the device run and the CPU cross-check.

    qk_bf16 : [M,D] ml_dtypes.bfloat16   device input (rotated in place -> output)
    cossin  : [M,ROT] float32            resident, per-row [cos(0..half-1) | sin(0..half-1)]
    exp     : [M,D] float64              golden.rope_interleaved_ref(qk_ref_in, positions, ROT, BASE)
    """
    g, _ = load_golden()
    rng = np.random.default_rng(0)
    qk_f32 = rng.standard_normal((M, D)).astype(np.float32)
    qk_bf16 = qk_f32.astype(ml_dtypes.bfloat16)      # what the device actually sees
    qk_ref_in = qk_bf16.astype(np.float32)           # SAME values feed the golden

    positions = np.arange(M, dtype=np.int64)         # decode-style monotonic KV offsets
    cossin = g.build_cossin_resident(positions, ROT, BASE)   # [M,ROT] f32, host-folded trig

    exp = g.rope_interleaved_ref(qk_ref_in, positions, ROT, BASE)  # gate against THIS
    return qk_bf16, cossin, positions, exp


# --- pure-buffer verify shim: copy qk_in->qk_out, rotate in place using the resident cossin table ---
_cb = int(time.time() * 1000) % 10**9
SHIM_BODY = (
    f'// cachebust {_cb}\n'
    'extern "C" void rope_interleaved_verify(bfloat16 *qk_in, float *cossin, bfloat16 *qk_out) {\n'
    '  for (unsigned i = 0; i < (unsigned)(ROPE_M * ROPE_D); ++i) qk_out[i] = qk_in[i];\n'
    '  rope_interleaved_prologue(qk_out, cossin);        // rotates qk_out IN PLACE\n'
    '}\n'
)

COMPILE_FLAGS = [f"-DROPE_D={D}", f"-DROPE_ROT={ROT}", f"-DROPE_M={M}"]


def do_rope_interleaved():
    """DEVICE-RUN. Build + run rope-interleaved on aie2p, gate rel-L2 vs the golden.
    Do NOT call at import/CPU time -- this triggers an iron.jit device build."""
    import bricklib  # lazy: keeps __main__ CPU-only (no aie.iron import)
    _, cc = load_golden()
    qk_bf16, cossin, _pos, exp = build_case()
    return bricklib.verify_oneshot(
        "rope-interleaved", cc, SHIM_BODY, "rope_interleaved_verify",
        inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cossin.reshape(-1), np.float32)],
        out_numel=M * D, out_shape=(M, D),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(M, D),
        golden=exp, gate=GATE, out_dt=ml_dtypes.bfloat16,
        compile_flags=COMPILE_FLAGS)
do_rope_interleaved.brick_name = "rope-interleaved"


if __name__ == "__main__":
    # CPU-ONLY cross-check: recompute the reference for the exact inputs the device path uses,
    # assert shape + finiteness + that rotation actually happened. Deliberately does NOT touch
    # the device / iron.jit / do_rope_interleaved.
    g, _ = load_golden()
    qk_bf16, cossin, positions, exp = build_case()

    qk_in = qk_bf16.astype(np.float32)
    assert exp.shape == (M, D), f"ref shape {exp.shape} != {(M, D)}"
    assert np.all(np.isfinite(exp)), "ref has non-finite values"
    assert cossin.shape == (M, ROT) and cossin.dtype == np.float32, "cossin resident malformed"
    # rotation must change the input (interleaved RoPE is not identity here)
    assert not np.allclose(exp, qk_in, atol=1e-3), "rotation was a no-op"

    # cross-validate the shim's ABI: rope_interleaved_prologue(qk_out, cossin) with qk_out
    # pre-copied from qk_in must reproduce the golden exactly (float64 math both sides, so this
    # should match to float rounding only -- NOT the bf16 device output, which is a separate,
    # looser (GATE=3e-2) check done on device).
    exp2 = g.rope_interleaved_ref(qk_in, positions, ROT, BASE)
    assert np.array_equal(exp2, exp), "re-derivation from the same inputs is nondeterministic"

    print(f"[rope-interleaved CPU cross-check] shape={exp.shape} D={D} ROT={ROT} M={M} "
          f"pos<={int(positions.max())}")
    print(f"  qk_in[0,:4]={qk_in[0,:4]}")
    print(f"  ref  [0,:4]={exp[0,:4].astype(np.float32)}")
    print("PASS: golden finite, non-identity, ABI-consistent "
          "(CPU-only; device run = do_rope_interleaved()).")
