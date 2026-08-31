#!/usr/bin/env python3
"""A COMPLETE decoder stage on device, stitched across windows, for stage 1, 2 or 3. Gate 3e-2.

    snake -> conv_transpose_1d(x stride) -> residual(dil 1) -> residual(dil 3) -> residual(dil 9)

The SAME graph verify_stage4_block.py already gates at rel-L2 8.869e-07 (`build_decoder_block` from
s2.cpp/src/s2_codec.cpp:501-516), run here at stage 1/2/3's real GGUF weights instead of stage 4's.
The kernels are unchanged and already device-green (bricks/snake, bricks/conv-transpose-1d, plus the
residual-unit kernels in codec_block/); what stages 1-3 need on top of stage 4 is window_driver's
ci_chunk/resident_depth chunking, because `c_in * k` -- the width of ONE streamed weight row -- grows
32x from stage 4 to stage 1 and blows a 64 KiB core tile well before the resident activation window
does. See stage_shapes.py for the arithmetic and the per-stage (ci_chunk, resident_depth) plan.

ALIGNMENT, exactly as in verify_stage4_block.py (same convention, same reasoning, generalized to a
stride-dependent L_IN so it still lands on a whole number of upsample output columns at stride 4/8):

    input   L_IN samples @ this stage's input rate
            --upsample, ctx 2 in --> (L_IN-2)*stride samples @ output rate
            --residual stack, ctx 78 @ output rate --> V valid outputs

so the first output lands at stream position (S+2)*stride + 78, not at S*stride. V is DERIVED from
L_IN (via ceil division so it is always >= 128), not hardcoded, because stride 8/4 does not divide
128+78 evenly the way stride 2 does -- forcing V=128 here would silently misreport how many outputs
were actually produced. This generalization was checked host-only (no device) against
codec_decoder_ref.py's own ops before being used here: for every stage the windowed-and-cropped
reconstruction matches the true stream at rel-L2 0.0 (bit-identical, pure numpy on both sides).

    python3 verify_stage123.py <1|2|3> [stage_io.npz]

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import window_driver as wd  # noqa: E402
import stage_shapes as ss  # noqa: E402
import gguf_extract as gx  # noqa: E402

GATE = 3e-2
V_TARGET = 128
# Start index (this stage's OWN input rate) for the segment, chosen well inside the stream -- same
# spirit as verify_stage4_block.py's S=20000 of 56320 (~35%), scaled to each stage's much shorter
# input (stage 1 input is only 220 samples: 220 latent frames through the head conv).
S_BY_STAGE = {1: 70, 2: 600, 3: 5000}

if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3"):
    raise SystemExit(
        f"usage: {sys.argv[0]} <1|2|3> [stage_io.npz]\n"
        "  stage 4 is already gated by verify_stage4_block.py -- this script covers 1, 2, 3.")
STAGE = int(sys.argv[1])
NPZ_PATH = sys.argv[2] if len(sys.argv) > 2 else "stage_io.npz"

GGUF = ss.GGUF
P = f"c.decoder.model.{STAGE}"


def g(name):
    return gx.load(GGUF, f"{P}.{name}").astype(np.float32)


def unit_weights(sub):
    return (g(f"{sub}.block.0.alpha").reshape(-1), g(f"{sub}.block.1.conv.weight"),
            g(f"{sub}.block.1.conv.bias").reshape(-1), g(f"{sub}.block.2.alpha").reshape(-1),
            g(f"{sub}.block.3.conv.weight"), g(f"{sub}.block.3.conv.bias").reshape(-1))


c_in, c_out, k, STRIDE = ss.stage_shape(STAGE)
c_res = ss.residual_channels(STAGE)
assert c_res == c_out, f"residual channel count {c_res} != stage c_out {c_out}"
plan = ss.plan(STAGE)          # {up_ci_chunk, res_ci_chunk, resident_depth}, self-checked vs L1_BUDGET
# Overridable the same way verify_stage4_block.py's CI_CHUNK is: forcing a DIFFERENT chunk size and
# getting the same rel-L2 is a correctness cross-check on the chunking/accumulation logic itself,
# not just a knob. Unset -> the self-checked stage_shapes.py plan above, unchanged.
if os.environ.get("UP_CI_CHUNK"):
    plan["up_ci_chunk"] = int(os.environ["UP_CI_CHUNK"])
if os.environ.get("RES_CI_CHUNK"):
    plan["res_ci_chunk"] = int(os.environ["RES_CI_CHUNK"])
if os.environ.get("RESIDENT_DEPTH"):
    plan["resident_depth"] = int(os.environ["RESIDENT_DEPTH"])
print(f"stage {STAGE}: c_in={c_in} c_out={c_out} k={k} stride={STRIDE}")
print(f"  plan: {plan}")

UP_CTX = ss.UP_CTX              # 2
RES_CTX = ss.RES_CTX            # 78, at the OUTPUT rate

S = S_BY_STAGE[STAGE]
# Ceil, not floor: stride 8/4 does not divide (V_TARGET + RES_CTX) evenly the way stage 4's stride 2
# does, so flooring would silently produce fewer than V_TARGET outputs. Ceiling guarantees >= target
# and V below is the ACTUAL count achieved, not assumed.
L_IN = UP_CTX + -(-(V_TARGET + RES_CTX) // STRIDE)

npz = np.load(NPZ_PATH)
x = np.ascontiguousarray(npz[f"stage{STAGE}_in"][:, S:S + L_IN]).astype(np.float32)
assert x.shape[0] == c_in, f"stage{STAGE}_in has {x.shape[0]} channels, expected {c_in}"

up_len = (L_IN - UP_CTX) * STRIDE
V = up_len - RES_CTX
assert V >= V_TARGET, f"V={V} < target {V_TARGET} -- widen L_IN"

# Where the surviving outputs land in the true stream, derived rather than tuned (see docstring).
out_start = (S + UP_CTX) * STRIDE + RES_CTX
truth = npz[f"stage{STAGE}_out"][:, out_start:out_start + V].astype(np.float32)
print(f"input {x.shape} @ stage-{STAGE} in-rate, from index {S}")
print(f"expect {V} outputs at stream [{out_start}, {out_start + V})")

wd.reset_stats()
up = wd.upsample_unfused(x, g("block.0.alpha").reshape(-1), g("block.1.conv.weight"),
                 g("block.1.conv.bias").reshape(-1), STRIDE, f"s{STAGE}up",
                 ci_chunk=plan["up_ci_chunk"], resident_depth=plan["resident_depth"])
print(f"  upsample -> {up.shape}", flush=True)

cur = up
for i, (sub, dil) in enumerate((("block.2", 1), ("block.3", 3), ("block.4", 9))):
    cur = wd.residual_unit(cur, unit_weights(sub), dil, f"s{STAGE}u{i}",
                           ci_chunk=plan["res_ci_chunk"], resident_depth=plan["resident_depth"])
    print(f"  residual dil {dil} -> {cur.shape}", flush=True)

assert cur.shape == truth.shape, f"{cur.shape} vs {truth.shape}"
num = np.linalg.norm((cur - truth).astype(np.float64))
den = np.linalg.norm(truth.astype(np.float64))
rl2 = float(num / den)
err = np.abs(cur - truth)
per_col = err.max(axis=0)
st = wd.stats()

print(f"\n  STAGE {STAGE} vs TRUE stream rel-L2 : {rl2:.3e}   (gate {GATE:.1e})")
print(f"  max abs err                       : {float(err.max()):.3e}")
print(f"  worst col / median col            : {float(per_col.max()):.3e} / "
      f"{float(np.median(per_col)):.3e}  (worst at col {int(per_col.argmax())})")
# A one-sample misalignment reads as a small rel-L2 on smooth audio, so check it explicitly:
# correlate against neighbouring shifts and require zero to win.
best = min(range(-2, 3),
           key=lambda s_: np.linalg.norm(cur[:, 2:V - 2] - truth[:, 2 + s_:V - 2 + s_]))
print(f"  best alignment shift              : {best} (0 means the index arithmetic is right)")
print(f"  dispatches {st['dispatches']}, useful {st['useful']}/{st['computed']} "
      f"-> {st['overhead'] * 100:.1f}% recompute")
assert best == 0, f"output is misaligned by {best} samples"
assert rl2 <= GATE, f"stage {STAGE} block gate failed: rel_l2={rl2}"
print("PASS")
