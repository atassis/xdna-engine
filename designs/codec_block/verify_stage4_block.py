#!/usr/bin/env python3
"""A COMPLETE decoder stage on device, stitched across windows. Gate 3e-2.

    snake -> conv_transpose_1d(x2) -> residual(dil 1) -> residual(dil 3) -> residual(dil 9)

i.e. `build_decoder_block` from s2.cpp/src/s2_codec.cpp:501-516, entirely on the NPU, at stage 4's
real GGUF weights, over a stream longer than any core tile holds. Gated against `stage4_out` from
the whole-decoder reference that reproduces codec_audio.bin -- the TRUE unblocked stream.

ALIGNMENT, which is the part that is easy to get quietly wrong. The upsample changes the time scale,
so the two halves carry context in different units:

    input   105 samples @ stage-4 input rate
            --upsample, ctx 2 in --> 206 samples @ output rate  (= (105-2) * 2)
            --residual stack, ctx 78 @ output rate --> 128 valid outputs

The first output therefore lands at stream position (S+2)*2 + 78, not at S*2. Getting that wrong
shifts the comparison by a few samples and still produces a small-looking rel-L2 on smooth audio,
which is exactly the failure this gate has to be built to catch -- hence the explicit index
arithmetic below rather than a slice offset chosen to make it pass.

    python3 verify_stage4_block.py <stage_io.npz>

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import window_driver as wd  # noqa: E402
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = codec_paths.gguf()
P = "c.decoder.model.4"
STRIDE = 2
UP_CTX = 2                                  # ceil((k-1)/stride) with k = 2*stride
RES_CTX = 6 * 1 + 6 * 3 + 6 * 9             # 78, at the OUTPUT rate
V = 128                                     # target valid outputs
S = 20000                                   # stage-4 INPUT index the segment starts at
L_IN = UP_CTX + (V + RES_CTX) // STRIDE     # 105
import os
# Force chunking on a stage that does not need it: same answer proves the split is correct.
CI_CHUNK = int(os.environ.get('CI_CHUNK', '0')) or None
GATE = 3e-2


def g(name):
    return gx.load(GGUF, f"{P}.{name}").astype(np.float32)


def unit_weights(sub):
    return (g(f"{sub}.block.0.alpha").reshape(-1), g(f"{sub}.block.1.conv.weight"),
            g(f"{sub}.block.1.conv.bias").reshape(-1), g(f"{sub}.block.2.alpha").reshape(-1),
            g(f"{sub}.block.3.conv.weight"), g(f"{sub}.block.3.conv.bias").reshape(-1))


npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
x = np.ascontiguousarray(npz["stage4_in"][:, S:S + L_IN]).astype(np.float32)

# Where the surviving outputs land in the true stream, derived rather than tuned.
out_start = (S + UP_CTX) * STRIDE + RES_CTX
truth = npz["stage4_out"][:, out_start:out_start + V].astype(np.float32)
print(f"input {x.shape} @ stage-4 in-rate, from index {S}")
print(f"expect {V} outputs at stream [{out_start}, {out_start + V})")

wd.reset_stats()
up = wd.upsample_unfused(x, g("block.0.alpha").reshape(-1), g("block.1.conv.weight"),
                 g("block.1.conv.bias").reshape(-1), STRIDE, "s4up")
print(f"  upsample -> {up.shape}", flush=True)

cur = up
for i, (sub, dil) in enumerate((("block.2", 1), ("block.3", 3), ("block.4", 9))):
    cur = wd.residual_unit(cur, unit_weights(sub), dil, f"s4u{i}", ci_chunk=CI_CHUNK)
    print(f"  residual dil {dil} -> {cur.shape}", flush=True)

assert cur.shape == truth.shape, f"{cur.shape} vs {truth.shape}"
num = np.linalg.norm((cur - truth).astype(np.float64))
den = np.linalg.norm(truth.astype(np.float64))
rl2 = float(num / den)
err = np.abs(cur - truth)
per_col = err.max(axis=0)
st = wd.stats()

print(f"\n  WHOLE STAGE vs TRUE stream rel-L2 : {rl2:.3e}   (gate {GATE:.1e})")
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
assert rl2 <= GATE, f"stage block gate failed: rel_l2={rl2}"
print("PASS")
