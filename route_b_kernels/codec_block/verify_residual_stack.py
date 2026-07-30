#!/usr/bin/env python3
"""A whole 3-unit residual STACK on device, stitched across windows. Gate 3e-2.

This is the first result where the device output is longer than anything a core tile holds: 128
output samples assembled from ~50 dispatches of at most 64 samples each, through three chained
residual units at dilations 1, 3 and 9. Everything before this gated a single window.

Compared against `stage4_res4` from the whole-decoder reference that reproduces codec_audio.bin --
the TRUE unblocked stream, not a host replay of the same windows. A replay would share every
stitching decision the driver made and agree with it while both were wrong.

Segment arithmetic (each causal op returns only what it can compute honestly, so lengths shrink):

    input  206  --unit dil 1, ctx  6-->  200
                --unit dil 3, ctx 18-->  182
                --unit dil 9, ctx 54-->  128 valid outputs

so a 206-sample lead-in buys 128 stitched outputs, and those must match stream positions
[S+78, S+206).

    python3 verify_residual_stack.py <stage_io.npz>

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
import gguf_extract as gx  # noqa: E402

GGUF = ("$WORKSPACE/"
        "s2.cpp/models/s2-pro-q6_k.gguf")
S = 40000
V = 128
UNITS = [("c.decoder.model.4.block.2", 1), ("c.decoder.model.4.block.3", 3),
         ("c.decoder.model.4.block.4", 9)]
TOTAL_CTX = sum(6 * d for _, d in UNITS)      # 78
L = V + TOTAL_CTX                             # 206
GATE = 3e-2


def weights(prefix):
    g = lambda n: gx.load(GGUF, f"{prefix}.{n}").astype(np.float32)
    return (g("block.0.alpha").reshape(-1), g("block.1.conv.weight"),
            g("block.1.conv.bias").reshape(-1), g("block.2.alpha").reshape(-1),
            g("block.3.conv.weight"), g("block.3.conv.bias").reshape(-1))


npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
x = np.ascontiguousarray(npz["stage4_upsample"][:, S:S + L]).astype(np.float32)
truth = npz["stage4_res4"][:, S + TOTAL_CTX:S + L].astype(np.float32)
print(f"segment {x.shape} -> expect {truth.shape} valid outputs, stream [{S + TOTAL_CTX}, {S + L})")

wd.reset_stats()
cur = x
for i, (prefix, dil) in enumerate(UNITS):
    cur = wd.residual_unit(cur, weights(prefix), dil, f"u{i}")
    print(f"  after unit {i} (dilation {dil}): {cur.shape}", flush=True)

assert cur.shape == truth.shape, f"{cur.shape} vs {truth.shape}"
num = np.linalg.norm((cur - truth).astype(np.float64))
den = np.linalg.norm(truth.astype(np.float64))
rl2 = float(num / den)
err = np.abs(cur - truth)

st = wd.stats()
print(f"\n  stitched output vs TRUE stream rel-L2 : {rl2:.3e}   (gate {GATE:.1e})")
print(f"  max abs err                          : {float(err.max()):.3e}")
# Stitching seams are where a driver bug would show: error concentrated at window boundaries rather
# than spread evenly. Report the worst column so a seam cannot hide in an aggregate.
per_col = err.max(axis=0)
print(f"  worst column / median column         : {float(per_col.max()):.3e} / "
      f"{float(np.median(per_col)):.3e}  (worst at col {int(per_col.argmax())})")
print(f"  dispatches {st['dispatches']}, samples computed {st['computed']}, "
      f"useful {st['useful']}, recompute overhead {st['overhead'] * 100:.1f}%")
assert rl2 <= GATE, f"residual stack stitched gate failed: rel_l2={rl2}"
print("PASS")
