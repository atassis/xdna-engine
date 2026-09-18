#!/usr/bin/env python3
"""HOST-ONLY (no device) simulation: what does the codec decoder's accuracy look like if every
conv and conv-transpose ran through the bf16 arm, everywhere, for the whole chain?

This is NOT a device measurement. It answers a narrower, honest question than
verify_conv_bf16_device_gate.py's per-op device numbers: composing the bf16 conv-1d/conv-transpose
cores into an actual on-device fused residual unit or upsample stage needs either moving the
intermediate into the resident-stream contract or reusing residual_unit.cc's static-buffer
two-phase pattern -- both out of scope for a kernel-authoring pass under this task's "no static L1
state" constraint (see this change's report for the tradeoff). This script instead reuses
scripts/codec_decoder_ref.py -- the SAME pure-numpy oracle every device rail is checked against --
and monkeypatches its two conv primitives with the bf16-quantizing models from
bricks/conv-1d/golden_bf16.py and codec_block/golden_conv_transpose_bf16.py, which encode the EXACT
same numerics my bf16 kernels implement (bf16 storage in, f64-accumulate, bf16 storage out, never
bf16-accumulated -- kb/bf16-norm-numerics-and-accumulation-guards). snake and the residual add stay
f32/host-numpy, matching this task's kernel design (snake's own cost is well under the per-tile
budget -- see sin.cc's header -- so narrowing it buys nothing and was never in scope).

So this is a BOUND on what the format lever alone would cost across the whole chain, assuming perfect
on-device execution of that exact numerics -- not a claim about real dispatch/DMA correctness, which
only the device gate script settles.

Usage: python3 host_bf16_codec_sim.py <dump_dir> [--gate 3e-2]
  (same dump_dir as codec_decoder_ref.py and verify_whole_decoder.py: needs codec_latent.{bin,shape}
  and codec_audio.{bin,shape}, e.g. ~/.cache/s2-oracle/dump80/)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "aie_kernels" / "conv-1d"))
sys.path.insert(0, str(HERE))

import codec_decoder_ref as R  # noqa: E402
from golden_bf16 import conv_1d_causal_bf16_model  # noqa: E402
from golden_conv_transpose_bf16 import conv_transpose_1d_bf16_model  # noqa: E402


def _bf16_conv_1d_causal(x, w, bias, dilation=1):
    return conv_1d_causal_bf16_model(x, w, bias, dilation)


def _bf16_conv_transpose_1d(x, w, bias, stride, crop_right):
    return conv_transpose_1d_bf16_model(x, w, bias, stride, crop_right)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--gguf", default=R.DEFAULT_GGUF)
    ap.add_argument("--gate", type=float, default=3e-2)
    args = ap.parse_args()

    cache = {}

    def W(name):
        if name not in cache:
            import gguf_extract as gx
            cache[name] = gx.load(args.gguf, name).astype(np.float32)
        return cache[name]

    zf, zshape = R.load_dump(args.dump_dir, "codec_latent")
    af, ashape = R.load_dump(args.dump_dir, "codec_audio")
    z = zf.reshape(zshape[1], zshape[0]).T.copy()
    ref_audio = af.reshape(-1)

    # f32 truth first, from the SAME unmodified decode(), so both numbers come from one process
    # and one weight cache -- no risk of a stale/second GGUF load skewing the comparison.
    got_f32 = R.decode(z, W).reshape(-1)
    num = np.linalg.norm((got_f32 - ref_audio).astype(np.float64))
    den = np.linalg.norm(ref_audio.astype(np.float64))
    rl2_f32 = float(num / den)
    print(f"f32 truth        : rel-L2 vs codec_audio.bin = {rl2_f32:.6e}")

    # Now monkeypatch the two conv primitives to their bf16-quantizing models and re-decode. snake,
    # residual_unit's add, and decode()'s own control flow are untouched -- only the two ops the
    # format lever actually targets change.
    orig_conv = R.conv_1d_causal
    orig_convt = R.conv_transpose_1d
    R.conv_1d_causal = _bf16_conv_1d_causal
    R.conv_transpose_1d = _bf16_conv_transpose_1d
    try:
        got_bf16 = R.decode(z, W).reshape(-1)
    finally:
        R.conv_1d_causal = orig_conv
        R.conv_transpose_1d = orig_convt

    assert got_bf16.shape == ref_audio.shape, f"length mismatch: {got_bf16.shape} vs {ref_audio.shape}"
    num_bf16 = np.linalg.norm((got_bf16 - ref_audio).astype(np.float64))
    rl2_bf16 = float(num_bf16 / den)
    num_vs_f32 = np.linalg.norm((got_bf16 - got_f32).astype(np.float64))
    den_f32 = np.linalg.norm(got_f32.astype(np.float64))
    rl2_vs_f32 = float(num_vs_f32 / den_f32) if den_f32 else float(num_vs_f32)

    print(f"bf16 host-sim    : rel-L2 vs codec_audio.bin = {rl2_bf16:.6e}   (gate {args.gate:.1e})")
    print(f"bf16 host-sim    : rel-L2 vs f32-truth decode = {rl2_vs_f32:.6e}  (format cost in isolation)")
    print(f"  max abs err (bf16 vs oracle) : {float(np.max(np.abs(got_bf16 - ref_audio))):.6e}")
    print(f"  oracle range                 : [{ref_audio.min():+.4f}, {ref_audio.max():+.4f}]")
    print(f"  bf16-sim range               : [{got_bf16.min():+.4f}, {got_bf16.max():+.4f}]")

    ok = rl2_bf16 <= args.gate
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
