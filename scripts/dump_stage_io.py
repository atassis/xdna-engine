#!/usr/bin/env python3
"""Capture each decoder stage's real input and output, so device kernels can be gated on real data.

Every device gate so far has fed the kernels random normal activations. That checks the arithmetic
but not the regime: real activations have their own magnitude distribution, and snake's argument
fold in particular is magnitude-sensitive (alpha*x reaching many periods is exactly where a f32 fold
loses digits). Gating a stage on the tensor the stage actually receives closes that gap.

It is also the input a time-blocked driver needs: a window taken out of the middle of a real stage
stream, with the left context that makes it bit-identical to the unblocked result.

Reuses scripts/codec_decoder_ref.py rather than reimplementing the graph -- that reference already
reproduces codec_audio.bin at rel-L2 3.919e-04, so it is the trustworthy source of intermediates.

    python3 scripts/dump_stage_io.py <dump-dir> <out.npz> [--gguf <path>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import codec_decoder_ref as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--gguf", default=R.DEFAULT_GGUF)
    args = ap.parse_args()

    cache = {}

    def W(name):
        if name not in cache:
            cache[name] = R.gx.load(args.gguf, name).astype(np.float32)
        return cache[name]

    zf, zshape = R.load_dump(args.dump_dir, "codec_latent")
    z = zf.reshape(zshape[1], zshape[0]).T.copy()

    saved = {}
    x = R.conv_1d_causal(z, W(f"{R.PREFIX}.model.0.conv.weight"),
                         W(f"{R.PREFIX}.model.0.conv.bias").reshape(-1), 1)

    for i, stride in enumerate(R.DECODER_RATES):
        p = f"{R.PREFIX}.model.{i + 1}"
        stage = i + 1
        saved[f"stage{stage}_in"] = x.copy()

        # After snake + the transposed conv, BEFORE the residual units: the fused upsample stage
        # kernel's exact output, so it can be gated on its own.
        up = R.conv_transpose_1d(R.snake(x, W(f"{p}.block.0.alpha").reshape(-1)),
                                 W(f"{p}.block.1.conv.weight"),
                                 W(f"{p}.block.1.conv.bias").reshape(-1), stride, crop_right=stride)
        saved[f"stage{stage}_upsample"] = up.copy()

        x = up
        for unit, dil in ((2, 1), (3, 3), (4, 9)):
            x = R.residual_unit(x, W, f"{p}.block.{unit}", dil)
            saved[f"stage{stage}_res{unit}"] = x.copy()
        saved[f"stage{stage}_out"] = x.copy()
        print(f"  stage {stage}: in {saved[f'stage{stage}_in'].shape} "
              f"upsample {up.shape} out {x.shape}", flush=True)

    np.savez_compressed(args.out_npz, **saved)
    total = sum(v.nbytes for v in saved.values())
    print(f"wrote {args.out_npz} ({len(saved)} arrays, {total / 1e6:.1f} MB uncompressed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
