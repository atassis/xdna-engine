#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fold an activation-aware per-channel scale into a dumped LLM weight tree.

WHY THIS IS A WEIGHT STEP AND NOT AN ENGINE FEATURE. Quantizing W is equivalent to quantizing
W*diag(s) and dividing the input by s, for any positive s -- but not after rounding, because s
decides which channels dominate their group's range. Choosing s from measured activation magnitude
(AWQ) spends the levels on the channels the activations actually excite. It is free at inference
ONLY if diag(1/s) folds into something upstream, and in Qwen3 it does, because the norm output
feeds nothing but the projection (the residual bypasses it):

    q,k,v_proj    <- input_layernorm.weight           one shared s, dim d_model
    gate,up_proj  <- post_attention_layernorm.weight  one shared s, dim d_model
    down_proj     <- up_proj's OUTPUT rows            the SwiGLU product is elementwise, so
                                                      scaling up's row c scales down's input c

o_proj is skipped: its input channels reach v_proj rows through the GQA repeat, so no per-channel
fold exists -- and quantizing it measures near-null anyway, so there is nothing to recover.

So the engine never learns about this. It reads a weights directory; this writes a different one.

MEASURED 2026-09-10, Qwen3-0.6B, whole model at affine int4/g32 (5.0 bits), 3000 paired positions
of natural prose: +23.33% perplexity without the fold, +12.06% with it -- a paired -9.14%
[-11.96, -6.23], t=-5.96, for ~2 minutes of host calibration and zero device change.

Needs torch + transformers, which .venv-iron deliberately does not carry. Use .venv-export:

  PYTHONPATH=designs/decode_fused .venv-export/bin/python scripts/awq_fold_llm_weights.py \
      --weights artifacts/qwen3-0.6b/weights --out artifacts/qwen3-0.6b/weights-awq \
      --calib some-corpus.txt --spec qwen3-0.6b
"""
import argparse
import json
import os
import sys

import numpy as np

HF_REPO = {"qwen3-0.6b": "Qwen/Qwen3-0.6B"}
# The formats the fold is worth doing for. It is a no-op at bf16 (the alpha search picks 0), so
# the caller states the format they intend to quantize to and the scale is optimised for it.
DEFAULT_SPEC = {"scheme": "affine", "nbits": 4, "group": 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="tree from dump_llm_weights.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", required=True, help="UTF-8 calibration corpus")
    ap.add_argument("--calib-tokens", type=int, default=512)
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--format", default=json.dumps(DEFAULT_SPEC),
                    help="the quantization the scale is optimised FOR, as JSON")
    ap.add_argument("--classes", default="mlp,qkv")
    ap.add_argument("--lab", default=None,
                    help="directory holding wq_formats.py/awq.py (the quality lab)")
    a = ap.parse_args()

    sys.path.insert(0, a.lab or os.environ.get("QLAB_DIR", "/mnt/data/qlab"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from awq import apply_awq                     # noqa: E402  the measured implementation

    repo = HF_REPO[a.spec]
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    ids = tok(open(a.calib, encoding="utf-8", errors="replace").read(),
              add_special_tokens=False)["input_ids"][:a.calib_tokens]
    if len(ids) < a.calib_tokens:
        raise SystemExit(f"{a.calib}: only {len(ids)} tokens")

    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                                 local_files_only=True)
    model.eval()
    fmt = json.loads(a.format)
    chosen = apply_awq(model, ids, fmt, classes=tuple(a.classes.split(",")))

    # Write ONLY the tensors the fold touched; everything else is copied, so the output tree is a
    # complete drop-in for --weights and a diff against it shows exactly what moved.
    os.makedirs(a.out, exist_ok=True)
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    moved = 0
    for fn in sorted(os.listdir(a.weights)):
        if not fn.endswith(".npy"):
            continue
        key = fn[:-len(".npy")]
        src = np.load(os.path.join(a.weights, fn))
        new = sd.get(key)
        if new is None or new.shape != src.shape:
            np.save(os.path.join(a.out, fn), src)
            continue
        if not np.array_equal(new, src):
            moved += 1
        np.save(os.path.join(a.out, fn), new.astype(src.dtype))

    json.dump({"source_weights": os.path.abspath(a.weights),
               "calib": os.path.abspath(a.calib), "calib_tokens": a.calib_tokens,
               "format": fmt, "classes": a.classes,
               "tensors_changed": moved,
               "alphas": [{"kind": k, "layer": i, "alpha": al} for k, i, al in chosen]},
              open(os.path.join(a.out, "awq_manifest.json"), "w"), indent=1)
    print(f"wrote {a.out}: {moved} tensors folded, manifest in awq_manifest.json")


if __name__ == "__main__":
    main()
