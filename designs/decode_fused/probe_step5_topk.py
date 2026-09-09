#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Top-k logits at ONE teacher-forced step, to tell a tie from a defect.

Both arms miss step 5 against the bf16 oracle in different ways, and the margin there is 0.0203 --
documented in scripts/llm_decode_bf16_oracle.py as a step where a faithful bf16 forward already
disagrees with HF f32. A near-tie and a broken op look identical in a token sequence. They do not
look alike in the logits: at a tie the arm's token sits within ~the margin of the oracle's, and at a
defect it does not.

Teacher-forcing means the INPUT state at the target step is identical across arms by construction,
so the logits are directly comparable.
"""
import argparse, json, os, sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, load_weight_buffer, isolate_build_dir  # noqa: E402
from verify_llm_decode import rope_row  # noqa: E402

BF16 = ml_dtypes.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    isolate_build_dir("probe")

    ref = json.load(open(a.ref))
    prompt_ids, gen_ids = ref["prompt_ids"], ref["gen_ids"]
    margins = ref.get("margins")

    sp, fused, weights, md = build_graph(a.spec, a.weights, None)
    HD, D, VOCAB = sp.head_dim, sp.d_model, sp.vocab
    c = fused.get_callable()
    params = c.params
    for name, arr in weights.items():
        load_weight_buffer(c.get_buffer(name), arr)
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")

    embed = np.load(os.path.join(a.weights, "model.embed_tokens.weight.npy")).astype(np.float32)
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    xin, rope_buf, out = c.get_buffer("x"), c.get_buffer("rope_global"), c.get_buffer("logits")

    fed = list(prompt_ids)
    tok = fed[0]
    for pos in range(len(fed) + a.step):
        with xin.overwrite() as _buf:
            _buf[:] = np.asarray(embed[tok] * scale, BF16).reshape(-1)
        with rope_buf.overwrite() as _buf:
            _buf[:] = rope_row(pos, HD, sp.rope_theta_global).reshape(-1)
        params.write("kv_off", int(pos * HD))
        params.write("sm_mask", int(pos + 1))
        params.sync()
        c()
        lg = np.asarray(out.data[:VOCAB], dtype=np.float32)
        i = pos - len(fed) + 1
        if i == a.step:
            order = np.argsort(-lg)[: a.topk]
            want = gen_ids[a.step]
            print(f"\n[step {a.step}] oracle token {want}, "
                  f"recorded margin {margins[a.step]:.4f}" if margins else "")
            print(f"{'rank':>4} {'token':>8} {'logit':>10} {'gap to top':>11}")
            for r, t in enumerate(order):
                print(f"{r:>4} {t:>8} {lg[t]:10.4f} {lg[order[0]]-lg[t]:11.4f}"
                      f"{'   <- ORACLE' if t == want else ''}")
            print(f"\ndevice top-1 {order[0]}, oracle {want}, "
                  f"logit gap {lg[order[0]] - lg[want]:.4f}")
            print("VERDICT: TIE (gap within the recorded margin)" if margins and
                  abs(lg[order[0]] - lg[want]) <= 3 * margins[a.step]
                  else "VERDICT: NOT a tie at this margin -- investigate")
            return 0
        # teacher-force
        tok = fed[pos + 1] if pos + 1 < len(fed) else gen_ids[i]
    return 1


if __name__ == "__main__":
    sys.exit(main())
