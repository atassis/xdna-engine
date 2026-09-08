#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a verify_llm_decode ref (prompt_ids / gen_ids / margins) from an HF checkpoint at bf16.

Why bf16 and not f32: greedy decode is chaotic wherever the top-2 logits are close, so gating a
bf16 device against an f32 reference charges it for ties it cannot win. scripts/llm_decode_bf16_oracle.py
makes that argument at length and measures it -- Qwen3-0.6B's step 5 has margin 0.0203 and a faithful
bf16 forward already disagrees with f32 there. That script hand-writes the forward in numpy, which
pins it to ONE architecture (silu, plain pre-norm, single-theta, `w` not `1+w`), so it cannot serve
Gemma. This runs the checkpoint's OWN modeling code at bf16 instead: less to re-derive, and it
generalises to any spec in llm_decode_spec.py rather than to the one it was written against.

`margins` is top1-top2 per step. Without it a mismatch cannot be classified, and a classifier with no
margins does not degrade to "unknown" -- verify_llm_decode.py's
`"KNIFE-EDGE" if margins and m < 0.25 else "REAL"` silently upgrades it to the most confident verdict
it has. Emitting margins is what makes the eventual mismatch readable.

The f32 contrast is OPT-IN (`--f32-contrast`) and the bf16 model is freed before it loads, because
the original form held BOTH resident and that is not survivable on the model this oracle exists for.
Measured 2026-09-08 on this box: 30 GB total RAM, ~13 GB available, and unsloth/gemma-4-12b-it is
23 GB of bf16 weights -- so bf16 alone is already borderline and bf16+f32 (~69 GB) cannot run at
all. A gate that cannot execute against the model it gates is not a gate.

  python scripts/llm_hf_bf16_ref.py --model unsloth/gemma-3-270m-it \
      --out tests/refs/gemma3-270m/bf16_oracle.json
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--f32-contrast", action="store_true",
                    help="also run an f32 pass for contrast (loads a SECOND copy of the weights; "
                         "see the module docstring for why this is off by default)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-flight, so an OOM kill is diagnosable instead of a bare 137. This prints rather than
    # refuses: the numbers are estimates and the caller may know better than the estimate.
    try:
        avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1])
        print(f"[ref] {avail / 1048576:.1f} GiB RAM available; a bf16 forward needs roughly the "
              f"checkpoint's on-disk size resident"
              + (", and --f32-contrast needs twice that again" if a.f32_contrast else ""))
    except (OSError, IndexError):
        pass

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()

    ids = tok(a.prompt, return_tensors="pt").input_ids
    prompt_ids = ids[0].tolist()

    # Greedy, one step at a time, feeding the model's OWN pick back -- the same trajectory the
    # device free-runs. Margins come off the same logit vector the pick came from.
    gen_ids, margins = [], []
    cur = ids
    with torch.no_grad():
        for _ in range(a.steps):
            logits = model(cur).logits[0, -1].float()
            top2 = torch.topk(logits, 2)
            gen_ids.append(int(top2.indices[0]))
            margins.append(float(top2.values[0] - top2.values[1]))
            cur = torch.cat([cur, top2.indices[:1].view(1, 1)], dim=1)

    # f32 contrast, NOT the gate -- kept so a reader can see where the two precisions already differ
    # before blaming the device for the difference. Opt-in, and the bf16 model is dropped FIRST:
    # holding both is what made this unrunnable on a 12B checkpoint.
    hf32 = None
    if a.f32_contrast:
        del model
        import gc
        gc.collect()
        f32 = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        hf32, cur = [], ids
        with torch.no_grad():
            for _ in range(a.steps):
                nxt = int(f32(cur).logits[0, -1].argmax())
                hf32.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

    ref = {
        "model": a.model,
        "prompt": a.prompt,
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "margins": margins,
        "hf_f32_gen_ids": hf32,   # null unless --f32-contrast; it is a contrast, never the gate
        "note": "Faithful bf16 host forward via the checkpoint's own modeling code -- the oracle a "
                "bf16 DEVICE should match 1:1. `margins` is top1-top2 per step: a mismatch at a "
                "margin near the device's own logit error is a knife-edge, not a defect. "
                "hf_f32_gen_ids is kept for contrast; it is NOT the gate.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"[ref] wrote {a.out}")
    print(f"[ref] bf16 gen_ids   = {gen_ids}")
    print(f"[ref] margins        = {[round(m, 4) for m in margins]}")
    if hf32 is None:
        print("[ref] f32 contrast   = skipped (--f32-contrast to run it)")
    else:
        disagree = [i for i, (b, f_) in enumerate(zip(gen_ids, hf32)) if b != f_]
        print(f"[ref] f32  gen_ids   = {hf32}")
        print(f"[ref] bf16 vs f32 disagree at steps {disagree or 'none'}"
              f"{'' if not disagree else ' -- those steps cannot gate a bf16 device against f32'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
