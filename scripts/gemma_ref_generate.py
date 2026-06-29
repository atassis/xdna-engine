#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gemma 3 host-CPU REFERENCE generation -- the correctness oracle for the NPU port (T7 phase 0).

This is the GROUND TRUTH our future on-NPU Gemma decode is validated against (the same role
scripts/parakeet_ref_encoder.py plays for the encoder). It runs the real Gemma 3 decoder on the HOST CPU
(transformers, torch CPU -- NEVER the dGPU), greedily generates tokens, and dumps a compact per-step oracle
(argmax token ids + the last-step logits + first-layer hidden) so the NPU path can be checked op-by-op and
end-to-end by token-sequence parity.

Model: ungated mirror `unsloth/gemma-3-270m-it` (270M; same weights as google/gemma-3-270m-it, which is
HF-gated). Swap --model unsloth/gemma-3-1b-it for the ~1B target once 270M is correct.

Honest framing: LLM DECODE is LPDDR-bandwidth-bound (weights stream per token), so the NPU win is
ENERGY / CPU-offload, not raw tok/s. The phase-0 milestone is CORRECT e2e generation through the engine,
not a speed record. See internal notes.

Usage (CPU only):
  CUDA_VISIBLE_DEVICES="" ~/npuvox-asr-bench/.venv/bin/python scripts/gemma_ref_generate.py \
      [--model unsloth/gemma-3-270m-it] [--prompt "..."] [--max-new 32] [--dump-oracle artifacts/gemma/oracle_270m]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # NPU-first engine; never the dGPU (oracle is host CPU)

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the Gemma chat template")
    ap.add_argument("--dump-oracle", default="", help="dir to write the per-step oracle (.npy/.json)")
    a = ap.parse_args()

    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gemma-ref] loading {a.model} on CPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    print(f"[gemma-ref] dims: d_model={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"q_heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim} "
          f"ffn={cfg.intermediate_size} vocab={cfg.vocab_size} act={cfg.hidden_activation}")

    if a.chat:
        msgs = [{"role": "user", "content": a.prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tok(a.prompt, return_tensors="pt").input_ids
    prompt_len = ids.shape[1]
    print(f"[gemma-ref] prompt ({prompt_len} tok): {a.prompt!r}")

    # Greedy generation (deterministic = the oracle). do_sample=False, no temperature.
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0]
    new_ids = seq[prompt_len:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    print(f"[gemma-ref] generated {len(new_ids)} tokens:")
    print(f"  ids   = {new_ids}")
    print(f"  text  = {text!r}")

    # Per-step argmax token ids = the token-sequence oracle (the automatable NPU-parity surrogate).
    step_argmax = [int(s[0].argmax()) for s in out.scores]
    print(f"[gemma-ref] per-step greedy argmax (oracle) = {step_argmax}")

    if a.dump_oracle:
        os.makedirs(a.dump_oracle, exist_ok=True)
        # last prompt-step logits (the first generated token's full logit vector) -- the tightest op-level oracle
        first_logits = out.scores[0][0].float().numpy()
        np.save(os.path.join(a.dump_oracle, "first_step_logits.npy"), first_logits)
        # first-layer hidden of the prompt forward (for per-node checks once the NPU block runs)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        np.save(os.path.join(a.dump_oracle, "prompt_hidden_layer1.npy"), hs[1][0].float().numpy())
        meta = {
            "model": a.model, "prompt": a.prompt, "prompt_len": prompt_len,
            "generated_ids": new_ids, "text": text, "step_argmax": step_argmax,
            "dims": {"d_model": cfg.hidden_size, "layers": cfg.num_hidden_layers,
                     "q_heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
                     "head_dim": cfg.head_dim, "ffn": cfg.intermediate_size, "vocab": cfg.vocab_size},
        }
        with open(os.path.join(a.dump_oracle, "oracle.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[gemma-ref] wrote oracle -> {a.dump_oracle} (first_step_logits.npy, prompt_hidden_layer1.npy, oracle.json)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
