#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TIER 2 of the rail's correctness gate: end-to-end top-k token-set inclusion.

THE RULE, and it is exactly one rule. Walk the greedy sequences in step order. If they agree for
all N tokens, PASS. At the FIRST step where they disagree, PASS if the reference's token is inside
the device's top-K, otherwise FAIL -- and stop, because free-running decode is not independent
between steps: one different token puts every later step on a different trajectory, so the tail
after a divergence measures the trajectory and not the forward pass.

WHY INCLUSION AND NOT EQUALITY. Two implementations of a transformer do not produce identical
logits, and greedy decode turns any difference into a token flip wherever the top two logits are
close. Measured on this rail's canonical prompt, step 5 has a top1-top2 margin of 0.0203 -- about
one bf16 quantum at that magnitude -- and transformers 4.57.6 in float32 flips it relative to the
float32 sequence an earlier run of transformers recorded in tests/refs/qwen3-0.6b/greedy_ref.json.
The reference disagrees with its own earlier self there. A gate demanding equality charges the
device for that coin flip and caps a perfect implementation below N/N.

Set inclusion at k=5 is what this domain uses -- it is the shape of vLLM's `check_logprobs_close`.
It is strictly weaker than equality and that is the point: it accepts a tie the precision cannot
resolve, and it still rejects a distribution that has genuinely moved, because a token that is not
in the top 5 of 151936 is not a rounding difference.

Both inputs are the JSON `scripts/gate_llm_reference.py` writes, so this also diffs two references
against each other -- which is how the numpy backend was validated against transformers.

  python3 scripts/gate_token_set.py --ref ref.json --npu npu_tokens.json
"""
import argparse
import json
import sys


def load(path, what):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: no {what} at {path}")
    for key in ("gen_ids", "prompt_ids"):
        if key not in d:
            raise SystemExit(f"ERROR: {path} has no `{key}` -- not a gate token file")
    return d


def judge(ref, npu, k):
    """Returns (passed, verdict_string, first_divergence_index or None)."""
    n = min(len(ref["gen_ids"]), len(npu["gen_ids"]))
    if n == 0:
        return False, "no tokens to compare", None
    for i in range(n):
        if ref["gen_ids"][i] == npu["gen_ids"][i]:
            continue
        tops = npu.get("topk_ids") or []
        if i >= len(tops):
            return False, (f"diverged at step {i} and the device file carries no top-{k} for it, "
                           f"so the gate cannot be evaluated -- re-run the capture with --topk"), i
        top = list(tops[i])[:k]
        if ref["gen_ids"][i] in top:
            rank = top.index(ref["gen_ids"][i])
            return True, (f"diverged at step {i}; the reference's token {ref['gen_ids'][i]} is the "
                          f"device's rank-{rank} candidate, inside top-{k}"), i
        return False, (f"diverged at step {i}; the reference's token {ref['gen_ids'][i]} is NOT in "
                       f"the device's top-{k} {top}"), i
    return True, f"exact token parity over all {n} steps", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference JSON from gate_llm_reference.py")
    ap.add_argument("--npu", required=True, help="device JSON from verify_llm_decode.py --emit-topk")
    ap.add_argument("--k", type=int, default=None, help="defaults to the device file's own k")
    a = ap.parse_args()

    ref, npu = load(a.ref, "reference"), load(a.npu, "device capture")
    k = a.k if a.k is not None else int(npu.get("k", ref.get("k", 5)))
    if ref["prompt_ids"] != npu["prompt_ids"]:
        raise SystemExit(f"ERROR: prompt mismatch -- reference {ref['prompt_ids']} vs device "
                         f"{npu['prompt_ids']}. Two different prompts is not a comparison.")
    n = min(len(ref["gen_ids"]), len(npu["gen_ids"]))
    if n < len(ref["gen_ids"]):
        print(f"[tier2] NOTE device produced {len(npu['gen_ids'])} tokens against the reference's "
              f"{len(ref['gen_ids'])}; judging the {n} they share")

    ok, verdict, first = judge(ref, npu, k)
    print(f"[tier2] reference : {ref.get('backend', '?')}")
    print(f"[tier2] device    : {npu.get('backend', 'npu')}")
    print(f"[tier2] prompt    : {ref.get('prompt', ref['prompt_ids'])}")
    print(f"[tier2] ref ids   : {ref['gen_ids'][:n]}")
    print(f"[tier2] npu ids   : {npu['gen_ids'][:n]}")
    agree = sum(1 for i in range(n) if ref["gen_ids"][i] == npu["gen_ids"][i])
    print(f"[tier2] raw token equality: {agree}/{n}  (a NOTE -- the gate is the rule below, "
          f"because equality is not achievable across two implementations)")
    if first is not None:
        m = ref.get("margins") or []
        if first < len(m):
            print(f"[tier2] step {first} reference margin (top1-top2): {m[first]:.4f}"
                  f"{'  -- a knife edge; this step was never decidable' if m[first] < 0.05 else ''}")
        tl = npu.get("topk_logits") or []
        if first < len(tl):
            pairs = ", ".join(f"{i}({v:.4f})" for i, v in
                              zip(npu["topk_ids"][first][:k], tl[first][:k]))
            print(f"[tier2] device top-{k} at step {first}: {pairs}")
    print(f"[tier2] {verdict}")
    print(f"[tier2] *** {'PASS' if ok else 'FAIL'} *** (N={n}, K={k})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
