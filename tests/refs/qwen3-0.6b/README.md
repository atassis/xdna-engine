# Qwen3-0.6B token references

Three files, three different jobs. Only one of them is the Tier 2 gate.

| file | what it is | what it gates |
|---|---|---|
| `gate_ref_n32.json` | 32 tokens, per-step top-5 ids + logits + top1-top2 margins. float32 arithmetic on the bf16 weights the device holds. | **TIER 2**, via `scripts/gate_token_set.py`. |
| `bf16_oracle.json` | 8 tokens from a faithful **bf16** host forward, with margins. | The OLD 1:1 token-parity gate (`designs/decode_fused/verify_llm_decode.py` without `--emit-topk`). |
| `greedy_ref.json` | 8 tokens from HF transformers in f32, no margins. | Nothing. It is the contrast the bf16 oracle is read against. |

Regenerate: `bash scripts/gate_llm.sh --make-ref` (device-free, ~20 s).

## Why the two 8-token files are not the gate, and why they are still here

**One prompt and 8 free-running tokens is too small.** After the first divergence a greedy sequence
is on a different trajectory, so the tokens past it measure the trajectory, not the forward pass --
which leaves 8 tokens carrying only a handful of independent argmaxes. This rail has already had a
dataflow arm that agreed on the first token and differed by the third.

**And 1:1 equality is the wrong shape of gate, not merely a short one.** Greedy decode is chaotic
wherever the top two logits are close. Step 5 of this prompt has a margin of 0.0203, about one bf16
quantum at that magnitude. Measured 2026-09-08: transformers 4.57.6 in float32 produces
`... 15344, 374, 21718` there, while `greedy_ref.json` -- recorded from an earlier transformers run
in float32 on the same prompt -- has `... 9625, 374, 1083`. The reference implementation disagrees
with its own earlier self at that step. A gate demanding equality charges the device for that.

They stay because they still answer a question `gate_ref_n32.json` does not: `bf16_oracle.json` is
matching-precision, so device-vs-oracle equality is a **determinism** check on a bf16 datapath, and
that is a real property to keep testing. Deleting them would delete that. What changed is which
file carries the CORRECTNESS verdict.

## The one thing to widen next

`gate_ref_n32.json` is still ONE prompt. Widening it needs the KV cache zeroed between prompts in
the device harness -- `verify_llm_decode.py` never resets it, because until now nothing ran two
prompts in one session. That is the change to make before adding prompts, not after.
