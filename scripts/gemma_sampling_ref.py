#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-CPU reference sampler -- the correctness oracle for `npu-gemma::sampling` (Rust).

Pure Python (stdlib `math` only, no numpy/torch): implements the IDENTICAL algorithm and the SAME
SplitMix64 PRNG, bit-for-bit, as `rust/npu-gemma/src/sampling.rs`. A fixed seed + fixed logits vector
must produce the identical token id in both languages -- this is the "not vibes" correctness gate the
Rust unit tests (`sampling::tests::cross_lang_*`) bake in as expected constants. Re-runnable by hand to
regenerate those constants if the algorithm ever changes (deliberately NOT wired into `cargo test` --
this repo has no Python-in-cargo-test harness; see other host oracles like `gemma_ref_generate.py` for
the same pattern of a standalone oracle script).

Order (mirrors the Rust module doc comment): repetition/frequency/presence penalties -> temperature ->
top-k -> top-p (nucleus) -> softmax -> inverse-CDF categorical draw. temperature<=0 short-circuits to
greedy argmax (no RNG draw, no penalty/top-k/top-p pass) -- the shipped decode default
(`gemma_ref_generate.py` uses `do_sample=False`, i.e. greedy).

All math is float64 (Python's native float) end to end, matching the Rust side's internal-f64 policy
(logits/config are f32 at the API boundary, promoted to f64 before any arithmetic) so the two
implementations do bit-identical IEEE-754 double arithmetic in the same operation order -- including
calling the platform libm `exp()` on Linux, the same function both languages ultimately invoke.

Usage: python3 scripts/gemma_sampling_ref.py
"""
import math

MASK64 = (1 << 64) - 1


class SplitMix64:
    """Deterministic 64-bit PRNG (Vigna's splitmix64). Mirrors `sampling::SplitMix64` in Rust exactly."""

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_f64(self) -> float:
        """Uniform draw in [0, 1) from the top 53 bits (standard PRNG-to-float recipe)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


def argmax(logits):
    best_i, best_v = 0, float("-inf")
    for i, v in enumerate(logits):
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def apply_penalties(logits, history, rep, freq, pres):
    """CTRL-style multiplicative repetition penalty + OpenAI-style additive frequency/presence."""
    if rep == 1.0 and freq == 0.0 and pres == 0.0:
        return list(logits)
    counts = {}
    for t in history:
        counts[t] = counts.get(t, 0) + 1
    out = list(logits)
    for tok, count in counts.items():
        l = out[tok]
        if rep != 1.0:
            l = l / rep if l > 0.0 else l * rep
        l -= freq * count
        l -= pres
        out[tok] = l
    return out


def filter_top_k(logits, k):
    if k == 0 or k >= len(logits):
        return list(logits)
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    out = list(logits)
    for i in order[k:]:
        out[i] = float("-inf")
    return out


def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) if v != float("-inf") else 0.0 for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def filter_top_p(logits, top_p):
    """Nucleus filter: keep the smallest prefix (by descending prob) whose cumulative mass >= top_p."""
    if top_p >= 1.0:
        return list(logits)
    probs = softmax(logits)
    order = sorted(range(len(logits)), key=lambda i: probs[i], reverse=True)
    cum = 0.0
    cutoff = len(order)
    for pos, i in enumerate(order):
        cum += probs[i]
        if cum >= top_p:
            cutoff = pos + 1
            break
    out = list(logits)
    for i in order[cutoff:]:
        out[i] = float("-inf")
    return out


def sample(logits, history, temperature, top_k, top_p, rep, freq, pres, rng):
    if temperature <= 0.0:
        return argmax(logits)
    work = apply_penalties(logits, history, rep, freq, pres)
    work = [v / temperature for v in work]
    work = filter_top_k(work, top_k)
    work = filter_top_p(work, top_p)
    probs = softmax(work)
    u = rng.next_f64()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u < cum:
            return i
    return len(probs) - 1  # float edge case: u landed in the last-ulp gap


if __name__ == "__main__":
    # Fixed vectors, mirrored 1:1 in rust/npu-gemma/src/sampling.rs unit tests (cross_lang_* / SAMPLE_LOGITS).
    logits = [1.0, 3.0, 2.0, 0.5, 4.0, -1.0, 2.5, 0.0]
    history = [1, 1, 4]

    rng = SplitMix64(42)
    tok = sample(logits, history, temperature=0.8, top_k=4, top_p=0.9,
                 rep=1.2, freq=0.3, pres=0.1, rng=rng)
    print("cross_lang_full_pipeline: seed=42 ->", tok)

    rng2 = SplitMix64(7)
    seq = [sample(logits, [], temperature=1.0, top_k=0, top_p=1.0,
                  rep=1.0, freq=0.0, pres=0.0, rng=rng2) for _ in range(8)]
    print("cross_lang_plain_softmax_seq: seed=7, n=8 ->", seq)

    # sanity: greedy path (temperature<=0) never touches the RNG or the history/penalty machinery.
    print("greedy (temperature=0):", sample(logits, history, 0.0, 0, 1.0, 1.0, 0.0, 0.0, SplitMix64(0)))
    print("argmax(logits) directly:", argmax(logits))
