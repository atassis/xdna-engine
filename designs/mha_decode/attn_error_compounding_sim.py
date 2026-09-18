#!/usr/bin/env python3
"""Does the mha_decode HD=128 bf16 compute error compound across the S2 slow transformer's 36
attention layers into a changed greedy-decoded token? Pure host/numpy, no device.

INPUT FACTS (device-measured, given by the task, not re-derived here): the mha_decode kernel
(HD=128, 32 heads, 8 KV heads -- `mlir-aie/aie_kernels/aie2p/mha_decode.cc`, gate comment at
line 14) reproduces a host f32 reference at rel-L2 3.361e-02 aggregate over 32 heads, per-head
range [1.650e-02, 5.354e-02], flat across n_tiles 1..4 (so not a flash-attention accumulation
artifact) and with both sides fed identical bf16-rounded inputs (so not input quantization). It is
therefore per-head bf16 COMPUTE error internal to the kernel. `scripts/s2_ar_ref.py:324`
(`ARHParams.block_count=36`) and `s2.cpp/src/s2_model.cpp:262` stack 36 of these per AR step.

METRIC CHOICE. Pricing a much larger perturbation (int4 weight requantization) found the
decision-relevant signal was not weight rel-L2 but whether temperature=0 greedy decode flips the
step-0 token -- every int4 config flipped it while int8 (rel-L2 0.0072) did not. That generalizes:
statistical error numbers are notes, not gates; what carries a ship/no-ship read is 1:1 comparison
against the reference at temperature 0. This script follows that precedent: greedy-token identity
at step 0 is the primary metric, logit-gap margin is the continuous variable that explains WHY (or
why not), and layer-by-layer hidden-state rel-L2 is what actually answers "does it compound".

INJECTION MODEL, and why it is built this way rather than reusing `slow_transformer_forward`
verbatim (the task requires using `s2_ar_ref.py`'s own op primitives, not reimplementing the
model -- this reproduces its per-layer op sequence exactly, function-for-function, adding only the
perturbation hook and the per-layer capture the monolithic function has no seam for):

  - Perturbation is injected on `causal_attention`'s output -- the mha_decode kernel's own
    boundary -- for every layer, per head, at a magnitude drawn per-head from
    Uniform[1.650e-02, 5.354e-02] (the measured range). This is realistic-relative and per-head,
    not a single Gaussian at the aggregate value, per the task's explicit design requirement.
  - The direction of the injected error is an isotropic random unit vector, scaled to hit the
    target rel-L2 EXACTLY (`rel_l2 = ||err|| / ||ctx|| `, `golden_hd128.rel_l2`'s own definition).
    We have no measurement of the true error's directional structure (systematic bf16
    round-off vs effectively-random across the 512x512 mmul-free MAC accumulation this kernel
    does) -- isotropic-random is the neutral assumption absent that evidence, and is explicitly
    flagged as a scope limit in the final report, not asserted as fact.
  - ONLY THE LAST TOKEN ROW is perturbed at each layer, not the whole sequence. This is not a
    simplification, it is the CORRECT model of AR decoding: causal attention means every row's
    computation depends only on itself and earlier rows, so a real KV cache's already-committed
    history is never retroactively touched by the current step's compute error -- rows 0..S-2 are
    bit-identical between the baseline and perturbed run at every layer, by construction, exactly
    as they would be on real hardware. Only the trajectory of the CURRENT query (which is also
    `hidden_last`, the thing the greedy decision is made from) is being asked "what happens to you
    after 36 rounds of this error".
  - Per-head magnitudes are redrawn independently every layer (`--eps-mode iid`, default) since
    there is no measurement of whether a given head's error is correlated across layers.
    `--eps-mode correlated` draws ONE per-head profile and reuses it at all 36 layers, as a
    sensitivity check on that assumption -- see the report for whether it changes the verdict.

USAGE (reproduces the reported numbers):
    python3 designs/mha_decode/attn_error_compounding_sim.py \\
        --n-tokens 16 32 64 --n-seeds 50 --eps-mode iid
    python3 designs/mha_decode/attn_error_compounding_sim.py \\
        --n-tokens 32 --n-seeds 50 --eps-mode correlated
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))
import s2_ar_ref as ar_ref  # noqa: E402
import golden_hd128 as golden  # noqa: E402  (reused only for its rel_l2 definition)

# Device-measured facts (task brief, 2026-09-02; not yet a committed KB slug -- see report caveat).
MEASURED_AGGREGATE = 3.361e-02
MEASURED_PER_HEAD_LO = 1.650e-02
MEASURED_PER_HEAD_HI = 5.354e-02


def build_prompt(hp: ar_ref.ARHParams, n_tokens: int, seed: int) -> np.ndarray:
    """Synthetic flat_tokens prompt, same construction as `s2_ar_ref.py`'s own __main__
    self-check (real tokenizer/prompt-template is out of scope, per that module's docstring)."""
    rng = np.random.default_rng(seed)
    flat = np.zeros((n_tokens, hp.codebook_dim), dtype=np.int64)
    flat[:, 0] = hp.semantic_begin_id + rng.integers(0, hp.codebook_size, size=n_tokens)
    for cb in range(hp.num_codebooks):
        flat[:, cb + 1] = rng.integers(0, hp.codebook_size, size=n_tokens)
    return flat


def perturb_last_row(ctx: np.ndarray, eps_per_head: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ctx: (n_tokens, n_head, head_dim) causal_attention output. Returns a copy with only the
    LAST row offset by an isotropic-random-direction vector whose rel-L2 against that row's
    original per-head vector is exactly eps_per_head[h] -- see module docstring for why only the
    last row."""
    out = ctx.copy()
    n_head, head_dim = ctx.shape[1], ctx.shape[2]
    last = ctx[-1].astype(np.float64)
    d = rng.standard_normal((n_head, head_dim))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    ctx_norm = np.linalg.norm(last, axis=-1, keepdims=True)
    out[-1] = (last + eps_per_head[:, None] * ctx_norm * d).astype(np.float32)
    return out


def run_step0(hp: ar_ref.ARHParams, w: ar_ref.ARWeights, flat_tokens: np.ndarray,
              noise_rng: np.random.Generator | None, eps_lo: float, eps_hi: float,
              eps_mode: str) -> dict:
    """Reproduces `slow_transformer_forward`'s per-layer op sequence op-for-op (same
    `s2_ar_ref` calls, same order), adding the attention-output perturbation hook and per-layer
    last-token capture that function has no seam for. `noise_rng=None` is the clean baseline path
    -- same code, zero injected error."""
    n_tokens = flat_tokens.shape[0]
    semantic_ids = flat_tokens[:, 0]
    is_semantic = (semantic_ids >= hp.semantic_begin_id) & (semantic_ids <= hp.semantic_end_id)

    x = w.embedding_rows(semantic_ids).astype(np.float32)
    codebook_sum = np.zeros_like(x)
    for cb in range(hp.num_codebooks):
        raw_ids = flat_tokens[:, cb + 1]
        ids = np.where(is_semantic, raw_ids + cb * hp.codebook_size, cb * hp.codebook_size)
        codebook_sum += w.codebook_embedding_rows(ids)
    x = x + codebook_sum * is_semantic[:, None].astype(np.float32)
    if hp.scale_codebook_embeddings:
        sem_scale = 1.0 / np.sqrt(hp.codebook_dim)
        token_scale = np.where(is_semantic, sem_scale, 1.0).astype(np.float32)
        x = x * token_scale[:, None]

    n_head, n_head_kv = hp.head_count, hp.head_count_kv
    head_dim = hp.head_dim
    q_size, kv_size = n_head * head_dim, n_head_kv * head_dim
    n_rep = n_head // n_head_kv
    scale = 1.0 / np.sqrt(head_dim)
    positions = np.arange(n_tokens)

    per_layer_hidden_last = []
    realized_eps = []
    # correlated mode: one per-head magnitude profile shared by all 36 layers.
    fixed_eps = (noise_rng.uniform(eps_lo, eps_hi, size=n_head)
                 if (noise_rng is not None and eps_mode == "correlated") else None)

    for il in range(hp.block_count):
        lw = w.slow_layer(il)
        attn_in = ar_ref.rms_norm(x, lw.attention_norm, hp.rms_norm_eps)
        qkv = ar_ref.linear(attn_in, lw.wqkv)
        q = qkv[:, 0:q_size].reshape(n_tokens, n_head, head_dim)
        k = qkv[:, q_size:q_size + kv_size].reshape(n_tokens, n_head_kv, head_dim)
        v = qkv[:, q_size + kv_size:q_size + 2 * kv_size].reshape(n_tokens, n_head_kv, head_dim)

        if hp.attention_qk_norm:
            q = ar_ref.rms_norm(q, lw.q_norm, hp.rms_norm_eps)
            k = ar_ref.rms_norm(k, lw.k_norm, hp.rms_norm_eps)

        q = ar_ref.rope_interleaved(q, positions, head_dim, hp.rope_freq_base)
        k = ar_ref.rope_interleaved(k, positions, head_dim, hp.rope_freq_base)

        k_rep = ar_ref.repeat_kv(k, n_rep)
        v_rep = ar_ref.repeat_kv(v, n_rep)
        attn = ar_ref.causal_attention(q, k_rep, v_rep, scale)  # (n_tokens, n_head, head_dim)

        if noise_rng is not None:
            eps_per_head = fixed_eps if fixed_eps is not None else noise_rng.uniform(eps_lo, eps_hi, size=n_head)
            attn_noisy = perturb_last_row(attn, eps_per_head, noise_rng)
            realized_eps.append(golden.rel_l2(attn_noisy[-1], attn[-1]))
            attn = attn_noisy

        attn = attn.reshape(n_tokens, q_size)
        attn_out = ar_ref.linear(attn, lw.wo)
        h = x + attn_out
        ff_in = ar_ref.rms_norm(h, lw.ffn_norm, hp.rms_norm_eps)
        ff_out = ar_ref.swiglu_ffn(ff_in, lw.w1, lw.w2, lw.w3)
        x = h + ff_out
        per_layer_hidden_last.append(x[-1].copy())

    slow_out = ar_ref.rms_norm(x, w.norm, hp.rms_norm_eps)
    hidden_last = slow_out[-1]
    return dict(hidden_last=hidden_last, per_layer_hidden_last=per_layer_hidden_last,
                realized_eps=realized_eps)


def masked_argmax_step0(hp: ar_ref.ARHParams, logits: np.ndarray, mask_row_ids: np.ndarray):
    """Same masking as `generate_greedy`'s step-0 call (`block_im_end=True`)."""
    biased = logits.copy()
    if hp.im_end_id >= 0:
        biased[mask_row_ids == hp.im_end_id] = -np.inf
    order = np.argsort(biased)[::-1]
    top1, top2 = order[0], order[1]
    return int(mask_row_ids[top1]), float(biased[top1] - biased[top2]), top1, top2, biased


def sweep(hp, w, mask_row_ids, mask_rows, n_tokens, n_seeds, eps_lo, eps_hi, eps_mode,
          prompt_seed, seed_base):
    flat = build_prompt(hp, n_tokens, prompt_seed)

    base = run_step0(hp, w, flat, None, eps_lo, eps_hi, eps_mode)
    base_logits = mask_rows @ base["hidden_last"]
    base_token, base_margin, base_top1, base_top2, _ = masked_argmax_step0(hp, base_logits, mask_row_ids)

    n_layers = hp.block_count
    layer_rel_l2 = np.zeros((n_seeds, n_layers), dtype=np.float64)
    final_rel_l2 = np.zeros(n_seeds, dtype=np.float64)
    flips = np.zeros(n_seeds, dtype=bool)
    pert_margin_at_pair = np.zeros(n_seeds, dtype=np.float64)
    realized_eps_all = []

    for s in range(n_seeds):
        rng = np.random.default_rng(seed_base + s)
        pert = run_step0(hp, w, flat, rng, eps_lo, eps_hi, eps_mode)
        for il in range(n_layers):
            layer_rel_l2[s, il] = golden.rel_l2(pert["per_layer_hidden_last"][il],
                                                 base["per_layer_hidden_last"][il])
        final_rel_l2[s] = golden.rel_l2(pert["hidden_last"], base["hidden_last"])

        pert_logits = mask_rows @ pert["hidden_last"]
        pert_biased = pert_logits.copy()
        if hp.im_end_id >= 0:
            pert_biased[mask_row_ids == hp.im_end_id] = -np.inf
        pert_token = int(mask_row_ids[np.argmax(pert_biased)])
        flips[s] = pert_token != base_token
        pert_margin_at_pair[s] = float(pert_biased[base_top1] - pert_biased[base_top2])
        realized_eps_all.extend(pert["realized_eps"])

    return dict(n_tokens=n_tokens, base_token=base_token, base_margin=base_margin,
                layer_rel_l2=layer_rel_l2, final_rel_l2=final_rel_l2, flips=flips,
                pert_margin_at_pair=pert_margin_at_pair,
                realized_eps_all=np.array(realized_eps_all))


def print_report(res, eps_mode):
    n_tokens = res["n_tokens"]
    n_seeds = len(res["flips"])
    print(f"\n=== n_tokens={n_tokens}  n_seeds={n_seeds}  eps_mode={eps_mode} ===")
    print(f"baseline step-0 greedy token = {res['base_token']}  "
          f"top1-top2 logit margin = {res['base_margin']:.4f}")

    re = res["realized_eps_all"]
    print(f"realized per-(layer,head) injected rel-L2: mean={re.mean():.3e} "
          f"min={re.min():.3e} max={re.max():.3e}  "
          f"(target range [{MEASURED_PER_HEAD_LO:.3e}, {MEASURED_PER_HEAD_HI:.3e}], "
          f"measured aggregate {MEASURED_AGGREGATE:.3e})")

    flips = res["flips"]
    print(f"greedy token flip rate: {flips.sum()}/{n_seeds} = {100.0 * flips.mean():.1f}%")

    pm = res["pert_margin_at_pair"]
    drift = res["base_margin"] - pm
    print(f"margin at baseline's top-2 pair under perturbation: "
          f"mean={pm.mean():.4f} min={pm.min():.4f} max={pm.max():.4f}")
    print(f"margin drift (positive = margin shrank toward a flip): "
          f"mean={drift.mean():.4f} min={drift.min():.4f} max={drift.max():.4f}")

    lrl = res["layer_rel_l2"]
    layer_mean = lrl.mean(axis=0)
    layer_max = lrl.max(axis=0)
    print("layer-by-layer hidden-state rel-L2 (last-token row, vs unperturbed run):")
    for il in range(0, lrl.shape[1], 4):
        print(f"  layer {il:2d}: mean={layer_mean[il]:.3e}  max={layer_max[il]:.3e}")
    last = lrl.shape[1] - 1
    print(f"  layer {last:2d}: mean={layer_mean[last]:.3e}  max={layer_max[last]:.3e}  (last)")
    fr = res["final_rel_l2"]
    print(f"post-final-norm hidden_last rel-L2: mean={fr.mean():.3e} max={fr.max():.3e}")

    early = layer_mean[:6].mean()
    late = layer_mean[-6:].mean()
    print(f"trend check: mean rel-L2 over layers 0-5 = {early:.3e}, over last 6 layers = {late:.3e}, "
          f"ratio late/early = {late / early:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-tokens", type=int, nargs="+", default=[32])
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--eps-mode", choices=["iid", "correlated"], default="iid")
    ap.add_argument("--eps-lo", type=float, default=MEASURED_PER_HEAD_LO)
    ap.add_argument("--eps-hi", type=float, default=MEASURED_PER_HEAD_HI)
    ap.add_argument("--prompt-seed", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    gguf_path = ar_ref.codec_paths.gguf()
    print(f"GGUF: {gguf_path}")
    gg = ar_ref.open_gguf(gguf_path)
    hp = ar_ref.read_ar_hparams(gg)
    print(f"hparams: block_count={hp.block_count} head_count={hp.head_count}/{hp.head_count_kv} "
          f"head_dim={hp.head_dim} attention_qk_norm={hp.attention_qk_norm}")
    assert hp.head_dim == 128 and hp.head_count == 32 and hp.head_count_kv == 8, (
        "this checkpoint's slow-attention shape does not match the measured mha_decode config "
        "(HD=128, 32/8 heads) -- the injected-error model below assumes it does")

    t0 = time.time()
    w = ar_ref.ARWeights(gg, hp)
    sem_lo, sem_hi = hp.semantic_begin_id, hp.semantic_end_id
    mask_row_ids = np.arange(sem_lo, sem_hi + 1)
    if hp.im_end_id >= 0:
        mask_row_ids = np.concatenate([mask_row_ids, [hp.im_end_id]])
    mask_rows = w.logits_rows(mask_row_ids)  # dequantized once, reused by every run below

    for n_tokens in args.n_tokens:
        res = sweep(hp, w, mask_row_ids, mask_rows, n_tokens, args.n_seeds, args.eps_lo,
                    args.eps_hi, args.eps_mode, args.prompt_seed, args.seed_base)
        print_report(res, args.eps_mode)

    print(f"\ntotal wall time (incl. one-time q6_k dequant of all {hp.block_count} layers): "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
