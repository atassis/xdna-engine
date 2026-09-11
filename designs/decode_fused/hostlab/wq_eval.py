"""Host-side model-level quality lab for weight formats, Qwen3-0.6B on CPU.

Why host: the device harness (designs/decode_fused/eval_llm_perplexity.py) costs an xclbin
build plus one NPU dispatch per token and is single-tenant, which makes a FORMAT SWEEP
impossible. A format changes only which numbers the weights hold, so the information it
destroys is measurable without the device. The device then confirms ONE chosen point.

The instrument is anchored, not assumed: validate_formats.py shows the `sym` path is
bit-identical to the shipped packer (iron/operators/gemv/quant.py) and reproduces the
recorded rel-L2 sweep to four decimals at every group size.

Metrics, and what each can and cannot see -- see the report; briefly:
  nll/ppl        distributional shift vs the TEXT. Corpus-dependent, and the ratio is
                 exp(delta mean NLL), so it depends on the delta only, not the base level.
  kl_ref_arm     distributional shift vs the BF16 MODEL. No ground truth involved, so it
                 measures what the format did rather than what the corpus rewards.
  top1_agree     fraction of positions where greedy decoding would pick the same token.
                 This is the metric that predicts whether generation diverges.
  ref_rank       rank the reference's argmax falls to under the arm. Sees near-misses.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wq_formats as F

MODEL = "Qwen/Qwen3-0.6B"
TARGETS = {
    "mlp":    ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
    "attn_o": ("self_attn.o_proj",),
    "qkv":    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
}


def load_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             local_files_only=True)
    m.eval()
    # Qwen3-0.6B ties lm_head to embed_tokens. Our engine SPLITS them: the device GEMV
    # reads a quantized W_head while the host gathers embeddings from a bf16 sidecar
    # (gen_llm_decode.py:1062-1073). Untie here so the split is reproduced and quantizing
    # the head never touches the embedding input.
    m.lm_head.weight = torch.nn.Parameter(m.lm_head.weight.detach().clone())
    return m


def apply_format(model, spec, targets):
    """Returns (n_tensors, n_params, bytes_before, bytes_after)."""
    if spec["scheme"] == "bf16" and not targets:
        return 0, 0, 0, 0
    bpe = (16.0 if spec["scheme"] == "bf16" else
           F.bits_per_element(spec.get("nbits", 4), spec["group"], spec["scheme"],
                              spec.get("scale_dtype", "bf16"),
                              spec.get("min_dtype", "bf16")))
    n_t = n_p = 0
    suffixes = tuple(s for t in targets if t != "head" for s in TARGETS[t])
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        hit = any(name.endswith(s) for s in suffixes) or \
              (name == "lm_head" and "head" in targets)
        if not hit:
            continue
        W = mod.weight.detach().numpy().astype(np.float32)
        mod.weight.data = torch.from_numpy(F.apply(W, spec))
        n_t += 1
        n_p += W.size
    return n_t, n_p, n_p * 2, int(n_p * bpe / 8)


def baseline_bf16(model):
    """Every Linear and the embedding rounded to bf16 -- the CONTROL. The device stores
    bf16 weights, so an f32 host model is not the right reference: it would fold a
    precision change into every format delta."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = torch.from_numpy(F._round_to(
                mod.weight.detach().numpy(), "bf16"))
    e = model.model.embed_tokens
    e.weight.data = torch.from_numpy(F._round_to(e.weight.detach().numpy(), "bf16"))


@torch.no_grad()
def run(model, ids, ref_mm=None, out_mm=None, slice_sz=256):
    """Teacher-forced over ONE sequence: position t is conditioned on 0..t, matching the
    device harness's growing-KV loop. Returns per-position arrays."""
    T = len(ids) - 1
    x = torch.tensor(np.asarray(ids[:T + 1])[None, :])
    h = model.model(input_ids=x).last_hidden_state[0]          # [T+1, d]
    tgt = np.asarray(ids[1:T + 1])
    nll = np.empty(T, np.float64); top1 = np.empty(T, np.int64)
    kl = np.full(T, np.nan); agree = np.full(T, np.nan); rrank = np.full(T, np.nan)
    for a in range(0, T, slice_sz):
        b = min(a + slice_sz, T)
        lg = model.lm_head(h[a:b]).float()
        lp = torch.log_softmax(lg.double(), dim=-1).numpy()
        nll[a:b] = -lp[np.arange(b - a), tgt[a:b]]
        top1[a:b] = lp.argmax(-1)
        if out_mm is not None:
            out_mm[a:b] = lp.astype(np.float32)
        if ref_mm is not None:
            rp = np.asarray(ref_mm[a:b], np.float64)
            p = np.exp(rp)
            kl[a:b] = (p * (rp - lp)).sum(-1)                  # KL(ref || arm), nats
            ra = rp.argmax(-1)
            agree[a:b] = (ra == top1[a:b])
            # rank the reference's argmax falls to under the arm (0 = still the argmax)
            rrank[a:b] = (lp > lp[np.arange(b - a), ra][:, None]).sum(-1)
        del lg, lp
    return dict(nll=nll, top1=top1, tgt=tgt, kl=kl, agree=agree, rrank=rrank)


def paired(d_arm, d_ref):
    """Paired stats on per-position NLL. The corpus cancels; a difference of means does not."""
    d = d_arm - d_ref
    n = len(d); mu = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    t = mu / se if se > 0 else float("nan")
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    return dict(n=n, mean_delta_nats=float(mu), se=float(se), t=float(t),
                ppl_ratio=float(np.exp(mu)),
                ppl_pct=float((np.exp(mu) - 1) * 100),
                ppl_pct_ci=[float((np.exp(lo) - 1) * 100), float((np.exp(hi) - 1) * 100)],
                frac_worse=float((d > 0).mean()), median_delta=float(np.median(d)))


def tokenize(path, n, tok):
    txt = open(path, encoding="utf-8", errors="replace").read()
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    if len(ids) < n + 1:
        raise SystemExit(f"{path}: only {len(ids)} tokens, need {n+1}")
    return ids[:n + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--spec", required=True, help='JSON, e.g. {"scheme":"affine","group":32}')
    ap.add_argument("--targets", default="", help="comma list: mlp,attn_o,qkv,head")
    ap.add_argument("--ref-mm", default=None, help="reference logprob memmap to compare against")
    ap.add_argument("--write-ref", default=None, help="write this run's logprobs as the reference")
    ap.add_argument("--ref-nll", default=None, help="reference per-position NLL .npy for pairing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("QLAB_THREADS", "18")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tokenize(a.corpus, a.tokens, tok)

    spec = json.loads(a.spec)
    targets = [t for t in a.targets.split(",") if t]
    t0 = time.time()
    model = load_model()
    baseline_bf16(model)
    n_t, n_p, b0, b1 = apply_format(model, spec, targets)

    T = a.tokens
    V = model.config.vocab_size
    out_mm = ref_mm = None
    if a.write_ref:
        out_mm = np.lib.format.open_memmap(a.write_ref, "w+", np.float32, (T, V))
    if a.ref_mm:
        ref_mm = np.load(a.ref_mm, mmap_mode="r")
    r = run(model, ids, ref_mm=ref_mm, out_mm=out_mm)
    if out_mm is not None:
        out_mm.flush()

    res = dict(corpus=os.path.basename(a.corpus), tokens=T, spec=spec, targets=targets,
               tensors=n_t, params=int(n_p), mb_before=b0 / 1e6, mb_after=b1 / 1e6,
               mean_nll=float(r["nll"].mean()), ppl=float(np.exp(r["nll"].mean())),
               top1_acc=float((r["top1"] == r["tgt"]).mean()),
               secs=round(time.time() - t0, 1))
    if ref_mm is not None:
        res.update(kl_mean=float(np.nanmean(r["kl"])),
                   kl_p50=float(np.nanpercentile(r["kl"], 50)),
                   kl_p99=float(np.nanpercentile(r["kl"], 99)),
                   top1_agree=float(np.nanmean(r["agree"])),
                   ref_rank_mean=float(np.nanmean(r["rrank"])))
    if a.ref_nll:
        res["paired"] = paired(r["nll"], np.load(a.ref_nll))
    if a.out:
        np.save(a.out + ".nll.npy", r["nll"])
        json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
