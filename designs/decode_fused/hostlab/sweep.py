"""Batch driver: one model, many format arms. Loads the model once and restores a pristine
bf16 copy of every touched tensor between arms, so an arm costs a forward pass rather than
a model load."""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import wq_formats as F
from wq_eval import load_model, baseline_bf16, run, paired, tokenize, TARGETS

BF16 = F.BF16


def touched(model, targets):
    sfx = tuple(s for t in targets if t != "head" for s in TARGETS[t])
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(name.endswith(s) for s in sfx) or (name == "lm_head" and "head" in targets):
            yield name, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--arms", required=True, help="JSON list of {name,spec,targets}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--no-kl", action="store_true", help="skip the logprob memmap (saves 1.2GB/corpus)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    arms = json.loads(open(a.arms).read()) if os.path.exists(a.arms) else json.loads(a.arms)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    ids = tokenize(a.corpus, a.tokens, tok)

    model = load_model()
    baseline_bf16(model)
    all_t = ["mlp", "attn_o", "qkv", "head"]
    pristine = {n: m.weight.detach().numpy().astype(BF16).copy()
                for n, m in touched(model, all_t)}
    print(f"pristine cache {sum(v.nbytes for v in pristine.values())/1e9:.2f} GB", flush=True)

    V = model.config.vocab_size
    refp = f"{QLAB}/runs/{a.tag}-ref.npy"
    out_mm = None if a.no_kl else np.lib.format.open_memmap(refp, "w+", np.float32,
                                                            (a.tokens, V))
    t0 = time.time()
    r0 = run(model, ids, out_mm=out_mm)
    np.save(f"{QLAB}/runs/{a.tag}--bf16-control.nll.npy", r0["nll"])
    if out_mm is not None:
        out_mm.flush(); del out_mm
    ref_mm = None if a.no_kl else np.load(refp, mmap_mode="r")
    base = dict(name="bf16-control", spec={"scheme": "bf16"}, targets=[],
                mean_nll=float(r0["nll"].mean()), ppl=float(np.exp(r0["nll"].mean())),
                top1_acc=float((r0["top1"] == r0["tgt"]).mean()), bits=16.0,
                weight_mb=1192.0, secs=round(time.time() - t0, 1))
    results = [base]
    print(json.dumps(base), flush=True)

    for arm in arms:
        t0 = time.time()
        # An arm may name ONE spec for a list of classes, or a per_class map so different weight
        # classes take different formats -- the per-class sensitivity sweep says a uniform format
        # is the wrong shape (damage per byte saved differs 2.6x across classes).
        per_class = arm.get("per_class")
        tg = list(per_class) if per_class else arm["targets"]
        for n, m in touched(model, all_t):          # restore everything, then quantize
            m.weight.data = torch.from_numpy(pristine[n].astype(np.float32))
        npar = 0; qbits = 0.0
        for cls in tg:
            sp_c = per_class[cls] if per_class else arm["spec"]
            b_c = (16.0 if sp_c["scheme"] == "bf16" else
                   F.bits_per_element(sp_c.get("nbits", 4), sp_c["group"], sp_c["scheme"],
                                      sp_c.get("scale_dtype", "bf16"),
                                      sp_c.get("min_dtype", "bf16")))
            for n, m in touched(model, [cls]):
                W = m.weight.detach().numpy()
                m.weight.data = torch.from_numpy(F.apply(W, sp_c))
                npar += W.size; qbits += W.size * b_c
        r = run(model, ids, ref_mm=ref_mm)
        sp = per_class or arm["spec"]
        bits = qbits / npar if npar else 16.0
        rec = dict(name=arm["name"], spec=sp, targets=tg, params=int(npar), bits=bits,
                   quant_mb=qbits / 8 / 1e6,
                   weight_mb=(596.05e6 - npar) * 2 / 1e6 + qbits / 8 / 1e6,
                   mean_nll=float(r["nll"].mean()), ppl=float(np.exp(r["nll"].mean())),
                   top1_acc=float((r["top1"] == r["tgt"]).mean()),
                   paired=paired(r["nll"], r0["nll"]), secs=round(time.time() - t0, 1))
        if ref_mm is not None:
            rec.update(kl_mean=float(np.nanmean(r["kl"])),
                       kl_p99=float(np.nanpercentile(r["kl"], 99)),
                       top1_agree=float(np.nanmean(r["agree"])),
                       ref_rank_mean=float(np.nanmean(r["rrank"])))
        np.save(f"{QLAB}/runs/{a.tag}--{arm['name']}.nll.npy", r["nll"])
        results.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(results, open(f"{QLAB}/runs/{a.tag}.json", "w"), indent=1)
    if not a.no_kl:
        os.remove(refp)


if __name__ == "__main__":
    main()
