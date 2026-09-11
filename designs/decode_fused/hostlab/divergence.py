"""Greedy-generation divergence: the thing perplexity cannot see.

Perplexity is a teacher-forced average -- every position is scored against the TRUE prefix, so a
format that shifts the distribution slightly at every step gets a small average penalty and the
measurement never lets those shifts compound. Real generation does let them compound: one flipped
argmax changes the prefix, and everything after it is a different trajectory.

So this measures the FIRST position at which greedy decoding diverges from the bf16 control, over
many prompts. Read it as a distribution, never as a single number, and never read "the outputs
differ" as a failure: greedy decoding is chaotic at ties, and this project has already recorded an
8-token oracle whose step 5 is a three-way bf16 tie broken by index. What the distribution shows
is the SCALE of trajectory stability -- diverging at token 3 and diverging at token 90 are
different products, and both are "different output".
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import wq_formats as F
from wq_eval import load_model, baseline_bf16, TARGETS
from awq import apply_awq


def quantize(model, spec, targets):
    sfx = tuple(s for t in targets if t != "head" for s in TARGETS[t])
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and (
                any(name.endswith(s) for s in sfx) or (name == "lm_head" and "head" in targets)):
            mod.weight.data = torch.from_numpy(
                F.apply(mod.weight.detach().numpy().astype(np.float32), spec))


@torch.no_grad()
def gen(model, batch, n_new):
    out = model.generate(**batch, max_new_tokens=n_new, do_sample=False, num_beams=1,
                         use_cache=True, pad_token_id=151643)
    return out[:, batch["input_ids"].shape[1]:].numpy()


def prompts_from(path, tok, n, plen):
    txt = open(path, encoding="utf-8", errors="replace").read()
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    step = max(plen, len(ids) // (n + 1))
    return [ids[i * step:i * step + plen] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=QLAB + "/corpora/natural-prose.txt")
    ap.add_argument("--calib", default=QLAB + "/corpora/wikitext2.txt")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--arms", required=True, help="JSON list of {name,spec,targets,awq}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=18)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    P = prompts_from(a.corpus, tok, a.prompts, a.prompt_len)
    cal = tok(open(a.calib).read(), add_special_tokens=False)["input_ids"][:512]
    batch = dict(input_ids=torch.tensor(P), attention_mask=torch.ones(len(P), a.prompt_len,
                                                                     dtype=torch.long))
    m = load_model(); baseline_bf16(m)
    ref = gen(m, batch, a.new)
    print(f"control generated {ref.shape}", flush=True)

    res = []
    for arm in json.loads(open(a.arms).read()):
        m = load_model(); baseline_bf16(m)
        if arm.get("awq"):
            apply_awq(m, cal, arm["spec"], classes=tuple(arm["awq"]), verbose=False)
        quantize(m, arm["spec"], arm["targets"])
        g = gen(m, batch, a.new)
        first = np.array([int(np.argmax(g[i] != ref[i])) if (g[i] != ref[i]).any() else a.new
                          for i in range(len(P))])
        rec = dict(name=arm["name"], n_prompts=len(P), n_new=a.new,
                   identical=int((first == a.new).sum()),
                   first_div_median=float(np.median(first)),
                   first_div_mean=float(first.mean()),
                   first_div_p10=float(np.percentile(first, 10)),
                   diverged_by_8=int((first < 8).sum()),
                   diverged_by_32=int((first < 32).sum()),
                   matched_token_frac=float((g == ref).mean()))
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(f"{QLAB}/runs/{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
