"""Two axes perplexity cannot see, and an honest statement of what each one can.

TASK ACCURACY (ARC-Easy, length-normalised log-likelihood over the four choices). Perplexity is
an average over ALL next tokens; most of them are syntax the model gets right regardless. Task
accuracy asks whether the model still ranks a correct ANSWER above three distractors -- a
different question, and the one a user notices. What it CANNOT see: generation quality, format
compliance, anything requiring more than one forward pass. At 0.6B absolute accuracy is modest,
so the number to read is RETENTION on the subset the bf16 control gets right; the rest is noise
about items the model never knew.

FORMAT COMPLIANCE. Instruction-tuned behaviour is the failure mode nobody measures: a scheme can
hold perplexity and stop obeying. These prompts have a mechanically checkable answer (exact token,
JSON, a fixed-length list) so compliance is a rate, not a judgement. What it CANNOT see: anything
about quality of open-ended text; it is a floor test, not a capability test. Small N, so read it
as a tripwire and never as a percentage with a decimal point.
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/qlab")
import argparse, json, re, sys
import numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import wq_formats as F
from wq_eval import load_model, baseline_bf16, TARGETS

PROBES = [
    ("Answer with exactly one word: what colour is a clear daytime sky?", r"^\W*blue\b"),
    ("Reply with only the number, no words: what is 17 plus 26?", r"^\W*43\b"),
    ("Output valid JSON with exactly the keys \"name\" and \"age\" for a person called Ada aged 36. "
     "Output only the JSON.", r"\{[^}]*\"name\"[^}]*\"age\"[^}]*\}"),
    ("List exactly three fruits, one per line, nothing else.", r"^\s*\S.*\n\S.*\n\S.*$"),
    ("Answer YES or NO only: is the Pacific larger than the Atlantic?", r"^\W*(YES|yes)\b"),
    ("Translate to French, output only the translation: the cat sleeps.",
     r"(?i)le\s+chat\s+dort"),
    ("Repeat this word back exactly five times separated by spaces: apple",
     r"(?i)(apple\s+){4}apple"),
    ("Write the first five positive even integers separated by commas, nothing else.",
     r"2\s*,\s*4\s*,\s*6\s*,\s*8\s*,\s*10"),
]


def quantize(model, arm):
    pc = arm.get("per_class")
    for cls, sp in (pc.items() if pc else [(t, arm["spec"]) for t in arm["targets"]]):
        sfx = tuple(TARGETS[cls]) if cls != "head" else ()
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and (
                    any(name.endswith(s) for s in sfx) or (name == "lm_head" and cls == "head")):
                mod.weight.data = torch.from_numpy(
                    F.apply(mod.weight.detach().numpy().astype(np.float32), sp))


@torch.no_grad()
def arc(model, tok, items, batch=16):
    """Length-normalised NLL of each choice continuation; argmin wins."""
    correct = np.zeros(len(items), bool)
    for i0 in range(0, len(items), batch):
        chunk = items[i0:i0 + batch]
        seqs, owner, gold = [], [], []
        for j, it in enumerate(chunk):
            q = "Question: " + it["question"] + "\nAnswer:"
            qi = tok(q, add_special_tokens=False)["input_ids"]
            for c in it["choices"]["text"]:
                ci = tok(" " + c, add_special_tokens=False)["input_ids"]
                seqs.append((qi, ci)); owner.append(j)
            gold.append(it["choices"]["label"].index(it["answerKey"])
                        if it["answerKey"] in it["choices"]["label"] else -1)
        L = max(len(a) + len(b) for a, b in seqs)
        ids = np.full((len(seqs), L), 151643, np.int64)
        mask = np.zeros((len(seqs), L), np.int64)
        tgt = np.full((len(seqs), L), -100, np.int64)
        for r, (a, b) in enumerate(seqs):                 # left pad
            s = L - len(a) - len(b)
            ids[r, s:] = a + b; mask[r, s:] = 1
            tgt[r, s + len(a):] = b
        lg = model(input_ids=torch.tensor(ids),
                   attention_mask=torch.tensor(mask)).logits.float()
        lp = torch.log_softmax(lg[:, :-1], -1)
        t = torch.tensor(tgt[:, 1:])
        keep = t >= 0
        g = torch.gather(lp, 2, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = (-(g * keep).sum(1) / keep.sum(1).clamp(min=1)).numpy()
        owner = np.array(owner)
        for j in range(len(chunk)):
            sel = owner == j
            if gold[j] >= 0:
                correct[i0 + j] = int(np.argmin(nll[sel])) == gold[j]
    return correct


@torch.no_grad()
def compliance(model, tok):
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
            for p, _ in PROBES]
    enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left")
    out = model.generate(**enc, max_new_tokens=48, do_sample=False, num_beams=1,
                         pad_token_id=151643)
    txt = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [bool(re.search(rx, t.strip(), re.M)) for t, (_, rx) in zip(txt, PROBES)], txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-arc", type=int, default=800)
    ap.add_argument("--threads", type=int, default=18)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    items = pq.read_table(QLAB + "/corpora/arc-easy-test.parquet").to_pylist()[:a.n_arc]

    m = load_model(); baseline_bf16(m)
    base = arc(m, tok, items)
    bc, btxt = compliance(m, tok)
    print(json.dumps(dict(arm="bf16-control", arc=float(base.mean()),
                          compliance=f"{sum(bc)}/{len(bc)}")), flush=True)
    res = []
    for arm in json.loads(open(a.arms).read()):
        m = load_model(); baseline_bf16(m); quantize(m, arm)
        c = arc(m, tok, items)
        cc, _ = compliance(m, tok)
        res.append(dict(arm=arm["name"], arc=float(c.mean()),
                        arc_delta_pts=float((c.mean() - base.mean()) * 100),
                        retention_on_base_correct=float(c[base].mean()),
                        recovered_from_base_wrong=float(c[~base].mean()),
                        compliance=f"{sum(cc)}/{len(cc)}",
                        compliance_lost=[i for i in range(len(cc)) if bc[i] and not cc[i]]))
        print(json.dumps(res[-1]), flush=True)
    json.dump(dict(control=dict(arc=float(base.mean()), compliance=f"{sum(bc)}/{len(bc)}",
                                texts=btxt), arms=res),
              open(f"{QLAB}/runs/{a.tag}.json", "w"), indent=1)


main()
