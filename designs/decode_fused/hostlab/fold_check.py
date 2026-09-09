"""Gate 0, done properly.

awq_eval's gate0 applies the fold with spec=bf16 and got max_abs_nll_delta = 0.0 -- but with no
quantization the alpha search picks alpha=0, s becomes all-ones, and the fold is the identity. It
proved nothing. This forces a NON-TRIVIAL per-channel scale and checks the model's output is
unchanged, which is what makes the AWQ arm interpretable at all: if diag(1/s) does not really fold
into the norm weight and up_proj's rows, arm B measures a broken model, not a calibrated one.
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/qlab")
import sys, numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wq_eval import load_model, baseline_bf16, run, tokenize
import awq

torch.set_num_threads(6)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
ids = tokenize(QLAB + "/corpora/wikitext2.txt", 512, tok)

m = load_model(); baseline_bf16(m)
r0 = run(m, ids)

# force a real scale: deterministic, per-channel, spanning 4x, geometric-mean normalised
def forced(Ws, X, absmean, spec, alphas=None):
    rng = np.random.default_rng(0)
    s = np.exp(rng.uniform(-0.7, 0.7, size=absmean.shape)).astype(np.float32)
    s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
    return 0.5, s

m2 = load_model(); baseline_bf16(m2)
orig = awq.search_alpha
awq.search_alpha = forced
awq.apply_awq(m2, ids, {"scheme": "bf16"}, classes=("mlp", "qkv"), verbose=False)
awq.search_alpha = orig
r1 = run(m2, ids)

d = np.abs(r1["nll"] - r0["nll"])
print(f"forced non-trivial fold (per-channel scale spanning "
      f"{np.exp(1.4):.1f}x): max |dNLL| = {d.max():.3e}  mean {d.mean():.3e}")
print(f"ppl {np.exp(r0['nll'].mean()):.4f} -> {np.exp(r1['nll'].mean()):.4f}")
print("VERDICT:", "PASS -- the fold is neutral" if d.max() < 2e-3 else "FAIL -- the fold changes the model")
