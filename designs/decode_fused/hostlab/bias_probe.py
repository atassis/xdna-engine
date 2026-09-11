"""Why does the affine format reconstruct BETTER yet score WORSE on perplexity at coarse groups?

Reconstruction error (rel-L2 on W) is not the quantity the model sees. The model sees the error in
W @ x. Those differ when the per-weight errors are CORRELATED along the reduction axis: a coherent
bias of e per weight contributes e*sum(x) to the dot product, while a zero-mean error contributes
only ~e*sqrt(sum(x^2)). At K=1024 that is a factor of ~32 between the two regimes, which is enough
to invert a ranking built on ||W_q - W||.

So measure three things per scheme, on real tensors and real activations:
  relL2_W      reconstruction error of the weight            (what the KB sweep measured)
  relL2_Wx     error of the actual matvec                    (what the model sees)
  bias_frac    the share of the matvec error explained by the per-group MEAN error, i.e. the
               coherent part -- computed by splitting each group's weight error into its mean
               (coherent) and its residual (incoherent) and propagating each separately.
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
import sys, glob, numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import wq_formats as F
from wq_eval import load_model, baseline_bf16, tokenize
from awq import collect_inputs

TARGETS = [(f"model.layers.{L}.mlp.{p}", ) for L in (0, 13, 27)
           for p in ("gate_proj", "up_proj", "down_proj")]
NAMES = [t[0] for t in TARGETS]


def main():
    torch.set_num_threads(8)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    ids = tokenize(QLAB + "/corpora/wikitext2.txt", 512, tok)
    m = load_model(); baseline_bf16(m)
    store = collect_inputs(m, ids, NAMES, max_rows=256)
    mods = dict(m.named_modules())

    print(f"{'scheme':16s} {'g':>4s} {'bits':>5s} {'relL2_W':>9s} {'relL2_Wx':>9s} "
          f"{'bias_frac':>9s}")
    for scheme in ("sym", "affine"):
        for g in (32, 64, 128, 256):
            aW = aWx = 0.0; nW = nWx = 0.0; bias_e = 0.0
            for n in NAMES:
                W = mods[n].weight.detach().numpy().astype(np.float32)
                X = store[n]["X"]
                spec = dict(scheme=scheme, nbits=4, group=g, kernel_round=False,
                            scale_dtype="f32" if scheme == "sym" else "bf16")
                Wq = F.apply(W, spec)
                E = Wq - W
                M, K = W.shape
                Eg = E.reshape(M, K // g, g)
                Emean = Eg.mean(axis=2, keepdims=True)              # coherent part per group
                Ecoh = np.broadcast_to(Emean, Eg.shape).reshape(M, K)
                aW += float(np.square(E).sum()); nW += float(np.square(W).sum())
                dY = E @ X.T
                dYc = Ecoh @ X.T
                aWx += float(np.square(dY).sum())
                nWx += float(np.square(W @ X.T).sum())
                bias_e += float(np.square(dYc).sum())
            print(f"{scheme:16s} {g:>4d} "
                  f"{F.bits_per_element(4, g, scheme, 'f32' if scheme=='sym' else 'bf16'):5.2f} "
                  f"{np.sqrt(aW/nW):9.4f} {np.sqrt(aWx/nWx):9.4f} {bias_e/aWx:9.3f}")


main()
