"""Activation-aware channel scaling (AWQ-style), measured on this model, and FOLDED so it
costs nothing at inference.

The format decides how many levels a group gets; the calibration decides which weights those
levels are spent on. Quantizing W is equivalent to quantizing W*diag(s) and dividing the input
by s, for any positive s -- but the two are NOT equivalent after rounding, because s changes
which channels dominate their group's range. AWQ picks s from activation magnitude so that
channels the activations actually excite get more effective resolution.

It is free ONLY if diag(1/s) can be folded into something upstream. In Qwen3 it can, for the
two classes that matter, because the norm output feeds nothing but the projection (the residual
bypasses it):
    q,k,v_proj   <- input_layernorm.weight            (one shared s, dim d_model)
    gate,up_proj <- post_attention_layernorm.weight   (one shared s, dim d_model)
    down_proj    <- up_proj's OUTPUT rows             (the SwiGLU product is elementwise, so
                                                       scaling up's row c scales down's input
                                                       channel c; dim d_ff)
o_proj is skipped: its input channels map to v_proj rows through the GQA repeat, so no per-channel
fold exists -- and it measures near-null under quantization anyway, so there is nothing to win.
"""
import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
import sys, numpy as np, torch
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import wq_formats as F


@torch.no_grad()
def collect_inputs(model, ids, names, max_rows=512):
    """Mean |x| per input channel, plus a subsample of raw rows for the MSE search."""
    store = {}
    hooks = []

    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            a = store.setdefault(n, {"absmean": torch.zeros(x.shape[1]), "n": 0, "rows": []})
            a["absmean"] += x.abs().sum(0)
            a["n"] += x.shape[0]
            if sum(r.shape[0] for r in a["rows"]) < max_rows:
                step = max(1, x.shape[0] // 128)
                a["rows"].append(x[::step].clone())
        return hook

    mods = dict(model.named_modules())
    for n in names:
        hooks.append(mods[n].register_forward_hook(mk(n)))
    model.model(input_ids=torch.tensor(np.asarray(ids)[None, :]))
    for h in hooks:
        h.remove()
    for n, a in store.items():
        a["absmean"] = (a["absmean"] / a["n"]).numpy()
        a["X"] = torch.cat(a["rows"])[:max_rows].numpy()
        del a["rows"]
    return store


def search_alpha(Ws, X, absmean, spec, alphas=np.arange(0, 1.01, 0.1)):
    """Pick the exponent minimising the GROUP's output error. One s for all Ws that share X."""
    am = np.maximum(absmean, 1e-8)
    best, best_a, best_s = None, 0.0, np.ones_like(am)
    for a in alphas:
        s = am ** a
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)   # AWQ's normalisation
        loss = 0.0
        for W in Ws:
            Weff = F.apply(W * s[None, :], spec) / s[None, :]
            loss += float(np.square((Weff - W) @ X.T).sum())
        if best is None or loss < best:
            best, best_a, best_s = loss, float(a), s
    return best_a, best_s


def apply_awq(model, ids, spec, classes=("mlp", "qkv"), verbose=True):
    """Fold an activation-aware scale into the model IN PLACE. Returns the chosen alphas."""
    L = len(model.model.layers)
    groups = []
    for i in range(L):
        p = f"model.layers.{i}."
        if "qkv" in classes:
            groups.append(("qkv", i, [p + "self_attn.q_proj", p + "self_attn.k_proj",
                                      p + "self_attn.v_proj"], p + "input_layernorm"))
        if "mlp" in classes:
            groups.append(("gu", i, [p + "mlp.gate_proj", p + "mlp.up_proj"],
                           p + "post_attention_layernorm"))
            groups.append(("down", i, [p + "mlp.down_proj"], p + "mlp.up_proj"))
    names = sorted({n for _, _, ns, _ in groups for n in ns})
    if verbose:
        print(f"collecting activations for {len(names)} linears...", flush=True)
    store = collect_inputs(model, ids, names)

    mods = dict(model.named_modules())
    chosen = []
    for kind, i, ns, host in groups:
        Ws = [mods[n].weight.detach().numpy().astype(np.float32) for n in ns]
        st = store[ns[0]]
        a, s = search_alpha(Ws, st["X"], st["absmean"], spec)
        for n, W in zip(ns, Ws):
            mods[n].weight.data = torch.from_numpy(W * s[None, :])
        h = mods[host]
        if isinstance(h, torch.nn.Linear):        # fold into the host's OUTPUT rows
            h.weight.data = torch.from_numpy(
                h.weight.detach().numpy().astype(np.float32) / s[:, None])
        else:                                     # an RMSNorm: fold into its per-channel weight
            h.weight.data = torch.from_numpy(
                h.weight.detach().numpy().astype(np.float32) / s)
        chosen.append((kind, i, a))
    if verbose:
        for k in ("qkv", "gu", "down"):
            al = [a for kk, _, a in chosen if kk == k]
            if al:
                print(f"  alpha[{k}]: mean {np.mean(al):.2f} min {min(al):.1f} max {max(al):.1f}")
    return chosen
