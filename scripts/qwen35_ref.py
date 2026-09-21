#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text-path reference in f32, holding one layer's weights at a time.

The gated delta rule runs in its RECURRENT form -- one token per step, state carried -- because that
is the form the NPU runs. HF runs the chunked form whenever there is no cache, so `--check-hf N`
compares the two at truncated depth: agreement there is the evidence this file is the model and not a
paraphrase of it. Streaming exists because the bf16 checkpoint is ~9 GB and this box has ~5 GB free
with the NPU service resident.

  python scripts/qwen35_ref.py --ckpt /mnt/data/xdna/artifacts/qwen3.5-4b/hf --check-hf 4
  python scripts/qwen35_ref.py --ckpt ... --prompt "The capital of France is" --out golden.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

PFX = "model.language_model."


# The matrices the decode rail packs; the 32-row in_proj_a/b gates, norms and conv stay bf16.
QUANT_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj")


def _iron_quant():
    """IRON's packer, loaded by path: the `iron` package imports the device toolchain, this file only
    numpy. IRON_DIR resolves as scripts/amd_paths.sh does."""
    import importlib.util
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.environ.get("IRON_DIR") or os.path.join(ws, "wt-iron-integ"),
                        "iron", "common", "quant.py")
    spec = importlib.util.spec_from_file_location("iron_quant", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def int4_roundtrip(w, group, clip_search=False, full_range=False):
    """Quantize-dequantize through the device packer, scale narrowed to bf16 as mv_quant.cc does."""
    q = _iron_quant()
    W = w.numpy()
    packed = q.quantize_weight(W, group, "int4", clip_search=clip_search, full_range=full_range)
    return torch.from_numpy(q.dequantize_weight(packed, *W.shape, group, "int4",
                                                emulate_kernel_scale_cast=True))


class Ckpt:
    def __init__(self, root, int4_group=0, clip_search=False, full_range=False):
        self.root = root
        self.int4_group = int4_group
        self.int4_kw = {"clip_search": clip_search, "full_range": full_range}
        self.cfg = json.load(open(os.path.join(root, "config.json")))["text_config"]
        wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.shard = wm
        self._open = {}
        self._memo = {}

    def _f(self, name):
        path = os.path.join(self.root, self.shard[name])
        if path not in self._open:
            self._open[path] = safe_open(path, framework="pt")
        return self._open[path]

    def w(self, name):
        """Memoised over one layer's worth of tensors, so a layer-major pass packs each matrix once."""
        if name not in self._memo:
            if len(self._memo) >= 16:
                self._memo.pop(next(iter(self._memo)))
            self._memo[name] = self._load(name)
        return self._memo[name]

    def _load(self, name):
        t = self._f(PFX + name).get_tensor(PFX + name).float()
        if self.int4_group and name.endswith(".weight") and name.split(".")[-2] in QUANT_LEAVES:
            t = int4_roundtrip(t, self.int4_group, **self.int4_kw)
        return t

    def rows(self, name, lo, hi):
        return self._f(PFX + name).get_slice(PFX + name)[lo:hi].float()


def rms(x, w, eps):
    """Zero-centred RMSNorm: every norm in this model but the DeltaNet output gate scales by 1 + w."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class DeltaNetState:
    def __init__(self, cfg):
        c = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"] * 2 + \
            cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        self.conv = torch.zeros(cfg["linear_conv_kernel_dim"] - 1, c)
        self.S = torch.zeros(cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                             cfg["linear_value_head_dim"])


class KVState:
    def __init__(self):
        self.k = None
        self.v = None


def deltanet(ck, i, x, st):
    cfg = ck.cfg
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    p = f"layers.{i}.linear_attn."
    T = x.shape[0]
    qkv = x @ ck.w(p + "in_proj_qkv.weight").T
    z = (x @ ck.w(p + "in_proj_z.weight").T).view(T, nv, dv)
    b = x @ ck.w(p + "in_proj_b.weight").T
    a = x @ ck.w(p + "in_proj_a.weight").T

    cw = ck.w(p + "conv1d.weight").squeeze(1)  # [C, K]
    K = cw.shape[1]
    xp = torch.cat([st.conv, qkv], 0)
    qkv = F.silu(sum(xp[j:j + T] * cw[:, j] for j in range(K)))
    st.conv = xp[-(K - 1):].clone()

    q, k, v = torch.split(qkv, [nk * dk, nk * dk, nv * dv], -1)
    rep = nv // nk
    q = (l2norm(q.view(T, nk, dk)) * dk ** -0.5).repeat_interleave(rep, 1)
    k = l2norm(k.view(T, nk, dk)).repeat_interleave(rep, 1)
    v = v.view(T, nv, dv)
    beta = torch.sigmoid(b)
    alpha = torch.exp(-ck.w(p + "A_log").exp() * F.softplus(a + ck.w(p + "dt_bias")))

    S = st.S
    o = torch.empty(T, nv, dv)
    for t in range(T):
        S = S * alpha[t, :, None, None]
        err = (v[t] - torch.einsum("hkv,hk->hv", S, k[t])) * beta[t, :, None]
        S = S + k[t, :, :, None] * err[:, None, :]
        o[t] = torch.einsum("hkv,hk->hv", S, q[t])
    st.S = S

    nw = ck.w(p + "norm.weight")
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg["rms_norm_eps"]) * nw * F.silu(z)
    return o.reshape(T, nv * dv) @ ck.w(p + "out_proj.weight").T


def rope_tables(cfg, pos):
    hd = cfg["head_dim"]
    rp = cfg["rope_parameters"]
    rd = int(hd * rp.get("partial_rotary_factor", 1.0))
    inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    f = torch.outer(torch.as_tensor(pos, dtype=torch.float64), inv)
    f = torch.cat([f, f], -1)
    return f.cos().float(), f.sin().float()


def rope(x, cos, sin):
    rd = cos.shape[-1]
    r, keep = x[..., :rd], x[..., rd:]
    half = torch.cat([-r[..., rd // 2:], r[..., :rd // 2]], -1)
    return torch.cat([r * cos[:, None] + half * sin[:, None], keep], -1)


def attention(ck, i, x, st, pos0):
    cfg = ck.cfg
    nh, nkv, hd, eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
    p = f"layers.{i}.self_attn."
    T = x.shape[0]
    qg = (x @ ck.w(p + "q_proj.weight").T).view(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    q = rms(q, ck.w(p + "q_norm.weight"), eps)
    k = rms((x @ ck.w(p + "k_proj.weight").T).view(T, nkv, hd), ck.w(p + "k_norm.weight"), eps)
    v = (x @ ck.w(p + "v_proj.weight").T).view(T, nkv, hd)
    cos, sin = rope_tables(cfg, range(pos0, pos0 + T))
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    st.k = k if st.k is None else torch.cat([st.k, k], 0)
    st.v = v if st.v is None else torch.cat([st.v, v], 0)
    n = st.k.shape[0]
    kk = st.k.repeat_interleave(nh // nkv, 1)
    vv = st.v.repeat_interleave(nh // nkv, 1)
    s = torch.einsum("thd,nhd->htn", q, kk) * hd ** -0.5
    mask = torch.arange(n)[None, :] > (torch.arange(T)[:, None] + n - T)
    s = s.masked_fill(mask[None], float("-inf"))
    o = torch.einsum("htn,nhd->thd", torch.softmax(s, -1), vv).reshape(T, nh * hd)
    return (o * torch.sigmoid(gate)) @ ck.w(p + "o_proj.weight").T


def mlp(ck, i, x):
    p = f"layers.{i}.mlp."
    return (F.silu(x @ ck.w(p + "gate_proj.weight").T) * (x @ ck.w(p + "up_proj.weight").T)) @ \
        ck.w(p + "down_proj.weight").T


class Model:
    def __init__(self, ck, n_layers=None):
        self.ck = ck
        self.L = n_layers or ck.cfg["num_hidden_layers"]
        self.types = ck.cfg["layer_types"][:self.L]
        self.state = [DeltaNetState(ck.cfg) if t == "linear_attention" else KVState() for t in self.types]
        self.pos = 0

    def embed(self, ids):
        emb = self.ck._f(PFX + "embed_tokens.weight").get_slice(PFX + "embed_tokens.weight")
        return torch.stack([emb[int(t):int(t) + 1][0].float() for t in ids])

    def forward(self, ids, keep_layers=False):
        """Run `ids` from the current position; returns the pre-final-norm hidden [T, D]."""
        ck, eps = self.ck, self.ck.cfg["rms_norm_eps"]
        x = self.embed(ids)
        per_layer = []
        for i, t in enumerate(self.types):
            h = rms(x, ck.w(f"layers.{i}.input_layernorm.weight"), eps)
            if t == "linear_attention":
                x = x + deltanet(ck, i, h, self.state[i])
            else:
                x = x + attention(ck, i, h, self.state[i], self.pos)
            x = x + mlp(ck, i, rms(x, ck.w(f"layers.{i}.post_attention_layernorm.weight"), eps))
            if keep_layers:
                per_layer.append(x.clone())
        self.pos += len(ids)
        return (x, per_layer) if keep_layers else x

    def forward_many(self, seqs):
        """Layer-major pass over independent sequences from position 0: each layer is loaded once."""
        ck, eps = self.ck, self.ck.cfg["rms_norm_eps"]
        xs = [self.embed(ids) for ids in seqs]
        for i, t in enumerate(self.types):
            for n, x in enumerate(xs):
                h = rms(x, ck.w(f"layers.{i}.input_layernorm.weight"), eps)
                if t == "linear_attention":
                    x = x + deltanet(ck, i, h, DeltaNetState(ck.cfg))
                else:
                    x = x + attention(ck, i, h, KVState(), 0)
                xs[n] = x + mlp(ck, i, rms(x, ck.w(f"layers.{i}.post_attention_layernorm.weight"), eps))
        return xs

    def final(self, x):
        return rms(x, self.ck.w("norm.weight"), self.ck.cfg["rms_norm_eps"])

    def logits(self, h, chunk=16384):
        """Tied head over the whole vocab, streamed in row chunks: the f32 table alone is 2.5 GB."""
        V = self.ck.cfg["vocab_size"]
        return torch.cat([h @ self.ck.rows("embed_tokens.weight", lo, min(lo + chunk, V)).T
                          for lo in range(0, V, chunk)], -1)


def check_hf(ck, ids, n):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = AutoConfig.from_pretrained(ck.root).text_config
    cfg.num_hidden_layers = n
    cfg.layer_types = cfg.layer_types[:n]
    cfg.vocab_size = 1  # fed inputs_embeds, so the 2.5 GB f32 table never materialises
    hf = Qwen3_5TextModel(cfg).float().eval()
    for name, p in hf.state_dict().items():
        if name != "embed_tokens.weight":
            p.copy_(ck.w(name))
    m = Model(ck, n)
    with torch.no_grad():
        want = hf(inputs_embeds=m.embed(ids)[None], use_cache=False).last_hidden_state[0]
    del hf
    got = m.final(m.forward(ids))
    rel = ((got - want).norm() / want.norm()).item()
    print(f"[check-hf] layers={n} types={cfg.layer_types} T={len(ids)} rel-L2 recurrent vs HF chunked = {rel:.3e}")
    return rel


def check_incremental(ck, ids, n):
    """Prefill-all-at-once and one-token-at-a-time must agree: the state carry is the decode protocol."""
    a = Model(ck, n)
    want = a.forward(ids)
    b = Model(ck, n)
    got = torch.cat([b.forward([t]) for t in ids], 0)
    rel = ((got - want).norm() / want.norm()).item()
    print(f"[check-incremental] layers={n} T={len(ids)} rel-L2 per-token vs one-shot = {rel:.3e}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/data/xdna/artifacts/qwen3.5-4b/hf")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--check-hf", type=int, default=0, metavar="N")
    ap.add_argument("--check-incremental", type=int, default=0, metavar="N")
    ap.add_argument("--steps", type=int, default=0, help="greedy continuation tokens")
    ap.add_argument("--int4-group", type=int, default=0,
                    help="run the projections through the int4 packer at this group size")
    ap.add_argument("--clip-search", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-logits", default=None,
                    help="npz of the logits at every fed position (prompt, then the greedy tokens) "
                         "in verify_llm_decode.py --dump-logits' own layout")
    ap.add_argument("--ref-json", default=None,
                    help="write {prompt_ids, gen_ids, margins} for designs/decode_fused/verify_llm_decode.py")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    ck = Ckpt(a.ckpt, a.int4_group, a.clip_search)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    ids = tok.encode(a.prompt, add_special_tokens=False)
    print(f"[ref] prompt={a.prompt!r} ids={ids}")

    rc = 0
    if a.check_hf:
        rc |= check_hf(ck, ids, a.check_hf) > 1e-4
    if a.check_incremental:
        rc |= check_incremental(ck, ids, a.check_incremental) > 1e-4
    if a.check_hf or a.check_incremental:
        return rc

    m = Model(ck, a.layers)
    x, per_layer = m.forward(ids, keep_layers=True)
    all_lg = [m.logits(m.final(x))] if a.dump_logits else []
    lg = (all_lg[0][-1] if all_lg else m.logits(m.final(x[-1:]))[0])
    gen, margins = [], []
    for _ in range(a.steps):
        top = torch.topk(lg, 2)
        gen.append(int(top.indices[0]))
        margins.append(float(top.values[0] - top.values[1]))
        lg = m.logits(m.final(m.forward([gen[-1]])))[0]
        if a.dump_logits:
            all_lg.append(lg[None])
    print(f"[ref] top5 next={torch.topk(lg if not gen else lg, 5).indices.tolist()} gen={gen} "
          f"text={tok.decode(gen)!r} margins={[round(x, 3) for x in margins]}")
    if a.dump_logits:
        np.savez(a.dump_logits, logits=torch.cat(all_lg).numpy(), tokens_fed=np.array(ids + gen))
        print(f"[ref] wrote {a.dump_logits}")
    if a.ref_json:
        json.dump({"prompt": a.prompt, "prompt_ids": ids, "gen_ids": gen, "margins": margins,
                   "layers": m.L, "source": "scripts/qwen35_ref.py f32 recurrent reference"},
                  open(a.ref_json, "w"))
        print(f"[ref] wrote {a.ref_json}")
    if a.out:
        np.savez(a.out, ids=np.array(ids), gen=np.array(gen), margins=np.array(margins),
                 hidden=torch.stack(per_layer).numpy(), final=m.final(x).numpy())
        print(f"[ref] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
