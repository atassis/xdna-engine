#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-node rel-L2 bisect of a fused decode ELF against an f32 host reference.

One dispatch runs the whole stack and every intermediate is its OWN named buffer in the arena, so a
single device run yields every node. Compare each against the host value computed from the SAME
weights, walk forward, and report the first node that leaves the bf16 noise floor -- that names the
kernel to look at instead of the model.

Node order per layer mirrors the runlist in gen_llm_decode.py. In-place ops (q/k through qk-norm and
RoPE, sc through scale, g through the activation, and -- under sandwich_norms -- a/d through their
own post-op RMSNorm) can only be read at their FINAL value, so those rows compare the end of the
chain, not each step. Run at pos 0 so attention is a no-op rotation and softmax is over one element:
that isolates the projections, norms and FFN from the KV path.
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, isolate_build_dir, COLS  # noqa: E402
from llm_decode_spec import k_chunks_for  # noqa: E402

BF16 = ml_dtypes.bfloat16


def rel_l2(ref, got):
    ref = np.asarray(ref, np.float64).ravel()
    got = np.asarray(got, np.float64).ravel()
    n = min(len(ref), len(got))
    d = np.linalg.norm(ref[:n] - got[:n])
    r = np.linalg.norm(ref[:n])
    return float(d / r) if r else float(d)


def identity_rope_row(width):
    """RoPE table row at pos 0: ang = pos*inv is zero for every frequency regardless of theta or
    partial-rotary width, so cos=1/sin=0 everywhere and rotation is the identity -- the whole
    reason this tool can probe at pos 0. Only the buffer's own WIDTH is table-specific (Gemma-4's
    rope_global is 512 wide, rope_local 256), so it is read off the device buffer, not a spec-wide
    head_dim.
    """
    row = np.zeros(width, np.float32)
    row[0::2] = 1.0
    return np.asarray(row, BF16)


def split_matvec(W, x, n):
    """Host mirror of gen_llm_decode.py's split_over_k: an unsplit GEMV, or K cut into `n` chunks,
    each partial rounded to bf16 and folded PAIRWISE in the same tree order the runlist builds --
    mv.cc rounds its f32 accumulator to bf16 once PER PARTIAL, so the fold order changes which
    roundings compound. Used for o_proj and the FFN down projection, the two GEMVs whose K does
    not fit L1 at some of Gemma-4's shapes (llm_decode_spec.k_chunks_for picks `n`).
    """
    if n == 1:
        return W @ x
    rnd = lambda v: np.asarray(v, BF16).astype(np.float32)  # noqa: E731
    K = x.shape[-1]
    cw = K // n
    level = [rnd(W[:, i * cw:(i + 1) * cw] @ x[i * cw:(i + 1) * cw]) for i in range(n)]
    while len(level) > 1:
        nxt = [rnd(level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="qwen3-0.6b")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--token", type=int, default=785)
    ap.add_argument("--detail-layers", type=int, default=2)
    ap.add_argument("--detail-from", type=int, default=None,
                    help="also print node detail for layers >= this index")
    a = ap.parse_args()
    # Same isolation verify_llm_decode does. Without it the build lands in CWD and
    # params.txt is not found, so the ParameterScratchpad never binds and every
    # per-token write fails -- which shows up as `params` being None at depth.
    isolate_build_dir("bisect")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers)
    NL, D, FF = md["NL"], sp.d_model, sp.ffn
    Hq, EPS = sp.n_q_heads, sp.eps
    down_chunks = k_chunks_for(D, FF, COLS)
    print(f"[bisect] {sp.name}: {NL} layers, token {a.token}, pos 0")

    c = fused.get_callable()
    params = c.params
    for n, arr in weights.items():
        with c.get_buffer(n).overwrite() as _buf:
            _buf[:] = np.asarray(arr, BF16).reshape(-1)

    def npy(n):
        # mmap_mode + copy=False, the same idiom gen_llm_decode.py's own npy() carries and for the
        # same reason: the dump is f32 on disk and the tied embedding is the whole 262144x3840
        # table, 3.75 GiB. Reading it into anonymous memory and then copying it again -- astype()
        # copies even when the dtype ALREADY matches -- is 7.5 GiB resident for one tensor. That is
        # what OOM-kills a 12B run, and depth does not reduce it. Every consumer here is read-only.
        return np.load(os.path.join(a.weights, f"{n}.npy"),
                       mmap_mode="r").astype(np.float32, copy=False)

    def load_norm(name):
        # Gemma-3 stores RMSNorm gain as w with the kernel computing x_hat*(1+w); Gemma-4 and
        # Qwen3 store it already absolute (x_hat*w) -- mirrors gen_llm_decode.py's load_norm.
        w = npy(name)
        return 1.0 + w if sp.norm_gain == "one_plus_w" else w

    embed = npy(f"{sp.weight_prefix}embed_tokens.weight")
    # embed_scale: Gemma multiplies the input embedding by sqrt(d_model) before layer 0; Qwen3
    # does not. Applies ONLY to the input gather, never to the tied lm-head multiply below.
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    x0 = embed[a.token].astype(np.float32) * scale
    with c.get_buffer("x").overwrite() as _buf:
        _buf[:] = np.asarray(x0, BF16).reshape(-1)
    # Every declared kv-offset slot and every declared RoPE table, not a single fixed pair:
    # Gemma-4's sliding and global layers keep their KV caches at different (slot, head_dim)
    # offsets, and its two RoPE tables differ in width as well as theta. At pos 0 the row content
    # is the identity regardless (see identity_rope_row), so only the SET of declared buffers and
    # their widths matter here.
    for slot_name, _ in md["kv_slots"]:
        params.write(slot_name, 0)
    for ang_name in ("rope_global", "rope_local"):
        if ang_name in md["inputs"]:
            with c.get_buffer(ang_name).overwrite() as _buf:
                _buf[:] = identity_rope_row(_buf.size)
    params.write("sm_mask", 1)
    params.sync()
    c()
    print("[bisect] dispatch complete\n")

    def dev(name):
        return np.asarray(c.get_buffer(name).data, np.float32)

    def rms(x, w=None):
        x = np.asarray(x, np.float32)
        xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
        return xn if w is None else xn * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -60, 60)))

    def gelu_tanh(x):
        x = np.asarray(x, np.float64)
        return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))).astype(np.float32)

    def act(x):
        return silu(x) if sp.act == "silu" else gelu_tanh(x)

    print(f"{'node':14} {'rel-L2':>11}   {'|host|':>10} {'|dev|':>10}")
    print("-" * 52)
    x = x0
    first_bad = None
    prev_e = 0.0
    for l in range(NL):
        p = f"L{l}_"
        detail = l < a.detail_layers or (a.detail_from is not None and l >= a.detail_from)
        hd, hkv = sp.head_dim_for(l), sp.n_kv_heads_for(l)
        has_v, gqa = sp.has_v_proj(l), Hq // hkv
        qd = Hq * hd
        o_chunks = k_chunks_for(D, qd, COLS)
        wp = f"{sp.weight_prefix}layers.{l}."

        def w(name):
            return npy(wp + name)

        nm = sp.norm_weight_names(l)
        n_in, n_pf = load_norm(nm["n_in"]), load_norm(nm["n_pf"])
        n_qn = load_norm(nm["n_qn"]) if sp.qk_norm else None
        n_kn = load_norm(nm["n_kn"]) if sp.qk_norm else None

        rows = []
        hn = rms(x, n_in);                                       rows.append(("hn", hn))
        q_raw = w("self_attn.q_proj.weight") @ hn
        k_raw = w("self_attn.k_proj.weight") @ hn
        # attention_k_eq_v: layers with no v_proj tensor read V straight off the RAW k
        # projection, BEFORE qk-norm/RoPE mutate k -- the checkpoint binds value_states to
        # key_states and rebinds key_states afterwards, so v_raw must come from k_raw here, not
        # from the normed/rotated `k` computed below.
        v_raw = (w("self_attn.v_proj.weight") @ hn) if has_v else k_raw
        if sp.qk_norm:
            q = np.concatenate([rms(q_raw.reshape(Hq, hd)[i], n_qn) for i in range(Hq)])
            k = np.concatenate([rms(k_raw.reshape(hkv, hd)[i], n_kn) for i in range(hkv)])
        else:
            q, k = q_raw, k_raw
        # v_norm: a GAINLESS RMSNorm (with_scale=False, no weight tensor) on every layer's value
        # path. RoPE never touches v.
        v = (np.concatenate([rms(v_raw.reshape(hkv, hd)[i]) for i in range(hkv)])
             if sp.v_norm else v_raw)
        rows += [("q", q), ("k", k), ("v", v)]
        # No attn_scale or softmax here: softmax over a SINGLE score (n_past+1 == 1 at pos 0) is
        # 1 regardless of the score's value, so ctx is exactly the GQA broadcast of v for any
        # attn_scale -- fixed (Gemma-4) or head_dim-derived (Qwen3) alike.
        ctx = np.stack([v.reshape(hkv, hd)[h // gqa] for h in range(Hq)]).reshape(-1)
        rows.append(("cx", ctx))
        av = split_matvec(w("self_attn.o_proj.weight"), ctx, o_chunks)
        if sp.sandwich_norms:
            av = rms(av, load_norm(nm["n_pa"]))            # in place on `a`, final value only
        rows.append(("a", av))
        x1 = x + av;                                             rows.append(("x1", x1))
        hf = rms(x1, n_pf);                                      rows.append(("hf", hf))
        g = act(w("mlp.gate_proj.weight") @ hf)
        u = w("mlp.up_proj.weight") @ hf
        rows += [("g", g), ("u", u)]
        gh = g * u;                                              rows.append(("gh", gh))
        d = split_matvec(w("mlp.down_proj.weight"), gh, down_chunks)
        if sp.sandwich_norms:
            d = rms(d, load_norm(nm["n_pff"]))              # in place on `d`, final value only
        rows.append(("d", d))
        xnext = x1 + d
        if sp.layer_scalar:
            # The trained per-layer scalar, applied to the block output AFTER both residual adds
            # -- it scales the residual stream itself, so it cannot fold into any op above.
            ls = float(npy(sp.layer_scalar_name(l))[0])
            xnext = ls * xnext
        x = xnext
        for nm_, ref in rows:
            if not detail:
                continue
            try:
                got = dev(p + nm_)
            except Exception as e:
                print(f"{p+nm_:14} unreadable: {type(e).__name__}")
                continue
            e = rel_l2(ref, got)
            gn = np.linalg.norm(np.asarray(got, np.float64).ravel()[:ref.size])
            print(f"{p+nm_:14} {e:11.4e}   {np.linalg.norm(ref):10.3f} {gn:10.3f}")
            if first_bad is None and e > 0.05:
                first_bad = (p + nm_, e)
        nxt = f"x{l+1}"
        try:
            gx = dev(nxt)
            e = rel_l2(x, gx)
            print(f"{'  -> ' + nxt:14} {e:11.4e}")
            # A jump in this curve has two very different causes and the per-quarter profile
            # tells them apart: a PARTIAL WRITE leaves one region at the previous value (or zero)
            # while the rest is exact, whereas a wrong computation is wrong roughly uniformly.
            # Print the profile only when the step is anomalous, so the normal curve stays
            # readable.
            if e > 2.5 * prev_e and prev_e > 0:
                qtr = len(x) // 4
                parts = " ".join(f"q{i}={rel_l2(x[i*qtr:(i+1)*qtr], gx[i*qtr:(i+1)*qtr]):.3e}"
                                 for i in range(4))
                zeros = int((np.abs(np.asarray(gx, np.float64).ravel()[:len(x)]) == 0).sum())
                print(f"{'':14} JUMP {e/prev_e:.1f}x -> {parts}  exact-zeros={zeros}/{len(x)}")
            prev_e = e
            if first_bad is None and e > 0.05:
                first_bad = (nxt, e)
        except Exception:
            pass
        # No early exit: the whole curve is the evidence. A single node above a threshold says
        # little when the error accumulates -- the SHAPE of the growth is what distinguishes a
        # biased per-op error from faithful bf16 rounding.

    # final norm + tied lm-head. `logits` is the RAW device output -- final_logit_softcapping is
    # a host-only post-process the runlist never applies (see verify_llm_decode.py), so neither
    # side of this comparison carries it.
    xf = rms(x, load_norm(f"{sp.weight_prefix}norm.weight"))
    try:
        print(f"{'  -> xf':14} {rel_l2(xf, dev('xf')):11.4e}")
    except Exception:
        pass
    # Chunked so the tied embedding stays a paged mmap rather than being pulled into one
    # anonymous 3.75 GiB temporary by the matmul.
    host_logits = np.empty(embed.shape[0], np.float32)
    for lo in range(0, embed.shape[0], 16384):
        hi = min(lo + 16384, embed.shape[0])
        host_logits[lo:hi] = np.asarray(embed[lo:hi], np.float32) @ xf
    try:
        dl = dev("logits")[:len(host_logits)]
        print(f"{'  -> logits':14} {rel_l2(host_logits, dl):11.4e}   "
              f"host argmax {int(np.argmax(host_logits))}  dev argmax {int(np.argmax(dl))}")
    except Exception:
        pass
    print()
    print(f"*** first node above 0.05: {first_bad[0]} at {first_bad[1]:.4e} ***" if first_bad
          else "*** no node above 0.05 ***")


if __name__ == "__main__":
    main()
