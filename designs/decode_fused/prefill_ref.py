#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The prefill dataflows in pure numpy, parameterised on WHERE the value is narrowed.

Two arms out of one definition, which is the whole point of the file:

  * `rnd=bf16` -- the device-faithful golden. Rounds to bf16 exactly where the device does, so it
    models the datapath. This is what the generators have always emitted and what the device-side
    rel-L2 probes compare against.
  * `rnd=f32`  -- the TIER 1 reference. Same dataflow, same bf16 INPUTS the device was handed, but
    no intermediate is narrowed. This is what `scripts/gate_numeric.py` gates against.

The second arm exists because the first cannot be a correctness reference for itself. A bf16 golden
and a bf16 device agree on precisely the rounding this gate is meant to measure; the residual it
would report is the difference between two roundings, not the distance from the right answer.
Measured on this rail, that distinction is worth a factor of 2.6: layer-0 V against a float64
reference is 1.658e-3 for the M=1 GEMV path and 4.332e-3 for the batched GEMM, while the two
devices' disagreement with EACH OTHER is 1.18 bf16 ULP and says nothing about which is closer.

No IRON import here, deliberately. The generators need IRON to build; refreshing an artifact's
goldens needs nothing but numpy, and `scripts/refresh_prefill_goldens.py` relies on that -- it can
re-gate an artifact that was built weeks ago on a toolchain that no longer exists.
"""
import json
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from gate_numeric import (ATOL_MARGIN, ATOL_RULE, RTOL, atol_for,  # noqa: E402
                          bf16_floor, mean_rel_l1, rms)

BF16 = ml_dtypes.bfloat16


def bf16(a):
    """Narrow to bf16 -- the device-faithful arm's `rnd`."""
    return np.asarray(a).astype(BF16)


def f32(a):
    """Keep the value -- the high-precision arm's `rnd`, which narrows nothing."""
    return np.asarray(a, np.float32)


# ------------------------------------------------------------------------------------------------
# Primitives. Each takes bf16 (or f32) operands, computes in f32, and narrows only via `rnd`.
# ------------------------------------------------------------------------------------------------
def rms_norm(v, w, eps, rnd):
    f = np.asarray(v, np.float32)
    s = f / np.sqrt((f * f).mean(-1, keepdims=True) + eps)
    return rnd(s * np.asarray(w, np.float32))


def matmul_t(a, b_nk, rnd):
    """A @ B^T with B stored `[Nout, K]` -- the `b_col_maj` read the device does.

    Our weights are stored `[Nout, K]` because that is what decode's GEMV wants, and prefill reads
    THE SAME BYTES out of the shared arena. A reference that transposed them first would be
    checking a layout the device never sees.
    """
    return rnd(np.asarray(a, np.float32) @ np.asarray(b_nk, np.float32).T)


def matmul(a, b, rnd):
    """A @ B with B already `[K, Nout]` -- the plain (non-`b_col_maj`) GEMM."""
    return rnd(np.asarray(a, np.float32) @ np.asarray(b, np.float32))


def rope_block(x, table, heads, rnd=bf16):
    """RoPE in `rope/design.py`'s BLOCK convention: token t's angle row covers `heads` rows.

    `x` is `[M, heads, HD]` token-major; `table` is `[M, HD]` interleaved `[cos, sin, ...]`.
    `rope/reference.py` TILES the angle table instead, and the two agree only at angle_rows 1 or
    rows -- decode only ever runs 1, which is why nothing noticed until a batch existed.
    """
    f = np.asarray(x, np.float32)
    ang = np.asarray(table, np.float32)
    cos, sin = ang[:, 0::2][:, None, :], ang[:, 1::2][:, None, :]
    half = f.shape[-1] // 2
    x1, x2 = f[..., :half], f[..., half:]
    return rnd(np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1))


def softmax_rows(s, widths, rnd=bf16):
    """Row-wise softmax with one unmasked width per row.

    `widths` is None (attend everything) or `[rows]` int: row i is softmaxed over `s[i, :w[i]]` and
    the tail is zero. That models `mask_bf16` writing -inf past the width and `softmax_bf16`
    exponentiating the whole row -- the MASK, not the device's rounding.
    """
    f = np.asarray(s, np.float32)
    if widths is not None:
        keep = np.arange(f.shape[-1])[None, :] < np.asarray(widths, np.int64)[:, None]
        f = np.where(keep, f, -np.inf)
    e = np.exp(f - f.max(-1, keepdims=True))
    e = np.nan_to_num(e, nan=0.0)
    return rnd(e / e.sum(-1, keepdims=True))


def gated_ffn_act(g, act, rnd):
    """SiLU or tanh-GELU on the gate branch.

    `exp(-g)` overflows f32 below g = -88, which a deep stack reaches; `g/inf` is -0.0, the right
    limit, so the warning is the only thing to suppress. NOT rewritten as a two-sided sigmoid: that
    is the same function with different f32 rounding, and it would move this reference away from
    the one the M=256 MLP block is gated on.
    """
    gf = np.asarray(g, np.float32)
    with np.errstate(over="ignore"):
        if act == "silu":
            return rnd(gf / (1.0 + np.exp(-gf)))
        return rnd(0.5 * gf * (1.0 + np.tanh(0.7978845608 * (gf + 0.044715 * gf ** 3))))


# ------------------------------------------------------------------------------------------------
# Blocks. One per built artifact, so each artifact's reference is a single named call.
# ------------------------------------------------------------------------------------------------
def mlp_block(X, nw, Wg, Wu, Wd, eps, act, rnd):
    """`gen_llm_prefill_mlp.py`'s block: RMSNorm -> gate/up GEMM -> act -> mul -> down GEMM."""
    hf = rms_norm(X, nw, eps, rnd)
    g = matmul_t(hf, Wg, rnd)
    u = matmul_t(hf, Wu, rnd)
    gs = gated_ffn_act(g, act, rnd)
    gh = rnd(np.asarray(gs, np.float32) * np.asarray(u, np.float32))
    return matmul_t(gh, Wd, rnd)


def attn_block(Q, KC, VC, rnd):
    """`gen_llm_prefill_attn.py`'s block: per-head scores -> softmax -> context, NON-causal.

    Q is `[Hq, M, HD]` head-major, KC/VC are `[Hkv, S, HD]`. The GQA group is derived from the two
    head counts rather than passed, so a caller cannot hand in a group that disagrees with the
    tensors it also handed in.
    """
    Hq, M, HD = Q.shape
    Hkv = KC.shape[0]
    grp = Hq // Hkv
    out = np.empty((Hq, M, HD), np.float32)
    for h in range(Hq):
        kv = h // grp
        s = matmul_t(Q[h], KC[kv], rnd)          # scores GEMM reads kc b_col_maj, stored [S, HD]
        p = softmax_rows(s, None, rnd)
        out[h] = np.asarray(matmul(p, VC[kv], rnd), np.float32)   # ctx GEMM is plain [S, HD]
    return rnd(out)


def npy_weights(weights_dir):
    """A `W(layer, name)` for `layer_stack`, reading the dumped .npy and narrowing to bf16 -- the
    dtype the device's shared arena actually holds, so the reference starts from the same
    quantisation the device did."""
    def W(layer, name):
        a = np.load(os.path.join(weights_dir, f"model.layers.{layer}.{name}.npy"))
        return bf16(np.asarray(a, np.float32))
    return W


def layer_stack(sp, W, NL, M, S, base, X, table, widths, rnd):
    """`gen_llm_prefill.py`'s full stack. Returns `(xout, [(kc_slab, vc_slab) per layer])`.

    `W(layer, name)` returns one weight already narrowed to the dtype the device holds -- the
    caller owns where the weights come from, because at build time they are `.npy` and at refresh
    time they may be the decode artifact's own `.bin`.

    kc/vc slabs are `[Hkv, M, HD]`: the rows THIS chunk writes, which is what a host gate compares
    against `cache[h, base:base+M, :]`. The rest of the cache is untouched and still zero.
    """
    D, HD = sp.d_model, sp.head_dim
    Hq, Hkv, grp = sp.n_q_heads, sp.n_kv_heads, sp.gqa_group
    x = np.asarray(X, np.float32)
    slabs = []
    for l in range(NL):
        # attn_scale rides on the q-norm gain, because that is how the SHARED decode buffer stores
        # it (SCALE_IN_QNORM). Applying it again here would double-scale against the device.
        n_in, n_pf = W(l, "input_layernorm.weight"), W(l, "post_attention_layernorm.weight")
        n_qn = bf16(np.asarray(W(l, "self_attn.q_norm.weight"), np.float32) * sp.attn_scale)
        n_kn = W(l, "self_attn.k_norm.weight")
        h = rms_norm(x, n_in, sp.eps, rnd)
        q = matmul_t(h, W(l, "self_attn.q_proj.weight"), rnd).reshape(M, Hq, HD)
        k = matmul_t(h, W(l, "self_attn.k_proj.weight"), rnd).reshape(M, Hkv, HD)
        v = matmul_t(h, W(l, "self_attn.v_proj.weight"), rnd).reshape(M, Hkv, HD)
        q = rope_block(rms_norm(q, n_qn, sp.eps, rnd), table, Hq, rnd)
        k = rope_block(rms_norm(k, n_kn, sp.eps, rnd), table, Hkv, rnd)
        kc = np.zeros((Hkv, S, HD), np.float32)
        vc = np.zeros((Hkv, S, HD), np.float32)
        kc[:, base:base + M] = np.asarray(k, np.float32).transpose(1, 0, 2)
        vc[:, base:base + M] = np.asarray(v, np.float32).transpose(1, 0, 2)
        slabs.append((rnd(kc[:, base:base + M]), rnd(vc[:, base:base + M])))
        cx = np.empty((Hq, M, HD), np.float32)
        for hh in range(Hq):
            kv = hh // grp
            s = matmul_t(np.asarray(q, np.float32)[:, hh], rnd(kc[kv]), rnd)
            p = softmax_rows(s, widths, rnd)
            cx[hh] = np.asarray(rnd(np.asarray(p, np.float32) @ vc[kv]), np.float32)
        cxt = cx.transpose(1, 0, 2).reshape(M, Hq * HD)
        x1 = rnd(x + np.asarray(matmul_t(cxt, W(l, "self_attn.o_proj.weight"), rnd), np.float32))
        hf = rms_norm(x1, n_pf, sp.eps, rnd)
        g = matmul_t(hf, W(l, "mlp.gate_proj.weight"), rnd)
        u = matmul_t(hf, W(l, "mlp.up_proj.weight"), rnd)
        gs = np.asarray(gated_ffn_act(g, sp.act, rnd), np.float32)
        gh = rnd(gs * np.asarray(u, np.float32))
        d = matmul_t(gh, W(l, "mlp.down_proj.weight"), rnd)
        x = np.asarray(rnd(np.asarray(x1, np.float32) + np.asarray(d, np.float32)), np.float32)
    return rnd(x), slabs


# ------------------------------------------------------------------------------------------------
# The gate block. One writer, used by the generators at build time and by the refresher after.
# ------------------------------------------------------------------------------------------------
GOLDEN_F32_DIR = "buffers/golden_f32"


def gate_block(art_dir, refs, floors, margin=ATOL_MARGIN):
    """Write the float32 references and return the `gate` dict for the artifact's meta.json.

    `refs` maps tensor name -> the f32 reference; `floors` maps the same names -> the bf16 arm of
    the same dataflow. Both are needed: the reference is what the device is measured against, and
    the floor is what a faithful bf16 implementation already costs, which is what sizes `atol`.
    Derived HERE, where the shape is picked, so the tolerance ships with the artifact instead of
    being typed on a command line where nothing records what it was.
    """
    gdir = os.path.join(art_dir, *GOLDEN_F32_DIR.split("/"))
    os.makedirs(gdir, exist_ok=True)
    missing = [n for n in refs if n not in floors]
    if missing:
        raise ValueError(f"no bf16 floor for {missing}; atol cannot be sized without it")
    tensors = {}
    for name, arr in refs.items():
        a = np.ascontiguousarray(np.asarray(arr, np.float32))
        a.tofile(os.path.join(gdir, f"{name}.bin"))
        fl = np.asarray(floors[name], np.float32)
        tensors[name] = {
            "ref": f"{GOLDEN_F32_DIR}/{name}.bin", "shape": list(a.shape), "rms": rms(a),
            "bf16_floor": bf16_floor(a, fl),
            "bf16_floor_mean_rel_L1": mean_rel_l1(fl, a),
            "atol": atol_for(a, fl, margin),
        }
    return {"rtol": RTOL, "atol_margin": margin, "atol_rule": ATOL_RULE,
            "ref_dtype": "float32", "device_dtype": "bfloat16", "tensors": tensors,
            "note": "TIER 1 of scripts/gate_llm.sh. The float32 references are the SAME dataflow "
                    "on the SAME bf16 inputs the device was handed, with no intermediate narrowed "
                    "-- not the bf16 goldens in buffers/, which model the datapath rather than "
                    "measure it."}


def merge_gate(meta_path, gate):
    """Put `gate` into an existing meta.json in place, leaving every other key alone."""
    meta = json.load(open(meta_path))
    meta["gate"] = gate
    json.dump(meta, open(meta_path, "w"), indent=2)
    return meta
