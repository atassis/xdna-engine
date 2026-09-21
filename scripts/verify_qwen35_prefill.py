#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One Qwen3.5 batched-prefill chunk from position 0 on device, against the CPU oracle.

The prefill ELF runs over decode's own arena: every shared weight is loaded from the decode
artifact's buffers, cw/S/kc/vc start at zero, and one dispatch primes M prompt tokens. Compared
per layer against scripts/qwen35_ref.py on the same bf16 embeddings with int4-roundtripped weights
(the decode build's own grid): the residual out of the stack, each DeltaNet layer's recurrent
state and conv history, each attention layer's K/V rows.

Run from the prefill build's work dir, in the build's IRON env and with its knobs (ACT_POLY,
GEMM_TILES_OVERRIDE), so build_graph links the compiled ELF instead of rebuilding it.
"""
import argparse
import json
import os
import sys

import numpy as np
import ml_dtypes
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "designs", "decode_fused"))
sys.path.insert(0, HERE)

BF16 = ml_dtypes.bfloat16


def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def oracle(hf, ids, NL, x0, int4_group, clip):
    """Residual after NL layers plus each layer's state, from bf16 embeddings x0."""
    from qwen35_ref import Ckpt, DeltaNetState, KVState, attention, deltanet, mlp, rms
    ck = Ckpt(hf, int4_group=int4_group, clip_search=clip)
    eps = ck.cfg["rms_norm_eps"]
    x = torch.from_numpy(np.asarray(x0, np.float32))
    states = {}
    for i in range(NL):
        h = rms(x, ck.w(f"layers.{i}.input_layernorm.weight"), eps)
        if ck.cfg["layer_types"][i] == "linear_attention":
            st = DeltaNetState(ck.cfg)
            x = x + deltanet(ck, i, h, st)
            states[i] = ("lin", st.S.numpy(), st.conv.numpy())
        else:
            st = KVState()
            x = x + attention(ck, i, h, st, 0)
            states[i] = ("attn", st.k.numpy(), st.v.numpy())
        x = x + mlp(ck, i, rms(x, ck.w(f"layers.{i}.post_attention_layernorm.weight"), eps))
        print(f"[oracle] layer {i} done", flush=True)
    return x.numpy(), states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--decode-meta", required=True)
    ap.add_argument("--hf", required=True, help="HF snapshot: tokenizer and oracle weights")
    ap.add_argument("--embed", required=True, help="model.embed_tokens.weight.npy (f32 dump)")
    ap.add_argument("--tasks", required=True, help="JevBench tasks .jsonl; prompts give the tokens")
    ap.add_argument("--int4-group", type=int, default=32)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--valid", type=int, default=None,
                    help="real tokens in the chunk (default M); the rest are pad rows")
    ap.add_argument("--oracle-cache", default=None,
                    help=".npz for the oracle's outputs: read if present, written if not")
    ap.add_argument("--oracle-only", action="store_true",
                    help="compute (and cache) the oracle, then stop before the device")
    a = ap.parse_args()
    NL, M, S = a.layers, a.batch, a.seq

    nv = a.valid or M
    if a.oracle_cache and os.path.isfile(a.oracle_cache):
        z = np.load(a.oracle_cache)
        x0, ref_x = z["x0"].view(BF16), z["x"]
        ref_st = {int(k.split("_")[1]): (str(z[k][()]), z[f"a_{k.split('_')[1]}"],
                                         z[f"b_{k.split('_')[1]}"]) for k in z.files if k.startswith("kind_")}
    else:
        from qwen35_decide_ref import prompt_ids
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.hf)
        ids = []
        for line in open(a.tasks):
            if len(ids) >= M:
                break
            ids += prompt_ids(tok, json.loads(line))[0]
        ids = ids[:M]
        emb = np.load(a.embed, mmap_mode="r")
        x0 = np.asarray(np.asarray(emb[ids], np.float32), BF16)
        ref_x, ref_st = oracle(a.hf, ids[:nv], NL, x0[:nv].astype(np.float32), a.int4_group,
                               not a.no_clip)
        if a.oracle_cache:
            np.savez(a.oracle_cache, x0=x0.view(np.uint16), x=ref_x,
                     **{f"kind_{i}": np.array(st[0]) for i, st in ref_st.items()},
                     **{f"a_{i}": st[1] for i, st in ref_st.items()},
                     **{f"b_{i}": st[2] for i, st in ref_st.items()})
    if a.oracle_only:
        return

    from gen_llm_prefill import build_graph, causal_widths, rope_table, SM_WIDTHS, GDR_COUNT
    from llm_decode_spec import SPECS
    sp = SPECS["qwen3.5-4b"]
    _, fused, dims = build_graph("qwen3.5-4b", NL, M, S, "rows", a.decode_meta, do_compile=True)
    sc = fused.get_callable()

    dmeta = json.load(open(a.decode_meta))
    bdir = os.path.join(os.path.dirname(a.decode_meta), "buffers")
    shared = set(dims["shared"])
    loaded = 0
    for name in dmeta["weights"]:
        if name not in shared:
            continue
        raw = np.fromfile(os.path.join(bdir, f"{name}.bin"), np.uint8)
        dst = sc.get_buffer(name).data.view(np.uint8)
        assert dst.nbytes == raw.nbytes, f"{name}: buffer {dst.nbytes} B, file {raw.nbytes} B"
        dst[:] = raw
        loaded += 1
    for name in dims["cache_names"]:
        sc.get_buffer(name).data.view(np.uint8)[:] = 0
    sc.scratch_buffer.device = "cpu"
    sc.scratch_buffer.to("npu")
    print(f"[device] {loaded} decode weight buffers loaded, {len(dims['cache_names'])} cache "
          f"buffers zeroed", flush=True)

    with sc.get_buffer("x").overwrite() as b:
        b[:] = x0.reshape(-1)
    with sc.get_buffer("rope").overwrite() as b:
        b[:] = rope_table(0, M, sp.rope_rotary_dim, sp.rope_theta_global).reshape(-1)
    sc.get_buffer(SM_WIDTHS).data.view(np.int32)[:] = causal_widths(0, M, S, sp.n_q_heads)
    if GDR_COUNT in dims["inputs"]:
        gt = int(os.environ.get("PREFILL_GDR_TOKENS", "16"))
        cb = sc.get_buffer(GDR_COUNT).data.view(np.int32).reshape(M // gt, -1)
        cb[:] = 0
        cb[:, 0] = np.clip(nv - np.arange(M // gt) * gt, 0, gt)
        lch = (2 * sp.lin_k_heads + sp.lin_v_heads) * sp.lin_head_dim
        sc.params.write("hist_off", nv * lch)
    for slot, *_ in dims["geom_slots"]:
        sc.params.write(slot, 0)
    sc.params.sync()
    import time
    t0 = time.perf_counter()
    sc()
    print(f"[device] one prefill dispatch ({NL} layers, M={M}): {time.perf_counter() - t0:.3f} s "
          f"host wall, input sync included")
    sc.scratch_buffer.device = "npu"
    sc.scratch_buffer.to("cpu")

    xout = np.asarray(sc.get_buffer("xout").data, np.float32).reshape(M, -1)[:nv]
    print(f"[cmp] xout after {NL} layers ({nv} of {M} rows real): rel-L2 {rel(xout, ref_x):.3e}")
    T = dims["kv_block"]
    for i, st in sorted(ref_st.items()):
        p = f"L{i}_"
        if st[0] == "lin":
            S_dev = sc.get_buffer(p + "S").data.view(np.float32).reshape(st[1].shape)
            cw = np.asarray(sc.get_buffer(p + "cw").data, np.float32).reshape(-1, st[2].shape[1])
            print(f"[cmp] L{i} DeltaNet  S rel-L2 {rel(S_dev, st[1]):.3e}   "
                  f"conv history rel-L2 {rel(cw[:st[2].shape[0]], st[2]):.3e}")
        else:
            hkv, hd = st[1].shape[1], st[1].shape[2]
            for tag, want in (("kc", st[1]), ("vc", st[2])):
                blk = np.asarray(sc.get_buffer(p + tag).data, np.float32)[:hkv * T * hd]
                got = blk.reshape(hkv, T, hd)[:, :nv].transpose(1, 0, 2)
                print(f"[cmp] L{i} attention {tag} rel-L2 {rel(got, want):.3e}")


if __name__ == "__main__":
    main()
