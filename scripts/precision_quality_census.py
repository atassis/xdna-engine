#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device-free weight-space cost of the precision plane, per SITE, on the REAL tensors.

Two questions per (site, dtype, group_size): is it BUILDABLE, and if so what does it cost in
rel-L2. Buildability is checked two ways, because they catch different failures:

  precision.check()       this tree's own P-rules (P001-P009): dtype support, fifo coupling,
                           channel budget. Device-free, lives in designs/decode_fused/precision.py.
  max_legal_vec_size(),
  + the 128-bit floor      the packer's own row-alignment search (iron/common/quant.py) has no
                           floor at the smallest representable vector: at K=3840 g128 it returns
                           VEC_SIZE=8, and `aie::vector<int8,8>` is 64 bits, under AIE2P's 128-bit
                           minimum -- a kernel COMPILE failure precision.check() does not see,
                           because it never calls max_legal_vec_size. This script adds the floor
                           check precision.py is missing.

rel-L2 uses the SHIPPED packer (iron.common.quant.quantize_weight/dequantize_weight) on the real
bf16 tensors dumped under --weights -- not a re-derived formula. Every tensor is mmap'd, never
np.load()'d whole: the tied embedding is 4 GB at f32, and a naive `W.astype(f64)` difference of
two full copies is 12 GB. Rows are processed in CHUNK_BUDGET_BYTES-sized slices instead -- legal
because group quantization is per-row, so a row chunk sees exactly what the kernel would.

  IRON=<wt-iron-integ> scripts/jobq.sh --mem 6G --class build -- \\
      .venv-iron/bin/python scripts/precision_quality_census.py \\
      --weights /mnt/data/xdna/artifacts/gemma4-12b/weights --spec gemma4-12b --layers 0-5
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "designs", "decode_fused"))
import precision as P                                             # noqa: E402
from llm_decode_spec import SPECS                                  # noqa: E402

IRON = os.environ.get("IRON") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "wt-iron-integ")
sys.path.insert(0, IRON)
from iron.common.quant import (quantize_weight, dequantize_weight,        # noqa: E402
                               max_legal_vec_size, is_affine)

BITS = {"int4": 4, "int8": 8, "int4a": 4, "int8a": 8}
GROUPS = (32, 64, 128, 256)
# absmax/clip are the same wire format at different scale values (host-side search only);
# zero_grid/free_min are the two affine offset conventions precision.py's SCALE_KINDS names.
FORMATS = [(dt, g, sk) for g in GROUPS for dt, sk in (
    ("int8", "absmax"), ("int8a", "zero_grid"), ("int8a", "free_min"),
    ("int4", "absmax"), ("int4", "clip"), ("int4a", "zero_grid"), ("int4a", "free_min"))]


def vec_legal(K, group, dtype):
    """(legal, detail). Reproduces the packer's own alignment search plus the 128-bit vector
    floor it does not enforce itself (see module docstring)."""
    if K % group:
        return False, f"K={K} not a multiple of group_size={group}"
    try:
        vec = max_legal_vec_size([K], group, dtype)
    except ValueError as exc:
        return False, f"packer: {exc}"
    load_bytes = vec // 2 if dtype in ("int4", "int4a") else vec
    if load_bytes * 8 < 128:
        return False, (f"derived VEC_SIZE={vec} loads {load_bytes*8} bits/access, under AIE2P's "
                       "128-bit minimum vector (no vector_storage specialisation -- a compile "
                       "failure, not caught by precision.check())")
    return True, f"VEC_SIZE={vec}, {load_bytes*8}-bit loads, row_stride legal"


# Real per-site tensor names off the dump tree (dump_llm_weights.py's own key scheme).
def site_tensors(sp, site, layers):
    pfx = sp.weight_prefix
    if site == "head":
        yield "head", f"{pfx}embed_tokens.weight"
        return
    for l in layers:
        if site == "mlp":
            yield f"L{l}.gate", f"{pfx}layers.{l}.mlp.gate_proj.weight"
            yield f"L{l}.up", f"{pfx}layers.{l}.mlp.up_proj.weight"
            yield f"L{l}.down", f"{pfx}layers.{l}.mlp.down_proj.weight"
        elif site == "qkv":
            yield f"L{l}.q", f"{pfx}layers.{l}.self_attn.q_proj.weight"
            yield f"L{l}.k", f"{pfx}layers.{l}.self_attn.k_proj.weight"
            if sp.has_v_proj(l):
                yield f"L{l}.v", f"{pfx}layers.{l}.self_attn.v_proj.weight"
        elif site == "attn_o":
            yield f"L{l}.o", f"{pfx}layers.{l}.self_attn.o_proj.weight"


def site_ks(sp, site, layers):
    """Every K this site's real tensors carry -- attn_o has two (sliding q_dim, global q_dim)."""
    if site == "mlp":
        return {sp.d_model, sp.ffn}
    if site == "head":
        return {sp.d_model}
    if site == "qkv":
        return {sp.d_model}
    if site == "attn_o":
        return {sp.q_dim_for(l) for l in layers}


def rel_l2(err_sq, orig_sq):
    return float(np.sqrt(err_sq / orig_sq)) if orig_sq else float("nan")


# Target bytes per row-chunk at f32 -- caps peak transient RSS regardless of K (15360 at mlp's
# down_proj, 3840 elsewhere). ~32 MB holds chunk + its f64 square + one format's packed/recon
# comfortably under jobq.sh's --mem cap even with ~20 legal formats swept per chunk (sequential,
# not concurrent, so their temporaries do not stack).
CHUNK_BUDGET_BYTES = 32 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="gemma4-12b", choices=sorted(SPECS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", default="0-5", help="layer indices for mlp/qkv/attn_o, e.g. 0-5")
    ap.add_argument("--sites", default="mlp,qkv,attn_o,head")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    sp = SPECS[a.spec]
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    sites = a.sites.split(",")

    print(f"[census] {sp.name}: layers {layers} (global: "
          f"{[l for l in layers if sp.is_global(l)]}), weights {a.weights}")

    legality = {}   # (site, dtype, group) -> (ok, detail)
    for site in sites:
        Ks = site_ks(sp, site, layers)
        for dtype, group, scale_kind in FORMATS:
            key = (site, dtype, group)
            if key in legality:
                continue
            oks = [vec_legal(K, group, dtype) for K in sorted(Ks)]
            ok = all(o for o, _ in oks)
            detail = "; ".join(f"K={K}: {d}" for K, (o, d) in zip(sorted(Ks), oks) if not o) \
                or f"legal at K={sorted(Ks)}: {oks[0][1]}"
            legality[key] = (ok, detail)

    results = {}   # (site, dtype, group, scale_kind) -> accumulators
    t0 = time.time()
    for site in sites:
        for tag, name in site_tensors(sp, site, layers):
            path = os.path.join(a.weights, name + ".npy")
            if not os.path.exists(path):
                print(f"[census]   MISSING {path}, skipping"); continue
            Wmm = np.load(path, mmap_mode="r")               # page cache, not anonymous memory
            M, K = Wmm.shape
            legal = [(dt, g, sk) for dt, g, sk in FORMATS
                    if legality[(site, dt, g)][0] and K % g == 0]
            chunk_rows = max(1, CHUNK_BUDGET_BYTES // (K * 4))
            print(f"[census]   {site:7} {tag:8} {name} {Wmm.shape} chunk_rows={chunk_rows} "
                 f"-- {time.time()-t0:.0f}s")
            orig_sq = 0.0
            err_sq = {fmt: 0.0 for fmt in legal}
            for r0 in range(0, M, chunk_rows):
                Wc = np.asarray(Wmm[r0:r0 + chunk_rows], dtype=np.float32)   # copies the slice only
                Wc64 = Wc.astype(np.float64)
                orig_sq += float(np.sum(Wc64 * Wc64))
                for dtype, group, scale_kind in legal:
                    kw = dict(affine_zero_on_grid=(scale_kind == "zero_grid")) if dtype in \
                        ("int4a", "int8a") else dict(clip_search=(scale_kind == "clip"))
                    packed = quantize_weight(Wc, group, dtype, **kw)
                    recon = dequantize_weight(packed, Wc.shape[0], K, group, dtype)
                    diff = Wc64 - recon.astype(np.float64)
                    err_sq[(dtype, group, scale_kind)] += float(np.sum(diff * diff))
                del Wc, Wc64
            for fmt in legal:
                rk = (site,) + fmt
                r = results.setdefault(rk, dict(err_sq=0.0, orig_sq=0.0, n_params=0, n_tensors=0))
                r["err_sq"] += err_sq[fmt]; r["orig_sq"] += orig_sq
                r["n_params"] += Wmm.size; r["n_tensors"] += 1
            del Wmm

    print(f"\n{'site':7} {'format':22} {'legal':5} {'rel-L2':>9} {'bits/w':>7}  detail")
    rows = []
    for site in sites:
        for dtype, group, scale_kind in FORMATS:
            ok, detail = legality[(site, dtype, group)]
            spec_str = f"{dtype}/g{group}/{scale_kind}"
            if not ok:
                print(f"{site:7} {spec_str:22} {'NO':5} {'':9} {'':7}  {detail}")
                rows.append(dict(site=site, dtype=dtype, group=group, scale_kind=scale_kind,
                                 legal=False, detail=detail))
                continue
            r = results.get((site, dtype, group, scale_kind))
            if r is None:
                continue
            rl2 = rel_l2(r["err_sq"], r["orig_sq"])
            # header is 4 B/group either family (f32 scale, or bf16 scale + bf16 min) = 32 bits/group.
            bits = BITS[dtype] + 32.0 / group
            print(f"{site:7} {spec_str:22} {'yes':5} {rl2*100:8.4f}% {bits:6.2f}b  "
                 f"{r['n_tensors']} tensors, {r['n_params']:,} params")
            rows.append(dict(site=site, dtype=dtype, group=group, scale_kind=scale_kind,
                             legal=True, rel_l2=rl2, bits_per_weight=bits,
                             n_tensors=r["n_tensors"], n_params=r["n_params"]))
    # Cross-site structural refusals (P002/P003/P006) are not a per-format question -- they come
    # from the SHIPPED GraphContext (fused_layer=True, FUSE_QKV_DP=0), so run them once here
    # rather than per legality row. This is what precision.check() adds beyond the packer floor.
    if a.spec in P.CENSUS_CTX:
        ctx = P.resolved_context(a.spec, fused_qkv_dp=False)
        print(f"\n[census] structural refusals under the shipped context ({a.spec}, "
              f"fused_layer={ctx.fused_layer} fused_qkv_dp={ctx.fused_qkv_dp}):")
        for label, plan in (
            ("qkv alone (int8a/g64)", {"qkv": "int8a/g64/free_min"}),
            ("qkv+kv matched (int8a/g64)", {"qkv": "int8a/g64/free_min", "kv": "int8a/g64"}),
            ("mlp+attn_o matched (int8a/g64)",
             {"mlp": "int8a/g64/zero_grid", "attn_o": "int8a/g64/zero_grid"}),
        ):
            plan_spec = {k: P.parse_spec(v, k) for k, v in plan.items()}
            try:
                P.check(plan_spec, ctx)
                print(f"  {label:32} OK")
            except P.PrecisionRefusal as exc:
                print(f"  {label:32} REFUSED {exc.rule.id}: {exc.rule.claim}")

    if a.out_json:
        json.dump(dict(spec=sp.name, layers=layers, weights=a.weights, rows=rows),
                  open(a.out_json, "w"), indent=1)
        print(f"\n[census] wrote {a.out_json}")


if __name__ == "__main__":
    main()
