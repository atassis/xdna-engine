#!/usr/bin/env python3
"""rf-lm-head-segment: reference model of the resident forward's LM head segment -- final
RMSNorm + tied int4 head pack (566 MB, `planar_int4g32_headpack`) + final logit softcap.

Two arithmetic paths over the SAME dequantised head weights, streamed in vocab chunks (the head
pack is 566 MB; this never materialises it at float precision):

  float64_head_logits  -- exact math: f64 dequant (int4 x bf16 scale) @ f64 hidden. The oracle.
  device_head_logits   -- the chain kernel's own arithmetic (mlp_ref.gemm_chain, which already
                           wraps chain_ref.mac_block, the device-fitted bfp16 accumulate rule):
                           bf16-rounded activations, bfp16-converted activations/weights, the
                           kernel's K-block accumulation order. K is consumed as one contiguous
                           [0, 3840) range in ascending order, which is the SAME order as 8
                           physical columns c=0..7 each owning [c*480, (c+1)*480) in increasing c
                           (see head_stream_layout.py) -- gemm_chain's single kb-loop over K//8
                           blocks IS "columns in order, k-blocks in order" for that column
                           assignment, so no separate per-column split is needed here. That column
                           order (and, within head_stream_layout.py, the per-column SUB_K chunk
                           order) is the part the device design has not fixed yet; both are named
                           parameters below (`column_order`), not hardcoded.

row_group is DERIVED from K (never read off the manifest's/quant.json's stored field, which was
computed for one K -- see weight_store.py:67 and int4_weights.py:25-34) and asserted equal to the
manifest's value as a consistency check, not a source of truth.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rf_paths                               # noqa: E402
from chain_ref import bf16, bf16_val          # noqa: E402
from mlp_ref import gemm_chain                # noqa: E402

STORE = os.environ.get("RF_STORE", str(rf_paths.ARTIFACTS / "store"))
MANIFEST = f"{STORE}/manifest.json"
CONFIG = os.environ.get("RF_HF_CONFIG", str(rf_paths.ARTIFACTS / "hf_config/config.json"))

GROUP_SIZE = 32  # int4 payload, bf16 scale -- checked against the manifest entry at load time


def blob_path(h):
    return f"{STORE}/blobs/{h}.bin"


def load_manifest():
    return json.load(open(MANIFEST))


def load_config():
    cfg = json.load(open(CONFIG))
    return cfg.get("text_config") or cfg


def row_group_for_k(K, group_size=GROUP_SIZE, vec_size=64):
    """Re-derive row_group from K (quant.json's/manifest's stored field was computed once for
    K=3840 and is wrong for other K -- weight_store.py's own comment). Same algorithm as
    int4_weights.row_group_for_k / weight_store.row_group_for_k, reproduced so this reference has
    no import dependency on either scratchpad module."""
    load = vec_size // 2
    n_groups = K // group_size
    payload = K // 2
    header = 2 * n_groups
    stride = header + payload
    g = 1
    while True:
        if (payload % load == 0) and ((g * stride) % load == 0):
            return g
        g += 1
        if g > 64:
            raise ValueError(f"no row_group <= 64 aligns int4/g{group_size} vec={vec_size} at K={K}")


def unplanar_chunk(raw, row0, row1, K, row_group):
    """int4 g32 row_group_planar bytes, rows [row0, row1) -> (q int8 [rows,K], scale f32
    [rows,K/32]). row0/row1 must land on row_group block boundaries (the dump-time packer's own
    chunking requirement, weight_store.py `pack_embedding`/int4_weights `_unplanar`)."""
    assert row0 % row_group == 0 and row1 % row_group == 0, (row0, row1, row_group)
    ng, pay = K // 32, K // 2
    stride = ng * 2 + pay
    nblk = (row1 - row0) // row_group
    blk0 = row0 // row_group
    blocks = raw[blk0 * row_group * stride: (blk0 + nblk) * row_group * stride]
    blocks = blocks.reshape(nblk, row_group * stride)
    rows = np.empty((nblk, row_group, stride), np.uint8)
    rows[:, :, 2 * ng:] = blocks[:, :row_group * pay].reshape(-1, row_group, pay)
    rows[:, :, :2 * ng] = blocks[:, row_group * pay:].reshape(-1, row_group, 2 * ng)
    rows = rows.reshape(row1 - row0, stride)
    sc = bf16_val(rows[:, :2 * ng].copy().view(np.uint16)).reshape(row1 - row0, ng)
    p = rows[:, 2 * ng:]
    lo = (p & 15).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    q = np.empty((row1 - row0, K), np.int8)
    q[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    q[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    return q, sc


def dequant_f64(q, sc, group=GROUP_SIZE):
    return q.astype(np.float64) * np.repeat(sc.astype(np.float64), group, axis=1)


def load_norm_weight(manifest):
    e = manifest["final_norm"]
    assert e["layout"] == "raw_f32"
    n = e["length"] // 4
    return np.fromfile(blob_path(e["blob"]), dtype=np.float32, count=n)


def rms_norm_f64(h, w, eps):
    """Gemma4RMSNorm: normed * weight, no +1 (measure.py:62-68). Computed once, identically, for
    both the reference and device paths below -- the design does not name a lower norm precision,
    so this is not modelled as a second bfp16 arithmetic site."""
    h = h.astype(np.float64)
    ms = np.mean(h * h) + eps
    return h * ms ** -0.5 * w.astype(np.float64)


def apply_softcap(logits, softcap):
    return np.tanh(logits / softcap) * softcap


def head_meta(manifest):
    e = manifest["embedding"]
    assert e["layout"] == "planar_int4g32_headpack"
    K, V, group = e["hidden"], e["vocab"], e["group_size"]
    rg = row_group_for_k(K, group)
    assert rg == e["row_group"], (
        f"derived row_group {rg} for K={K} disagrees with manifest's stored {e['row_group']}")
    return e, K, V, group, rg


def float64_head_logits(xf64, manifest, chunk_rows=8192):
    """Exact reference: f64 dequant @ f64 hidden, streamed in row-group-aligned chunks so the
    566 MB pack is never expanded to float at full width."""
    e, K, V, group, rg = head_meta(manifest)
    raw = np.memmap(blob_path(e["blob"]), dtype=np.uint8, mode="r", shape=(e["length"],))
    step = max(rg, chunk_rows - chunk_rows % rg)
    logits = np.empty(V, np.float64)
    for r0 in range(0, V, step):
        r1 = min(r0 + step, V)
        q, sc = unplanar_chunk(raw, r0, r1, K, rg)
        w = dequant_f64(q, sc, group)
        logits[r0:r1] = w @ xf64
    return logits


def device_head_logits(xf32, manifest, chunk_rows=8192, column_order=None):
    """Device path: bf16-round the activation (gemm_chain's own contract -- "xa already the
    bfp16-exact A values' source"), pad to the array's 8-row tile (row 0 real, 1-7 zero, matching
    the pad-row convention every other resident piece uses), then run mlp_ref.gemm_chain per vocab
    chunk -- the chain kernel's bfp16 mac + accumulate rule, chain_ref.mac_block underneath.

    `column_order` parameterises which physical K sub-range the kernel visits first; None (the
    default) is the natural ascending order gemm_chain's single kb-loop already walks, which
    equals 8 columns c=0..7 owning [c*480,(c+1)*480) in increasing c (head_stream_layout.py). Any
    other permutation of K blocks is legal input to gemm_chain and would model a different device
    column-dispatch order once one is fixed -- this file does not pick that order, only exposes it.
    """
    e, K, V, group, rg = head_meta(manifest)
    raw = np.memmap(blob_path(e["blob"]), dtype=np.uint8, mode="r", shape=(e["length"],))
    xa = np.zeros((8, K), np.float32)
    xa[0] = bf16(xf32.astype(np.float32))
    if column_order is not None:
        xa = xa[:, column_order]
    step = max(rg, chunk_rows - chunk_rows % rg)
    logits = np.empty(V, np.float32)
    for r0 in range(0, V, step):
        r1 = min(r0 + step, V)
        q, sc = unplanar_chunk(raw, r0, r1, K, rg)
        if column_order is not None:
            q = q[:, column_order]
            sc = sc[:, np.asarray(column_order) // group]  # only exact for a column-aligned perm
        out = gemm_chain(xa, q, sc, group_size=group)
        logits[r0:r1] = out[0]
    return logits


def rel_l2(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    den = np.linalg.norm(b64)
    return float(np.linalg.norm(a64 - b64) / den) if den else float("nan")


def stage_breakdown(xf64, manifest, row0=0, row1=16384):
    """Isolate which stage of device_head_logits's error against the float64 oracle dominates, on
    one vocab chunk [row0,row1): activation bf16-rounding alone, weight bfp16-conversion alone,
    both-but-plain-summation (no chain accumulate order), and the full chain (mac_block's fitted
    accumulate rule) -- the gap between the last two isolates the accumulate order's own
    contribution. Softcap is excluded (monotonic, so it cannot change which stage dominates -- see
    run_report's separate pre/post-softcap rel_l2, which never diverge in ranking)."""
    from bfp16_model import f32_to_bfp16, bfp16_to_f32

    e, K, V, group, rg = head_meta(manifest)
    raw = np.memmap(blob_path(e["blob"]), dtype=np.uint8, mode="r", shape=(e["length"],))
    q, sc = unplanar_chunk(raw, row0, row1, K, rg)
    w64 = dequant_f64(q, sc, group)
    ref = w64 @ xf64

    xf_bf16 = bf16(xf64.astype(np.float32)).astype(np.float64)
    w32 = w64.astype(np.float32)
    w_bfp16 = bfp16_to_f32(f32_to_bfp16(w32.reshape(-1))).reshape(w32.shape).astype(np.float64)

    act_only = w64 @ xf_bf16
    weight_only = w_bfp16 @ xf64
    both_naive = w_bfp16 @ xf_bf16          # bf16 act + bfp16 weight, plain (non-chain) summation

    xa = np.zeros((8, K), np.float32)
    xa[0] = bf16(xf64.astype(np.float32))
    full_chain = gemm_chain(xa, q, sc, group_size=group)[0].astype(np.float64)

    return dict(
        row0=row0, row1=row1,
        rel_l2_activation_bf16_only=rel_l2(act_only, ref),
        rel_l2_weight_bfp16_only=rel_l2(weight_only, ref),
        rel_l2_both_plain_sum=rel_l2(both_naive, ref),
        rel_l2_full_chain=rel_l2(full_chain, ref),
        rel_l2_accumulate_order_contribution=rel_l2(full_chain, ref) - rel_l2(both_naive, ref),
    )


def run_report(seeds=(0, 1, 2), chunk_rows=16384, out_path=None):
    manifest = load_manifest()
    cfg = load_config()
    eps = cfg.get("rms_norm_eps", 1e-6)
    softcap = cfg.get("final_logit_softcapping", 30.0)
    K = manifest["embedding"]["hidden"]
    norm_w = load_norm_weight(manifest)

    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Synthesised pre-norm hidden state -- see the task worklog: the byte-skipping int4
        # reference (prototypes/byte-skipping/int4_forward.py) needs a resident WeightStore of
        # ~8-10 GB, over the 3 GB systemd MemoryMax cap this task runs under, so it could not be
        # used to source a real final hidden state. RMSNorm is scale-invariant, so only the
        # DIRECTION matters for the norm+head math below; a unit-Gaussian draw is not shaped like
        # a real residual stream (no outlier dims) but is an honest, clearly-labelled stand-in.
        h_raw = rng.standard_normal(K)
        xf64 = rms_norm_f64(h_raw, norm_w, eps)
        xf32 = xf64.astype(np.float32)

        ref = float64_head_logits(xf64, manifest, chunk_rows=chunk_rows)
        dev = device_head_logits(xf32, manifest, chunk_rows=chunk_rows)

        r2_pre = rel_l2(dev, ref)
        ref_cap = apply_softcap(ref, softcap)
        dev_cap = apply_softcap(dev.astype(np.float64), softcap)
        r2_post = rel_l2(dev_cap, ref_cap)
        argmax_ref = int(np.argmax(ref))
        argmax_dev = int(np.argmax(dev))
        results.append(dict(seed=seed, rel_l2_pre_softcap=r2_pre, rel_l2_post_softcap=r2_post,
                             argmax_ref=argmax_ref, argmax_dev=argmax_dev,
                             argmax_match=argmax_ref == argmax_dev))
        print(f"seed={seed} rel_l2(pre-softcap)={r2_pre:.6e} rel_l2(post-softcap)={r2_post:.6e} "
              f"argmax ref={argmax_ref} dev={argmax_dev} match={argmax_ref == argmax_dev}")

    n_match = sum(r["argmax_match"] for r in results)
    summary = dict(vocab=manifest["embedding"]["vocab"], hidden=K, softcap=softcap, eps=eps,
                    n_seeds=len(results), argmax_matches=n_match, argmax_total=len(results),
                    results=results)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nargmax agreement: {n_match}/{len(results)}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--chunk-rows", type=int, default=16384)
    ap.add_argument("--out", default=str(rf_paths.BUILD_ROOT / "scratch/head_ref_report.json"))
    ap.add_argument("--breakdown", action="store_true",
                     help="run stage_breakdown on one chunk (seed 0) instead of the full report")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.breakdown:
        manifest = load_manifest()
        norm_w = load_norm_weight(manifest)
        eps = load_config().get("rms_norm_eps", 1e-6)
        K = manifest["embedding"]["hidden"]
        rng = np.random.default_rng(0)
        xf64 = rms_norm_f64(rng.standard_normal(K), norm_w, eps)
        print(json.dumps(stage_breakdown(xf64, manifest, row0=0, row1=args.chunk_rows), indent=2))
    else:
        run_report(seeds=args.seeds, chunk_rows=args.chunk_rows, out_path=args.out)
