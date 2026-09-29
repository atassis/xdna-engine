#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare decode's arena int4 codes against prefill's dump, per layer/matrix (spec Step 0.1).

Gemma-4-12B geometry (conventions section of the plan / meta.json `dims`): D=3840, FF=15360,
q_heads=16, kv_heads 8 (sliding, head_dim 256) / 1 (global, head_dim 512, no v), sw_pattern=6,
a layer is global when `(layer_idx + 1) % sw_pattern == 0` (llm_decode_spec.py `is_global`).

`quant_source_files`/`merge_kchunk_rows` below are copies of `gen_llm_prefill.py`'s functions of
the same name (down_proj always 4-way K-chunked, global-layer o_proj 2-way) -- reimplemented
rather than imported because that module pulls in the full IRON/AIE toolchain at import time,
which this host-only check does not need.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weight_views as wv  # noqa: E402
from iron.common.quant import row_stride_bytes  # noqa: E402

D = 3840
FF = 15360
HQ = 16
HKV_SLIDING, HD_SLIDING = 8, 256
HKV_GLOBAL, HD_GLOBAL = 1, 512
SW_PATTERN = 6
GROUP = 32
WEIGHT_DTYPE = "int4"
DECODE_SCALE_DTYPE = "bf16"
PREFILL_SCALE_DTYPE = "f32"

QUANT_TENSOR = {
    "Wq": "self_attn.q_proj.weight", "Wk": "self_attn.k_proj.weight",
    "Wv": "self_attn.v_proj.weight", "Wo": "self_attn.o_proj.weight",
    "Wg": "mlp.gate_proj.weight", "Wu": "mlp.up_proj.weight", "Wd": "mlp.down_proj.weight",
}


def is_global(layer):
    return (layer + 1) % SW_PATTERN == 0


def geometry(layer):
    if is_global(layer):
        return {"hd": HD_GLOBAL, "hkv": HKV_GLOBAL, "has_v": False}
    return {"hd": HD_SLIDING, "hkv": HKV_SLIDING, "has_v": True}


def quant_source_files(src_dir, prefix, tensor):
    """See gen_llm_prefill.py's function of the same name."""
    plain = f"{prefix}{tensor}.npy"
    if os.path.exists(os.path.join(src_dir, plain)):
        return [plain]
    files, n = [], 0
    while os.path.exists(os.path.join(src_dir, f"{prefix}{tensor}.kchunk{n}.npy")):
        files.append(f"{prefix}{tensor}.kchunk{n}.npy")
        n += 1
    return files


def merge_kchunk_rows(rows, N, K, group_size, scale_dtype):
    """See gen_llm_prefill.py's function of the same name -- exact splice, not a requant."""
    n = len(rows)
    chunk_k = K // n
    sb = {"f32": 4, "bf16": 2}[scale_dtype]
    hdr, pay = (chunk_k // group_size) * sb, chunk_k // 2
    cs = [np.asarray(r).view(np.uint8).reshape(N, hdr + pay) for r in rows]
    merged = np.concatenate([np.concatenate([c[:, :hdr] for c in cs], axis=1),
                             np.concatenate([c[:, hdr:] for c in cs], axis=1)], axis=1)
    return merged.reshape(-1).view(np.int8)


def prefill_rows(dump_dir, prefix, tensor, N, K, group_size=GROUP, scale_dtype=PREFILL_SCALE_DTYPE):
    files = quant_source_files(dump_dir, prefix, tensor)
    if not files:
        raise FileNotFoundError(f"{dump_dir}: no dump file for {prefix}{tensor}[.kchunkN].npy")
    loaded = [np.load(os.path.join(dump_dir, f)) for f in files]
    packed = loaded[0] if len(loaded) == 1 else merge_kchunk_rows(
        loaded, N, K, group_size, scale_dtype)
    stride = row_stride_bytes(K, group_size, WEIGHT_DTYPE, scale_dtype)
    return np.asarray(packed).view(np.uint8).reshape(N, stride)


def bf16_round(x):
    import ml_dtypes
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def compare_matrix(name, decode_rows, prefill_rows_, N, K):
    dcodes, dscales = wv.codes_and_scales(decode_rows, N, K, scale_dtype=DECODE_SCALE_DTYPE)
    pcodes, pscales = wv.codes_and_scales(prefill_rows_, N, K, scale_dtype=PREFILL_SCALE_DTYPE)
    codes_equal = bool(np.array_equal(dcodes, pcodes))
    n_diff = int(np.sum(dcodes != pcodes))
    pscales_bf16 = bf16_round(pscales)
    scale_diff = np.abs(dscales - pscales_bf16)
    max_scale_diff = float(scale_diff.max())
    scales_match = bool(np.array_equal(dscales, pscales_bf16))
    return {
        "name": name, "N": N, "K": K, "codes_equal": codes_equal, "n_diff_codes": n_diff,
        "total_codes": int(dcodes.size), "max_scale_diff": max_scale_diff,
        "scales_match_bf16_round": scales_match,
    }


def check_layer(layer, decode_dir, prefill_dump, prefix="model.language_model."):
    g = geometry(layer)
    hd, hkv, has_v = g["hd"], g["hkv"], g["has_v"]
    qd, kvd = HQ * hd, hkv * hd
    p = f"L{layer}_"
    lp = f"{prefix}layers.{layer}."
    results = []

    wqkv_rows = qd + kvd + (kvd if has_v else 0)
    arena_qkv = wv.arena_weight_rows(decode_dir, p + "Wqkv", wqkv_rows, D)
    arena_parts = wv.qkv_rows(arena_qkv, qd, kvd, has_v)
    for key, part_key in (("Wq", "q"), ("Wk", "k"), ("Wv", "v")):
        if part_key not in arena_parts:
            continue
        decode_part = arena_parts[part_key]
        N = decode_part.shape[0]
        pref = prefill_rows(prefill_dump, lp, QUANT_TENSOR[key], N, D)
        results.append(compare_matrix(f"L{layer}.{key}", decode_part, pref, N, D))

    o_K = qd
    arena_o = wv.arena_weight_rows(decode_dir, p + "Wo", D, o_K)
    pref_o = prefill_rows(prefill_dump, lp, QUANT_TENSOR["Wo"], D, o_K)
    results.append(compare_matrix(f"L{layer}.Wo", arena_o, pref_o, D, o_K))

    for key, (N, K) in (("Wg", (FF, D)), ("Wu", (FF, D)), ("Wd", (D, FF))):
        arena_w = wv.arena_weight_rows(decode_dir, p + key, N, K)
        pref_w = prefill_rows(prefill_dump, lp, QUANT_TENSOR[key], N, K)
        results.append(compare_matrix(f"L{layer}.{key}", arena_w, pref_w, N, K))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode-dir", required=True)
    ap.add_argument("--prefill-dump", required=True)
    ap.add_argument("--layers", default="0-47",
                    help="e.g. '0,5,47' or '0-47' or a mix '0,5,10-15'")
    ap.add_argument("--out", default=None, help="write the report here too (in addition to stdout)")
    a = ap.parse_args()

    layers = []
    for part in a.layers.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# decode-dir={a.decode_dir}")
    emit(f"# prefill-dump={a.prefill_dump}")
    emit(f"# layers={sorted(set(layers))}")
    emit(f"{'matrix':<12}{'N':>7}{'K':>7}{'codes_equal':>13}{'n_diff':>10}{'total':>10}"
        f"{'max_scale_diff':>16}{'scales_bf16_match':>19}")

    any_code_mismatch = False
    any_scale_mismatch = False
    for layer in sorted(set(layers)):
        for r in check_layer(layer, a.decode_dir, a.prefill_dump):
            emit(f"{r['name']:<12}{r['N']:>7}{r['K']:>7}{str(r['codes_equal']):>13}"
                f"{r['n_diff_codes']:>10}{r['total_codes']:>10}{r['max_scale_diff']:>16.6e}"
                f"{str(r['scales_match_bf16_round']):>19}")
            if not r["codes_equal"]:
                any_code_mismatch = True
            if not r["scales_match_bf16_round"]:
                any_scale_mismatch = True

    emit()
    emit(f"any_code_mismatch={any_code_mismatch}")
    emit(f"any_scale_mismatch_beyond_bf16_round={any_scale_mismatch}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    if any_code_mismatch:
        print("STOP: at least one matrix has differing int4 codes -- the pure-reorder premise "
             "is false. See spec Step 0.1 / plan Task 2 Step 3.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
