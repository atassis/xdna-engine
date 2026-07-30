#!/usr/bin/env python3
"""Host-side S2 AR weight prep: GGUF (q6_k + f16) -> resident bf16 weight blob + JSON manifest.

WHY this exists: `scripts/s2_ar_ref.py` established that every big AR weight matrix in S2-Pro
(`embeddings.weight`, `codebook_embeddings.weight`, every `wqkv`/`wo`/`w1`/`w2`/`w3` across all 36
slow + 4 fast layers, `fast_embeddings.weight`, `fast_output.weight` -- 204 tensors) is GGUF type
q6_k, and only the small RMSNorm gamma vectors (154 tensors) are f16. That module's own docstring
recommends dequantizing on the HOST, ONCE, at load time rather than porting q6_k's 6-bit/
16-scales-per-256-superblock layout to an AIE2P kernel -- q6_k has no relationship to this
project's existing `dequant-int4-group` brick, and this conversion runs once per model load, not
once per token. This script IS that host conversion: it reads the shipped GGUF and writes a flat,
alignment-friendly binary blob plus a JSON manifest that a future device-side loader can mmap and
upload directly, with no GGUF re-parsing and no re-running dequant. See
`docs/s2-weight-blob-format.md` for the on-disk format, the alignment rule, and how to verify a
blob independently of this tool.

This script does NOT touch the NPU, does NOT build a device-side loader/upload path, and does NOT
re-quantize to int8/int4 (a real future FORMAT-lever option, but a separate gated decision -- see
the doc's "Future format options" section). It is pure numpy, reusing `s2_ar_ref.py`'s GGUF parser
and validated q6_k dequant AS A MODULE (imported, not copied) -- this file adds only: dtype-cast
policy, the blob writer with 64B-aligned tensor offsets, the manifest schema, an analytic full-set
size report (no dequant needed -- it only reads the GGUF's tensor directory), and a self-test that
re-reads sampled tensors from the written blob and checks them against an independent re-dequant
from the source GGUF, bit-exact against the SAME cast the writer used, with an explicit shape
assertion against `s2_ar_ref.tensor_numpy_shape` (an element-count-only check would pass a
transposed tensor -- see that module's ne-vs-numpy-shape note).

USAGE:
    python3 scripts/s2_weight_prep.py --out /path/to/outdir                  # full 358-tensor set
    python3 scripts/s2_weight_prep.py --out /tmp/smoke --layers 2            # smoke test: top-level
                                                                              # tensors + 2 slow +
                                                                              # 2 fast layers
    python3 scripts/s2_weight_prep.py --out DIR --gguf /path/to/model.gguf   # override GGUF path
    python3 scripts/s2_weight_prep.py --out DIR --no-selftest                # skip the round-trip
                                                                              # self-test (not
                                                                              # recommended)

Path resolution for --gguf follows `codec_paths.gguf()` (same as s2_ar_ref.py): $S2_GGUF env var
first, then the sibling `s2.cpp/models/` checkout layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codec_paths  # noqa: E402  (sibling script, path-resolution only)
import s2_ar_ref as ar  # noqa: E402  (reuse GGUF parsing + validated q6_k dequant -- do not copy)

FORMAT_VERSION = 1

# 64 bytes is the floor this project already treats as a real hardware alignment tier, not a
# round number picked for this tool: docs/aie2p-brick-catalog.md's load_v/store_v row documents
# 16/32/64B as the AIE2P vector load/store alignment classes, and a 16-lane f32 (or 32-lane bf16)
# vector register store is exactly 64B. Aligning every tensor to 64B means a future device-side
# loader can DMA any tensor's bytes straight off the mmap with no bounce-copy to fix up alignment,
# regardless of which vector width the consuming kernel uses. See docs/s2-weight-blob-format.md.
ALIGNMENT_BYTES = 64

BLOB_FILENAME = "s2_ar_weights.blob"
MANIFEST_FILENAME = "manifest.json"

# ggml type id -> (dequantized) numpy/ml_dtypes output dtype this tool emits, and the reasoning:
#   q6_k (14) -> bf16    the FORMAT this tool's job is to unblock: 6-bit quantized -> dense bf16.
#   f16  (1)  -> f16     RMSNorm gamma vectors are ALREADY f16 in the source. Casting them to bf16
#                        would COST precision (f16 has 10 mantissa bits, bf16 has 7) for ZERO size
#                        benefit (both are 2 bytes) -- pure information loss with no upside. Kept
#                        as native f16; a consumer widens to f32 for the elementwise multiply the
#                        same way it would from bf16 (one astype, either direction).
# Anything else raises loud rather than guessing -- this checkpoint has only these two source
# types (see s2_ar_ref.py's own report: 204 q6_k + 154 f16, zero of anything else), and a future
# checkpoint with a different quant type should fail here, not silently emit wrong bytes.
_OUT_DTYPE_BY_SRC_TYPE = {14: "bf16", 1: "f16"}
_NP_DTYPE_BY_OUT_NAME = {"bf16": ml_dtypes.bfloat16, "f16": np.float16}


def choose_out_dtype(src_gguf_type: int) -> str:
    try:
        return _OUT_DTYPE_BY_SRC_TYPE[src_gguf_type]
    except KeyError:
        type_name = ar.GGUF_TYPE_NAMES.get(src_gguf_type, str(src_gguf_type))
        raise ValueError(
            f"s2_weight_prep has no output-dtype policy for source GGUF type "
            f"{type_name} ({src_gguf_type}) -- this checkpoint was only ever observed to use "
            f"q6_k and f16 for AR tensors; extend _OUT_DTYPE_BY_SRC_TYPE if a future checkpoint "
            f"needs another type, don't silently guess") from None


# --------------------------------------------------------------------------------------------
# Tensor name selection. Reuses s2_ar_ref's own lists/generators so the set of tensors this tool
# writes is defined in exactly one place (that module), not re-derived here.
# --------------------------------------------------------------------------------------------

def select_tensor_names(hp: ar.ARHParams, layers: "int | None") -> tuple[list[str], int, int]:
    """Returns (names, n_slow_layers_included, n_fast_layers_included). `layers=None` means the
    full model (36 slow + 4 fast in this checkpoint, 358 tensors total); `layers=N` caps EACH of
    slow/fast at N layers (fast only has 4 to begin with) so a smoke test pays for a handful of
    tensors, not all 358 -- while still including every top-level tensor (the two big embedding
    tables + 4 more), so the self-test below has both large 2D q6_k matrices and small f16 vectors
    to sample from even in the smallest smoke test."""
    names = list(ar.SLOW_TOP_LEVEL)
    if hp.has_fast_decoder:
        names += list(ar.FAST_TOP_LEVEL)

    n_slow = hp.block_count if layers is None else min(layers, hp.block_count)
    n_fast = hp.fast_block_count if layers is None else min(layers, hp.fast_block_count)

    for il in range(n_slow):
        names += ar.slow_layer_names(il, hp.attention_qk_norm)
    if hp.has_fast_decoder:
        for il in range(n_fast):
            names += ar.fast_layer_names(il, hp.fast_attention_qk_norm)
    return names, n_slow, n_fast


# --------------------------------------------------------------------------------------------
# Analytic full-set size report. Reads ONLY the GGUF tensor directory (gg.infos), no dequant, so
# this is cheap (microseconds) and can always be printed regardless of --layers -- it answers "how
# big is the FULL blob" even when this run only converts a subset.
# --------------------------------------------------------------------------------------------

def analytic_size_report(gg: ar.GGUFFile, hp: ar.ARHParams) -> dict:
    names, _, _ = select_tensor_names(hp, layers=None)  # the FULL 358-tensor set, always
    q6k_elems = q6k_src_bytes = f16_elems = 0
    n_q6k = n_f16 = 0
    for name in names:
        dims, ty, _off = gg.infos[name]
        n = ar.tensor_numel(dims)
        if ty == 14:
            assert n % ar.QK_K == 0, f"{name}: numel {n} not a multiple of {ar.QK_K}"
            q6k_elems += n
            q6k_src_bytes += (n // ar.QK_K) * ar.Q6K_BLOCK_BYTES
            n_q6k += 1
        elif ty == 1:
            f16_elems += n
            n_f16 += 1
        else:
            choose_out_dtype(ty)  # raises with a clear message; keeps this scan honest too

    bf16_out_bytes = q6k_elems * 2       # q6_k source -> bf16 output, 2 bytes/element
    f16_out_bytes = f16_elems * 2        # f16 source -> f16 output, native passthrough
    blob_bytes = bf16_out_bytes + f16_out_bytes  # + <= (n_tensors * (ALIGNMENT-1)) padding, negligible
    src_bytes = q6k_src_bytes + f16_out_bytes
    return {
        "tensor_count": len(names),
        "q6k_tensor_count": n_q6k,
        "f16_tensor_count": n_f16,
        "q6k_elements": q6k_elems,
        "f16_elements": f16_elems,
        "total_elements": q6k_elems + f16_elems,
        "q6k_source_bytes": q6k_src_bytes,
        "f16_source_bytes": f16_out_bytes,
        "source_bytes_total": src_bytes,
        "bf16_output_bytes": bf16_out_bytes,
        "f16_output_bytes": f16_out_bytes,
        "blob_bytes_total": blob_bytes,
        "expansion_ratio": blob_bytes / src_bytes,
    }


def print_size_report(rep: dict) -> None:
    ggib = 1024 ** 3
    print(f"\n--- full-set ({rep['tensor_count']} tensors) size arithmetic (analytic, no dequant) ---")
    print(f"  q6_k source: {rep['q6k_tensor_count']} tensors, {rep['q6k_elements']:,} elements, "
          f"{rep['q6k_source_bytes']:,} B ({rep['q6k_source_bytes']/ggib:.3f} GiB)")
    print(f"  f16  source: {rep['f16_tensor_count']} tensors, {rep['f16_elements']:,} elements, "
          f"{rep['f16_source_bytes']:,} B ({rep['f16_source_bytes']/ggib:.5f} GiB)")
    print(f"  bf16 output for the q6_k tensors: {rep['bf16_output_bytes']:,} B "
          f"({rep['bf16_output_bytes']/ggib:.3f} GiB) -- {rep['bf16_output_bytes']/rep['q6k_source_bytes']:.3f}x "
          f"the q6_k source bytes for the SAME elements (q6_k is ~6.56 bit/weight packed; bf16 is 16)")
    print(f"  blob total (bf16 matrices + native-f16 gammas): {rep['blob_bytes_total']:,} B "
          f"({rep['blob_bytes_total']/ggib:.3f} GiB)")
    print(f"  vs total AR source bytes as read from the GGUF: {rep['source_bytes_total']:,} B "
          f"({rep['source_bytes_total']/ggib:.3f} GiB) -- {rep['expansion_ratio']:.3f}x expansion")
    if rep["blob_bytes_total"] > 8 * ggib:
        print(f"  >>> {rep['blob_bytes_total']/ggib:.1f} GiB is a real number: this is the resident "
              f"weight set a device-side loader would need to stage, not a rounding artifact.")


# --------------------------------------------------------------------------------------------
# Conversion + blob writer.
# --------------------------------------------------------------------------------------------

def _pad_len(cursor: int) -> int:
    return (-cursor) % ALIGNMENT_BYTES


def convert(gg: ar.GGUFFile, names: list[str], blob_path: Path) -> tuple[list[dict], int, float]:
    """Reads each named tensor via s2_ar_ref.read_tensor (validated GGUF decode + q6_k dequant),
    casts to the policy dtype, and appends it to `blob_path` at a 64B-aligned offset. Returns
    (manifest tensor entries, final blob size in bytes, wall-clock seconds)."""
    entries: list[dict] = []
    cursor = 0
    t0 = time.time()
    with open(blob_path, "wb") as bf:
        for name in names:
            dims, ty, _off = gg.infos[name]
            numpy_shape = ar.tensor_numpy_shape(dims)

            arr_f32 = ar.read_tensor(gg, name)
            assert arr_f32.shape == numpy_shape, (
                f"{name}: read_tensor shape {arr_f32.shape} != tensor_numpy_shape(ne) {numpy_shape}")

            out_dtype_name = choose_out_dtype(ty)
            np_dtype = _NP_DTYPE_BY_OUT_NAME[out_dtype_name]
            arr_out = arr_f32.astype(np_dtype)
            raw = arr_out.tobytes()

            pad = _pad_len(cursor)
            if pad:
                bf.write(b"\x00" * pad)
                cursor += pad
            offset = cursor
            bf.write(raw)
            cursor += len(raw)

            entries.append({
                "name": name,
                "dtype": out_dtype_name,
                "numpy_shape": list(numpy_shape),
                "ggml_ne": list(dims),
                "source_gguf_type": ar.GGUF_TYPE_NAMES.get(ty, str(ty)),
                "offset": offset,
                "nbytes": len(raw),
            })
    elapsed = time.time() - t0
    return entries, cursor, elapsed


# --------------------------------------------------------------------------------------------
# Manifest.
# --------------------------------------------------------------------------------------------

def _hash_header(path: str, nbytes: int) -> str:
    """SHA-256 over the first `nbytes` of the GGUF (header + KV metadata + tensor directory, i.e.
    exactly gg.data_start bytes -- NOT the tensor payload, which is multiple GB). Identifies which
    GGUF this blob was converted from without hashing 4+ GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def build_manifest(gg: ar.GGUFFile, hp: ar.ARHParams, entries: list[dict], blob_bytes: int,
                    n_slow_layers: int, n_fast_layers: int, full_set: bool) -> dict:
    real_path = os.path.realpath(gg.path)
    return {
        "format_version": FORMAT_VERSION,
        "alignment_bytes": ALIGNMENT_BYTES,
        "blob_file": BLOB_FILENAME,
        "blob_bytes_total": blob_bytes,
        "tensor_count": len(entries),
        "full_set": full_set,  # False if this manifest was produced with --layers (a subset)
        "slow_layers_included": n_slow_layers,
        "fast_layers_included": n_fast_layers,
        "slow_layers_total": hp.block_count,
        "fast_layers_total": hp.fast_block_count,
        "generated_at_unix": time.time(),
        "source_gguf": {
            "path": real_path,
            "size_bytes": os.path.getsize(real_path),
            "data_start": gg.data_start,
            "header_sha256": _hash_header(gg.path, gg.data_start),
            "header_hash_covers_bytes": gg.data_start,
        },
        "hparams": {
            "architecture": gg.kv.get("general.architecture"),
            "block_count": hp.block_count,
            "embedding_length": hp.embedding_length,
            "feed_forward_length": hp.feed_forward_length,
            "head_count": hp.head_count,
            "head_count_kv": hp.head_count_kv,
            "head_dim": hp.head_dim,
            "vocab_size": hp.vocab_size,
            "codebook_size": hp.codebook_size,
            "num_codebooks": hp.num_codebooks,
            "attention_qk_norm": hp.attention_qk_norm,
            "scale_codebook_embeddings": hp.scale_codebook_embeddings,
            "fast_block_count": hp.fast_block_count,
            "fast_embedding_length": hp.fast_embedding_length,
            "fast_feed_forward_length": hp.fast_feed_forward_length,
            "fast_head_count": hp.fast_head_count,
            "fast_head_count_kv": hp.fast_head_count_kv,
            "fast_head_dim": hp.fast_head_dim,
            "fast_attention_qk_norm": hp.fast_attention_qk_norm,
        },
        "tensors": entries,
    }


# --------------------------------------------------------------------------------------------
# Self-test: proves the blob round-trips, not just that the tool ran without crashing.
# --------------------------------------------------------------------------------------------

def self_test(gg: ar.GGUFFile, manifest: dict, blob_path: Path, n_samples: int, seed: int) -> None:
    entries = manifest["tensors"]
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(entries))
    idxs = rng.choice(len(entries), size=k, replace=False)

    print(f"\n--- self-test: {k} of {len(entries)} tensors, seed={seed} ---")
    checked_dtypes = set()
    with open(blob_path, "rb") as bf:
        for i in idxs:
            e = entries[int(i)]
            name = e["name"]
            expected_shape = tuple(e["numpy_shape"])

            # 1. shape must match tensor_numpy_shape(ne) exactly, independently recomputed --
            #    catches a transposed tensor, which an element-count-only check would miss.
            dims, ty, _off = gg.infos[name]
            recomputed_shape = ar.tensor_numpy_shape(dims)
            assert expected_shape == recomputed_shape, (
                f"{name}: manifest numpy_shape {expected_shape} != "
                f"tensor_numpy_shape(ne) {recomputed_shape}")
            assert tuple(e["ggml_ne"]) == dims, f"{name}: manifest ggml_ne drifted from GGUF ne"

            # 2. re-read the tensor bytes straight from the blob at the manifest's offset.
            np_dtype = _NP_DTYPE_BY_OUT_NAME[e["dtype"]]
            bf.seek(e["offset"])
            raw = bf.read(e["nbytes"])
            blob_arr = np.frombuffer(raw, dtype=np_dtype).reshape(expected_shape)
            assert blob_arr.shape == expected_shape, (
                f"{name}: blob array shape {blob_arr.shape} != manifest shape {expected_shape}")

            # 3. independently re-dequantize the SAME tensor from the source GGUF (fresh read,
            #    not reusing anything cached from the conversion pass) and apply the SAME cast
            #    policy; require BIT-EXACT equality, not a numeric tolerance -- the cast is
            #    deterministic, so anything less than exact equality means the write or the read
            #    path corrupted something (wrong offset, wrong dtype view, off-by-one in padding).
            ref_f32 = ar.read_tensor(gg, name)
            assert ref_f32.shape == expected_shape
            expected_rounded = ref_f32.astype(np_dtype)
            assert np.array_equal(blob_arr, expected_rounded), (
                f"{name}: blob bytes do NOT round-trip to the same {e['dtype']} values as an "
                f"independent re-dequant + re-cast from the source GGUF")

            # secondary, human-readable metric: how far bf16 rounding moved the values (informational).
            rel_err = float(np.max(np.abs(
                blob_arr.astype(np.float32) - ref_f32) / (np.abs(ref_f32) + 1e-8)))
            checked_dtypes.add(e["dtype"])
            print(f"  [ok] {name}: shape={expected_shape} dtype={e['dtype']} "
                  f"bit-exact round-trip, max-rel-err-vs-f32={rel_err:.2e}")

    assert checked_dtypes, "self-test sampled zero tensors"
    print(f"  [ok] all {k} sampled tensors: shapes exact, bytes bit-exact against independent "
          f"re-dequant. dtypes exercised: {sorted(checked_dtypes)}")


# --------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=None, help="path to the S2-Pro GGUF (default: codec_paths.gguf())")
    ap.add_argument("--out", required=True, help="output directory for the blob + manifest.json")
    ap.add_argument("--layers", type=int, default=None,
                     help="cap slow AND fast layers each at N (plus all top-level tensors) for a "
                          "smoke test; omit for the full model (36 slow + 4 fast, 358 tensors)")
    ap.add_argument("--selftest-samples", type=int, default=12,
                     help="number of tensors to round-trip-verify after writing (default 12)")
    ap.add_argument("--selftest-seed", type=int, default=0)
    ap.add_argument("--no-selftest", action="store_true", help="skip the round-trip self-test (not recommended)")
    args = ap.parse_args()

    gguf_path = args.gguf or codec_paths.gguf()
    print(f"GGUF: {gguf_path}")
    gg = ar.open_gguf(gguf_path)
    hp = ar.read_ar_hparams(gg)
    print(f"hparams: block_count={hp.block_count} fast_block_count={hp.fast_block_count} "
          f"embedding_length={hp.embedding_length} attention_qk_norm={hp.attention_qk_norm} "
          f"fast_attention_qk_norm={hp.fast_attention_qk_norm}")

    print_size_report(analytic_size_report(gg, hp))

    names, n_slow, n_fast = select_tensor_names(hp, args.layers)
    full_set = args.layers is None
    print(f"\nconverting {len(names)} tensors (slow_layers={n_slow}/{hp.block_count}, "
          f"fast_layers={n_fast}/{hp.fast_block_count}, full_set={full_set})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = out_dir / BLOB_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    entries, blob_bytes, elapsed = convert(gg, names, blob_path)
    total_elements = sum(int(np.prod(e["numpy_shape"])) for e in entries)
    rate = total_elements / elapsed if elapsed > 0 else float("inf")
    ggib = 1024 ** 3
    print(f"\n--- conversion done ---")
    print(f"  wrote {len(entries)} tensors, {blob_bytes:,} B ({blob_bytes/ggib:.3f} GiB) -> {blob_path}")
    print(f"  {total_elements:,} elements in {elapsed:.2f}s ({rate/1e6:.1f}M elements/s)")
    full_rep = analytic_size_report(gg, hp)
    extrapolated_full_s = full_rep["total_elements"] / rate if rate > 0 else float("nan")
    print(f"  extrapolated full-358-tensor-set time at this rate: {extrapolated_full_s:.1f}s")

    manifest = build_manifest(gg, hp, entries, blob_bytes, n_slow, n_fast, full_set)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote manifest ({len(entries)} tensor entries) -> {manifest_path}")

    if not args.no_selftest:
        self_test(gg, manifest, blob_path, args.selftest_samples, args.selftest_seed)
    else:
        print("\n--no-selftest: skipped round-trip verification")

    print("\nAll done.")


if __name__ == "__main__":
    main()
