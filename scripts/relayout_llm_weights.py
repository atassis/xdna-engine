#!/usr/bin/env python3
"""Re-layout a packed weight dump -- to the row_group its reader derives, or off planar
entirely (`--layout header_first`) -- without requantizing.

`row_group_planar` is a REORDER of header-first rows (`_rows_to_planar`'s own docstring: "only
where a byte lives changes, never what it is"), so a dump packed at the wrong row_group is
recoverable by permutation -- no safetensors read, no quantize, no clip search, and every value
bit-identical, which is why a parity result taken on the old dump still holds on the new one.

Why a dump can be wrong: derive_row_group takes scale_dtype (it sets the row stride), and
dump_llm_weights.py omitted it, defaulting to f32. At int4/g32/K=3840 that is row_group 1 against
the bf16 answer of 2. Fixed there; this recovers the dumps already on disk.

K comes from the bf16 dump's own shapes, never from packed file size: at int4/g32 stride(3840)
divides o_proj's total exactly, so size alone reports K=3840 for a K=4096 tensor.
"""
import argparse, importlib.util, json, os, shutil, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="packed dump to read")
ap.add_argument("--bf16", required=True, help="bf16 dump supplying tensor shapes")
ap.add_argument("--out", required=True)
ap.add_argument("--iron", default=os.environ.get("IRON_DIR", ""))
ap.add_argument("--src-row-group", type=int,
                help="row_group the SOURCE was packed at, when its quant.json does not record one. "
                     "There is no safe default: a wrong value is a silent garbage permutation that "
                     "every self-consistency check still passes. Confirm it by dequantizing one "
                     "tensor against the bf16 dump -- the wrong value reads as nan or noise.")
ap.add_argument("--layout", default="row_group_planar",
                choices=("row_group_planar", "header_first"),
                help="layout to WRITE. row_group_planar (default) re-derives the row_group, which is "
                     "what a decode GEVM dump needs. header_first undoes the planar reorder entirely: "
                     "prefill's GEMM path reads the row-packed form and iron.common.quant."
                     "repack_gemm_weight has no planar parser.")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
# quant.py by path, not `from iron.common.quant import ...`: the layout math is numpy-only, while
# `iron.common.__init__` pulls in `aie.utils` and so a provisioned MLIR distro this tool never uses.
_qp = os.path.join(a.iron or ".", "iron", "common", "quant.py")
if not os.path.isfile(_qp):
    sys.exit(f"no iron/common/quant.py under --iron {a.iron!r}")
_spec = importlib.util.spec_from_file_location("_iron_quant", _qp)
_quant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_quant)
widest_chunk, derive_row_group = _quant.widest_chunk, _quant.derive_row_group
row_stride_bytes = _quant.row_stride_bytes
_rows_to_planar, _planar_to_rows = _quant._rows_to_planar, _quant._planar_to_rows

man = json.load(open(os.path.join(a.src, "quant.json")))
G, DT, SDT = man["group_size"], man["dtype"], man.get("scale_dtype", "f32")
assert man["layout"] == "row_group_planar", man["layout"]
packed = set(man["packed"])

def shape_of(name):
    """(M, K) for a packed tensor. A `.kchunkI` splits its base's K by the chunk count."""
    base, _, tail = name.rpartition(".kchunk")
    if not base:
        base, n_chunks = name, 1
    else:
        n_chunks = 1 + max(int(p.rpartition(".kchunk")[2]) for p in packed
                           if p.startswith(base + ".kchunk"))
    src = base[:-len(".headpack")] if base.endswith(".headpack") else base
    arr = np.load(os.path.join(a.bf16, src + ".npy"), mmap_mode="r")
    M, K = arr.shape[0], arr.shape[-1]
    assert K % n_chunks == 0, (name, K, n_chunks)
    return M, K // n_chunks

# A wrong source row_group permutes every row into garbage and still round-trips cleanly, so it is
# refused rather than defaulted. Measured 2026-09-15: weights_int8g64_planar records no row_group and
# was packed at 4, where the former default of 1 dequantizes to nan against the bf16 dump.
if "row_group" in man:
    SRC_RG = int(man["row_group"])
    if a.src_row_group is not None and a.src_row_group != SRC_RG:
        sys.exit(f"--src-row-group {a.src_row_group} contradicts {a.src}/quant.json's {SRC_RG}")
elif a.src_row_group is not None:
    SRC_RG = a.src_row_group
else:
    sys.exit(f"{a.src}/quant.json records no row_group and this tool will not guess one -- "
             f"pass --src-row-group. Confirm the value by dequantizing one tensor against the "
             f"bf16 dump at each candidate; only the right one is not nan.")

os.makedirs(a.out, exist_ok=True)
moved = same = 0
for f in sorted(os.listdir(a.src)):
    name, src = f[:-4] if f.endswith(".npy") else None, os.path.join(a.src, f)
    dst = os.path.join(a.out, f)
    if name not in packed:
        if not a.dry_run:
            (shutil.copy2 if f.endswith(".json") else os.link)(src, dst)
        continue
    M, K = shape_of(name)
    stride = row_stride_bytes(K, G, DT, SDT)
    new_rg = derive_row_group([K], G, DT, vec_size=widest_chunk(G, DT), scale_dtype=SDT)
    old_rg = SRC_RG
    blob = np.load(src, mmap_mode="r")
    assert blob.nbytes == M * stride, (name, blob.nbytes, M, stride)
    if new_rg == old_rg and a.layout == "row_group_planar":
        same += 1
        if not a.dry_run:
            os.link(src, dst)
        continue
    rows = _planar_to_rows(np.asarray(blob), M, K, stride, DT, old_rg)
    if a.layout == "header_first":
        out = rows.ravel()
        # The round-trip runs the other way here: re-planarizing must reproduce the input BYTES.
        # Compared as uint8 on both sides -- the dump is int8 on disk and these helpers return
        # uint8, so comparing values instead of bytes reports every high byte as unequal.
        assert np.array_equal(_rows_to_planar(rows, K, DT, old_rg).view(np.uint8).ravel(),
                              np.asarray(blob).view(np.uint8).ravel()), name
        print(f"  {name:70} K={K:5} M={M:6} rg {old_rg}->header_first")
    else:
        out = _rows_to_planar(rows, K, DT, new_rg)
        # Lossless by construction, checked anyway: both layouts must decode to the SAME rows.
        back = _planar_to_rows(out, M, K, stride, DT, new_rg)
        assert np.array_equal(back, rows), name
        print(f"  {name:70} K={K:5} M={M:6} rg {old_rg}->{new_rg}")
    moved += 1
    if not a.dry_run:
        np.save(dst, out.view(np.int8))

if a.layout == "header_first":
    man["layout"] = "header_first"
    man.pop("row_group", None)
    man["note"] = man.get("note", "") + " | de-planarized by relayout_llm_weights.py"
else:
    man["row_group"] = derive_row_group([3840], G, DT, vec_size=widest_chunk(G, DT), scale_dtype=SDT)
    man["note"] = man.get("note", "") + " | row_group is per-K; see relayout_llm_weights.py"
if not a.dry_run:
    json.dump(man, open(os.path.join(a.out, "quant.json"), "w"), indent=2)
print(f"\nre-laid out {moved} tensor(s), {same} already correct"
      + (" (dry run, nothing written)" if a.dry_run else f" -> {a.out}"))
