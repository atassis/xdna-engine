#!/usr/bin/env python3
"""Re-layout a packed weight dump to the row_group its reader derives, without requantizing.

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
import argparse, json, os, shutil, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="packed dump to read")
ap.add_argument("--bf16", required=True, help="bf16 dump supplying tensor shapes")
ap.add_argument("--out", required=True)
ap.add_argument("--iron", default=os.environ.get("IRON_DIR", ""))
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if a.iron:
    sys.path.insert(0, a.iron)
from iron.common.quant import (widest_chunk, derive_row_group, row_stride_bytes,  # noqa: E402
                               _rows_to_planar, _planar_to_rows)

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
    old_rg = man.get("row_group", 1)          # unrecorded in legacy dumps; 1 is what they wrote
    blob = np.load(src, mmap_mode="r")
    assert blob.nbytes == M * stride, (name, blob.nbytes, M, stride)
    if new_rg == old_rg:
        same += 1
        if not a.dry_run:
            os.link(src, dst)
        continue
    rows = _planar_to_rows(np.asarray(blob), M, K, stride, DT, old_rg)
    out = _rows_to_planar(rows, K, DT, new_rg)
    # Lossless by construction, checked anyway: both layouts must decode to the SAME rows.
    back = _planar_to_rows(out, M, K, stride, DT, new_rg)
    assert np.array_equal(back, rows), name
    moved += 1
    print(f"  {name:70} K={K:5} M={M:6} rg {old_rg}->{new_rg}")
    if not a.dry_run:
        np.save(dst, out.view(np.int8))

man["row_group"] = derive_row_group([3840], G, DT, vec_size=widest_chunk(G, DT), scale_dtype=SDT)
man["note"] = man.get("note", "") + " | row_group is per-K; see relayout_llm_weights.py"
if not a.dry_run:
    json.dump(man, open(os.path.join(a.out, "quant.json"), "w"), indent=2)
print(f"\nre-laid out {moved} tensor(s), {same} already correct"
      + (" (dry run, nothing written)" if a.dry_run else f" -> {a.out}"))
