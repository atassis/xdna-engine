#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump a HuggingFace decoder-LLM checkpoint to the flat .npy tree gen_llm_decode.py reads.

One .npy per tensor, named by its HF key. Only the tensors the spec's graph actually consumes are
written, and every one is CHECKED against the spec's dims before writing -- a silently transposed or
mis-shaped projection is the failure mode that survives every build gate and shows up only as
drifting token parity.

  python scripts/dump_llm_weights.py --spec qwen3-0.6b --out artifacts/qwen3-0.6b/weights

`--quant` packs the PROJECTION matrices on the way out, for models whose f32 dump does not fit on
disk: a 12B is 43 GB at f32 and 6 GB at int4 g64. Packed tensors are np.int8 ON-WIRE BYTES, not
values, so the dump declares itself in `quant.json` (dtype, group_size, and the exact set of packed
names) and the generator READS that declaration. Agreeing via matching env vars instead would make a
group-size disagreement silent numerical garbage rather than a load error -- the packer and the
kernel share a byte layout, and nothing else checks it.

  python scripts/dump_llm_weights.py --spec qwen3-0.6b --out ... --quant int4 --quant-group 64
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "designs", "decode_fused"))
from llm_decode_spec import SPECS, k_chunks_for  # noqa: E402

HF_REPO = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "gemma3-270m": "unsloth/gemma-3-270m-it",
           "gemma4-12b": "unsloth/gemma-4-12b-it"}

# The projection leaves, i.e. everything that is a [out, in] matrix rather than a norm gain or the
# embedding table. Only these are packable.
EXP_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# The tied embedding/lm-head. Handled separately from EXP_LEAVES: it always stays dumped f32 (the
# host embedding-gather reads it directly), and packing it ADDS a sidecar rather than replacing
# the array -- see _pack_head_chunked.
HEAD_LEAF = "embed_tokens"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None, help="override the HF repo id")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="read safetensors from this local directory instead of resolving --repo "
                         "through the HF cache. For a checkpoint already fetched to a local_dir, "
                         "which the cache does not see and would otherwise re-download.")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--quant", default="bf16", choices=("bf16", "int4", "int8"),
                    help="pack the PROJECTION matrices at this width (norms stay f32 and readable; "
                         "the embedding stays f32 and readable too unless --quant-leaves names "
                         "'embed_tokens'). bf16 writes the plain f32 dump.")
    ap.add_argument("--quant-group", type=int, default=64)
    ap.add_argument("--quant-full-range", action="store_true",
                    help="quantize onto all 2**n levels rather than the symmetric subset. "
                         "Needed to land on the grid of a checkpoint QAT-trained for this "
                         "width; worth ~0.5 points on any other checkpoint (K025).")
    ap.add_argument("--quant-scale-dtype", default="f32", choices=("f32", "bf16"),
                    help="stored width of the per-group scale. mv_quant.cc casts it to bfloat16 "
                         "before the MAC either way, so f32 stores 2 B/group the core discards -- "
                         "1 bit/weight at int4 g32. The kernel must be built to match "
                         "(GEMV(scale_dtype=...) -> -DSCALE_BF16), so the two move together.")
    ap.add_argument("--resume", action="store_true",
                    help="skip keys already written, per this dir's _dump_progress.json. A full "
                         "dump is ~1h and has no other way to survive an interruption; the "
                         "progress file records the format it was written under and refuses to "
                         "resume into a different one.")
    ap.add_argument("--quant-clip-search", action="store_true",
                    help="per-group MSE-optimal scale instead of amax/qmax. Pair with "
                         "--quant-full-range: the grid-exact scale is a search candidate.")
    ap.add_argument("--quant-layout", default="header_first",
                    choices=("header_first", "row_group_planar"),
                    help="on-wire row layout (iron/common/quant.py). row_group_planar moves the "
                         "per-row scale out of the row, which is what lets g64 reach a 512-bit "
                         "load at K=3840 instead of the 128-bit header_first gives it -- see "
                         "quant.max_legal_vec_size's docstring. ROW_GROUP is DERIVED per tensor "
                         "(K022), never chosen here.")
    ap.add_argument("--cols", type=int, default=8,
                    help="num_aie_columns the CONSUMER will build with. Only used to decide which "
                         "packed tensors must be pre-chunked over K, via the same k_chunks_for the "
                         "generator calls -- a mismatch here silently produces chunks it cannot read.")
    # Default: exactly the leaves gen_llm_decode.py can CONSUME packed today -- the MLP class
    # (QUANT_MLP_DTYPE) and Wo (QUANT_ATTN_DTYPE). q/k/v are excluded because their GEMVs are built
    # without quant kwargs (`gemv(QD, D, ctx)`, and the concatenated Wqkv likewise), so a packed
    # q_proj has no reader. Packing them is allowed but the build will refuse it by name rather
    # than quietly widening the bytes.
    ap.add_argument("--quant-leaves", default="gate_proj,up_proj,down_proj,o_proj",
                    help="comma-separated projection leaves to pack (default: the ones the "
                         "generator can read back). Add 'embed_tokens' to also pack the tied "
                         "lm-head/embedding into a '<key>.headpack' sidecar (P009).")
    a = ap.parse_args()
    sp = SPECS[a.spec]
    repo = a.repo or HF_REPO[a.spec]
    NL = a.layers if a.layers is not None else sp.n_layers
    os.makedirs(a.out, exist_ok=True)

    # HF stores nn.Linear weights as [out, in].
    D_, FF_, V_ = sp.d_model, sp.ffn, sp.vocab

    quant_leaves = {x for x in a.quant_leaves.split(",") if x}
    quantize_weight = None
    derive_row_group = None
    if a.quant != "bf16":
        unknown = quant_leaves - set(EXP_LEAVES) - {HEAD_LEAF}
        if unknown:
            ap.error(f"--quant-leaves has unknown leaves {sorted(unknown)}; expected a subset of "
                     f"{sorted(EXP_LEAVES)} plus {HEAD_LEAF!r}")
        # Imported from IRON rather than reimplemented here, because the on-wire row layout
        # ([n_groups x f32 scale][packed payload]) is shared with the matvec kernel. A second copy
        # of it is a seam with no owner, which is the class this whole manifest exists to close.
        # The packer MOVED in IRON 6a347dc; try both sides (see gen_llm_decode.py's identical
        # compat shim). derive_row_group is post-move only -- row_group_planar needs it.
        try:
            from iron.common.quant import quantize_weight, derive_row_group
        except ModuleNotFoundError:
            from iron.operators.gemv.quant import quantize_weight
            if a.quant_layout == "row_group_planar":
                ap.error("--quant-layout row_group_planar needs iron.common.quant "
                         "(derive_row_group); this IRON tree only has the pre-move "
                         "iron.operators.gemv.quant. Point IRON at a tree past 6a347dc.")

    from safetensors import safe_open

    if a.checkpoint_dir:
        path = a.checkpoint_dir
        if not any(f.endswith(".safetensors") for f in os.listdir(path)):
            ap.error(f"--checkpoint-dir {path} holds no .safetensors")
        repo = f"{path} (local)"
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    shards = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    index = {}
    for s in shards:
        # framework="pt": these checkpoints are bf16, which safetensors' numpy backend cannot read
        # ("data type 'bfloat16' not understood"). torch reads it and .float() widens losslessly.
        with safe_open(s, framework="pt") as f:
            for k in f.keys():
                index[k] = s
    print(f"{repo}: {len(index)} tensors across {len(shards)} shard(s)")

    def get(key):
        if key not in index:
            raise KeyError(f"{key!r} not in checkpoint (have e.g. {sorted(index)[:3]})")
        with safe_open(index[key], framework="pt") as f:
            return f.get_tensor(key).float().numpy()

    # EVERY name comes off the spec, never a literal. `weight_prefix` is "model." on a text-only
    # checkpoint and "model.language_model." on Gemma-4-12B, whose text stack sits beside a vision
    # and an audio embedder -- a hardcoded prefix reports that as a missing tensor and names a path
    # instead of the axis.
    want = {}
    exp_per_key = {}   # key -> expected [out, in], since the two geometries do not share one
    for l in range(NL):
        want.update({v: None for v in sp.norm_weight_names(l).values()})
        # Per-layer, not per-spec: Gemma-4-12B's global layers are head_dim 512 / 1 kv head where
        # its sliding layers are 256 / 8, so q_proj is [8192, D] on one and [4096, D] on the other.
        # Checking both against the uniform value would reject the correct checkpoint.
        qd, kvd, hd = sp.q_dim_for(l), sp.kv_dim_for(l), sp.head_dim_for(l)
        leaves = {"self_attn.q_proj": (qd, D_), "self_attn.k_proj": (kvd, D_),
                  "self_attn.o_proj": (D_, qd), "mlp.gate_proj": (FF_, D_),
                  "mlp.up_proj": (FF_, D_), "mlp.down_proj": (D_, FF_)}
        # attention_k_eq_v: a layer whose V is the raw k projection HAS no v_proj tensor. Demanding
        # one turns a correctly-dumped checkpoint into a KeyError.
        if sp.has_v_proj(l):
            leaves["self_attn.v_proj"] = (kvd, D_)
        for t, shape in leaves.items():
            key = f"{sp.weight_prefix}layers.{l}.{t}.weight"
            want[key] = None
            exp_per_key[key] = shape
        for nm in sp.norm_weight_names(l).values():
            # q_norm/k_norm are per-HEAD, so they follow the layer's head_dim; every other norm is
            # d_model-wide.
            exp_per_key[nm] = (hd,) if nm.endswith(("q_norm.weight", "k_norm.weight")) else (D_,)
        if sp.layer_scalar:
            # A register_buffer -- in the checkpoint, absent from config.json, and scalar-shaped, so
            # it gets no shape assertion beyond "it is there".
            want[sp.layer_scalar_name(l)] = None
    want[f"{sp.weight_prefix}norm.weight"] = None
    exp_per_key[f"{sp.weight_prefix}norm.weight"] = (D_,)
    want[f"{sp.weight_prefix}embed_tokens.weight"] = None
    exp_per_key[f"{sp.weight_prefix}embed_tokens.weight"] = (V_, D_)

    def _layout_kw(K):
        """quantize_weight kwargs for one tensor's OWN K (post-chunking) -- row_group is a
        pure function of (K, group_size, dtype, SCALE_DTYPE), so this always agrees with what
        GEMV's own __post_init__ derives for a GEMV built at the same four (no shared state,
        no quant.json round-trip needed for the value itself). The scale width is in it
        because it sets the row stride: at int4/g32/K=3840 an f32 scale gives stride 2400 and
        row_group 1, a bf16 scale gives 2160 and row_group 2. Omitting it here defaulted the
        derivation to f32 and packed every bf16-scale dump one block-shape off what the
        kernel reads -- silently, because the bytes are a permutation and every value is
        still there. The scale-selection axes ride here
        too so that every packing path in this script -- dense, K-chunked and head-chunked --
        goes through one funnel and cannot disagree about the format."""
        kw = {"full_range": a.quant_full_range, "clip_search": a.quant_clip_search,
              "scale_dtype": a.quant_scale_dtype}
        if a.quant_layout != "row_group_planar":
            return kw
        # Same rule the operator derives row_group from -- see widest_chunk's docstring.
        from iron.common.quant import widest_chunk
        vec = widest_chunk(a.quant_group, a.quant)
        kw.update(layout=a.quant_layout,
                  row_group=derive_row_group([K], a.quant_group, a.quant, vec_size=vec,
                                             scale_dtype=a.quant_scale_dtype))
        return kw

    # Row-chunk budget for packing the tied head/embedding -- bounds quantize_weight's OWN
    # transient arrays (its abs/div/round/clip/astype chain each allocates a full chunk-sized f32
    # temporary) regardless of K. A dense call over Gemma-4-12B's 262144x3840 table peaks near
    # 22 GiB and OOMs a 30 GiB box; chunked, each temporary is ~64 MiB. Quantization is per-row
    # (a group never spans rows) and row_group_planar's block rearrange is per ROW_GROUP-row
    # block, so any chunk size that is a multiple of row_group concatenates byte-for-byte equal
    # to one dense call -- this is a memory fix, not a format change.
    HEAD_CHUNK_BUDGET_BYTES = 64 * 1024 * 1024

    def _pack_head_chunked(w):
        M, K = w.shape
        row_group = _layout_kw(K).get("row_group", 1)
        chunk_rows = max(row_group,
                         (HEAD_CHUNK_BUDGET_BYTES // (K * 4) // row_group) * row_group)
        parts = [quantize_weight(w[r0:min(r0 + chunk_rows, M)], a.quant_group, a.quant,
                                 **_layout_kw(K))
                for r0 in range(0, M, chunk_rows)]
        return np.concatenate(parts)

    prog_path = os.path.join(a.out, "_dump_progress.json")
    fmt_key = json.dumps({k: getattr(a, k) for k in (
        "quant", "quant_group", "quant_layout", "quant_scale_dtype", "quant_full_range",
        "quant_clip_search", "quant_leaves", "layers", "cols")}, sort_keys=True)
    done = {}
    if a.resume and os.path.isfile(prog_path):
        _prev = json.load(open(prog_path))
        if _prev.get("fmt") != fmt_key:
            raise SystemExit(
                f"[dump] --resume refused: {prog_path} was written under a different format.\n"
                f"  there: {_prev.get('fmt')}\n  here:  {fmt_key}\n"
                f"  delete the directory and start over rather than mixing two formats in it")
        done = _prev.get("done", {})
        print(f"[dump] resuming: {len(done)} keys already written")

    def _record(key, names):
        """Mark one key complete. Written per key, because the cost of losing the position is an
        hour and the cost of the write is a few hundred bytes."""
        done[key] = names
        with open(prog_path, "w") as f:
            json.dump({"fmt": fmt_key, "done": done}, f)

    n, packed = 0, []
    for key in sorted(want):
        if key in done:
            packed.extend(done[key])
            n += 1
            continue
        _mark = len(packed)
        w = get(key)
        leaf = key.rsplit(".", 2)[-2]
        want_shape = exp_per_key.get(key)
        if want_shape is not None and w.shape != want_shape:
            raise ValueError(f"{key}: shape {w.shape} != expected {want_shape} for spec {sp.name}")
        # Shapes are checked ABOVE, on the f32 array, before any packing -- a packed tensor is a
        # flat byte run and has no shape left to check.
        if leaf == HEAD_LEAF:
            # Always dumped f32 -- the host's embedding-gather table reads this file directly,
            # tied or not (P009). Packing ADDS a sidecar rather than replacing it.
            np.save(os.path.join(a.out, f"{key}.npy"), w)
            if quantize_weight is not None and HEAD_LEAF in quant_leaves:
                sidecar = f"{key}.headpack"
                np.save(os.path.join(a.out, f"{sidecar}.npy"), _pack_head_chunked(w))
                packed.append(sidecar)
                n += 1
        elif quantize_weight is not None and leaf in quant_leaves:
            # PRE-CHUNK before packing when the generator will split this tensor over K.
            # A packed tensor is a flat byte run with no axis left to slice, so a K-split
            # CANNOT happen after quantizing -- it would cut through a quantization group and
            # renumber the payload. The generator says so and reads `<tensor>.kchunkN` when
            # they exist (gen_llm_decode.py: "Splitting AFTER packing would cut through a
            # group"), falling back to splitting an UNPACKED tensor itself. Without these it
            # cannot consume a packed down_proj at all: np.split on the flat run dies with
            # "object of type 'int' has no len()".
            #
            # The chunk count must match the generator's exactly or the bytes are silently
            # wrong, so it comes from the same k_chunks_for the generator uses, on the same
            # (D, K, cols) -- never a second copy of the arithmetic.
            nch = k_chunks_for(D_, w.shape[1], a.cols) if w.ndim == 2 else 1
            if nch > 1:
                for i, part in enumerate(np.split(w, nch, axis=1)):
                    name = f"{key}.kchunk{i}"
                    np.save(os.path.join(a.out, f"{name}.npy"),
                            quantize_weight(np.ascontiguousarray(part), a.quant_group, a.quant,
                                            **_layout_kw(part.shape[1])))
                    packed.append(name)
                n += nch - 1
                _record(key, packed[_mark:])
                continue
            np.save(os.path.join(a.out, f"{key}.npy"),
                    quantize_weight(w, a.quant_group, a.quant, **_layout_kw(w.shape[1])))
            packed.append(key)
        else:
            np.save(os.path.join(a.out, f"{key}.npy"), w)
        n += 1
        _record(key, packed[_mark:])

    manifest = {
        "dtype": a.quant,
        "group_size": a.quant_group,
        "layout": a.quant_layout,
        "scale_dtype": a.quant_scale_dtype,
        "full_range": a.quant_full_range,
        "clip_search": a.quant_clip_search,
        "packed": sorted(packed),
        "note": "packed arrays are np.int8 on-wire bytes, NOT values -- never .astype()",
    }
    with open(os.path.join(a.out, "quant.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    tail = ""
    if packed:
        _scale_bits = 32 if a.quant_scale_dtype == "f32" else 16
        bits = ((4 if a.quant == "int4" else 8) * a.quant_group + _scale_bits) / a.quant_group
        tail = f", {len(packed)} packed at {a.quant} g{a.quant_group} scale {a.quant_scale_dtype} = {bits:.2f} bits/weight"
    print(f"wrote {n} tensors to {a.out} (spec {sp.name}, {NL} layers) -- all shapes checked{tail}")


if __name__ == "__main__":
    main()
