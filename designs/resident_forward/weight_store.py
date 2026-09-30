"""rf-one-weight-store-per-model: pack gemma4-12b's shipped int4g32 planar dump into ONE
content-addressed store, chain-resident layout, one copy per matrix.

Reuses the resident-forward prototypes' own stream builders bit-for-bit (mlp_ref.gateup_stream /
down_stream, qkv_ref.qkv_stream / stream / head_gain_elements, o_ref.o_stream / stream, rmlp_ref.stream)
so the packed bytes are the SAME function calls that produced the gated wstream.npy files per
prototype module (m1/m2/qkv/o/rmlp_ref), generalised from layer 0 to all 48 layers.

Scope note (real, not hypothetical): the chain QKV/O layout in qkv_ref.py/o_ref.py hardcodes the
SLIDING-window head geometry (HQ=16, HKV=8, HD=256, K_O=4096) that layer 0 has. Gemma-4 12B's 8
full-attention layers (5, 11, 17, 23, 29, 35, 41, 47 -- confirmed via config.json's `layer_types` and
via the on-disk shapes: q_proj is 2x the sliding-layer byte count, k_proj 1/4, o_proj split into two
K=4096 kchunks, v_proj absent) have a DIFFERENT (M, K) geometry that no prototype has ever chain-laid
out or gated. Inventing a layout for it here would be undesigned, ungated code the loader would have
no way to trust. Full-attention layers' Q/K/V/O therefore go into the store as their ORIGINAL planar
int4g32 bytes, unchanged (still int4, still one copy, just not chain-resident) -- `layout:
"planar_int4g32_passthrough"` in the manifest -- and this is logged as an open follow-up, not silently
papered over. MLP (gate/up/down) is geometry-invariant across all 48 layers and IS fully chain-laid
out everywhere.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_paths  # noqa: E402
from chain_ref import bf16, bf16_bits, bf16_val  # noqa: E402
import mlp_ref  # noqa: E402
import qkv_ref  # noqa: E402
import o_ref  # noqa: E402
import rmlp_ref  # noqa: E402

# Overridable so recipes/gemma4_data.sh can build a store into a scratch root for verification
# without touching the live artifact; defaults are this repo's artifacts/build trees (rf_paths.py).
WDIR = os.environ.get("RF_WDIR", mlp_ref.WDIR)
D, FF, NCOL, NROW = mlp_ref.D, mlp_ref.FF, mlp_ref.NCOL, mlp_ref.NROW
NLAYERS = 48
FULL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41, 47}   # config.json layer_types, confirmed on-disk
TOWERS_DIR = os.environ.get("RF_TOWERS_DIR", str(rf_paths.ARTIFACTS / "towers_qat"))
STORE = os.environ.get("RF_STORE", str(rf_paths.ARTIFACTS / "store"))
BLOBS = f"{STORE}/blobs"
CHECKPOINT_DIR = os.environ.get("RF_CHECKPOINT_DIR", str(rf_paths.DATA_ROOT / "artifacts/gemma4-12b-qat/checkpoint"))
SCRATCH = os.environ.get("RF_SCRATCH", str(rf_paths.BUILD_ROOT / "scratch"))


# ---- content addressing --------------------------------------------------
def put_blob(data: bytes) -> str:
    """Write bytes to blobs/<sha256>.bin if not already present; return the hash."""
    h = hashlib.sha256(data).hexdigest()
    p = f"{BLOBS}/{h}.bin"
    if not os.path.exists(p):
        tmp = f"{p}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
    return h


def entry(blob_bytes, layout, **meta):
    h = put_blob(blob_bytes)
    e = {"blob": h, "offset": 0, "length": len(blob_bytes), "layout": layout}
    e.update(meta)
    return e


# ---- row_group re-derivation (never trust quant.json's single top-level field: it was derived
# only for K=3840 -- see prototypes/byte-skipping/int4_weights.py::row_group_for_k, reproduced here
# so this packer has no runtime dependency on that directory) --------------------------------------
GROUP, SCALE_BYTES = 32, 2


def row_group_for_k(K, group_size=GROUP, vec_size=64):
    load = vec_size // 2
    n_groups = K // group_size
    payload = K // 2
    header = SCALE_BYTES * n_groups
    stride = header + payload
    g = 1
    while True:
        if (payload % load == 0) and ((g * stride) % load == 0):
            return g
        g += 1
        if g > 64:
            raise ValueError(f"no row_group <= 64 aligns int4/g{group_size} vec={vec_size} at K={K}")


def chunk_files(prefix, name):
    single = f"{WDIR}/{prefix}{name}.npy"
    if os.path.exists(single):
        return [single]
    chunks = sorted(glob.glob(f"{WDIR}/{prefix}{name}.kchunk*.npy"),
                     key=lambda p: int(p.rsplit("kchunk", 1)[1].split(".")[0]))
    if not chunks:
        raise FileNotFoundError(f"no packed file for {prefix}{name}")
    return chunks


def unplanar_matrix(li, name, M, K):
    """(layer, tensor name, M, K) -> (q int8 [M,K], scale f32 [M,K/32]), K-chunks concatenated.
    row_group re-derived per chunk from that chunk's own K (mlp_ref._unplanar's algorithm)."""
    prefix = f"model.language_model.layers.{li}."
    files = chunk_files(prefix, name)
    kc = K // len(files)
    qs, scs = [], []
    for f in files:
        rg = row_group_for_k(kc)
        pk = np.load(f).view(np.uint8)
        q, sc = _unplanar_bytes(pk, M, kc, rg)
        qs.append(q)
        scs.append(sc)
    return np.concatenate(qs, 1), np.concatenate(scs, 1)


def _unplanar_bytes(pk, M, K, row_group):
    """Same algorithm as mlp_ref._unplanar, parameterised on raw bytes + explicit row_group."""
    ng, pay = K // 32, K // 2
    stride = ng * 2 + pay
    blocks = pk.reshape(M // row_group, row_group * stride)
    rows = np.empty((M // row_group, row_group, stride), np.uint8)
    rows[:, :, 2 * ng:] = blocks[:, :row_group * pay].reshape(-1, row_group, pay)
    rows[:, :, :2 * ng] = blocks[:, row_group * pay:].reshape(-1, row_group, 2 * ng)
    rows = rows.reshape(M, stride)
    sc = bf16_val(rows[:, :2 * ng].copy().view(np.uint16)).reshape(M, ng)
    p = rows[:, 2 * ng:]
    lo = (p & 15).astype(np.int8)
    hi = (p >> 4).astype(np.int8)
    q = np.empty((M, K), np.int8)
    q[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    q[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    return q, sc


def raw_load(li, name):
    return np.load(f"{WDIR}/model.language_model.layers.{li}.{name}.npy")


# ---- per-layer loaders, generalised from mlp_ref.load_layer0 / qkv_ref.load_qkv / o_ref.load_o ---
def load_mlp(li):
    out = {}
    for k, name in (("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight")):
        out[k + "_q"], out[k + "_s"] = unplanar_matrix(li, name, FF, D)
    dq, ds = zip(*[unplanar_matrix(li, f"mlp.down_proj.weight.kchunk{i}", D, D) for i in range(4)])
    out["down_q"], out["down_s"] = np.concatenate(dq, 1), np.concatenate(ds, 1)
    out["g_pre"] = raw_load(li, "pre_feedforward_layernorm.weight")
    out["g_post"] = raw_load(li, "post_feedforward_layernorm.weight")
    out["ls"] = np.float32(raw_load(li, "layer_scalar")[0])
    return out


def load_qkv_sliding(li):
    HQ, HKV, HD = qkv_ref.HQ, qkv_ref.HKV, qkv_ref.HD
    out = {}
    for k, name, rows in (("q", "self_attn.q_proj.weight", HQ * HD),
                          ("k", "self_attn.k_proj.weight", HKV * HD),
                          ("v", "self_attn.v_proj.weight", HKV * HD)):
        out[k + "_q"], out[k + "_s"] = unplanar_matrix(li, name, rows, D)
    out["g_in"] = raw_load(li, "input_layernorm.weight")
    out["q_norm"] = raw_load(li, "self_attn.q_norm.weight")
    out["k_norm"] = raw_load(li, "self_attn.k_norm.weight")
    return out


def load_o_sliding(li):
    KO = o_ref.KO
    q, s = unplanar_matrix(li, "self_attn.o_proj.weight", D, KO)
    return dict(o_q=q, o_s=s, g_post_attn=raw_load(li, "post_attention_layernorm.weight"))


# ---- per-layer packers ----------------------------------------------------
def pack_mlp_layer(li):
    """mlp_stream: pre-gain + gate/up + down + post-gain, chain-resident, all 48 layers (rmlp_ref.stream
    generalised from layer 0 -- the shape is geometry-invariant: D/FF are model-wide, not per-layer)."""
    W = load_mlp(li)
    gu = np.stack([mlp_ref.gateup_stream(W, c) for c in range(NCOL)])
    blob = rmlp_ref.stream(W, gu)
    return entry(blob.tobytes(), "chain_resident_v1_mlp", ncol=NCOL, d=D, ff=FF)


def pack_attn_sliding(li):
    """attn_in_stream (pre-attn-norm + QKV + per-column head-gains) and attn_out_stream
    (O + post-attn-norm), chain-resident -- qkv_ref.stream/head_gain_elements/o_ref.stream
    generalised from layer 0."""
    Wq = load_qkv_sliding(li)
    s = qkv_ref.stream(Wq).reshape(NCOL, -1)
    hg = qkv_ref.head_gain_elements(Wq)
    attn_in = np.concatenate([np.concatenate([s[c], hg]) for c in range(NCOL)])
    Wo = load_o_sliding(li)
    attn_out = o_ref.stream(Wo)
    return (entry(attn_in.tobytes(), "chain_resident_v1_attn_in", ncol=NCOL, hq=qkv_ref.HQ,
                  hkv=qkv_ref.HKV, hd=qkv_ref.HD),
            entry(attn_out.tobytes(), "chain_resident_v1_attn_out", ncol=NCOL, ko=o_ref.KO))


def pack_attn_full_passthrough(li):
    """Full-attention layer: Q/K/V/O and their norms go in verbatim (still int4 for the projections,
    still one copy), not chain-resident -- see module docstring. `matrices` mirrors the source
    tensor names instead of the fused stream names sliding layers use."""
    prefix = f"model.language_model.layers.{li}."
    mats = {}
    for name in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight"):
        files = chunk_files(prefix, name)
        chunks = []
        for f in files:
            raw = np.load(f).view(np.uint8)
            k_chunk_bytes = raw.nbytes
            chunks.append(entry(raw.tobytes(), "planar_int4g32_passthrough",
                                 group_size=GROUP, scale_dtype="bf16", n_bytes=int(k_chunk_bytes)))
        mats[name.replace(".weight", "")] = chunks if len(chunks) > 1 else chunks[0]
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        a = raw_load(li, name)
        mats[name.replace(".weight", "")] = entry(a.astype(np.float32).tobytes(), "raw_f32",
                                                  shape=list(a.shape))
    return mats


def pack_embedding():
    """Reuse the existing int4 head pack verbatim (planar int4g32, row_group derived for K=D=3840
    -> 2, matching quant.json). Rows are dequantised on lookup by the loader, never expanded whole."""
    p = f"{WDIR}/model.language_model.embed_tokens.weight.headpack.npy"
    raw = np.load(p).view(np.uint8)
    vocab_cfg = json.load(open(f"{CHECKPOINT_DIR}/config.json"))
    vocab = (vocab_cfg.get("text_config") or vocab_cfg).get("vocab_size", 262144)
    rg = row_group_for_k(D)
    return entry(raw.tobytes(), "planar_int4g32_headpack", vocab=vocab, hidden=D,
                group_size=GROUP, scale_dtype="bf16", row_group=rg)


def pack_towers():
    out = {}
    for p in sorted(glob.glob(f"{TOWERS_DIR}/*.npy")):
        name = os.path.basename(p)[:-4]
        a = np.load(p)
        out[name] = entry(a.tobytes(), "raw_native", shape=list(a.shape), dtype=str(a.dtype))
    return out


# ---- byte-identity gate against the prototypes' own gated wstream files ---------------------------
def gate_layer0():
    """Compare layer-0 output against every wstream.npy the prototypes already gated on device.
    m1/m2's wstreams are gate-less partial builds (no norm gains) used only for the earliest GEMM
    probes; rmlp/qkv/o's wstreams are the full chain-resident streams this packer reproduces."""
    results = []

    def cmp(label, made, path):
        ref = np.load(path).reshape(-1).view(np.uint8)
        made = np.asarray(made).reshape(-1).view(np.uint8)
        ok = made.shape == ref.shape and np.array_equal(made, ref)
        results.append((label, ok, int(made.nbytes), int(ref.nbytes) if made.shape != ref.shape else int(made.nbytes)))
        return ok

    W = load_mlp(0)
    gu = [mlp_ref.gateup_stream(W, c) for c in range(NCOL)]
    cmp("m1/wstream.npy (gate/up, no gains)", np.concatenate(gu), f"{SCRATCH}/m1/wstream.npy")
    m2 = np.concatenate([np.concatenate([gu[c], mlp_ref.down_stream(W, c)]) for c in range(NCOL)])
    cmp("m2/wstream.npy (gate/up+down, no gains)", m2, f"{SCRATCH}/m2/wstream.npy")
    guW = np.stack(gu)
    cmp("rmlp/wstream.npy (full mlp_stream)", rmlp_ref.stream(W, guW), f"{SCRATCH}/rmlp/wstream.npy")

    Wq = load_qkv_sliding(0)
    cmp("qkv/wstream.npy (qkv_ref.stream)", qkv_ref.stream(Wq), f"{SCRATCH}/qkv/wstream.npy")
    s = qkv_ref.stream(Wq).reshape(NCOL, -1)
    hg = qkv_ref.head_gain_elements(Wq)
    wh = np.concatenate([np.concatenate([s[c], hg]) for c in range(NCOL)])
    cmp("qkv/wstream_h.npy (attn_in_stream)", wh, f"{SCRATCH}/qkv/wstream_h.npy")

    Wo = load_o_sliding(0)
    cmp("o/wstream.npy (attn_out_stream)", o_ref.stream(Wo), f"{SCRATCH}/o/wstream.npy")
    return results


# ---- driver ----------------------------------------------------------------
def main():
    os.makedirs(BLOBS, exist_ok=True)
    t0 = time.time()

    print("=== gate: byte-identity against prototypes' own wstream.npy files (layer 0) ===", flush=True)
    all_ok = True
    for label, ok, nbytes, ref_bytes in gate_layer0():
        all_ok &= ok
        print(f"  {'OK ' if ok else 'FAIL'} {label}: {nbytes} B (ref {ref_bytes} B)", flush=True)
    if not all_ok:
        print("GATE FAILED -- not writing the store.", flush=True)
        sys.exit(1)
    print(f"gate passed in {time.time() - t0:.1f}s", flush=True)

    manifest = {
        "model": "gemma4-12b",
        "created": time.strftime("%Y-%m-%d"),
        "source_dir": WDIR,
        "num_layers": NLAYERS,
        "hidden_size": D,
        "intermediate_size": FF,
        "full_attention_layers": sorted(FULL_ATTN_LAYERS),
        "layers": {},
    }

    for li in range(NLAYERS):
        t = time.time()
        is_full = li in FULL_ATTN_LAYERS
        mats = {"mlp_stream": pack_mlp_layer(li)}
        if is_full:
            mats.update(pack_attn_full_passthrough(li))
            attn_type = "full"
        else:
            attn_in, attn_out = pack_attn_sliding(li)
            mats["attn_in_stream"] = attn_in
            mats["attn_out_stream"] = attn_out
            attn_type = "sliding"
        manifest["layers"][str(li)] = {"attn_type": attn_type, "matrices": mats}
        print(f"layer {li:2d} ({attn_type:7s}) packed in {time.time() - t:.1f}s", flush=True)

    manifest["embedding"] = pack_embedding()
    manifest["final_norm"] = entry(
        np.load(f"{WDIR}/model.language_model.norm.weight.npy").astype(np.float32).tobytes(),
        "raw_f32", shape=[D])
    manifest["towers"] = pack_towers()

    with open(f"{STORE}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    blob_bytes = sum(os.path.getsize(f"{BLOBS}/{n}") for n in os.listdir(BLOBS))
    print(f"\ndone in {time.time() - t0:.1f}s", flush=True)
    print(f"store size: {blob_bytes / 1e9:.3f} GB across {len(os.listdir(BLOBS))} blobs", flush=True)
    print(f"manifest: {STORE}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
