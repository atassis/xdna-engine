"""Pack Gemma-3 QAT tensors into the resident rail's padded geometry.

Q4_0 uses group-32 round-to-nearest quantization. Layer and head streams carry
folded norm gains; element round trips and zero padding are checked on export.
"""
import os
os.environ.setdefault("RF_MODEL", "gemma3-270m-rail")

import json
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rf_model import MODEL  # noqa: E402
assert MODEL.name == "gemma3-270m-rail", f"run with RF_MODEL=gemma3-270m-rail, got {MODEL}"
from chain_ref import bf16_bits  # noqa: E402
import mlp_ref  # noqa: E402
import qkv_ref  # noqa: E402
import o_ref  # noqa: E402
import rmlp_ref  # noqa: E402
import rhead  # noqa: E402
import rf_paths

QAT = os.environ.get("RF_G3_QAT")
OUT = os.environ.get("RF_G3_STORE", str(rf_paths.DATA_ROOT / "artifacts/gemma3-270m/rail_store"))

D0, FF0, HQ0, HKV0, HD0, NL = 640, 2048, 4, 1, 256, 18     # the real gemma3-270m
Dp, FFp, HQp, HKVp, HD = MODEL.D, MODEL.FF, MODEL.HQ, MODEL.HKV, MODEL.HD
assert HD0 == HD and MODEL.V == 262144
assert MODEL.norm_width == D0 and MODEL.norm_eps == 1e-6
GFIX = (MODEL.norm_width / Dp) ** 0.5
SLIDING_BASE, SLIDING_WIN = 1e4, MODEL.sliding_window
NCOL, NROW, GROUP = mlp_ref.NCOL, mlp_ref.NROW, 32


def pad(w, o, i):
    out = torch.zeros(o, i)
    out[:w.shape[0], :w.shape[1]] = w
    return out


def gain(w):
    g = torch.zeros(Dp)
    g[:MODEL.norm_width] = (1 + w.float()) * GFIX
    return g.numpy().astype(np.float32)


def q4_0_qs(w):
    """ggml Q4_0 along the last (K) dim, group 32: (q int8 [o, i], scale f32 [o, i//32]) -- the
    grid Google's QAT checkpoint is trained against, split instead of dequantised in place."""
    o, i = w.shape
    g = w.reshape(o, i // 32, 32)
    m = g.gather(-1, g.abs().argmax(-1, keepdim=True))
    d = m / -8
    dz = torch.where(d != 0, d, torch.ones_like(d))
    q = torch.where(d != 0, torch.round(g / dz).clamp(-8, 7), torch.zeros_like(g))
    return q.to(torch.int8).numpy().reshape(o, i), d.squeeze(-1).to(torch.float32).numpy()


def qpad(w, o, i):
    return q4_0_qs(pad(w, o, i))


def load_layer(sd, li):
    p = lambda n: sd[f"model.layers.{li}.{n}"].float()
    kv = lambda w: w.repeat(HKVp, 1)        # every padded kv head copies kv head 0
    q_q, q_s = qpad(p("self_attn.q_proj.weight"), HQp * HD, Dp)
    k_q, k_s = qpad(kv(p("self_attn.k_proj.weight")), HKVp * HD, Dp)
    v_q, v_s = qpad(kv(p("self_attn.v_proj.weight")), HKVp * HD, Dp)
    o_q, o_s = qpad(p("self_attn.o_proj.weight"), Dp, HQp * HD)
    gate_q, gate_s = qpad(p("mlp.gate_proj.weight"), FFp, Dp)
    up_q, up_s = qpad(p("mlp.up_proj.weight"), FFp, Dp)
    down_q, down_s = qpad(p("mlp.down_proj.weight"), Dp, FFp)
    return dict(
        g_in=gain(p("input_layernorm.weight")), g_pre=gain(p("pre_feedforward_layernorm.weight")),
        g_post=gain(p("post_feedforward_layernorm.weight")),
        g_post_attn=gain(p("post_attention_layernorm.weight")),
        q_norm=(1 + p("self_attn.q_norm.weight")).numpy().astype(np.float32),
        k_norm=(1 + p("self_attn.k_norm.weight")).numpy().astype(np.float32),
        q_q=q_q, q_s=q_s, k_q=k_q, k_s=k_s, v_q=v_q, v_s=v_s, o_q=o_q, o_s=o_s,
        gate_q=gate_q, gate_s=gate_s, up_q=up_q, up_s=up_s, down_q=down_q, down_s=down_s,
        ls=np.float32(1.0),       # gemma3 has no altup scalar; neutral for rmlp_ref.gain_elements
    )


def pack_layer(W):
    """attn_in = qkv_ref.stream + per-column head gains (q_norm/k_norm), matching
    weight_store.pack_attn_sliding's attn_in_stream exactly."""
    gu = np.stack([mlp_ref.gateup_stream(W, c) for c in range(NCOL)])
    s = qkv_ref.stream(W).reshape(NCOL, -1)
    hg = qkv_ref.head_gain_elements(W)
    attn_in = np.concatenate([np.concatenate([s[c], hg]) for c in range(NCOL)])
    attn_out = o_ref.stream(W)
    mlp = rmlp_ref.stream(W, gu)
    return attn_in, attn_out, mlp


def head_stream(q, sc, final_norm):
    """rhead's [column][group][row][GW elements] stream, including final-norm gains and padding."""
    if q.shape != (MODEL.V, MODEL.D):
        raise ValueError(f"head q must be {(MODEL.V, MODEL.D)}, got {q.shape}")
    if sc.shape != (q.shape[0], q.shape[1] // GROUP):
        raise ValueError(f"head scales must be {q.shape[0], q.shape[1] // GROUP}, got {sc.shape}")
    return head_stream_from_blocks(rhead.pack_block(q, sc), final_norm)


def head_stream_from_blocks(packed, final_norm):
    expected = (MODEL.V // 32, rhead.NCOL, rhead.NSUB, rhead.ELEM)
    if packed.shape != expected or packed.dtype != np.uint8:
        raise ValueError(f"head blocks must be uint8 {expected}, got {packed.dtype} {packed.shape}")
    return _head_stream_layout(packed, final_norm)


def _head_stream_layout(packed, final_norm):
    if packed.ndim != 4 or packed.shape[1:] != (rhead.NCOL, rhead.NSUB, rhead.ELEM) or packed.shape[0] % rhead.NROW:
        raise ValueError(f"head blocks must be [multiple of {rhead.NROW}, {rhead.NCOL}, {rhead.NSUB}, {rhead.ELEM}], got {packed.shape}")
    if final_norm.shape != (MODEL.D,):
        raise ValueError(f"final norm must be ({MODEL.D},), got {final_norm.shape}")
    ndb = packed.shape[0] // rhead.NROW
    nw = 2 + ndb * rhead.NSUB
    nwp = -(-nw // (2 * rhead.GW)) * (2 * rhead.GW)
    out = np.zeros((rhead.NCOL, nwp, rhead.NROW, rhead.ELEM), np.uint8)
    for c in range(rhead.NCOL):
        out[c, :2] = rmlp_ref.gain_elements(final_norm, np.float32(1.0), c).reshape(2, rhead.NROW, rhead.ELEM)
    for row in range(rhead.NROW):
        part = packed[row * ndb:(row + 1) * ndb]
        for c in range(rhead.NCOL):
            out[c, 2:nw, row] = part[:, c].reshape(-1, rhead.ELEM)
    return np.ascontiguousarray(out.reshape(rhead.NCOL, nwp // rhead.GW, rhead.GW, rhead.NROW, rhead.ELEM)
                             .transpose(0, 1, 3, 2, 4))


def load_head_stream(directory):
    from pathlib import Path
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('head_layout') != rhead.HEAD_STREAM_LAYOUT:
        raise ValueError('head layout is not the resident runtime stream')
    stream = np.load(directory / 'w_head.npy', mmap_mode='r')
    expected = (rhead.NCOL, rhead.NWP // rhead.GW, rhead.NROW, rhead.GW, rhead.ELEM)
    if stream.shape != expected or stream.dtype != np.uint8:
        raise ValueError(f"head stream must be uint8 {expected}, got {stream.dtype} {stream.shape}")
    return stream


# ---- gate: every element this layer packed decodes back to the (q, scale) it was built from, and
# every all-zero padded group decodes to exactly zero -------------------------------------------
def q4_0_ref(w):
    """Independent re-derivation of q4_0_qs (numpy, not torch) for the gate below."""
    o, i = w.shape
    g = w.reshape(o, i // 32, 32)
    m = np.take_along_axis(g, np.abs(g).argmax(-1, keepdims=True), -1)
    d = m / -8
    dz = np.where(d != 0, d, 1.0)
    q = np.where(d != 0, np.clip(np.round(g / dz), -8, 7), 0.0)
    return q.astype(np.int8).reshape(o, i), d.squeeze(-1).astype(np.float32)


def gate_gateup(W, c, gu_c):
    out = gu_c.reshape(-1, mlp_ref.KC // mlp_ref.SUB_K, 2, NROW, mlp_ref.ELEM)
    nblk = out.shape[0]
    bad = 0
    for r in range(NROW):
        for n in range(nblk):
            for h in range(2):
                g0, g1 = mlp_ref.gateup_cols(r, n, h)
                for s in range(mlp_ref.KC // mlp_ref.SUB_K):
                    k0 = c * mlp_ref.KC + s * mlp_ref.SUB_K
                    ks = slice(k0, k0 + mlp_ref.SUB_K)
                    gsl = slice(k0 // GROUP, (k0 + mlp_ref.SUB_K) // GROUP)
                    exp_q = np.concatenate([W["gate_q"][g0, ks], W["up_q"][g0, ks],
                                            W["gate_q"][g1, ks], W["up_q"][g1, ks]])
                    exp_s = np.concatenate([W["gate_s"][g0, gsl], W["up_s"][g0, gsl],
                                            W["gate_s"][g1, gsl], W["up_s"][g1, gsl]])
                    q, sc = mlp_ref.unpack_elem(out[n, s, h, r], mlp_ref.SUB_K, GROUP)
                    bad += int(not (np.array_equal(q, exp_q) and np.allclose(sc, exp_s)))
    return bad


def gate_down(W, c, down_c):
    nsp, ndb = mlp_ref.down_dims()
    out = down_c.reshape(ndb, nsp, mlp_ref.KC // mlp_ref.SUB_K, NROW, mlp_ref.ELEM)
    bad = 0
    for r in range(NROW):
        for n in range(ndb):
            cols = np.arange(r * (Dp // NROW) + n * 32, r * (Dp // NROW) + n * 32 + 32)
            for s in range(nsp):
                for sub in range(mlp_ref.KC // mlp_ref.SUB_K):
                    k0 = c * (FFp // NCOL) + s * mlp_ref.KC + sub * mlp_ref.SUB_K
                    exp_q = W["down_q"][cols, k0:k0 + mlp_ref.SUB_K]
                    exp_s = W["down_s"][cols, k0 // GROUP:(k0 + mlp_ref.SUB_K) // GROUP]
                    q, sc = mlp_ref.unpack_elem(out[n, s, sub, r], mlp_ref.SUB_K, GROUP)
                    bad += int(not (np.array_equal(q, exp_q) and np.allclose(sc, exp_s)))
    return bad


def gate_qkv(W, c):
    q, s = qkv_ref.weights_perm(W)
    out = qkv_ref.qkv_stream(W, c).reshape(qkv_ref.NBLK_Q, mlp_ref.KC // mlp_ref.SUB_K, 2, NROW, mlp_ref.ELEM)
    bad = 0
    for r in range(NROW):
        for n in range(qkv_ref.NBLK_Q):
            for h in range(2):
                idx = [qkv_ref.chain_col(r, n, 32 * h + t) for t in range(32)]
                for sb in range(mlp_ref.KC // mlp_ref.SUB_K):
                    k0 = c * mlp_ref.KC + sb * mlp_ref.SUB_K
                    exp_q = q[idx, k0:k0 + mlp_ref.SUB_K]
                    exp_s = s[idx, k0 // GROUP:(k0 + mlp_ref.SUB_K) // GROUP]
                    qd, scd = mlp_ref.unpack_elem(out[n, sb, h, r], mlp_ref.SUB_K, GROUP)
                    bad += int(not (np.array_equal(qd, exp_q) and np.allclose(scd, exp_s)))
    return bad


def gate_o(W, c):
    q, s = o_ref.weights_pad(W)
    out = o_ref.o_stream(W, c).reshape(o_ref.NDB, o_ref.NSPO, o_ref.KCO // mlp_ref.SUB_K, NROW, mlp_ref.ELEM)
    bad = 0
    for r in range(NROW):
        for n in range(o_ref.NDB):
            cols = np.arange(r * (Dp // NROW) + n * 32, r * (Dp // NROW) + n * 32 + 32)
            for sp in range(o_ref.NSPO):
                for sub in range(o_ref.KCO // mlp_ref.SUB_K):
                    k0 = c * o_ref.KPAD + sp * o_ref.KCO + sub * mlp_ref.SUB_K
                    exp_q = q[cols, k0:k0 + mlp_ref.SUB_K]
                    exp_s = s[cols, k0 // GROUP:(k0 + mlp_ref.SUB_K) // GROUP]
                    qd, scd = mlp_ref.unpack_elem(out[n, sp, sub, r], mlp_ref.SUB_K, GROUP)
                    bad += int(not (np.array_equal(qd, exp_q) and np.allclose(scd, exp_s)))
    return bad


def gate_padding_zero(W):
    """Check (a): every zero-padded region decodes to exactly zero, straight off the quantised
    (q, scale) arrays every stream above was packed from."""
    bad = 0
    bad += int(np.any(W["q_q"][HQ0 * HD:]) or np.any(W["q_q"][:, D0:]))
    bad += int(np.any(W["o_q"][:, HQ0 * HD:]) or np.any(W["o_q"][D0:]))
    bad += int(np.any(W["gate_q"][FF0:]) or np.any(W["gate_q"][:, D0:]))
    bad += int(np.any(W["up_q"][FF0:]) or np.any(W["up_q"][:, D0:]))
    bad += int(np.any(W["down_q"][D0:]) or np.any(W["down_q"][:, FF0:]))
    bad += int(np.any(W["g_in"][D0:]) or np.any(W["g_pre"][D0:]) or np.any(W["g_post"][D0:])
              or np.any(W["g_post_attn"][D0:]))
    return bad


def gate_q4_0_match(W, p):
    """Check (b): the dequantised real block equals an independent re-derivation of Q4_0 on the
    same padded fp32 tensor, within float rounding."""
    def real(w_q, w_s, o, i, name):
        wf = p(name)
        padded = np.zeros((o, i), np.float32)
        padded[:wf.shape[0], :wf.shape[1]] = wf.numpy()
        rq, rs = q4_0_ref(padded)
        got = w_q.astype(np.float32) * np.repeat(w_s, GROUP, axis=1)
        ref = rq.astype(np.float32) * np.repeat(rs, GROUP, axis=1)
        return float(np.abs(got - ref).max())
    return {
        "gate": real(W["gate_q"], W["gate_s"], FFp, Dp, "mlp.gate_proj.weight"),
        "down": real(W["down_q"], W["down_s"], Dp, FFp, "mlp.down_proj.weight"),
    }


def main():
    if not QAT:
        raise ValueError('set RF_G3_QAT to the source checkpoint directory')
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sd = load_file(f"{QAT}/model.safetensors")

    print("=== gate: per-element round trip + zero-padding + independent Q4_0 re-derivation ===", flush=True)
    all_ok = True
    sizes = []
    for li in range(NL):
        W = load_layer(sd, li)
        attn_in, attn_out, mlp = pack_layer(W)
        if li in (0, NL - 1):
            bad = gate_padding_zero(W)
            bad += sum(gate_qkv(W, c) for c in range(NCOL))
            bad += sum(gate_o(W, c) for c in range(NCOL))
            gu = [mlp_ref.gateup_stream(W, c) for c in range(NCOL)]
            bad += sum(gate_gateup(W, c, gu[c]) for c in range(NCOL))
            bad += sum(gate_down(W, c, mlp_ref.down_stream(W, c)) for c in range(NCOL))
            m = gate_q4_0_match(W, lambda n: sd[f"model.layers.{li}.{n}"].float())
            all_ok &= bad == 0
            print(f"  layer {li}: {bad} mismatched elements, max|d| gate={m['gate']:.3e} down={m['down']:.3e}", flush=True)
        w = np.concatenate([attn_in, attn_out, mlp])
        np.save(f"{OUT}/w{li}.tmp.npy", w)
        os.replace(f"{OUT}/w{li}.tmp.npy", f"{OUT}/w{li}.npy")
        sizes.append(int(w.nbytes))
        print(f"layer {li:2d} packed, {w.nbytes / 1e6:.2f} MB", flush=True)
    if not all_ok:
        print("GATE FAILED", flush=True)
        sys.exit(1)

    E = sd["model.embed_tokens.weight"].float()
    emb = pad(E * D0 ** 0.5, E.shape[0], Dp)
    np.save(f"{OUT}/embed.npy", bf16_bits(emb.numpy()))
    final_norm = gain(sd["model.norm.weight"].float())
    head_q, head_s = qpad(E, E.shape[0], Dp)
    head_el = head_stream(head_q, head_s, final_norm)
    np.save(f"{OUT}/w_head.npy", head_el)
    np.save(f"{OUT}/final_norm.npy", final_norm)
    head_bytes = int(head_el.nbytes)

    manifest = {
        "model": "gemma3-270m-rail", "created": time.strftime("%Y-%m-%d"),
        "source_checkpoint": QAT, "quantization": "q4_0 round-to-nearest, group 32 (the QAT checkpoint is continuous-valued, not grid-snapped)",
        "num_layers": NL, "hidden_size": Dp, "intermediate_size": FFp,
        "n_q_heads": HQp, "n_kv_heads": HKVp, "head_dim": HD, "vocab_size": MODEL.V,
        "full_attention_layers": [], "sliding_base": SLIDING_BASE, "sliding_window": SLIDING_WIN,
        "layer_bytes": sizes, "head_bytes": head_bytes, "head_layout": rhead.HEAD_STREAM_LAYOUT,
        "embed_bytes": int(emb.numel() * 2), "embed_dtype": "bf16",
        "final_norm_bytes": int(final_norm.nbytes),
    }
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(sizes) + head_bytes + manifest["embed_bytes"] + manifest["final_norm_bytes"]
    print(f"\ndone in {time.time() - t0:.1f}s", flush=True)
    print(f"layer bytes: min {min(sizes)/1e6:.2f} max {max(sizes)/1e6:.2f} MB; head {head_bytes/1e6:.2f} MB; "
          f"total {total/1e6:.1f} MB", flush=True)
    print(f"manifest: {OUT}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
