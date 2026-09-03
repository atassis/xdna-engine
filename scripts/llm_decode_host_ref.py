"""f32 host reference of the EXACT graph gen_llm_decode.py emits, in numpy.

Splits the failure: if this reproduces HF's tokens, the schedule/weight conventions in LlmSpec are
right and the divergence is on device; if it does not, the graph definition itself is wrong and no
amount of device debugging will help. Same split that cracked the Gemma bring-up.
"""
import json, os, sys
import numpy as np

W = sys.argv[1] if len(sys.argv) > 1 else "artifacts/qwen3-0.6b/weights"
REF = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "artifacts/qwen3-0.6b/refs/greedy_ref.json"))
NL, D, FF, Hq, Hkv, HD, EPS = 28, 1024, 3072, 16, 8, 128, 1e-6
THETA, STEPS = 1e6, 4

def npy(n): return np.load(os.path.join(W, f"{n}.npy")).astype(np.float32)
def rms(x, w):                       # Qwen3: x_hat * w  (NOT 1+w)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w
def silu(x): return x / (1.0 + np.exp(-x))

def rope(v, pos):                    # two-halves, matching HF rotate_half
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float64)[:half] / HD))
    c, s = np.cos(pos * inv).astype(np.float32), np.sin(pos * inv).astype(np.float32)
    v = v.reshape(-1, HD)
    x1, x2 = v[:, :half], v[:, half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(-1)

Wt = {}
for l in range(NL):
    p = f"model.layers.{l}."
    Wt[l] = dict(
        n_in=npy(p + "input_layernorm.weight"), n_pf=npy(p + "post_attention_layernorm.weight"),
        n_qn=npy(p + "self_attn.q_norm.weight"), n_kn=npy(p + "self_attn.k_norm.weight"),
        Wq=npy(p + "self_attn.q_proj.weight"), Wk=npy(p + "self_attn.k_proj.weight"),
        Wv=npy(p + "self_attn.v_proj.weight"), Wo=npy(p + "self_attn.o_proj.weight"),
        Wg=npy(p + "mlp.gate_proj.weight"), Wu=npy(p + "mlp.up_proj.weight"),
        Wd=npy(p + "mlp.down_proj.weight"))
n_final, embed = npy("model.norm.weight"), npy("model.embed_tokens.weight")
scale = HD ** -0.5
S = 64
kc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]
vc = [np.zeros((Hkv, S, HD), np.float32) for _ in range(NL)]

fed, produced = list(REF["prompt_ids"]), []
tok = fed[0]
for pos in range(len(fed) + STEPS - 1):
    x = embed[tok].copy()
    for l in range(NL):
        w = Wt[l]
        h = rms(x, w["n_in"])
        q, k, v = w["Wq"] @ h, w["Wk"] @ h, w["Wv"] @ h
        q = np.concatenate([rms(q.reshape(Hq, HD)[i], w["n_qn"]) for i in range(Hq)])
        k = np.concatenate([rms(k.reshape(Hkv, HD)[i], w["n_kn"]) for i in range(Hkv)])
        q, k = rope(q, pos), rope(k, pos)
        kc[l][:, pos, :] = k.reshape(Hkv, HD); vc[l][:, pos, :] = v.reshape(Hkv, HD)
        qh = q.reshape(Hq, HD)
        ctx = np.empty((Hq, HD), np.float32)
        for hh in range(Hq):
            kvh = hh // (Hq // Hkv)
            sc = (kc[l][kvh, :pos + 1] @ qh[hh]) * scale
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            ctx[hh] = sc @ vc[l][kvh, :pos + 1]
        x = x + w["Wo"] @ ctx.reshape(-1)
        hf = rms(x, w["n_pf"])
        x = x + w["Wd"] @ (silu(w["Wg"] @ hf) * (w["Wu"] @ hf))
    nxt = int(np.argmax(embed @ rms(x, n_final)))
    if pos + 1 < len(fed):
        tok = fed[pos + 1]
    else:
        produced.append(nxt); tok = nxt
    if len(produced) >= STEPS: break

print("HF ref :", REF["gen_ids"][:STEPS])
print("host   :", produced)
print("MATCH" if produced == REF["gen_ids"][:STEPS] else "DIVERGES -> the GRAPH DEFINITION is wrong")
