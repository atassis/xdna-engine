"""P3.1 model: pre-attention RMSNorm and the QKV projection on the chain rail for a sliding layer
(16 q heads, 8 kv heads, head 256). MemTile m holds q heads 2m and 2m+1, k head m and v head m,
1024 columns row-major; chain row r feeds MemTiles 2r and 2r+1."""
import os
import numpy as np
import rf_paths
from chain_ref import bf16, bf16_bits
from mlp_ref import _unplanar, pack_elem, gemm_chain, WDIR, PFX, D, KC, SUB_K, ELEM, NCOL, NROW
from rmlp_ref import rmsnorm, gain_elements

HD, HQ, HKV = 256, 16, 8
NQKV = (HQ + 2 * HKV) * HD            # 8192
NBLK_Q = NQKV // NROW // 64           # 32 N blocks of 64 per chain row
CACHE = str(rf_paths.BUILD_ROOT / "scratch/w0/layer0_qkv.npz")


def load_qkv():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    out = {}
    for k, n, rows in (("q", "self_attn.q_proj.weight", HQ * HD), ("k", "self_attn.k_proj.weight", HKV * HD),
                       ("v", "self_attn.v_proj.weight", HKV * HD)):
        out[k + "_q"], out[k + "_s"] = _unplanar(n, rows, D)
    out["g_in"] = np.load(f"{WDIR}/{PFX}input_layernorm.weight.npy")
    for k in ("q_norm", "k_norm"):
        out[k] = np.load(f"{WDIR}/{PFX}self_attn.{k}.weight.npy")
    np.savez(CACHE, **out)
    return out


def perm_rows():
    """chain column order -> (tensor, row) of the q/k/v weight."""
    cols = []
    for m in range(NCOL):
        cols += [("q", (2 * m) * HD + i) for i in range(HD)] + [("q", (2 * m + 1) * HD + i) for i in range(HD)]
        cols += [("k", m * HD + i) for i in range(HD)] + [("v", m * HD + i) for i in range(HD)]
    return cols                                   # MemTile m owns cols[m*1024:(m+1)*1024]


def weights_perm(W):
    cols = perm_rows()
    q = np.stack([W[t + "_q"][r] for t, r in cols])
    s = np.stack([W[t + "_s"][r] for t, r in cols])
    return q, s                                   # [8192, 3840], [8192, 120]


def chain_col(r, n, t):
    """chain row r, N block n, column t in the block -> index into perm order."""
    m = 2 * r + n // 16
    return m * 1024 + (n % 16) * 64 + t


def qkv_stream(W, c):
    """Column c's QKV weight stream: [n 32][s 5][h 2][r 4] elements of 1728 B."""
    q, s = weights_perm(W)
    out = np.empty((NBLK_Q, KC // SUB_K, 2, NROW, ELEM), np.uint8)
    for r in range(NROW):
        for n in range(NBLK_Q):
            for h in range(2):
                idx = [chain_col(r, n, 32 * h + t) for t in range(32)]
                for sb in range(KC // SUB_K):
                    k0 = c * KC + sb * SUB_K
                    out[n, sb, h, r] = pack_elem(q[idx, k0:k0 + SUB_K], s[idx, k0 // 32:(k0 + SUB_K) // 32])
    return out.reshape(-1)


def model(x, W):
    """x bf16-valued [rows, 3840] -> q/k/v bf16 bits per MemTile [8][rows][1024]."""
    xn = rmsnorm(x, bf16(W["g_in"]))
    q, s = weights_perm(W)
    y = bf16_bits(gemm_chain(xn, q, s))           # [rows, 8192] in perm order
    return y.reshape(x.shape[0], NCOL, 1024).transpose(1, 0, 2)


def stream(W):
    return np.concatenate([np.concatenate([gain_elements(W["g_in"], np.float32(1.0), c), qkv_stream(W, c)])
                           for c in range(NCOL)])


# ---- P3.2: per-head RMSNorm (q, k with gains; v gainless) and rotate-half RoPE on q and k
from chain_ref import acc_add, bf16_val  # noqa: E402

THETA_LOCAL = 10000.0


def rope_table(pos):
    """positions -> bf16 bits [rows][256] = [cos 128 | sin 128] (the host's table; the device
    only consumes it)."""
    inv = 1.0 / (THETA_LOCAL ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    return bf16_bits(np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32))


def head_ss(xh):
    """rf_head.cc head_ss on [n, 256] bf16-valued rows."""
    sq = (xh * xh).astype(np.float32)
    a = sq[:, 0:32]
    for j in range(1, HD // 32):
        a = acc_add(a, sq[:, 32 * j:32 * j + 32])
    h = acc_add(a[:, :16], a[:, 16:])
    t = h[:, 0]
    for k in range(1, 16):
        t = acc_add(t, h[:, k])
    return t


def head_pass(qkv_bits, rope_bits, gq, gk):
    """qkv bits [8][rows][1024] -> the head-pass output bits, same layout."""
    x = bf16_val(qkv_bits).reshape(NCOL, -1, 4, HD).astype(np.float32)
    c = bf16_val(rope_bits[:, :HD // 2]).astype(np.float32)
    s = bf16_val(rope_bits[:, HD // 2:]).astype(np.float32)
    out = np.empty_like(x)
    eps = np.float32(1e-6)
    for h in range(4):
        xh = x[:, :, h]                                        # [8][rows][256]
        ss = head_ss(xh.reshape(-1, HD)).reshape(xh.shape[:2])
        m = acc_add((ss / np.float32(256)).astype(np.float32), np.full_like(ss, eps))
        r = bf16((np.float32(1) / np.sqrt(m)).astype(np.float32))[..., None]
        t = bf16(xh * r)
        if h == 3:
            out[:, :, h] = t
            continue
        n = bf16(t * bf16((gq if h < 2 else gk).astype(np.float32)))
        n1, n2 = n[..., :HD // 2], n[..., HD // 2:]
        out[:, :, h, :HD // 2] = bf16(acc_add((n1 * c).astype(np.float32), -(n2 * s).astype(np.float32)))
        out[:, :, h, HD // 2:] = bf16(acc_add((n2 * c).astype(np.float32), (n1 * s).astype(np.float32)))
    return bf16_bits(out.reshape(NCOL, -1, 1024))


def head_gain_elements(W):
    """8 elements, core r gets the r-th: q_norm then k_norm (bf16), the rest zero."""
    e = np.zeros((2 * NROW, ELEM), np.uint8)
    body = np.concatenate([bf16_bits(W["q_norm"]), bf16_bits(W["k_norm"])]).view(np.uint8)
    e[:NROW, :body.size] = body
    return e.reshape(-1)
