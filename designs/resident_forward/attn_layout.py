"""Attention chain layouts from the layer's geometry, for the weight store: the QKV stream
(attn_in) and the O stream (attn_out) of any attention layer, sliding or global.

QKV: MemTile m holds q heads [m*hq/8, (m+1)*hq/8), then its share of each kv tensor: whole heads
when num_kv_heads >= 8 (sliding: k head m, v head m), else head dims split 8 ways (global: K dims
m*hd/8 ..; V absent when k_eq_v, the head pass derives it from K). The norm across a split head
rides the row-0 cascade, as the layer norm does. Chain row r feeds MemTiles 2r and 2r + 1.

O: column c's K slice is its q heads' context, cut into sub-passes of KC_O = 288 in the x V
workers' order: sub-pass h = [head 0 of the column, dims h*128 ..; head 1, dims h*128 ..; 32 zero].
Sliding (hd 256) is 2 sub-passes, global (hd 512) 4.

`gate()` checks the sliding builders byte-for-byte against qkv_ref / o_ref (whose streams were
device-gated) and that the global layouts are bijections onto the weights' rows and columns."""
import numpy as np
from mlp_ref import pack_elem, D, KC, SUB_K, ELEM, NCOL, NROW
from rmlp_ref import gain_elements

KC_O = 288            # O's K slice per chain position (o_ref.KCO)
HW = 128              # head dims per x V worker sub-pass


class Geo:
    def __init__(self, hq, hkv, hd, k_eq_v):
        self.hq, self.hkv, self.hd, self.k_eq_v = hq, hkv, hd, k_eq_v
        assert hq % NCOL == 0, "q heads split evenly over the MemTiles"
        assert hkv >= NCOL and hkv % NCOL == 0 or (hkv * hd) % NCOL == 0

    def kv_tensors(self):
        return ("k",) if self.k_eq_v else ("k", "v")

    def memtile_cols(self, m):
        """MemTile m's columns, in order: (tensor, weight row)."""
        qh = self.hq // NCOL
        cols = [("q", h * self.hd + i) for h in range(m * qh, (m + 1) * qh) for i in range(self.hd)]
        for t in self.kv_tensors():
            if self.hkv >= NCOL:
                kh = self.hkv // NCOL
                cols += [(t, h * self.hd + i) for h in range(m * kh, (m + 1) * kh) for i in range(self.hd)]
            else:
                w = self.hkv * self.hd // NCOL
                if getattr(self, "kpair", False):     # half from each RoPE half: a pair stays in a column
                    h = w // 2
                    cols += [(t, m * h + i) for i in range(h)] + [(t, self.hd // 2 + m * h + i) for i in range(h)]
                else:
                    cols += [(t, m * w + i) for i in range(w)]
        return cols

    def mt_width(self):
        w = len(self.memtile_cols(0))
        assert w % 64 == 0, "whole 64-column N blocks per MemTile"
        return w

    def nblk(self):
        """N blocks of 64 per chain row (two MemTiles)."""
        return 2 * self.mt_width() // 64

    def perm(self):
        return [c for m in range(NCOL) for c in self.memtile_cols(m)]

    # ---- O
    def o_nsp(self):
        return self.hq // NCOL * self.hd // (2 * HW) if self.hq // NCOL == 2 else None

    def o_kpad(self):
        return self.o_nsp() * KC_O

    def o_k_index(self):
        """K position (column c, o_kpad each) -> O input column, -1 for padding."""
        kp, nsp = self.o_kpad(), self.o_nsp()
        idx = np.full((NCOL, kp), -1, np.int64)
        for c in range(NCOL):
            for j in range(kp):
                h, jj = divmod(j, KC_O)
                if jj < 2 * HW:
                    hh, d = divmod(jj, HW)
                    idx[c, j] = (2 * c + hh) * self.hd + h * HW + d
        return idx.reshape(-1)


SLIDING = Geo(16, 8, 256, False)
GLOBAL = Geo(16, 1, 512, True)


def weights_perm(geo, W):
    cols = geo.perm()
    q = np.stack([W[t + "_q"][r] for t, r in cols])
    s = np.stack([W[t + "_s"][r] for t, r in cols])
    return q, s


def qkv_stream(geo, W, c):
    """Column c's QKV weight elements: [n nblk][s KC/SUB_K][h 2][r 4] of 1728 B."""
    q, s = weights_perm(geo, W)
    nb, half, mw = geo.nblk(), geo.nblk() // 2, geo.mt_width()
    out = np.empty((nb, KC // SUB_K, 2, NROW, ELEM), np.uint8)
    for r in range(NROW):
        for n in range(nb):
            m = 2 * r + n // half
            for h in range(2):
                idx = [m * mw + (n % half) * 64 + 32 * h + t for t in range(32)]
                for sb in range(KC // SUB_K):
                    k0 = c * KC + sb * SUB_K
                    out[n, sb, h, r] = pack_elem(q[idx, k0:k0 + SUB_K], s[idx, k0 // 32:(k0 + SUB_K) // 32])
    return out.reshape(-1)


def head_gain_elements(geo, W):
    """8 elements, core r gets the r-th and (r+4)-th: q_norm then k_norm (bf16), the rest zero."""
    from chain_ref import bf16_bits
    e = np.zeros((2 * NROW, ELEM), np.uint8)
    q, k = bf16_bits(W["q_norm"]).view(np.uint8), bf16_bits(W["k_norm"]).view(np.uint8)
    if q.size + k.size <= ELEM:
        e[:NROW, :q.size + k.size] = np.concatenate([q, k])
    else:                          # global (head 512): q_norm in core r's first element, k_norm in its second
        e[:NROW, :q.size] = q
        e[NROW:, :k.size] = k
    return e.reshape(-1)


def attn_in_stream(geo, W):
    """Per column: pre-attention gains, QKV, head gains (the attn_in blob)."""
    hg = head_gain_elements(geo, W)
    return np.concatenate([np.concatenate([gain_elements(W["g_in"], np.float32(1.0), c), qkv_stream(geo, W, c), hg])
                           for c in range(NCOL)])


def o_weights_pad(geo, W):
    idx = geo.o_k_index()
    q = np.zeros((D, idx.size), W["o_q"].dtype)
    q[:, idx >= 0] = W["o_q"][:, idx[idx >= 0]]
    gidx = idx[::32]
    s = np.zeros((D, idx.size // 32), W["o_s"].dtype)
    s[:, gidx >= 0] = W["o_s"][:, gidx[gidx >= 0] // 32]
    return q, s


def o_stream(geo, W, c, q=None, s=None):
    """Column c's O weight elements: [n 30][s nsp][sub 3][r 4] of 1728 B."""
    if q is None:
        q, s = o_weights_pad(geo, W)
    nsp, kp, ndb = geo.o_nsp(), geo.o_kpad(), D // NROW // 32
    out = np.empty((ndb, nsp, KC_O // SUB_K, NROW, ELEM), np.uint8)
    for r in range(NROW):
        for n in range(ndb):
            cols = np.arange(r * (D // NROW) + n * 32, r * (D // NROW) + n * 32 + 32)
            for sp in range(nsp):
                for sub in range(KC_O // SUB_K):
                    k0 = c * kp + sp * KC_O + sub * SUB_K
                    out[n, sp, sub, r] = pack_elem(q[cols, k0:k0 + SUB_K], s[cols, k0 // 32:(k0 + SUB_K) // 32])
    return out.reshape(-1)


def attn_out_stream(geo, W):
    """Per column: O, then the post-attention gains (the attn_out blob)."""
    q, s = o_weights_pad(geo, W)
    return np.concatenate([np.concatenate([o_stream(geo, W, c, q, s), gain_elements(W["g_post_attn"], np.float32(1.0), c)])
                           for c in range(NCOL)])


def load_global(li):
    """A full-attention layer's weights, from the planar int4 source (weight_store's reader)."""
    import weight_store as ws
    W = {}
    W["q_q"], W["q_s"] = ws.unplanar_matrix(li, "self_attn.q_proj.weight", GLOBAL.hq * GLOBAL.hd, D)
    W["k_q"], W["k_s"] = ws.unplanar_matrix(li, "self_attn.k_proj.weight", GLOBAL.hkv * GLOBAL.hd, D)
    W["o_q"], W["o_s"] = ws.unplanar_matrix(li, "self_attn.o_proj.weight", D, GLOBAL.hq * GLOBAL.hd)
    for k, n in (("g_in", "input_layernorm"), ("q_norm", "self_attn.q_norm"), ("k_norm", "self_attn.k_norm"),
                 ("g_post_attn", "post_attention_layernorm")):
        W[k] = ws.raw_load(li, n + ".weight")
    return W


def gate():
    import qkv_ref
    import o_ref
    assert SLIDING.perm() == qkv_ref.perm_rows(), "sliding QKV layout"
    Wq = qkv_ref.load_qkv()
    for c in (0, 7):
        assert np.array_equal(qkv_stream(SLIDING, Wq, c), qkv_ref.qkv_stream(Wq, c)), c
    Wo = o_ref.load_o()
    o_ref.KORDER = "workers"
    assert np.array_equal(SLIDING.o_k_index(), o_ref.k_index()), "sliding O K order"
    for c in (0, 7):
        assert np.array_equal(o_stream(SLIDING, Wo, c), o_ref.o_stream(Wo, c)), c
    # global: the layouts cover every weight row / O input column exactly once
    p = GLOBAL.perm()
    assert sorted(r for t, r in p if t == "q") == list(range(16 * 512))
    assert sorted(r for t, r in p if t == "k") == list(range(512))
    ki = GLOBAL.o_k_index()
    assert sorted(ki[ki >= 0].tolist()) == list(range(16 * 512))
    print(f"sliding: QKV and O streams byte-identical to qkv_ref / o_ref (workers); "
          f"global: {GLOBAL.nblk()} N blocks per chain row ({GLOBAL.mt_width()} columns per MemTile), "
          f"O {GLOBAL.o_nsp()} sub-passes of {KC_O} per column ({GLOBAL.o_kpad() - 1024} zero rows)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:          # `attn_layout.py <full-attention layer>`: build its two streams
        W = load_global(int(sys.argv[1]))
        a_in, a_out = attn_in_stream(GLOBAL, W), attn_out_stream(GLOBAL, W)
        print(f"layer {sys.argv[1]}: attn_in {a_in.size} B ({a_in.size // NCOL // ELEM} elements per column), "
              f"attn_out {a_out.size} B")
    else:
        gate()
