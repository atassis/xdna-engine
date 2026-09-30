"""P3.4 model: O projection + post-attention RMSNorm + residual on the chain rail.

O's K (16 heads x 256) splits as MemTile c = heads 2c, 2c+1 (512 columns of the context), but a chain
position's K slice must be whole 96-row weight elements, so each column carries 576 = 2 sub-passes of
288, the last 64 rows zero (12.5% of O's weight bytes). The chain end accumulates the two sub-passes
in f32 as the down projection does; everything after the GEMM is rmlp_ref's post pass with the layer
scalar 1."""
import os
import numpy as np
from bfp16_model import f32_to_bfp16
from chain_ref import bf16, bf16_bits, mac_block, acc_add
from mlp_ref import _unplanar, pack_elem, bfp16_blocks, WDIR, PFX, D, SUB_K, ELEM, NCOL, NROW
from rmlp_ref import rmsnorm, gain_elements
import rf_paths

HD, HQ = 256, 16
KO = HQ * HD                      # 4096
KCOL = KO // NCOL                 # 512 real context columns per MemTile
KCO, NSPO = 288, 2                # chain K slice and sub-passes per column
KPAD = KCO * NSPO                 # 576
NDB = D // NROW // 32             # 30 N blocks of 32 per chain row
ABO = 16 * KCO * 9 // 8           # one 16-row A block of a sub-pass: 5184
CACHE = str(rf_paths.BUILD_ROOT / "scratch/w0/layer0_o.npz")


def load_o():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    import sys
    sys.path.insert(0, rf_paths.iron_dir())
    from iron.common.quant import derive_row_group, widest_chunk
    rg = derive_row_group([KO], 32, "int4", vec_size=widest_chunk(32, "int4"), scale_dtype="bf16")
    q, s = _unplanar("self_attn.o_proj.weight", D, KO, rg)
    out = dict(o_q=q, o_s=s, g_post_attn=np.load(f"{WDIR}/{PFX}post_attention_layernorm.weight.npy"))
    np.savez(CACHE, **out)
    return out


KORDER = "heads"           # "heads": column c = heads 2c, 2c+1 in order, padding at the end;
                           # "workers": sub-pass h = [head 2c dims h*128.., head 2c+1 dims h*128.., 32 zero]
                           # (the fused layer's x V worker h writes sub-pass h)


def k_index():
    """New K position (column c, 576 each) -> original O input column, or -1 for padding."""
    idx = np.full((NCOL, KPAD), -1, np.int64)
    for c in range(NCOL):
        for j in range(KPAD):
            if KORDER == "heads":
                if j < KCOL:
                    idx[c, j] = c * KCOL + j
            else:
                h, jj = divmod(j, KCO)
                if jj < 2 * 128:
                    hh, d = divmod(jj, 128)
                    idx[c, j] = (2 * c + hh) * HD + h * 128 + d
    return idx.reshape(-1)


def pad_k(a, fill=0):
    """[..., 4096] -> [..., 8 * 576]: each column's 512 then 64 of `fill`."""
    idx = k_index()
    out = np.full(a.shape[:-1] + (NCOL * KPAD,), fill, a.dtype)
    out[..., idx >= 0] = a[..., idx[idx >= 0]]
    return out


def weights_pad(W):
    q = pad_k(W["o_q"])                                           # [3840, 4608]
    gidx = k_index()[::32]                                        # a group is 32 aligned columns
    s = np.zeros((D, NCOL * KPAD // 32), W["o_s"].dtype)
    s[:, gidx >= 0] = W["o_s"][:, gidx[gidx >= 0] // 32]
    return q, s


def o_stream(W, c):
    """Column c's O weight stream: [n 30][s 2][sub 3][r 4] elements of 1728 B."""
    q, s = weights_pad(W)
    out = np.empty((NDB, NSPO, KCO // SUB_K, NROW, ELEM), np.uint8)
    for r in range(NROW):
        for n in range(NDB):
            cols = np.arange(r * (D // NROW) + n * 32, r * (D // NROW) + n * 32 + 32)
            for sp in range(NSPO):
                for sub in range(KCO // SUB_K):
                    k0 = c * KPAD + sp * KCO + sub * SUB_K
                    out[n, sp, sub, r] = pack_elem(q[cols, k0:k0 + SUB_K], s[cols, k0 // 32:(k0 + SUB_K) // 32])
    return out.reshape(-1)


def stream(W):
    return np.concatenate([np.concatenate([o_stream(W, c), gain_elements(W["g_post_attn"], np.float32(1.0), c)])
                           for c in range(NCOL)])


def gemm_o_chain(ctx, W):
    """ctx bf16-valued [rows, 4096] -> f32 [rows, 3840] in the chain's order."""
    q, s = weights_pad(W)
    am, ae = bfp16_blocks(pad_k(ctx))
    wm, we = bfp16_blocks(q.astype(np.float32) * np.repeat(s, 32, axis=1))
    wmf = wm.astype(np.float64)
    y = None
    for sp in range(NSPO):
        acc = np.zeros((ctx.shape[0], D), np.float32)
        for c in range(NCOL):
            for kk in range(KCO // 8):
                kb = (c * KPAD + sp * KCO) // 8 + kk
                S = (am[:, kb, :].astype(np.float64) @ wmf[:, kb, :].T).astype(np.int64)
                acc = mac_block(acc, S, ae[:, kb][:, None] + we[:, kb][None, :] - 266)
        y = acc if y is None else acc_add(y, acc)
    return y


def block(x, ctx, W):
    y = bf16(gemm_o_chain(ctx, W))
    yn = rmsnorm(y, bf16(W["g_post_attn"]))
    return bf16(acc_add(x.astype(np.float32), yn)), dict(y=y, yn=yn)


def ctx_memtile(ctx, t_blocks):
    """ctx bf16-valued [16 t, 4096] -> per MemTile c: [c 8][t][s 2][kb 36][r 2][72] bytes."""
    a = pad_k(ctx).reshape(t_blocks, 2, 8, NCOL, NSPO, KCO // 8, 8)     # [t][r][row][c][s][kb][k]
    a = a.transpose(3, 0, 4, 5, 1, 2, 6)                                 # [c][t][s][kb][r][row][k]
    return f32_to_bfp16(np.ascontiguousarray(a, np.float32).reshape(-1))
