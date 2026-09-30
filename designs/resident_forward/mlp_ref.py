"""Layer-0 gemma4-12b MLP weights (shipped int4 g32 planar dump) and the bit-level reference of the
resident MLP: chain-order bfp16 GEMM with the device-fitted mac rule (chain_ref.mac_block), the
GELU*up epilogue of chain_mm_bfp16.cc, and the DDR/MemTile layouts the designs use."""
import os
import numpy as np
import rf_paths
from bfp16_model import f32_to_bfp16
from chain_ref import bf16, bf16_bits, bf16_val, mac_block, acc_add

WDIR = os.environ.get("RF_WDIR", str(rf_paths.ARTIFACTS / "weights_int4g32sbf16_planar_qat_rg"))
CACHE = os.environ.get("RF_W0_CACHE", str(rf_paths.BUILD_ROOT / "scratch/w0"))
PFX = "model.language_model.layers.0."
D, FF, KC, NCOL, NROW = 3840, 15360, 480, 8, 4
SUB_K, ELEM = 96, 1728          # weight element: 96 K rows x 32 columns, int4 g32 + bf16 scales


def _unplanar(name, M, K, row_group=2):
    """planar int4 g32 bf16-scale rows -> (q int8 [M, K], scale f32 [M, K/32])."""
    pk = np.load(f"{WDIR}/{PFX}{name}.npy").view(np.uint8)
    ng, pay = K // 32, K // 2
    stride = ng * 2 + pay
    blocks = pk.reshape(M // row_group, row_group * stride)
    rows = np.empty((M // row_group, row_group, stride), np.uint8)
    rows[:, :, 2 * ng:] = blocks[:, :row_group * pay].reshape(-1, row_group, pay)
    rows[:, :, :2 * ng] = blocks[:, row_group * pay:].reshape(-1, row_group, 2 * ng)
    rows = rows.reshape(M, stride)
    sc = bf16_val(rows[:, :2 * ng].copy().view(np.uint16)).reshape(M, ng)
    p = rows[:, 2 * ng:]
    lo = (p & 15).astype(np.int8); hi = (p >> 4).astype(np.int8)
    q = np.empty((M, K), np.int8)
    q[:, 0::2] = np.where(lo >= 8, lo - 16, lo)
    q[:, 1::2] = np.where(hi >= 8, hi - 16, hi)
    return q, sc


def load_layer0():
    """dict of q/scale for gate, up ([FF, D]), down ([D, FF]) and the two FFN norm gains."""
    f = f"{CACHE}/layer0.npz"
    if os.path.exists(f):
        return dict(np.load(f))
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for k, name in (("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight")):
        out[k + "_q"], out[k + "_s"] = _unplanar(name, FF, D)
    dq, ds = zip(*[_unplanar(f"mlp.down_proj.weight.kchunk{i}", D, D) for i in range(4)])
    out["down_q"], out["down_s"] = np.concatenate(dq, 1), np.concatenate(ds, 1)
    out["g_pre"] = np.load(f"{WDIR}/{PFX}pre_feedforward_layernorm.weight.npy")
    out["g_post"] = np.load(f"{WDIR}/{PFX}post_feedforward_layernorm.weight.npy")
    np.savez(f, **out)
    return out


# ---- weight element (nb*(sub_k//8)*8/2 + n_groups*nb*2 bytes; 1728 B at nb=32, sub_k=96,
# group=32): scales [n_groups][32] bf16, nibbles [4 n-subtiles][sub_k/8][32 B], nibble i of a
# 32-byte subtile = (n = i/8, k = i%8), low nibble first. sub_k is read off q's own shape, not a
# module constant, so a caller can pack elements of any size the ring fits (chain_partition.py).
def pack_elem(q, sc):
    """q int8 [32 cols, sub_k], sc f32 [32, n_groups] (weight rows = output columns) -> bytes."""
    sub_k = q.shape[1]
    s = np.ascontiguousarray(bf16_bits(sc.T)).view(np.uint8).reshape(-1)   # [g][n]
    t = q.reshape(4, 8, sub_k // 8, 8).transpose(0, 2, 1, 3)             # [ns][kb][n][k]
    lanes = (t.reshape(4, sub_k // 8, 64) & 15).astype(np.uint8)
    nib = lanes[..., 0::2] | (lanes[..., 1::2] << 4)
    return np.concatenate([s, nib.reshape(-1)])


def unpack_elem(b, sub_k, group_size):
    """pack_elem's exact inverse: bytes -> (q int8 [32, sub_k], sc f32 [32, n_groups]). Used only
    by the partition round-trip check (rf-chain-partition-from-model-dims); the device kernel's
    own decode is chain_ref.decode_sub, which still hardcodes group 32."""
    b = np.asarray(b, np.uint8)
    n_groups = sub_k // group_size
    s_bytes = n_groups * 32 * 2
    sc = bf16_val(b[:s_bytes].view(np.uint16)).reshape(n_groups, 32).T   # [32, n_groups]
    nib = b[s_bytes:].reshape(4, sub_k // 8, 32)
    lo = (nib & 15).astype(np.int8)
    hi = ((nib >> 4) & 15).astype(np.int8)
    lanes = np.empty((4, sub_k // 8, 64), np.int8)
    lanes[..., 0::2], lanes[..., 1::2] = lo, hi
    lanes = np.where(lanes > 7, lanes - 16, lanes)
    t = lanes.reshape(4, sub_k // 8, 8, 8).transpose(0, 2, 1, 3)         # [ns][n][kb][k]
    q = t.reshape(32, sub_k)
    return q, sc


def gateup_cols(r, n, h, ffn=FF, nrow=NROW):
    """(gate rows, up rows) of the 32 columns of half h of N block n of chain row r, in the
    element's column order: [gate p, up p, gate p+1, up p+1] x 8 with p = 2h."""
    base = r * (ffn // nrow) + n * 32 + 16 * h
    g0, g1 = np.arange(base, base + 8), np.arange(base + 8, base + 16)
    return g0, g1


def gateup_stream(W, c, d_model=D, ffn=FF, ncol=NCOL, nrow=NROW, sub_k=SUB_K, elem=ELEM,
                  group_size=32, kc=KC):
    """Column c's gate/up weight stream: [n][s][h 2][r] elements of `elem` bytes each. `n` ranges
    over ffn // nrow // 32 N blocks. Defaults reproduce the 12B stream ([n 120][s 5][h 2][r 4] of
    1728 B) unchanged."""
    nblk = ffn // nrow // 32
    out = np.empty((nblk, kc // sub_k, 2, nrow, elem), np.uint8)
    gq, gs, uq, us = W["gate_q"], W["gate_s"], W["up_q"], W["up_s"]
    for r in range(nrow):
        for n in range(nblk):
            for h in range(2):
                g0, g1 = gateup_cols(r, n, h, ffn, nrow)
                for s in range(kc // sub_k):
                    k0 = c * kc + s * sub_k
                    ks = slice(k0, k0 + sub_k)
                    gsl = slice(k0 // group_size, (k0 + sub_k) // group_size)
                    q = np.concatenate([gq[g0, ks], uq[g0, ks], gq[g1, ks], uq[g1, ks]])
                    sc = np.concatenate([gs[g0, gsl], us[g0, gsl], gs[g1, gsl], us[g1, gsl]])
                    out[n, s, h, r] = pack_elem(q, sc)
    return out.reshape(-1)


# ---- bfp16 GEMM in chain order (K blocks of 8 in order, all columns), device mac rule
def bfp16_blocks(v):
    """f32 [..., K] -> (mant int64 [..., K/8, 8], exp int64 [..., K/8]) via the G1.2 model."""
    sh = v.shape
    b = f32_to_bfp16(np.ascontiguousarray(v, np.float32).reshape(-1)).reshape(-1, 9)
    m = b[:, 1:].view(np.int8).astype(np.int64).reshape(*sh[:-1], sh[-1] // 8, 8)
    e = b[:, 0].astype(np.int64).reshape(*sh[:-1], sh[-1] // 8)
    return m, e


def gemm_chain(xa, wq, ws, group_size=32):
    """xa f32 [rows, K] (already the bfp16-exact A values' source: bf16), wq int8 [N, K],
    ws [N, K/group_size] -> f32 [rows, N] exactly as the chain computes it. `group_size` is the
    weight's own quantisation grid (32 for gemma4-12b/qwen3-0.6b, 64 for gemma3-270m); the K-shape
    of `xa`/`wq` is read from the arrays, not from a module constant."""
    am, ae = bfp16_blocks(xa)
    wf = wq.astype(np.float32) * np.repeat(ws, group_size, axis=1)
    wm, we = bfp16_blocks(wf)
    rows, N, KBn = xa.shape[0], wq.shape[0], xa.shape[1] // 8
    acc = np.zeros((rows, N), np.float32)
    wmf = wm.astype(np.float64)
    for kb in range(KBn):
        S = (am[:, kb, :].astype(np.float64) @ wmf[:, kb, :].T).astype(np.int64)
        s = ae[:, kb][:, None] + we[:, kb][None, :] - 266
        acc = mac_block(acc, S, s)
    return acc


GELU_C = np.array([9.773047566e-01, 5.426859856e-02, -5.423979461e-02, 2.708265744e-02,
                   -4.168297164e-03, -2.585965674e-03, 1.167817041e-03, 5.876845535e-05,
                   -7.190783072e-05], np.float32)


def gelu_bf16(x):
    """chain_mm_bfp16.cc gelu32 on bf16 values, step by step (f32 accumulate, bf16 RNE)."""
    x = bf16(x)
    a = np.minimum(np.abs(x), np.float32(4.0))
    y = bf16(a - np.float32(2.0))
    p = bf16(np.full_like(x, GELU_C[8]))
    for i in range(7, -1, -1):
        p = bf16(GELU_C[i] + y * p)
    pn = bf16(np.float32(1.0) - p)
    s = np.where(x < 0, pn, p)
    return bf16(x * s)


def h_from_gateup(gate, up):
    """f32 accumulators -> h f32 values (pre-bfp16): bf16 park, GELU(g) * u."""
    return gelu_bf16(bf16(gate)) * bf16(up)


# ---- layouts
def a_layout(x, t_blocks):
    """bf16-exact f32 x [16*t, K] -> 16-row interleaved bfp16 per column slice:
    [c 8][t][kb 60][r 2][72] bytes."""
    K = x.shape[1]
    kc = K // NCOL
    t = x.reshape(t_blocks, 2, 8, NCOL, kc // 8, 8)                    # [t][r][row][c][kb][k]
    t = t.transpose(3, 0, 4, 1, 2, 5)                                    # [c][t][kb][r][row][k]
    return f32_to_bfp16(np.ascontiguousarray(t).reshape(-1))


def h_memtile_layout(hv, t_blocks):
    """h f32 values [16*t, FF] -> per MemTile m: [m 8][t][kb 240][r 2][72]."""
    t = hv.reshape(t_blocks, 2, 8, NCOL, 240, 8).transpose(3, 0, 4, 1, 2, 5)
    return f32_to_bfp16(np.ascontiguousarray(t).reshape(-1))


# ---- down projection (Nb = 32, K in 4 sub-passes of KC per column, chain end accumulates in f32)
NSP = FF // NCOL // KC          # K sub-passes per column: 4
NDB = D // NROW // 32           # N blocks of 32 per chain row: 30


def down_dims(d_model=D, ffn=FF, ncol=NCOL, nrow=NROW, kc=KC):
    """(nsp, ndb): K sub-passes per column and N blocks of 32 per chain row, generalised off the
    matrix's own d_model/ffn/ncol/nrow rather than the 12B module globals NSP/NDB. `kc` is the
    per-sub-pass K width (a MemTile transfer-size constant, not derived from d_model here -- see
    chain_partition.py for the weight-element-level partition rule)."""
    return ffn // ncol // kc, d_model // nrow // 32


def down_stream(W, c, d_model=D, ffn=FF, ncol=NCOL, nrow=NROW, sub_k=SUB_K, elem=ELEM,
                group_size=32, kc=KC):
    """Column c's down weight stream: [n][s][sub][r] elements of `elem` bytes each. Defaults
    reproduce the 12B stream ([n 30][s 4][sub 5][r 4] of 1728 B) unchanged."""
    nsp, ndb = down_dims(d_model, ffn, ncol, nrow, kc)
    out = np.empty((ndb, nsp, kc // sub_k, nrow, elem), np.uint8)
    dq, ds = W["down_q"], W["down_s"]
    for r in range(nrow):
        for n in range(ndb):
            cols = np.arange(r * (d_model // nrow) + n * 32, r * (d_model // nrow) + n * 32 + 32)
            for s in range(nsp):
                for sub in range(kc // sub_k):
                    k0 = c * (ffn // ncol) + s * kc + sub * sub_k
                    out[n, s, sub, r] = pack_elem(dq[cols, k0:k0 + sub_k],
                                                  ds[cols, k0 // group_size:(k0 + sub_k) // group_size])
    return out.reshape(-1)


def gemm_down_chain(hv, wq, ws, group_size=32, d_model=D, ffn=FF, ncol=NCOL, kc=KC):
    """hv f32 [rows, ffn] (h values, bfp16-exact after bfp16_blocks), down q/s -> f32 [rows,
    d_model]: per sub-pass s a zero-seeded chain over columns 0..ncol-1 (each its K range
    c*(ffn/ncol) + s*kc), then the chain end's accfloat add across sub-passes (chain_ref.acc_add).
    Defaults reproduce the 12B chain (d_model=3840, ffn=15360, ncol=8, kc=480) unchanged."""
    nsp = ffn // ncol // kc
    am, ae = bfp16_blocks(hv)
    wf = wq.astype(np.float32) * np.repeat(ws, group_size, axis=1)
    wm, we = bfp16_blocks(wf)
    wmf = wm.astype(np.float64)
    rows = hv.shape[0]
    y = None
    for s in range(nsp):
        acc = np.zeros((rows, d_model), np.float32)
        for c in range(ncol):
            for kk in range(kc // 8):
                kb = (c * (ffn // ncol) + s * kc) // 8 + kk
                S = (am[:, kb, :].astype(np.float64) @ wmf[:, kb, :].T).astype(np.int64)
                acc = mac_block(acc, S, ae[:, kb][:, None] + we[:, kb][None, :] - 266)
        y = acc if y is None else acc_add(y, acc)
    return y


def y_memtile_layout(y, t_blocks):
    """y_d f32 [16*t, D] -> bf16 bytes per MemTile m: [m 8][t][n 15][cs 4][r 2][row 8][col 8]."""
    t = bf16_bits(y).reshape(t_blocks, 2, 8, NCOL, 15, 4, 8)             # [t][r][row][m][n][cs][col]
    return np.ascontiguousarray(t.transpose(3, 0, 4, 5, 1, 2, 6)).view(np.uint8).reshape(-1)
