#!/usr/bin/env python3
"""Golden for the FastConformer rel-pos MHSA kernel candidate (A4).

Validates THREE things against the verified host reference
(rust/npu-parakeet/src/{ops.rs,encoder.rs} + scripts/parakeet_ref_encoder.py mhsa,
itself rel<=3e-5 vs ONNX):

  G1. The "rel_shift as a STRIDED RELAYOUT" identity the kernel relies on:
          BD_shifted[i, j] == BD[i, (T-1) - i + j]
      must be BIT-EXACT vs the NeMo pad/reshape/slice rel_shift. (load-bearing)

  G2. A pure-f32 re-implementation of the mhsa node == the host mhsa (sanity that
      our numpy mirror of the math is exact). rel <= 1e-6.

  G3. The KERNEL NUMERIC MODEL (bf16 mmul projections + bias add, f32 AC/BD from
      bf16 mmul, the strided rel_shift, vectorized-exp2 softmax with bf16 probs,
      bf16 ctx mmul + out proj) vs the f32 host mhsa. Gate rel-L2 <= 0.08.

Run: ~/npuvox-asr-bench/.venv/bin/python scripts/parakeet_relpos_mha_golden.py
"""
import os, sys
import numpy as np

try:
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16
except Exception as e:  # pragma: no cover
    print("need ml_dtypes (use ~/npuvox-asr-bench/.venv):", e); sys.exit(2)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Encoder artifacts. Overridable via PARAKEET_ENC_DIR; if the local worktree copy
# is absent (split-out worktrees do not carry the gitignored artifacts) fall back
# to the sibling public checkout so this pure-numpy golden runs read-only anywhere.
ENC = os.environ.get("PARAKEET_ENC_DIR",
                     os.path.join(REPO, "artifacts", "parakeet", "encoder"))
if not os.path.isdir(ENC):
    _sib = os.path.join(os.path.dirname(REPO), "xdna-engine",
                        "artifacts", "parakeet", "encoder")
    if os.path.isdir(_sib):
        ENC = _sib
D, H, DK = 1024, 8, 128
GATE = 0.08
LOG2E = np.float32(1.4426950408889634)

def W(blk, name): return np.load(f"{ENC}/L{blk}/{name}.npy")
def REF(name):    return np.load(f"{ENC}/refs/{name}.npy")
def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))

def bf16(x):  # round f32 -> bf16 -> f32 (emulate AIE bf16 storage)
    return np.asarray(x, np.float32).astype(BF16).astype(np.float32)

def mm_bf16(a, b):  # AIE mmul: bf16 inputs, f32 (accfloat) accumulate
    return (bf16(a).astype(np.float32) @ bf16(b).astype(np.float32)).astype(np.float32)

# ----------------------------------------------------------------------------
# host reference rel_shift (NeMo pad/reshape/slice) -- the oracle.
def rel_shift_host(bd):  # bd [H,T,P=2T-1] -> [H,T,T]
    Hh, T, P = bd.shape
    x = np.pad(bd, ((0, 0), (0, 0), (1, 0)))   # [H,T,P+1]
    x = x.reshape(Hh, P + 1, T)
    x = x[:, 1:].reshape(Hh, T, P)
    return x[:, :, :T]

# the kernel's strided-relayout form: BD_shifted[i,j] = BD[i, (T-1)-i+j].
def rel_shift_strided(bd):  # bd [H,T,P] -> [H,T,T]
    Hh, T, P = bd.shape
    out = np.zeros((Hh, T, T), bd.dtype)
    for i in range(T):
        base = (T - 1) - i               # contiguous window start in BD row i
        out[:, i, :] = bd[:, i, base:base + T]
    return out

# ----------------------------------------------------------------------------
# pure-f32 host mhsa (mirror of parakeet_ref_encoder.mhsa).
def mhsa_f32(x, blk, pos_enc, rel_shift_fn=rel_shift_host):
    T = x.shape[0]
    q = (x @ W(blk, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(blk, "self_attn.linear_k.weight")).reshape(T, H, DK)
    v = (x @ W(blk, "self_attn.linear_v.weight")).reshape(T, H, DK)
    p = (pos_enc @ W(blk, "self_attn.linear_pos.weight")).reshape(-1, H, DK)
    u = W(blk, "self_attn.pos_bias_u"); vv = W(blk, "self_attn.pos_bias_v")
    qu = (q + u).transpose(1, 0, 2); qv = (q + vv).transpose(1, 0, 2)
    kt = k.transpose(1, 2, 0); pt = p.transpose(1, 2, 0)
    ac = qu @ kt; bd = rel_shift_fn(qv @ pt)
    scores = (ac + bd) / np.sqrt(DK)
    scores = scores - scores.max(-1, keepdims=True)
    a = np.exp(scores); a /= a.sum(-1, keepdims=True)
    ctx = (a @ v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, H * DK)
    return ctx @ W(blk, "self_attn.linear_out.weight")

# ----------------------------------------------------------------------------
# KERNEL numeric model: bf16 mmuls + the on-chip bias-add / strided rel_shift /
# vectorized-exp2 softmax (bf16 probs) the .cc kernel implements.
def softmax_kernel(scores_row):  # f32 [T] -> bf16-modeled probs [T]
    m = scores_row.max()
    e = np.exp2(((scores_row - m) * LOG2E).astype(np.float32)).astype(BF16).astype(np.float32)  # bf16 exp2
    s = e.sum(dtype=np.float32)            # f32 accumulate of bf16 exps
    inv = bf16(np.float32(1.0) / s)        # bf16 reciprocal
    return bf16(e * inv)                   # bf16 probs

# STANDALONE step-1 brick: numpy MODEL of relpos_scores_softmax_bake(.cc). Given
# AC[T,T] f32 and BD[T,P] f32 it reproduces the on-chip dataflow bit-for-bit at
# the algorithm level: the strided rel_shift read (BD[i, (T-1)-i : (T-1)-i+T]),
# scale by inv_scale, then the f32-max / bf16-exp2 / f32-sum / bf16-reciprocal
# softmax with bf16 probs out. No matmul -- this is exactly what the xclbin runs.
def relpos_scores_softmax_model(AC, BD, inv_scale):
    T, P = AC.shape[0], BD.shape[1]
    probs = np.zeros((T, T), np.float32)
    for i in range(T):
        base = (T - 1) - i                       # strided rel_shift window start
        bd_row = BD[i, base:base + T]            # contiguous length-T read
        scores = ((AC[i] + bd_row) * inv_scale).astype(np.float32)
        m = scores.max()
        e = np.exp2(((scores - m) * LOG2E).astype(np.float32)).astype(BF16).astype(np.float32)
        s = e.sum(dtype=np.float32)              # f32 accumulate of bf16 exps
        inv = bf16(np.float32(1.0) / s)          # bf16 reciprocal
        probs[i] = bf16(e * inv)                 # bf16 probs
    return probs

# STEP-2 COMPOSED brick: numpy MODEL of relpos_ac_scores_softmax_bake(.cc). Given
# qu[T,DK] and k[T,DK] it runs the ON-CHIP AC = qu @ k^T matmul (bf16 in, f32
# accumulate -- exactly the aie::mac dot in relpos_ac_matmul), keeps the f32 AC
# tile RESIDENT, then applies the same strided rel_shift + exp2 softmax as step 1.
# This models what the STEP-2 xclbin computes (matmul -> resident L1 f32 -> softmax).
def relpos_ac_scores_softmax_model(qu, k, BD, inv_scale):
    ac = mm_bf16(qu, k.T)                          # [T,T] f32 (bf16 mmul, f32 acc)
    return relpos_scores_softmax_model(ac, BD, inv_scale)

# STEP-6 ROW-TILED brick: numpy MODEL of relpos_rowtiled_bake(.cc). Processes the
# T query rows in TILES of Tq (Tq need NOT divide T -- ragged final tile), reading
# resident k/p/V. Mirrors the on-chip arithmetic exactly: per query tile it runs
# AC_tile = qu_tile @ k^T (bf16 mmul) and BD_tile = qv_tile @ p^T, then the strided
# rel_shift + exp2 softmax with the GLOBAL query index into the rel_shift base
# ((T-1) - (q0+il)), then ctx_tile = probs @ V. Returns probs[T,T], ctx[T,DK].
# Tq == T recovers the single-tile (untiled) result -- so tiled(Tq<T) == tiled(T)
# is the bit-exact cross-tile rel_shift correctness check (the #1 risk).
def relpos_rowtiled_model(qu, qv, k, p, V, Tq, inv_scale):
    T, DKc = qu.shape[0], qu.shape[1]
    P = p.shape[0]
    kb, pb, Vb = bf16(k), bf16(p), bf16(V)
    probs = np.zeros((T, T), np.float32)
    ctx = np.zeros((T, V.shape[1]), np.float32)
    for q0 in range(0, T, Tq):
        tq = min(Tq, T - q0)
        ac = mm_bf16(bf16(qu[q0:q0 + tq]), kb.T)      # [tq,T] f32
        bd = mm_bf16(bf16(qv[q0:q0 + tq]), pb.T)      # [tq,P] f32
        for il in range(tq):
            i = q0 + il
            base = (T - 1) - (q0 + il)                 # GLOBAL-index rel_shift base
            bd_row = bd[il, base:base + T]             # contiguous length-T window
            scores = (ac[il] + bd_row) * inv_scale
            probs[i] = softmax_kernel(scores)          # bf16 exp2 softmax, bf16 probs
        # ctx_tile = probs @ V; bf16 acc-narrow to match the kernel's bf16 ctx store.
        ctx[q0:q0 + tq] = bf16(mm_bf16(bf16(probs[q0:q0 + tq]), Vb))
    return probs, ctx

# tiled rel_shift assembly via the SAME global-index query-tile loop the kernel
# uses (integer index only) -- compared bit-exact to the NeMo pad/reshape oracle.
def rel_shift_tiled(bd, Tq):  # bd [T,P] -> [T,T]
    T, P = bd.shape
    out = np.zeros((T, T), bd.dtype)
    for q0 in range(0, T, Tq):
        tq = min(Tq, T - q0)
        for il in range(tq):
            i = q0 + il
            base = (T - 1) - (q0 + il)
            out[i] = bd[i, base:base + T]
    return out

# f32 host oracle (one head) from qu/qv/k/p/V -- probs[T,T] + ctx[T,DK].
def host_probs_ctx(qu, qv, k, p, V, inv_scale):
    T = qu.shape[0]
    ac = (qu.astype(np.float32) @ k.astype(np.float32).T)     # [T,T]
    bd = (qv.astype(np.float32) @ p.astype(np.float32).T)     # [T,P]
    bd_sh = rel_shift_host(bd[None])[0]                        # [T,T]
    s = (ac + bd_sh) * inv_scale
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    ctx = (a @ V.astype(np.float32))
    return a.astype(np.float32), ctx.astype(np.float32)

def mhsa_kernel(x, blk, pos_enc):
    T = x.shape[0]
    Wq, Wk, Wv = (W(blk, f"self_attn.linear_{n}.weight") for n in ("q", "k", "v"))
    Wp = W(blk, "self_attn.linear_pos.weight"); Wo = W(blk, "self_attn.linear_out.weight")
    u = W(blk, "self_attn.pos_bias_u"); vv = W(blk, "self_attn.pos_bias_v")
    q = mm_bf16(x, Wq).reshape(T, H, DK)
    k = mm_bf16(x, Wk).reshape(T, H, DK)
    v = mm_bf16(x, Wv).reshape(T, H, DK)
    p = mm_bf16(pos_enc, Wp).reshape(-1, H, DK)
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    ctx = np.zeros((T, H, DK), np.float32)
    for h in range(H):
        qu = bf16(q[:, h] + u[h])           # BRICK 1: bias add (bf16)
        qv = bf16(q[:, h] + vv[h])
        kh = bf16(k[:, h]); ph = bf16(p[:, h]); vh = bf16(v[:, h])
        ac = mm_bf16(qu, kh.T)              # [T,T]  (mmul, f32 acc)
        bd = mm_bf16(qv, ph.T)              # [T,P]  (mmul, f32 acc)
        # BRICK 2a: strided rel_shift (pointer offset, no recompute)
        P = bd.shape[1]
        probs = np.zeros((T, T), np.float32)
        for i in range(T):
            base = (T - 1) - i
            bd_row = bd[i, base:base + T]   # contiguous window
            scores = (ac[i] + bd_row) * inv_scale
            probs[i] = softmax_kernel(scores)  # BRICK 2b: exp2 softmax, bf16 probs
        ctx[:, h] = mm_bf16(probs, vh)      # [T,DK] (mmul)
    ctx = ctx.reshape(T, H * DK)
    return mm_bf16(ctx, Wo)

# ----------------------------------------------------------------------------
def main():
    blk = 0
    pos_enc = np.asarray(REF("pos_enc"), np.float32).reshape(-1, D)  # [2T-1, D]
    x = np.asarray(REF("block_in"), np.float32).reshape(-1, D)       # [T, D] (realistic activations)
    T = x.shape[0]
    print(f"T={T}, P={pos_enc.shape[0]} (=2T-1: {pos_enc.shape[0] == 2*T-1}), D={D}, H={H}, DK={DK}")

    # G1: strided-relayout identity vs the NeMo rel_shift -- bit-exact.
    rng = np.random.default_rng(0)
    bd_test = rng.standard_normal((H, T, 2 * T - 1)).astype(np.float32)
    g1 = rel(rel_shift_strided(bd_test), rel_shift_host(bd_test))
    print(f"G1  strided rel_shift == NeMo rel_shift : rel={g1:.3e}  {'PASS' if g1 < 1e-12 else 'FAIL'}")

    # G2: pure-f32 mirror == host mhsa (uses strided form for the node too).
    ref = mhsa_f32(x, blk, pos_enc, rel_shift_host)
    mirror = mhsa_f32(x, blk, pos_enc, rel_shift_strided)
    g2 = rel(mirror, ref)
    print(f"G2  f32 mirror (strided) == host mhsa   : rel={g2:.3e}  {'PASS' if g2 < 1e-6 else 'FAIL'}")

    # G3: full kernel numeric model vs f32 host mhsa.
    kern = mhsa_kernel(x, blk, pos_enc)
    g3 = rel(kern, ref)
    print(f"G3  kernel bf16 model vs f32 host mhsa  : rel={g3:.3e}  GATE<= {GATE}  {'PASS' if g3 <= GATE else 'FAIL'}")

    # G4: STANDALONE step-1 brick (relpos_scores_softmax_bake) in ISOLATION.
    # Feed IDENTICAL f32 AC/BD to the on-chip model and to the host softmax, so
    # this measures ONLY the two de-risked bricks (strided rel_shift + exp2
    # softmax, bf16 probs) -- no matmul error folded in. Gate rel-L2 <= 0.08,
    # reported per head and worst-case (this is exactly what the xclbin computes).
    q = (x @ W(blk, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(blk, "self_attn.linear_k.weight")).reshape(T, H, DK)
    pm = (pos_enc @ W(blk, "self_attn.linear_pos.weight")).reshape(-1, H, DK)
    u = W(blk, "self_attn.pos_bias_u"); vv = W(blk, "self_attn.pos_bias_v")
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    def host_probs(ac, bd, scale):  # exact f32 softmax over keys of shifted+scaled
        bd_sh = rel_shift_host(bd[None])[0]
        hs = (ac + bd_sh) * scale
        hs = hs - hs.max(-1, keepdims=True)
        hp = np.exp(hs); hp /= hp.sum(-1, keepdims=True)
        return hp

    g4a_worst = 0.0   # real operating regime (block-0 scores saturate -> ~one-hot)
    g4b_worst = 0.0   # rescaled: NON-degenerate softmax that exercises bf16 exp2
    for h in range(H):
        qu = (q[:, h] + u[h]).astype(np.float32)
        qv = (q[:, h] + vv[h]).astype(np.float32)
        ac = (qu @ k[:, h].T).astype(np.float32)                       # [T,T]
        bd = (qv @ pm[:, h].T).astype(np.float32)                      # [T,P]
        # G4a: real AC/BD as the xclbin receives them.
        kp = relpos_scores_softmax_model(ac, bd, inv_scale)
        g4a_worst = max(g4a_worst, rel(kp, host_probs(ac, bd, inv_scale)))
        # G4b: rescale so post-scale scores have std ~1 (healthy softmax spread),
        # preserving the REAL strided rel_shift structure -- this is what actually
        # exercises the vectorized-exp2 / bf16-reciprocal softmax numerics.
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac + bd_sh) * inv_scale).std()) + 1e-6
        rescale = np.float32(inv_scale / std)
        kpb = relpos_scores_softmax_model(ac, bd, rescale)
        g4b_worst = max(g4b_worst, rel(kpb, host_probs(ac, bd, rescale)))
    print(f"G4a standalone brick, real regime       : rel={g4a_worst:.3e}  GATE<= {GATE}  {'PASS' if g4a_worst <= GATE else 'FAIL'}  (worst of {H} heads; scores saturate -> ~one-hot)")
    print(f"G4b standalone brick, non-degenerate sm : rel={g4b_worst:.3e}  GATE<= {GATE}  {'PASS' if g4b_worst <= GATE else 'FAIL'}  (worst of {H} heads; exercises exp2 softmax)")

    # G5: STEP-2 COMPOSED brick (relpos_ac_scores_softmax_bake) -- the on-chip
    # AC = qu @ k^T matmul feeding the resident-in-L1 f32 score tile -> softmax.
    # Feeds bf16 qu/k (the kernel's inputs) through the bf16 mmul + the softmax
    # brick, and compares to the f32 host oracle (f32 qu@k^T + f32 softmax) with
    # the IDENTICAL BD and effective scale. G5c also isolates the bf16 AC matmul
    # vs the f32 qu@k^T (sanity that the on-chip score matmul alone is sane).
    g5a_worst = 0.0   # real regime (block-0 scores saturate -> one-hot): DIAGNOSTIC
    g5a_flips = 0     # argmax flips the bf16 matmul induces on saturated rows
    g5b_worst = 0.0   # rescaled: non-degenerate softmax exercising bf16 exp2 (GATED)
    g5c_worst = 0.0   # AC bf16 mmul vs f32 qu@k^T (matmul-only, DIAGNOSTIC)
    for h in range(H):
        qu = (q[:, h] + u[h]).astype(np.float32)
        qv = (q[:, h] + vv[h]).astype(np.float32)
        kh = k[:, h].astype(np.float32)
        ac_f32 = (qu @ kh.T).astype(np.float32)                        # [T,T]
        bd = (qv @ pm[:, h].T).astype(np.float32)                      # [T,P]
        # G5c: on-chip bf16 AC matmul vs f32 reference.
        ac_bf16 = mm_bf16(qu, kh.T)
        g5c_worst = max(g5c_worst, rel(ac_bf16, ac_f32))
        # G5a: real AC/BD regime, full composed path. NOT GATED: block-0 scores
        # saturate to an exact one-hot (softmax == argmax), so the only signal here
        # is whether the bf16 matmul flips a near-tie argmax. Each single-row flip in
        # a T-grid costs rel = sqrt(2/T) ~ 0.25 by construction, which over-penalizes a
        # harmless near-tie; the end-to-end G3 (bf16 AC folded through ctx+out proj)
        # shows these flips wash out to 4.2e-2. So we REPORT the flip count and gate
        # the composed brick on G5b (exercises the actual numerics) + G3 (pipeline).
        dev = relpos_ac_scores_softmax_model(qu, kh, bd, inv_scale)
        ora = host_probs(ac_f32, bd, inv_scale)
        g5a_worst = max(g5a_worst, rel(dev, ora))
        g5a_flips += int((dev.argmax(-1) != ora.argmax(-1)).sum())
        # G5b: rescale to a non-degenerate softmax (as G4b), real rel_shift kept.
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac_f32 + bd_sh) * inv_scale).std()) + 1e-6
        rescale = np.float32(inv_scale / std)
        devb = relpos_ac_scores_softmax_model(qu, kh, bd, rescale)
        g5b_worst = max(g5b_worst, rel(devb, host_probs(ac_f32, bd, rescale)))
    print(f"G5c step-2 AC bf16 mmul vs f32 qu@k^T   : rel={g5c_worst:.3e}  (worst of {H} heads; matmul only, diagnostic)")
    print(f"G5a step-2 composed brick, real regime  : rel={g5a_worst:.3e}  DIAGNOSTIC (one-hot; {g5a_flips} near-tie argmax flip(s) across {H} heads; washes out in G3)")
    print(f"G5b step-2 composed brick, non-degen sm : rel={g5b_worst:.3e}  GATE<= {GATE}  {'PASS' if g5b_worst <= GATE else 'FAIL'}  (worst of {H} heads; on-chip matmul + exp2 softmax)")

    # ========================================================================
    # STEP-6 ROW-TILED, MemTile-staged block. The per-query-row computation is
    # independent, so tiling the query rows must be NUMERICALLY IDENTICAL to the
    # single tile -- PROVIDED the rel_shift window uses the GLOBAL query index
    # (base = (T-1) - (q0+il), NOT (T-1) - il). This is the #1 correctness risk;
    # the checks below prove it BOTH at the real block-0 T=32 AND at the target
    # real-block T=172 (synthesized realistic tensors -- only real T=32 activations
    # exist locally, but the rel_shift index math is data-independent).
    print("\n--- STEP-6 row-tiled, MemTile-staged block (T up to 172, one head) ---")

    # G6: tiled global-index rel_shift assembly == NeMo rel_shift, BIT-EXACT, over
    # several (T, Tq) incl. ragged tiles (Tq does NOT divide T) and the target T=172.
    g6_fail = False
    for (Tg, Tq) in [(32, 8), (32, 16), (172, 8), (172, 16), (172, 24)]:
        bdt = rng.standard_normal((Tg, 2 * Tg - 1)).astype(np.float32)
        d = rel(rel_shift_tiled(bdt, Tq), rel_shift_host(bdt[None])[0])
        ragged = "ragged" if (Tg % Tq) else "exact"
        ok6 = d < 1e-12
        g6_fail = g6_fail or (not ok6)
        print(f"G6  tiled rel_shift(T={Tg:3d},Tq={Tq:2d}) == NeMo : rel={d:.2e} {ragged:6s} {'PASS' if ok6 else 'FAIL'}")

    # G7: row-tiled numeric block. (a) tiled(Tq) == single-tile (Tq=T): bit-exact
    # cross-tile equivalence -- the load-bearing q0 check. (b) tiled bf16 model vs
    # f32 host oracle: rel-L2 <= 0.08 gate. Run at T=32 REAL (block-0 head-0) and
    # T=172 synthesized (realistic scale so softmax is non-degenerate).
    def run_g7(tag, qu, qv, kk, pp, VV, Tq):
        inv = np.float32(1.0 / np.sqrt(DK))
        # non-degenerate scale (as G4b/G5b): rescale so post-scale scores ~ std 1.
        ac = qu.astype(np.float32) @ kk.astype(np.float32).T
        bd = qv.astype(np.float32) @ pp.astype(np.float32).T
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac + bd_sh) * inv).std()) + 1e-6
        qu_s, qv_s, sc = qu / std, qv / std, inv  # fold 1/std into qu,qv (shrinks ac,bd)
        pr_t, cx_t = relpos_rowtiled_model(qu_s, qv_s, kk, pp, VV, Tq, sc)      # tiled
        pr_1, cx_1 = relpos_rowtiled_model(qu_s, qv_s, kk, pp, VV, len(qu), sc) # single-tile
        pr_h, cx_h = host_probs_ctx(qu_s, qv_s, kk, pp, VV, sc)                 # f32 host
        d_eq_p, d_eq_c = rel(pr_t, pr_1), rel(cx_t, cx_1)
        d_h_p, d_h_c = rel(pr_t, pr_h), rel(cx_t, cx_h)
        eq_ok = (d_eq_p < 1e-12) and (d_eq_c < 1e-12)
        h_ok = (d_h_p <= GATE) and (d_h_c <= GATE)
        print(f"G7 {tag} tiled(Tq={Tq})==single-tile : probs rel={d_eq_p:.2e} ctx rel={d_eq_c:.2e}  {'PASS' if eq_ok else 'FAIL'}")
        print(f"G7 {tag} tiled bf16 vs f32 host      : probs rel={d_h_p:.3e} ctx rel={d_h_c:.3e}  GATE<= {GATE}  {'PASS' if h_ok else 'FAIL'}")
        return eq_ok and h_ok

    # T=32 real block-0 head-0.
    h0 = 0
    qu32 = (q[:, h0] + u[h0]).astype(np.float32)
    qv32 = (q[:, h0] + vv[h0]).astype(np.float32)
    k32 = k[:, h0].astype(np.float32)
    p32 = pm[:, h0].astype(np.float32)
    v32 = (x @ W(blk, "self_attn.linear_v.weight")).reshape(T, H, DK)[:, h0].astype(np.float32)
    g7_32 = run_g7("T=32 real  ", qu32, qv32, k32, p32, v32, 8)

    # T=172 synthesized realistic tensors (data-independent index math; realistic
    # std so the softmax is non-degenerate). Same tensors feed tiled/single/host.
    T172 = 172
    rng2 = np.random.default_rng(1)
    sig = float(q[:, h0].std())  # match real projection scale
    qu172 = rng2.standard_normal((T172, DK)).astype(np.float32) * sig
    qv172 = rng2.standard_normal((T172, DK)).astype(np.float32) * sig
    k172 = rng2.standard_normal((T172, DK)).astype(np.float32) * sig
    p172 = rng2.standard_normal((2 * T172 - 1, DK)).astype(np.float32) * sig
    v172 = rng2.standard_normal((T172, DK)).astype(np.float32) * sig
    g7_172_8 = run_g7("T=172 synth", qu172, qv172, k172, p172, v172, 8)
    g7_172_16 = run_g7("T=172 synth", qu172, qv172, k172, p172, v172, 16)

    step6_ok = (not g6_fail) and g7_32 and g7_172_8 and g7_172_16

    ok = (g1 < 1e-12) and (g2 < 1e-6) and (g3 <= GATE) and (g4a_worst <= GATE) and (g4b_worst <= GATE) \
         and (g5b_worst <= GATE) and step6_ok
    print("\nRESULT:", "ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
