#!/usr/bin/env python3
"""Reproduce the STEP=8 block-decomposed + --stream-packed path in numpy and
compare to the proven monolithic tiled model (relpos_rowtiled_model). If they
diverge, the bug is in the block bricks / packing logic (not the device replay)."""
import numpy as np
from ml_dtypes import bfloat16 as BF16

DK = 128
LOG2E = np.float32(1.4426950408889634)


def bf16(x): return np.asarray(x, np.float32).astype(BF16).astype(np.float32)
def mm_bf16(a, b): return (bf16(a).astype(np.float32) @ bf16(b).astype(np.float32)).astype(np.float32)


def softmax_kernel(scores_row):
    m = scores_row.max()
    e = np.exp2(((scores_row - m) * LOG2E).astype(np.float32)).astype(BF16).astype(np.float32)
    s = e.sum(dtype=np.float32)
    inv = bf16(np.float32(1.0) / s)
    return bf16(e * inv)


# ---- PROVEN monolithic tiled model (from the golden, ctx-returning) ----
def monolithic(qu, qv, k, p, V, TQ, inv_scale):
    T, P = qu.shape[0], p.shape[0]
    kb_, pb_, Vb_ = bf16(k), bf16(p), bf16(V)
    probs = np.zeros((T, T), np.float32)
    ctx = np.zeros((T, DK), np.float32)
    for q0 in range(0, T, TQ):
        tq = min(TQ, T - q0)
        ac = mm_bf16(bf16(qu[q0:q0 + tq]), kb_.T)
        bd = mm_bf16(bf16(qv[q0:q0 + tq]), pb_.T)
        for il in range(tq):
            i = q0 + il
            base = (T - 1) - (q0 + il)
            scores = (ac[il] + bd[il, base:base + T]) * inv_scale
            probs[i] = softmax_kernel(scores)
        ctx[q0:q0 + tq] = bf16(mm_bf16(bf16(probs[q0:q0 + tq]), Vb_))
    return ctx


# ---- BLOCK-DECOMPOSED + --stream-packed model (mirrors the .cc bricks) ----
def ceildiv(a, b): return (a + b - 1) // b


def pad_rows(x, n):
    r = x.shape[0]
    if r < n:
        x = np.concatenate([x, np.zeros((n - r, DK), np.float32)], 0)
    return x


def dot_block(A, Bblk, out, tq, kb, j0, ncol):
    # out[il, j0+jj] = f32-acc dot(bf16(A[il]), bf16(Bblk[jj]))
    for il in range(tq):
        for jj in range(kb):
            out[il, j0 + jj] = np.dot(bf16(A[il]).astype(np.float32),
                                      bf16(Bblk[jj]).astype(np.float32))


def block_stream(qu, qv, k, p, V, TQ, KB, inv_scale):
    T, P = qu.shape[0], p.shape[0]
    n_kb = ceildiv(T, KB); n_pb = ceildiv(P, KB)
    Tp, Pp = n_kb * KB, n_pb * KB
    # --stream padded L2 layout: k_pad | p_pad | V_pad
    k_pad, p_pad, V_pad = pad_rows(k, Tp), pad_rows(p, Pp), pad_rows(V, Tp)
    Tk_full = (T // KB) * KB; k_rag = T - Tk_full
    Pp_full = (P // KB) * KB; p_rag = P - Pp_full
    Tq_full = (T // TQ) * TQ; q_rag = T - Tq_full

    ctx = np.zeros((T, DK), np.float32)

    def emit_tile(tq, q0):
        # QUV tile-interleaved packing: qu_tile / qv_tile padded to TQ rows
        qu_t = pad_rows(qu[q0:q0 + tq], TQ)
        qv_t = pad_rows(qv[q0:q0 + tq], TQ)
        AC = np.zeros((TQ, T), np.float32)
        BD = np.zeros((TQ, P), np.float32)
        # phase K: k full-blocks + ragged
        for j0 in range(0, Tk_full, KB):
            dot_block(qu_t, k_pad[j0:j0 + KB], AC, tq, KB, j0, T)
        if k_rag:
            dot_block(qu_t, k_pad[Tk_full:Tk_full + KB], AC, tq, k_rag, Tk_full, T)
        # phase P
        for j0 in range(0, Pp_full, KB):
            dot_block(qv_t, p_pad[j0:j0 + KB], BD, tq, KB, j0, P)
        if p_rag:
            dot_block(qv_t, p_pad[Pp_full:Pp_full + KB], BD, tq, p_rag, Pp_full, P)
        # softmax (global-index rel_shift)
        probs = np.zeros((TQ, T), np.float32)
        for il in range(tq):
            base = (T - 1) - (q0 + il)
            scores = (AC[il] + BD[il, base:base + T]) * inv_scale
            probs[il] = softmax_kernel(scores)
        # phase V: block-accumulate ctx in f32, narrow at end
        ctxf = np.zeros((TQ, DK), np.float32)
        for j0 in range(0, Tk_full, KB):
            block_ctx(probs, V_pad[j0:j0 + KB], ctxf, tq, KB, j0)
        if k_rag:
            block_ctx(probs, V_pad[Tk_full:Tk_full + KB], ctxf, tq, k_rag, Tk_full)
        ctx[q0:q0 + tq] = bf16(ctxf[:tq])

    def block_ctx(probs, Vblk, ctxf, tq, kb, j0):
        for il in range(tq):
            acc = np.zeros(DK, np.float32)
            for jj in range(kb):
                acc += bf16(probs[il, j0 + jj]).astype(np.float32) * bf16(Vblk[jj]).astype(np.float32)
            ctxf[il] += acc

    for q0 in range(0, Tq_full, TQ):
        emit_tile(TQ, q0)
    if q_rag:
        emit_tile(q_rag, Tq_full)
    return ctx


def run(T, TQ=8, KB=43, seed=1):
    rng = np.random.default_rng(seed)
    P = 2 * T - 1
    sig = 0.3
    qu = rng.standard_normal((T, DK)).astype(np.float32) * sig
    qv = rng.standard_normal((T, DK)).astype(np.float32) * sig
    k = rng.standard_normal((T, DK)).astype(np.float32) * sig
    p = rng.standard_normal((P, DK)).astype(np.float32) * sig
    V = rng.standard_normal((T, DK)).astype(np.float32) * sig
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    m = monolithic(qu, qv, k, p, V, TQ, inv_scale)
    b = block_stream(qu, qv, k, p, V, TQ, KB, inv_scale)
    rel = float(np.linalg.norm(b - m) / (np.linalg.norm(m) + 1e-12))
    corr = float(np.corrcoef(b.ravel(), m.ravel())[0, 1])
    print(f"T={T} TQ={TQ} KB={KB}: block-vs-monolithic rel-L2={rel:.4e} corr={corr:.6f}  "
          f"{'MATCH' if rel < 1e-3 else 'DIVERGE'}")
    if rel >= 1e-3:
        print(f"  ctx_block[0,:4]={b[0,:4]}  ctx_mono[0,:4]={m[0,:4]}")
    return rel


run(32)
run(172)


# ---- DELIVERY-BUG HYPOTHESES: block_stream but the forward mis-delivers ----
def block_stream_buggy(qu, qv, k, p, V, TQ, KB, inv_scale, mode):
    T, P = qu.shape[0], p.shape[0]
    n_kb = ceildiv(T, KB); n_pb = ceildiv(P, KB)
    Tp, Pp = n_kb * KB, n_pb * KB
    k_pad, p_pad, V_pad = pad_rows(k, Tp), pad_rows(p, Pp), pad_rows(V, Tp)
    kpv = np.concatenate([k_pad, p_pad, V_pad], 0)   # [Tp+Pp+Tp, DK] flat L2 layout
    nblk = kpv.shape[0] // KB
    Tk_full = (T // KB) * KB; k_rag = T - Tk_full
    Pp_full = (P // KB) * KB; p_rag = P - Pp_full
    Tq_full = (T // TQ) * TQ; q_rag = T - Tq_full
    ctx = np.zeros((T, DK), np.float32)

    def get_block(idx):
        # idx = sequential block index the core acquires within a tile (0..nblk-1)
        if mode == "first":      # forward delivers block 0 repeated
            return kpv[0:KB]
        if mode == "seq":        # correct address-order
            return kpv[idx * KB:(idx + 1) * KB]
        raise ValueError(mode)

    def block_ctx(probs, Vblk, ctxf, tq, kb, j0):
        for il in range(tq):
            acc = np.zeros(DK, np.float32)
            for jj in range(kb):
                acc += bf16(probs[il, j0 + jj]).astype(np.float32) * bf16(Vblk[jj]).astype(np.float32)
            ctxf[il] += acc

    def emit_tile(tq, q0):
        qu_t = pad_rows(qu[q0:q0 + tq], TQ); qv_t = pad_rows(qv[q0:q0 + tq], TQ)
        AC = np.zeros((TQ, T), np.float32); BD = np.zeros((TQ, P), np.float32)
        bi = 0  # sequential block index consumed this tile
        for j0 in range(0, Tk_full, KB):
            dot_block(qu_t, get_block(bi), AC, tq, KB, j0, T); bi += 1
        if k_rag:
            dot_block(qu_t, get_block(bi), AC, tq, k_rag, Tk_full, T); bi += 1
        for j0 in range(0, Pp_full, KB):
            dot_block(qv_t, get_block(bi), BD, tq, KB, j0, P); bi += 1
        if p_rag:
            dot_block(qv_t, get_block(bi), BD, tq, p_rag, Pp_full, P); bi += 1
        probs = np.zeros((TQ, T), np.float32)
        for il in range(tq):
            base = (T - 1) - (q0 + il)
            probs[il] = softmax_kernel((AC[il] + BD[il, base:base + T]) * inv_scale)
        ctxf = np.zeros((TQ, DK), np.float32)
        for j0 in range(0, Tk_full, KB):
            block_ctx(probs, get_block(bi), ctxf, tq, KB, j0); bi += 1
        if k_rag:
            block_ctx(probs, get_block(bi), ctxf, tq, k_rag, Tk_full); bi += 1
        ctx[q0:q0 + tq] = bf16(ctxf[:tq])

    for q0 in range(0, Tq_full, TQ):
        emit_tile(TQ, q0)
    if q_rag:
        emit_tile(q_rag, Tq_full)
    return ctx


def run_hyp(T, TQ=8, KB=43, seed=1):
    rng = np.random.default_rng(seed); P = 2 * T - 1; sig = 0.3
    qu = rng.standard_normal((T, DK)).astype(np.float32) * sig
    qv = rng.standard_normal((T, DK)).astype(np.float32) * sig
    k = rng.standard_normal((T, DK)).astype(np.float32) * sig
    p = rng.standard_normal((P, DK)).astype(np.float32) * sig
    V = rng.standard_normal((T, DK)).astype(np.float32) * sig
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    m = monolithic(qu, qv, k, p, V, TQ, inv_scale)
    for mode in ("seq", "first"):
        b = block_stream_buggy(qu, qv, k, p, V, TQ, KB, inv_scale, mode)
        rel = float(np.linalg.norm(b - m) / (np.linalg.norm(m) + 1e-12))
        corr = float(np.corrcoef(b.ravel(), m.ravel())[0, 1])
        print(f"T={T} mode={mode:6s}: rel-L2={rel:.4e} corr={corr:.6f}  ctx[0,:4]={b[0,:4]}")
    print(f"  ref ctx[0,:4]={m[0,:4]}")


print("--- delivery hypotheses ---")
run_hyp(32)
run_hyp(172)
