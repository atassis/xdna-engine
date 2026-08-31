#!/usr/bin/env python3
"""On-device validation of the STEP-6 ROW-TILED, MemTile-staged rel-pos MHA block:
the ENTIRE per-head node (AC+BD matmuls -> rel_shift+softmax -> ctx matmul) with
the T query rows processed in TILES of Tq, reading resident k/p/V. Drives PACKED
quv[2T,DK] bf16 (qu=quv[0:T], qv=quv[T:2T]) and PACKED kpv[(2T+P),DK] bf16
(k=kpv[0:T], p=kpv[T:T+P], V=kpv[T+P:2T+P]) through relpos_rowtiled_bake (built
STEP=6) on the XDNA2 NPU via pyxrt; reads ctx[T,DK] bf16 and compares to the fp32
host per-head ctx.

Regime: block-0 scores saturate to one-hot, so by default the host pre-scales
qu/qv by 1/std(scores) (identical to run_npu_relpos_qkp.py) so the softmax is
non-degenerate and actually exercises the exp2 path + the row-tiling; --raw drives
the true saturating regime.

--synth-T N runs a SYNTHESIZED N-frame case (realistic scale) for a pure
device-vs-host numeric gate at the TARGET shape (e.g. --synth-T 172) when real
N-frame activations are not on disk (only the T=32 block-0 refs ship locally).
The row-tiling index math is data-independent (golden G6/G7), so a synthesized
gate at T=172 validates the full device pipeline at the target shape.

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the fp32 host ctx.
ABI (opcode 3): kernel(op, instr[gid1,cacheable], n, QUV[gid3], KPV[gid4], CTX[gid5]).
QUV=[2*T*DK] bf16, KPV=[(2*T+P)*DK] bf16, CTX=[T*DK] bf16.
Build MUST use the SAME T (and a TQ that the kernel was built with): the .cc bakes
RELPOS_T / RELPOS_TQ, so make STEP=6 T=<N> TQ=<tq> before running with matching N.

STATUS (round 19, relpos-uses-8-of-32-cores): --stream (STEP=8, relpos_rowtiled_stream) drives an
xclbin built HEADS=8 (H parallel cores, ONE dispatch drives ALL 8 heads at once -- confirmed
against the compiled .mlir's aie.runtime_sequence arg sizes and each head's per-head DMA offset
stride). This script's ABI above describes the ORIGINAL, single-head STEP=6 design; --stream was
added later without updating that assumption, so every --stream dispatch sent a buffer sized for
ONE head against an ABI that expects 8 heads concatenated head-major -- a straight buffer-size
mismatch (FIXED: --stream now builds+sends all H heads' QUV/KPV and de-concatenates the CTX
readback per head). That fix alone is NOT sufficient to pass the gate yet: a SEPARATE, unresolved
issue remains, present even on the untouched, genuinely single-core STEP=6 path (--stream is not
required to reproduce it) -- per-row error is cleanly bimodal (~0.002, bf16-noise floor, or ~1.3-1.6,
essentially uncorrelated), with the FIRST query tile (rows 0..TQ-1) consistently clean and later
tiles scattered bad, which shape-matches the KPV block-REPLAY mechanism (BUILD-AND-BENCH.md 3f: one
shim BD re-reads padded kpv from DDR n_qt times) rather than the multi-head fix above. NOT
isolated to harness vs kernel this pass; the real pipeline (rust/npu-parakeet's RelposK, same xclbin
family) gives correct transcripts on 2 real clips (round 18), which argues against a kernel defect
but does not prove the standalone probe's remaining wrongness is harness-only. Next decisive test:
localize which query tile / which host-side step introduces the divergence (compare the reference's
rel_shift_host output tile-by-tile against a device dump, or bisect --stream against STEP=7's
kpvstream design, which BUILD-AND-BENCH.md's 3f section names as the block-arithmetic isolator).
Do not trust this script's --stream gate as fully green yet -- the buffer-size class of failure
(NaN, non-deterministic) is gone, but the remaining failure is real and unresolved.
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

D, H, DK = 1024, 8, 128
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(REPO, "mlir-aie/programming_examples/ml/relpos_mha/build")
ENC = os.environ.get("PARAKEET_ENC_DIR",
                     os.path.join(REPO, "artifacts/parakeet/encoder"))
if not os.path.isdir(ENC):
    _sib = os.path.join(os.path.dirname(REPO), "xdna-engine", "artifacts/parakeet/encoder")
    if os.path.isdir(_sib):
        ENC = _sib


def W(blk, name): return np.load(f"{ENC}/L{blk}/{name}.npy")
def REF(name):    return np.load(f"{ENC}/refs/{name}.npy")


def rel_shift_host(bd):  # [H,T,P] -> [H,T,T]  (NeMo pad/reshape/slice oracle)
    Hh, T, P = bd.shape
    x = np.pad(bd, ((0, 0), (0, 0), (1, 0)))
    x = x.reshape(Hh, P + 1, T)
    x = x[:, 1:].reshape(Hh, T, P)
    return x[:, :, :T]


def host_probs_ctx(qu, qv, k, p, V, scale):  # f32 oracle: probs[T,T], ctx[T,DK]
    ac = qu @ k.T
    bd = qv @ p.T
    bd_sh = rel_shift_host(bd[None])[0]
    s = (ac + bd_sh) * scale
    s = s - s.max(-1, keepdims=True)
    a = np.exp(s); a /= a.sum(-1, keepdims=True)
    return a.astype(np.float32), (a @ V).astype(np.float32)


def build_head(block, head, synth_T, real_tiled_T=0):
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    if real_tiled_T:
        # DISCRIMINATOR mode: take the REAL block-0 head tensors and tile/repeat
        # them up to real_tiled_T frames. Device input AND reference then derive
        # from ONE trusted array set (no synth RNG), so a multi-block failure here
        # is a genuine device bug, not a synth-data / packing divergence. The tiled
        # data is not physically meaningful (repeated frames) but is a valid
        # numeric parity test: both sides compute attention on the same arrays.
        _, qu0, qv0, k0, p0, V0, _ = build_head(block, head, 0)
        Tn = real_tiled_T
        Pn = 2 * Tn - 1
        def tile(x, n): return np.resize(x, (n, DK)).astype(np.float32)
        return Tn, tile(qu0, Tn), tile(qv0, Tn), tile(k0, Tn), tile(p0, Pn), \
            tile(V0, Tn), inv_scale
    if synth_T:
        T = synth_T
        rng = np.random.default_rng(1)
        # match a realistic per-projection scale from the real weights' output std.
        x = np.asarray(REF("block_in"), np.float32).reshape(-1, D)
        q0 = (x @ W(block, "self_attn.linear_q.weight")).reshape(-1, H, DK)[:, head]
        sig = float(q0.std())
        qu = rng.standard_normal((T, DK)).astype(np.float32) * sig
        qv = rng.standard_normal((T, DK)).astype(np.float32) * sig
        kh = rng.standard_normal((T, DK)).astype(np.float32) * sig
        ph = rng.standard_normal((2 * T - 1, DK)).astype(np.float32) * sig
        Vh = rng.standard_normal((T, DK)).astype(np.float32) * sig
        return T, qu, qv, kh, ph, Vh, inv_scale
    pos = np.asarray(REF("pos_enc"), np.float32).reshape(-1, D)
    x = np.asarray(REF("block_in"), np.float32).reshape(-1, D)
    T = x.shape[0]
    q = (x @ W(block, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(block, "self_attn.linear_k.weight")).reshape(T, H, DK)
    v = (x @ W(block, "self_attn.linear_v.weight")).reshape(T, H, DK)
    pm = (pos @ W(block, "self_attn.linear_pos.weight")).reshape(-1, H, DK)
    u = W(block, "self_attn.pos_bias_u"); vv = W(block, "self_attn.pos_bias_v")
    qu = (q[:, head] + u[head]).astype(np.float32)
    qv = (q[:, head] + vv[head]).astype(np.float32)
    return T, qu, qv, k[:, head].astype(np.float32), pm[:, head].astype(np.float32), \
        v[:, head].astype(np.float32), inv_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="true saturating (one-hot) regime")
    ap.add_argument("--synth-T", type=int, default=0,
                    help="synthesize an N-frame case (e.g. 172) instead of the real block")
    ap.add_argument("--real-tiled-T", type=int, default=0,
                    help="DISCRIMINATOR: tile the REAL block-0 tensors up to N frames so "
                         "device input + reference derive from ONE array (no synth RNG). A "
                         "failure here at multi-block N is a real device bug, not a synth issue.")
    ap.add_argument("--stream", action="store_true",
                    help="STEP=8 packing: tile-interleaved QUV + padded KPV/CTX for the "
                         "MemTile-streamed block (relpos_rowtiled_stream). Requires --tq/--kb "
                         "matching the build.")
    ap.add_argument("--tq", type=int, default=8, help="query-tile rows (must match -DRELPOS_TQ)")
    ap.add_argument("--kb", type=int, default=43, help="key-block rows (must match -DRELPOS_KB)")
    a = ap.parse_args()

    def bf(x): return np.ascontiguousarray(x, np.float32).astype(bfloat16).reshape(-1)

    def ceildiv(x, y): return (x + y - 1) // y

    def pad_rows(x, n):  # [r,DK] float32 -> [n,DK] bf16, zero-padded (pad rows unread by core)
        r = x.shape[0]
        if r < n:
            x = np.concatenate([x, np.zeros((n - r, DK), np.float32)], 0)
        return x

    def prep_head(head):
        """Per-head data + non-degenerate-softmax prescale (--raw skips it) + the f32 golden
        ctx for that head. Factored out so --stream can call it once per head (see BUG NOTE
        below), while non-stream (STEP=6) keeps calling it exactly once for --head."""
        T_, qu_, qv_, kh_, ph_, Vh_, inv_scale_ = build_head(a.block, head, a.synth_T, a.real_tiled_T)
        if a.raw:
            qu_d_, qv_d_, sc_ = qu_, qv_, inv_scale_
        else:
            ac_ = qu_ @ kh_.T; bd_ = qv_ @ ph_.T
            bd_sh_ = rel_shift_host(bd_[None])[0]
            std_ = float(((ac_ + bd_sh_) * inv_scale_).std()) + 1e-6
            qu_d_, qv_d_, sc_ = qu_ / std_, qv_ / std_, inv_scale_
        _, ctx_ref_ = host_probs_ctx(qu_d_, qv_d_, kh_, ph_, Vh_, sc_)  # [T,DK] f32 golden
        return T_, qu_d_, qv_d_, kh_, ph_, Vh_, sc_, ctx_ref_

    T, qu_d, qv_d, kh, ph, Vh, sc, ctx_ref = prep_head(a.head)
    P = 2 * T - 1

    if a.stream:
        # STEP=8 MemTile-streamed packing (relpos_rowtiled_stream_iron.py):
        #   QUV tile-interleaved [qu_t0, qv_t0, qu_t1, qv_t1, ...], each tile TQ rows
        #     (ragged final tile zero-padded to TQ); CTX read back n_qt*TQ rows -> [:T].
        #   KPV = k(pad Tp) | p(pad Pp) | V(pad Tp), Tp=n_kb*KB, Pp=n_pb*KB.
        #
        # BUG FOUND round 19: the xclbin this drives is built HEADS=8 (H parallel cores, ONE
        # dispatch drives ALL 8 heads at once -- see rust/npu-parakeet/src/npu.rs's own comment
        # "The BOs concatenate all H heads; each head's Worker reads/writes its own slice via a
        # per-head shim OFFSET tap"). Confirmed against the compiled .mlir's own
        # `aie.runtime_sequence(%arg0: memref<327680xbf16>, %arg1: memref<655360xbf16>,
        # %arg2: memref<163840xbf16>)` and each head's `aiex.dma_configure_task_for @kpv{i}`
        # offset (0, 81920, 163840, ... -- uniform stride, head-major): every one of those sizes
        # is EXACTLY 8x a single head's QUV/KPV/CTX (327680/8=40960 QUV elems/head,
        # 655360/8=81920 KPV elems/head, 163840/8=20480 CTX elems/head, all matching this
        # function's own per-head math below). The ORIGINAL code here built and dispatched
        # ONE head's worth of data against that 8-head ABI -- a straight buffer-size mismatch
        # (the STEP=6 assumption baked into main(): one head per dispatch, --head selects which).
        # STEP=6's own xclbin genuinely IS single-head (Makefile's GEN_EXTRA only adds --heads
        # under STEP=8), so the non-stream path below this block is correct as originally
        # written and is NOT touched by this fix.
        TQ, KB = a.tq, a.kb
        n_qt = ceildiv(T, TQ); n_kb = ceildiv(T, KB); n_pb = ceildiv(P, KB)
        Tp, Pp = n_kb * KB, n_pb * KB
        ctx_rows = n_qt * TQ

        def pack_one_head(qu_h, qv_h, kh_h, ph_h, Vh_h):
            quv_tiles = []
            for q in range(n_qt):
                q0 = q * TQ
                quv_tiles.append(pad_rows(qu_h[q0:q0 + TQ], TQ))
                quv_tiles.append(pad_rows(qv_h[q0:q0 + TQ], TQ))
            quv_h = bf(np.concatenate(quv_tiles, 0))            # [2*n_qt*TQ, DK], one head
            kpv_h = np.concatenate([bf(pad_rows(kh_h, Tp)),
                                    bf(pad_rows(ph_h, Pp)),
                                    bf(pad_rows(Vh_h, Tp))])    # [Tp+Pp+Tp, DK], one head
            return quv_h, kpv_h

        quv_h0, kpv_h0 = pack_one_head(qu_d, qv_d, kh, ph, Vh)
        quv_parts, kpv_parts, ctx_ref_parts = [quv_h0], [kpv_h0], [ctx_ref]
        for h in range(H):
            if h == a.head:
                continue
            _, qu_h, qv_h, kh_h, ph_h, Vh_h, _, ctx_ref_h = prep_head(h)
            quv_h, kpv_h = pack_one_head(qu_h, qv_h, kh_h, ph_h, Vh_h)
            quv_parts.append(quv_h); kpv_parts.append(kpv_h); ctx_ref_parts.append(ctx_ref_h)
        # head-major order 0..H-1, matching the .mlir's uniform per-head offset stride
        # (a.head's chunk was computed first above, so re-sort back to head-id order).
        head_ids = [a.head] + [h for h in range(H) if h != a.head]
        reorder = np.argsort(head_ids)
        QUV = np.concatenate([quv_parts[i] for i in reorder])
        KPV = np.concatenate([kpv_parts[i] for i in reorder])
        ctx_ref = np.concatenate([ctx_ref_parts[i] for i in reorder], axis=0)  # [H*T, DK]
        ctx_rows_per_head = ctx_rows  # n_qt*TQ, may exceed T (padded tail, cropped below)
        ctx_rows = H * ctx_rows_per_head

        # SELF-CHECK (a.head's own slice only, matching the original check's scope): de-pack
        # the EXACT bytes about to be sent for a.head and assert they reconstruct qu_d/qv_d/
        # kh/ph/Vh byte-for-byte.
        def _bf_rows(x): return np.frombuffer(bf(x).tobytes(), np.uint16).view(bfloat16).reshape(-1, DK).astype(np.float32)
        QUVr = np.frombuffer(quv_h0.view(np.uint16).tobytes(), np.uint16).view(bfloat16).reshape(2 * n_qt, TQ, DK).astype(np.float32)
        KPVr = np.frombuffer(kpv_h0.view(np.uint16).tobytes(), np.uint16).view(bfloat16).reshape(-1, DK).astype(np.float32)
        qu_dp = np.zeros((T, DK), np.float32); qv_dp = np.zeros((T, DK), np.float32)
        for q in range(n_qt):
            q0 = q * TQ; tq = min(TQ, T - q0)
            qu_dp[q0:q0 + tq] = QUVr[2 * q][:tq]; qv_dp[q0:q0 + tq] = QUVr[2 * q + 1][:tq]
        checks = {"qu": (qu_dp, _bf_rows(qu_d)), "qv": (qv_dp, _bf_rows(qv_d)),
                  "k": (KPVr[0:Tp][:T], _bf_rows(kh)),
                  "p": (KPVr[Tp:Tp + Pp][:P], _bf_rows(ph)),
                  "V": (KPVr[Tp + Pp:2 * Tp + Pp][:T], _bf_rows(Vh))}
        bad = {n: float(np.abs(a - r).max()) for n, (a, r) in checks.items() if not np.array_equal(a, r)}
        if bad:
            print(f"[pack-check] FAIL: device-BO de-pack != reference: {bad}")
        else:
            print(f"[pack-check] OK: QUV/KPV bytes de-pack to the reference qu/qv/k/p/V exactly")
    else:
        QUV = np.concatenate([bf(qu_d), bf(qv_d)])             # [2T,DK]
        KPV = np.concatenate([bf(kh), bf(ph), bf(Vh)])         # [(2T+P),DK]
        ctx_rows = T

    import pyxrt
    instr = np.fromfile(a.insts, dtype=np.uint32)
    QUVbytes, KPVbytes, Cbytes = QUV.nbytes, KPV.nbytes, (ctx_rows * DK * 2)
    d = pyxrt.device(0)
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    _mode = 'real-tiled' if a.real_tiled_T else ('synth' if a.synth_T else 'real')
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}  T={T} P={P} "
          f"regime={'raw' if a.raw else 'rescaled'} data={_mode}")
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    kk = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_quv = pyxrt.bo(d, QUVbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_kpv = pyxrt.bo(d, KPVbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_cx = pyxrt.bo(d, Cbytes, pyxrt.bo.host_only, kk.group_id(5))

    bo_instr.write(instr.tobytes(), 0);                bo_instr.sync(TO)
    bo_quv.write(QUV.view(np.uint16).tobytes(), 0);    bo_quv.sync(TO)
    bo_kpv.write(KPV.view(np.uint16).tobytes(), 0);    bo_kpv.sync(TO)

    def once():
        r = kk(3, bo_instr, instr.size, bo_quv, bo_kpv, bo_cx); r.wait()

    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_cx.sync(FROM)
    CXraw = np.frombuffer(bo_cx.read(Cbytes, 0), dtype=np.uint16).view(bfloat16).reshape(-1, DK).astype(np.float32)
    if a.stream:
        # [H*n_qt*TQ, DK] -> per-head [n_qt*TQ, DK], crop each head's padded tail to T, re-stack
        # head-major to match ctx_ref's [H*T, DK] layout.
        CX = CXraw.reshape(H, ctx_rows_per_head, DK)[:, :T, :].reshape(H * T, DK)
    else:
        CX = CXraw[:T]
    print(f"[diag] CX stats: min={CX.min():.3g} max={CX.max():.3g} "
          f"finite={np.isfinite(CX).sum()}/{CX.size}  shape={CX.shape}")
    print(f"[diag] ref stats: min={ctx_ref.min():.3g} max={ctx_ref.max():.3g}  shape={ctx_ref.shape}")
    d_abs = np.abs(CX - ctx_ref)
    worst = np.unravel_index(np.argmax(d_abs), d_abs.shape)
    print(f"[diag] worst abs err at {worst}: dev={CX[worst]:.4g} ref={ctx_ref[worst]:.4g} "
          f"|d|={d_abs[worst]:.4g}")
    row_rel = np.linalg.norm(d_abs, axis=1) / (np.linalg.norm(np.abs(ctx_ref), axis=1) + 1e-9)
    print(f"[diag] per-row rel err (non-stream shape [T,DK]): {np.round(row_rel, 3).tolist()}")
    if a.stream:
        CXh = CX.reshape(H, T, DK); RFh = ctx_ref.reshape(H, T, DK)
        for h in range(H):
            d = CXh[h] - RFh[h]
            rl2 = float(np.linalg.norm(d) / (np.linalg.norm(RFh[h]) + 1e-12))
            print(f"[diag] head {h}: rel-L2={rl2:.4e}  dev[0,:3]={CXh[h,0,:3]}  ref[0,:3]={RFh[h,0,:3]}")
    af, rf = CX.ravel(), ctx_ref.ravel()
    rel_l2 = float(np.linalg.norm(af - rf) / (np.linalg.norm(rf) + 1e-12))
    corr = float(np.corrcoef(af, rf)[0, 1])
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (T={T}; row-tiled AC+BD+softmax+ctx)")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}")
    print(f"[run] ctx[0,:4]={CX[0,:4]}  ref={ctx_ref[0,:4]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] relpos_rowtiled (row-tiled MHA block) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
