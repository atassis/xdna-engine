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


def build_head(block, head, synth_T):
    inv_scale = np.float32(1.0 / np.sqrt(DK))
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
    a = ap.parse_args()

    T, qu, qv, kh, ph, Vh, inv_scale = build_head(a.block, a.head, a.synth_T)
    P = 2 * T - 1

    # non-degenerate softmax: fold 1/std(scores) into qu/qv (shrinks AC and BD by
    # the same factor); the oracle uses the identical effective scale. --raw skips it.
    if a.raw:
        qu_d, qv_d, sc = qu, qv, inv_scale
    else:
        ac = qu @ kh.T; bd = qv @ ph.T
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac + bd_sh) * inv_scale).std()) + 1e-6
        qu_d, qv_d, sc = qu / std, qv / std, inv_scale

    _, ctx_ref = host_probs_ctx(qu_d, qv_d, kh, ph, Vh, sc)  # [T,DK] f32 golden

    def bf(x): return np.ascontiguousarray(x, np.float32).astype(bfloat16).reshape(-1)
    QUV = np.concatenate([bf(qu_d), bf(qv_d)])                 # [2T,DK]
    KPV = np.concatenate([bf(kh), bf(ph), bf(Vh)])            # [(2T+P),DK]

    import pyxrt
    instr = np.fromfile(a.insts, dtype=np.uint32)
    QUVbytes, KPVbytes, Cbytes = QUV.nbytes, KPV.nbytes, (T * DK * 2)
    d = pyxrt.device(0)
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}  T={T} P={P} "
          f"regime={'raw' if a.raw else 'rescaled'}{' synth' if a.synth_T else ''}")
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
    CX = np.frombuffer(bo_cx.read(Cbytes, 0), dtype=np.uint16).view(bfloat16).reshape(T, DK).astype(np.float32)
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
