#!/usr/bin/env python3
"""On-device validation of the STEP-3 resident rel-pos block: BOTH score matmuls
on chip. Drives PACKED qk[2T,DK] bf16 (qu=qk[0:T], k=qk[T:2T]) and PACKED
qvp[(T+P),DK] bf16 (qv=qvp[0:T], p=qvp[T:T+P]) through relpos_qkp_scores_softmax_bake
(built STEP=3) on the XDNA2 NPU via pyxrt. The core computes AC=qu@k^T [T,T] AND
BD=qv@p^T [T,P] into resident L1 tiles, then rel_shift(BD)+scale+exp2 softmax ->
probs[T,T] bf16. No host score buffer at all. Compared to the fp32 host softmax.

Non-degenerate regime by default (block-0 scores saturate to one-hot): host pre-
scales qu and qv by 1/std(scores) so AC and BD both shrink, landing a softmax that
exercises the on-chip matmuls + exp2. --raw drives the true saturating regime.

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the fp32 host softmax.

ABI (opcode 3): kernel(op, instr[gid1,cacheable], n_instr, QK[gid3], QVP[gid4], PROBS[gid5]).
QK=[2*T*DK] bf16, QVP=[(T+P)*DK] bf16, PROBS=[T*T] bf16.
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
    _sib = os.path.join(os.path.dirname(REPO), "xdna-engine",
                        "artifacts/parakeet/encoder")
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


def host_probs(ac, bd, scale):  # exact f32 softmax over keys of shifted+scaled
    bd_sh = rel_shift_host(bd[None])[0]
    hs = (ac + bd_sh) * scale
    hs = hs - hs.max(-1, keepdims=True)
    hp = np.exp(hs); hp /= hp.sum(-1, keepdims=True)
    return hp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="drive the real saturating (one-hot) regime")
    a = ap.parse_args()

    pos = np.asarray(REF("pos_enc"), np.float32).reshape(-1, D)
    x = np.asarray(REF("block_in"), np.float32).reshape(-1, D)
    T = x.shape[0]
    P = 2 * T - 1
    inv_scale = np.float32(1.0 / np.sqrt(DK))

    q = (x @ W(a.block, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(a.block, "self_attn.linear_k.weight")).reshape(T, H, DK)
    pm = (pos @ W(a.block, "self_attn.linear_pos.weight")).reshape(-1, H, DK)
    u = W(a.block, "self_attn.pos_bias_u"); vv = W(a.block, "self_attn.pos_bias_v")

    h = a.head
    qu = (q[:, h] + u[h]).astype(np.float32)   # [T,DK]
    qv = (q[:, h] + vv[h]).astype(np.float32)  # [T,DK]
    kh = k[:, h].astype(np.float32)            # [T,DK]
    ph = pm[:, h].astype(np.float32)           # [P,DK]
    ac_f32 = (qu @ kh.T).astype(np.float32)    # [T,T]
    bd = (qv @ ph.T).astype(np.float32)        # [T,P]

    # AC and BD are BOTH computed on device (bf16 mmul). To hit a non-degenerate
    # softmax, pre-divide qu and qv by std(scores) -- scaling qu shrinks AC and
    # scaling qv shrinks BD by the same factor; the oracle uses the identical scale.
    if a.raw:
        qu_dev, qv_dev, oracle_scale = qu, qv, inv_scale
    else:
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac_f32 + bd_sh) * inv_scale).std()) + 1e-6
        qu_dev, qv_dev, oracle_scale = qu / std, qv / std, inv_scale / std

    ref = host_probs(ac_f32, bd, oracle_scale)  # fp32 golden [T,T]

    QK = np.concatenate([
        np.ascontiguousarray(qu_dev, dtype=np.float32).astype(bfloat16).reshape(-1),
        np.ascontiguousarray(kh, dtype=np.float32).astype(bfloat16).reshape(-1),
    ])
    QVP = np.concatenate([
        np.ascontiguousarray(qv_dev, dtype=np.float32).astype(bfloat16).reshape(-1),
        np.ascontiguousarray(ph, dtype=np.float32).astype(bfloat16).reshape(-1),
    ])

    import pyxrt
    instr = np.fromfile(a.insts, dtype=np.uint32)
    QKbytes, QVPbytes, Pbytes = QK.nbytes, QVP.nbytes, (T * T * 2)
    d = pyxrt.device(0)
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}  T={T} P={P} regime={'raw' if a.raw else 'rescaled'}")
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    kk = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_qk = pyxrt.bo(d, QKbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_qvp = pyxrt.bo(d, QVPbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_pr = pyxrt.bo(d, Pbytes, pyxrt.bo.host_only, kk.group_id(5))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_qk.write(QK.view(np.uint16).tobytes(), 0);    bo_qk.sync(TO)
    bo_qvp.write(QVP.view(np.uint16).tobytes(), 0);  bo_qvp.sync(TO)

    def once():
        r = kk(3, bo_instr, instr.size, bo_qk, bo_qvp, bo_pr)
        r.wait()

    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_pr.sync(FROM)
    PR = np.frombuffer(bo_pr.read(Pbytes, 0), dtype=np.uint16).view(bfloat16).reshape(T, T)
    yf = PR.astype(np.float32)

    a_flat, r_flat = yf.ravel(), ref.ravel()
    rel_l2 = float(np.linalg.norm(a_flat - r_flat) / (np.linalg.norm(r_flat) + 1e-12))
    corr = float(np.corrcoef(a_flat, r_flat)[0, 1])
    rowsum = yf.sum(-1)
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (T={T}; on-chip AC + BD matmuls + resident softmax)")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}  probs rowsum min/max={rowsum.min():.4f}/{rowsum.max():.4f}")
    print(f"[run] probs[0,:5]={yf[0,:5]}  ref={ref[0,:5]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] relpos_qkp_scores_softmax on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
