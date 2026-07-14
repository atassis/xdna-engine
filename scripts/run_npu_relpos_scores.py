#!/usr/bin/env python3
"""On-device validation of the STANDALONE rel-pos scores->softmax step-1 kernel.

Drives one head/block of AC[T,T] f32 + BD[T,P] f32 through the
relpos_scores_softmax_bake IRON design (route_b_kernels/relpos_mha) on the XDNA2
NPU via pyxrt and compares the bf16 probs readback against the fp32 host softmax
(scripts/parakeet_relpos_mha_golden.relpos_scores_softmax_model oracle).

The kernel bakes inv_scale = 1/sqrt(DK) internally. Block-0's raw scores saturate
(~one-hot softmax), which would only test rel_shift + argmax; so by default the
host PRE-SCALES AC/BD by 1/std to land a NON-DEGENERATE softmax that actually
exercises the on-chip vectorized-exp2 / bf16-reciprocal path (the oracle is scaled
identically). Pass --raw to drive the true saturating regime instead.

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the fp32 host softmax.

IRON host ABI (dwconv1d/eltwise_mul test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n_instr, AC[gid3], BD[gid4], PROBS[gid5])
AC = [T*T] f32, BD = [T*P] f32, PROBS = [T*T] bf16.
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
LOG2E = np.float32(1.4426950408889634)


def W(blk, name): return np.load(f"{ENC}/L{blk}/{name}.npy")
def REF(name):    return np.load(f"{ENC}/refs/{name}.npy")
def bf16(x):      return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


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
                    help="drive the real saturating regime (default: rescale to a "
                         "non-degenerate softmax that exercises exp2)")
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
    qu = (q[:, h] + u[h]).astype(np.float32)
    qv = (q[:, h] + vv[h]).astype(np.float32)
    ac = (qu @ k[:, h].T).astype(np.float32)                   # [T,T]
    bd = (qv @ pm[:, h].T).astype(np.float32)                  # [T,P]

    # The kernel applies inv_scale on-chip. To hit a non-degenerate softmax we
    # pre-divide the INPUTS by std(scores) host-side (equivalent to scaling the
    # effective temperature); the oracle uses the identical effective scale.
    if a.raw:
        ac_dev, bd_dev, oracle_scale = ac, bd, inv_scale
    else:
        bd_sh = rel_shift_host(bd[None])[0]
        std = float(((ac + bd_sh) * inv_scale).std()) + 1e-6
        ac_dev, bd_dev, oracle_scale = ac / std, bd / std, inv_scale / std

    ref = host_probs(ac, bd, oracle_scale)                     # fp32 golden [T,T]

    AC = np.ascontiguousarray(ac_dev, dtype=np.float32).reshape(-1)
    BD = np.ascontiguousarray(bd_dev, dtype=np.float32).reshape(-1)

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}  T={T} P={P} regime={'raw' if a.raw else 'rescaled'}")

    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    kk = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    Pbytes = T * T * 2
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_ac = pyxrt.bo(d, AC.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_bd = pyxrt.bo(d, BD.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_pr = pyxrt.bo(d, Pbytes, pyxrt.bo.host_only, kk.group_id(5))

    bo_instr.write(instr.tobytes(), 0);        bo_instr.sync(TO)
    bo_ac.write(AC.tobytes(), 0);              bo_ac.sync(TO)
    bo_bd.write(BD.tobytes(), 0);              bo_bd.sync(TO)

    def once():
        r = kk(3, bo_instr, instr.size, bo_ac, bo_bd, bo_pr)
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
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (T={T} x T={T})")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}  probs rowsum min/max={rowsum.min():.4f}/{rowsum.max():.4f}")
    print(f"[run] probs[0,:5]={yf[0,:5]}  ref={ref[0,:5]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] relpos_scores_softmax on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
