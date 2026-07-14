#!/usr/bin/env python3
"""On-device validation of the STEP-4 context matmul: ctx = probs @ V (the AV
half), host-fed probs (mirrors step 1's host-fed scores). Drives probs[T,T] bf16
and V[T,DK] bf16 through relpos_ctx_bake (built STEP=4) on the XDNA2 NPU via pyxrt;
reads ctx[T,DK] bf16 and compares to the fp32 host probs @ V.

probs are the REAL block-0 head-0 attention weights (softmax of rel_shift(BD)+AC,
non-degenerate rescaled regime by default so the weights are not one-hot).

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the fp32 host ctx.
ABI (opcode 3): kernel(op, instr[gid1,cacheable], n, PROBS[gid3], V[gid4], CTX[gid5]).
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


def rel_shift_host(bd):
    Hh, T, P = bd.shape
    x = np.pad(bd, ((0, 0), (0, 0), (1, 0)))
    x = x.reshape(Hh, P + 1, T)
    x = x[:, 1:].reshape(Hh, T, P)
    return x[:, :, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()

    pos = np.asarray(REF("pos_enc"), np.float32).reshape(-1, D)
    x = np.asarray(REF("block_in"), np.float32).reshape(-1, D)
    T = x.shape[0]
    inv_scale = np.float32(1.0 / np.sqrt(DK))
    h = a.head

    q = (x @ W(a.block, "self_attn.linear_q.weight")).reshape(T, H, DK)
    k = (x @ W(a.block, "self_attn.linear_k.weight")).reshape(T, H, DK)
    v = (x @ W(a.block, "self_attn.linear_v.weight")).reshape(T, H, DK)
    pm = (pos @ W(a.block, "self_attn.linear_pos.weight")).reshape(-1, H, DK)
    u = W(a.block, "self_attn.pos_bias_u"); vv = W(a.block, "self_attn.pos_bias_v")

    qu = (q[:, h] + u[h]).astype(np.float32)
    qv = (q[:, h] + vv[h]).astype(np.float32)
    ac = (qu @ k[:, h].astype(np.float32).T)
    bd = (qv @ pm[:, h].astype(np.float32).T)
    bd_sh = rel_shift_host(bd[None])[0]
    scores = (ac + bd_sh) * inv_scale
    if not a.raw:
        scores = scores / (float(scores.std()) + 1e-6)
    scores = scores - scores.max(-1, keepdims=True)
    probs = np.exp(scores); probs /= probs.sum(-1, keepdims=True)  # [T,T] f32
    Vh = v[:, h].astype(np.float32)                                # [T,DK]
    ctx_ref = (probs @ Vh).astype(np.float32)                      # [T,DK] f32 golden

    PROBS = np.ascontiguousarray(probs, np.float32).astype(bfloat16).reshape(-1)
    V = np.ascontiguousarray(Vh, np.float32).astype(bfloat16).reshape(-1)

    import pyxrt
    instr = np.fromfile(a.insts, dtype=np.uint32)
    Pbytes, Vbytes, Cbytes = PROBS.nbytes, V.nbytes, (T * DK * 2)
    d = pyxrt.device(0)
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}  T={T} regime={'raw' if a.raw else 'rescaled'}")
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    kk = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_pr = pyxrt.bo(d, Pbytes, pyxrt.bo.host_only, kk.group_id(3))
    bo_v = pyxrt.bo(d, Vbytes, pyxrt.bo.host_only, kk.group_id(4))
    bo_cx = pyxrt.bo(d, Cbytes, pyxrt.bo.host_only, kk.group_id(5))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_pr.write(PROBS.view(np.uint16).tobytes(), 0); bo_pr.sync(TO)
    bo_v.write(V.view(np.uint16).tobytes(), 0);      bo_v.sync(TO)

    def once():
        r = kk(3, bo_instr, instr.size, bo_pr, bo_v, bo_cx); r.wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_cx.sync(FROM)
    CX = np.frombuffer(bo_cx.read(Cbytes, 0), dtype=np.uint16).view(bfloat16).reshape(T, DK).astype(np.float32)
    a_flat, r_flat = CX.ravel(), ctx_ref.ravel()
    rel_l2 = float(np.linalg.norm(a_flat - r_flat) / (np.linalg.norm(r_flat) + 1e-12))
    corr = float(np.corrcoef(a_flat, r_flat)[0, 1])
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (T={T}; ctx = probs @ V, on-chip AV matmul)")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}")
    print(f"[run] ctx[0,:4]={CX[0,:4]}  ref={ctx_ref[0,:4]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] relpos_ctx (AV matmul) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
