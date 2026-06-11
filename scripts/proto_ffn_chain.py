#!/usr/bin/env python3
"""Prototype: keep the FFN intermediate H ON-DEVICE between the two matmul dispatches
(BO-chaining), eliminating the host round-trip of the 3072-wide intermediate — without any
new IRON kernel. mm1 (silu epilogue) writes H (bf16 [M,3072]) to a BO; mm2 (plain matmul)
reads that SAME BO as its A input. Validates (a) a BO can be shared across two whole-array
xclbin/hw-context dispatches, and (b) numerical correctness.

Run on a freed NPU:  .venv-iron/bin/python scripts/proto_ffn_chain.py
"""
import os, time
import numpy as np
from ml_dtypes import bfloat16
import pyxrt

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
M, K, P, N = 512, 768, 3072, 768   # FFN: [M,K]@W1[K,P] -silu-> H[M,P] @ W2[P,N] -> C[M,N]
TILE = "32x32x32"


def silu_bf16(z):
    bf = lambda x: x.astype(bfloat16).astype(np.float32)
    x = bf(z); hx = bf(x * 0.5); th = bf(np.tanh(hx)); s = bf(bf(th + 1.0) * 0.5)
    return bf(x * s)


def main():
    rng = np.random.RandomState(0)
    A = rng.uniform(-1, 1, (400, K)).astype(bfloat16)         # real rows (400), pad to M=512
    W1 = rng.uniform(-1, 1, (K, P)).astype(bfloat16); b1 = rng.uniform(-1, 1, (P,)).astype(bfloat16)
    W2 = rng.uniform(-1, 1, (P, N)).astype(bfloat16); b2 = rng.uniform(-1, 1, (N,)).astype(bfloat16)

    # host reference
    Hf = silu_bf16(A.astype(np.float32) @ W1.astype(np.float32) + b1.astype(np.float32))
    C_ref = Hf @ W2.astype(np.float32) + b2.astype(np.float32)

    d = pyxrt.device(0)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    def load(name):
        xb = pyxrt.xclbin(f"{WA}/{name}"); d.register_xclbin(xb)
        ctx = pyxrt.hw_context(d, xb.get_uuid())
        return pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

    k1 = load(f"final_512x800x3072_{TILE}_8c_silu.xclbin")     # mm1: Kaug=800, N=3072, bf16 out
    k2 = load(f"final_512x3072x768_{TILE}_8c.xclbin")          # mm2: plain, f32 out
    i1 = np.fromfile(f"{WA}/insts_512x800x3072_{TILE}_8c_silu.txt", np.uint32)
    i2 = np.fromfile(f"{WA}/insts_512x3072x768_{TILE}_8c.txt", np.uint32)

    # mm1 K-augmented operands (bias rider)
    Kaug = K + 32
    A_aug = np.zeros((M, Kaug), bfloat16); A_aug[:400, :K] = A; A_aug[:, K] = bfloat16(1.0)
    B1_aug = np.zeros((Kaug, P), bfloat16); B1_aug[:K] = W1; B1_aug[K] = b1
    W2b = np.ascontiguousarray(W2)

    # --- BOs. bo_H is shared: mm1's C (gid5) AND mm2's A (gid3). ---
    bo_i1 = pyxrt.bo(d, i1.nbytes, pyxrt.bo.cacheable, k1.group_id(1))
    bo_a1 = pyxrt.bo(d, A_aug.size * 2, pyxrt.bo.host_only, k1.group_id(3))
    bo_b1 = pyxrt.bo(d, B1_aug.size * 2, pyxrt.bo.host_only, k1.group_id(4))
    bo_H = pyxrt.bo(d, M * P * 2, pyxrt.bo.host_only, k1.group_id(5))   # bf16 [M,P], STAYS on device
    bo_t1 = pyxrt.bo(d, 1, pyxrt.bo.host_only, k1.group_id(6)); bo_tr1 = pyxrt.bo(d, 4, pyxrt.bo.host_only, k1.group_id(7))

    bo_i2 = pyxrt.bo(d, i2.nbytes, pyxrt.bo.cacheable, k2.group_id(1))
    bo_b2 = pyxrt.bo(d, W2b.size * 2, pyxrt.bo.host_only, k2.group_id(4))
    bo_C = pyxrt.bo(d, M * N * 4, pyxrt.bo.host_only, k2.group_id(5))   # f32 [M,N]
    bo_t2 = pyxrt.bo(d, 1, pyxrt.bo.host_only, k2.group_id(6)); bo_tr2 = pyxrt.bo(d, 4, pyxrt.bo.host_only, k2.group_id(7))

    bo_i1.write(i1.tobytes(), 0); bo_i1.sync(TO)
    bo_a1.write(np.ascontiguousarray(A_aug).view(np.uint16).tobytes(), 0); bo_a1.sync(TO)
    bo_b1.write(np.ascontiguousarray(B1_aug).view(np.uint16).tobytes(), 0); bo_b1.sync(TO)
    bo_i2.write(i2.tobytes(), 0); bo_i2.sync(TO)
    bo_b2.write(W2b.view(np.uint16).tobytes(), 0); bo_b2.sync(TO)

    # chained dispatch: mm1 -> bo_H (no host read), mm2 reads bo_H (no host write)
    def chained():
        k1(3, bo_i1, i1.size, bo_a1, bo_b1, bo_H, bo_t1, bo_tr1).wait()
        k2(3, bo_i2, i2.size, bo_H, bo_b2, bo_C, bo_t2, bo_tr2).wait()

    chained()  # warm
    iters = 50; t0 = time.perf_counter()
    for _ in range(iters):
        chained()
    dt = (time.perf_counter() - t0) / iters
    bo_C.sync(FROM)
    C = np.frombuffer(bo_C.read(M * N * 4, 0), np.float32).reshape(M, N)[:400] + b2.astype(np.float32)

    diff = np.abs(C - C_ref)
    rel = diff.max() / (np.abs(C_ref).max() + 1e-9)
    ok = rel < 0.05 and not np.isnan(C).any()
    print(f"[chained] 2 dispatches, H kept on-device. time/iter (both mm) = {dt*1e3:.3f} ms")
    print(f"[chained] C[0,:4]={C[0,:4]}  ref={C_ref[0,:4]}")
    print(f"[chained] max_rel vs host FFN ref = {rel:.3e}  -> {'PASS (BO-chaining works!)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
