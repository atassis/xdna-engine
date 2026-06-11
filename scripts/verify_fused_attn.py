#!/usr/bin/env python3
"""Fused Conformer MHSA (attention) sub-block on XDNA2, verified vs ONNX.

GigaAM block-0 self-attention. The four BIG projections (q/k/v/out, all 768x768)
run on the NPU as whole-array (8-col) fused-epilogue matmuls with bias carried by
K-augmentation (Kaug = 768 + 32 = 800) -- the SAME proven pattern as the fused
FFN (scripts/verify_fused_ffn.py / run_npu_mm_silu_wa.py). The cheap glue stays
on the host: LayerNorm, the (unusual) per-head RoPE, head reshape, the small
per-head score/ctx matmuls, softmax, and the residual.

The MHSA recipe (verified fp32 in scripts/block0_numpy.py):
  x   = after_ffn1                               [400,768]
  ln  = LayerNorm_{norm_self_att}(x)             [400,768]   (== ref att_ln)
  RoPE on ln, per head (NH=16, HD=48, rotate_half split at 24), BEFORE q/k proj:
    xr   = ln.reshape(400,1,16,48)
    rope = xr*cos + concat([-xr[...,24:], xr[...,:24]], -1)*sin   (cos/sin [400,1,1,48])
    qk_in = rope.reshape(400,768)
  q = qk_in @ Wq + bq ; k = qk_in @ Wk + bk           <- NPU (rope'd input)
  v = ln    @ Wv + bv                                  <- NPU (PLAIN ln, NOT rope!)
  heads [16,400,48]; scores = (qh @ kh^T)/sqrt(48); softmax; ctx = probs @ vh
  ctx -> [400,768]; out = ctx @ Wout + bout            <- NPU
  x + out  (residual)                                  == ref after_mhsa

NPU vs HOST split:
  NPU  : q, k, v, out projections (4x whole_array 512x800x768 bias xclbin, ONE file)
  HOST : LayerNorm, RoPE, head reshape, scores, softmax, ctx, residual

The per-head score/ctx matmuls are [400,48]x[48,400] and [400,400]x[400,48] -- the
contracted dim is HD=48 (or T=400). Putting those on the NPU is awkward: HD=48 is
not a multiple of the 32-tile and the natural pad is to 64, and there are 16 of
them per side; they are cheap relative to the 768x768 projections, so they (and
softmax) stay on host for now. Documented as future work.

Build the bias xclbin first (no NPU; see run_npu_mm_silu_wa.py header):
  source scripts/iron_env.sh
  MM=$PWD/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
  rm -f $MM/build/mm_32x32x32.o $MM/build/mm_silu_epilogue_32x32x32.o
  make -f $MM/Makefile.silu -C $MM NPU2=1 M=512 K=800 N=768 n_aie_cols=8 no_silu=1 \
       build/final_512x800x768_32x32x32_8c_bias.xclbin

Usage:
  scripts/verify_fused_attn.py --dry      # wiring + numpy correctness, no NPU
  scripts/verify_fused_attn.py            # dispatch on a freed NPU (single-tenant)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

A = "artifacts"
WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
M_PAD, K, D = 512, 768, 768          # padded rows, inner dim, proj out dim (all 768)
NH, HD, T = 16, 48, 400              # heads, head_dim, seq len
f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)
W = lambda k: np.load(f"{A}/weights/{k}.npy")
R = lambda k: np.load(f"{A}/refs/{k}.npy")


def wa_bias(M, K, N, A_real, B_real, bias, dry):
    """One whole-array fused matmul + bias (K-augmented) -> C[M,N] bf16 (f32).

    On --dry, skip the NPU and emulate the device exactly: bf16-round inputs,
    f32 accumulate, bf16-narrow the result (the 'bias' narrow epilogue). Returns
    (C_f32, device_dt_seconds | 0.0)."""
    m = k = n = 32
    Kaug = K + k
    suffix = f"{M}x{Kaug}x{N}_{m}x{k}x{n}_8c_bias"
    xclbin = f"{WA}/final_{suffix}.xclbin"; insts = f"{WA}/insts_{suffix}.txt"
    A_aug = np.zeros((M, Kaug), bfloat16); A_aug[:, :K] = bf(A_real); A_aug[:, K] = bfloat16(1.0)
    B_aug = np.zeros((Kaug, N), bfloat16); B_aug[:K, :] = bf(B_real); B_aug[K, :] = bf(bias)
    if dry:
        for p in (xclbin, insts):
            print(f"[dry]   {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        # device-matched math: bf16 in, f32 accumulate, bf16-narrow out.
        z = A_aug.astype(np.float32) @ B_aug.astype(np.float32)
        return z.astype(bfloat16).astype(np.float32), 0.0
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build the bias xclbin (see header)")
    import pyxrt
    instr = np.fromfile(insts, np.uint32)
    xb = pyxrt.xclbin(xclbin); kname = xb.get_kernels()[0].get_name()
    d = pyxrt.device(0); d.register_xclbin(xb)
    kk = pyxrt.kernel(pyxrt.hw_context(d, xb.get_uuid()), kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    Ab = np.ascontiguousarray(A_aug).view(np.uint16); Bb = np.ascontiguousarray(B_aug).view(np.uint16)
    bi = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    ba = pyxrt.bo(d, Ab.nbytes, pyxrt.bo.host_only, kk.group_id(3))
    bb = pyxrt.bo(d, Bb.nbytes, pyxrt.bo.host_only, kk.group_id(4))
    bc = pyxrt.bo(d, M * N * 2, pyxrt.bo.host_only, kk.group_id(5))
    bt = pyxrt.bo(d, 1, pyxrt.bo.host_only, kk.group_id(6)); btr = pyxrt.bo(d, 4, pyxrt.bo.host_only, kk.group_id(7))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    ba.write(Ab.tobytes(), 0); ba.sync(TO)
    bb.write(Bb.tobytes(), 0); bb.sync(TO)
    t0 = time.perf_counter()
    kk(3, bi, instr.size, ba, bb, bc, bt, btr).wait()
    dt = time.perf_counter() - t0
    bc.sync(FROM)
    C = np.frombuffer(bc.read(M * N * 2, 0), np.uint16).view(bfloat16).reshape(M, N)
    return f32(C), dt


def rel(a, b):
    a, b = f32(a), f32(b)
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="no NPU; device-matched numpy")
    a = ap.parse_args()
    dry = a.dry

    pad = lambda v: (lambda p: (p.__setitem__((slice(0, T)), v), p)[1])(np.zeros((M_PAD, v.shape[1]), np.float32))

    # --- HOST: input + LayerNorm(norm_self_att) ---
    # Reconstruct the attention input from after_ffn1 + the norm_self_att affine,
    # and cross-check against the captured att_ln ref. Use att_ln as canonical.
    x = f32(R("after_ffn1")[0])                            # [400,768]
    g, beta = W("norm_self_att.weight"), W("norm_self_att.bias")
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    ln_recon = (x - mu) / np.sqrt(var + 1e-5) * g + beta
    ln = f32(R("att_ln")[0])                               # canonical LN'd input
    r_ln = rel(ln_recon, ln)

    # --- HOST: RoPE on ln, per head, BEFORE q/k projection ---
    xr = ln.reshape(T, 1, NH, HD)
    cos, sin = f32(R("pos_cos")), f32(R("pos_sin"))        # [400,1,1,48]
    half = HD // 2
    rot = np.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    rope = xr * cos + rot * sin
    r_rope = rel(rope, R("att_rope"))
    qk_in = rope.reshape(T, NH * HD)                       # [400,768]  q/k input

    # --- NPU: q, k, v, out projections (whole_array bias, Kaug=800) ---
    Wq, bq = W("self_attn.linear_q.weight"), W("self_attn.linear_q.bias")
    Wk, bk = W("self_attn.linear_k.weight"), W("self_attn.linear_k.bias")
    Wv, bv = W("self_attn.linear_v.weight"), W("self_attn.linear_v.bias")
    Wo, bo_ = W("self_attn.linear_out.weight"), W("self_attn.linear_out.bias")

    print("[fused MHSA] q/k/v/out = 4x whole_array 512x768x768 bias (Kaug=800); RoPE/score/softmax/ctx/residual host")
    q_pad, tq = wa_bias(M_PAD, K, D, pad(qk_in), Wq, bq, dry); q = q_pad[:T]
    k_pad, tk = wa_bias(M_PAD, K, D, pad(qk_in), Wk, bk, dry); k = k_pad[:T]
    v_pad, tv = wa_bias(M_PAD, K, D, pad(ln),    Wv, bv, dry); v = v_pad[:T]   # PLAIN ln
    r_q, r_k, r_v = rel(q, R("q")[0]), rel(k, R("k")[0]), rel(v, R("v")[0])

    # --- HOST: heads, scores, softmax, ctx ---
    qh = q.reshape(T, NH, HD).transpose(1, 0, 2)
    kh = k.reshape(T, NH, HD).transpose(1, 0, 2)
    vh = v.reshape(T, NH, HD).transpose(1, 0, 2)
    scores = (qh @ kh.transpose(0, 2, 1)) * (1.0 / np.sqrt(HD))    # [16,400,400]
    r_sc = rel(scores[None], R("scores"))
    p = scores - scores.max(-1, keepdims=True)
    p = np.exp(p); p /= p.sum(-1, keepdims=True)
    r_pr = rel(p[None], R("attn_probs"))
    ctx = p @ vh                                                   # [16,400,48]
    r_ctx = rel(ctx[None], R("attn_ctx"))
    ctx2 = ctx.transpose(1, 0, 2).reshape(T, NH * HD)              # [400,768]

    # --- NPU: output projection ---
    out_pad, to = wa_bias(M_PAD, K, D, pad(ctx2), Wo, bo_, dry); out = out_pad[:T]
    r_out = rel(out, R("attn_out")[0])

    # --- HOST: residual ---
    after = bf(x + out)
    r_after = rel(after, R("after_mhsa")[0])

    def line(name, r):
        return f"  {name:18s} rel vs ONNX = {r:.2e}  ({'PASS' if r < 0.05 else 'FAIL'})"

    if not dry:
        print(f"  device time: q={tq*1e3:.3f} k={tk*1e3:.3f} v={tv*1e3:.3f} out={to*1e3:.3f} ms (host: LN/RoPE/score/softmax/ctx/resid)")
    print(line("LN recon (host)", r_ln))   # ln_recon vs att_ln (sanity, host-only)
    print(line("att_rope (host)", r_rope))
    print(line("q (NPU)", r_q))
    print(line("k (NPU)", r_k))
    print(line("v (NPU)", r_v))
    print(line("scores (host)", r_sc))
    print(line("attn_probs (host)", r_pr))
    print(line("attn_ctx (host)", r_ctx))
    print(line("attn_out (NPU)", r_out))
    print(line("after_mhsa", r_after))
    rels = [r_ln, r_rope, r_q, r_k, r_v, r_sc, r_pr, r_ctx, r_out, r_after]
    ok = all(r < 0.05 for r in rels)
    print(f"[fused MHSA] {'PASS' if ok else 'FAIL'} (after_mhsa rel={r_after:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
