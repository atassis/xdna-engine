#!/usr/bin/env python3
# golden_subsample.py -- numpy GOLDEN for the Parakeet conv2D /8 subsample front-end
# reformulated as im2col -> mmul GEMM (patch-embed idiom) with a fused ReLU epilogue.
#
# WHAT THIS PROVES (CPU-only, no NPU):
#   (1) FORMULATION: every conv2d in the dw_striding subsample is exactly an
#       im2col (gather receptive-field patches) -> GEMM (A[M,K] @ B[K,Cout]) with
#       a per-Cout bias and (for conv.0/conv.3/conv.6) a fused ReLU activation.
#       The pointwise (1x1) convs are a degenerate im2col (K = Cin) = a pure GEMM.
#       The depthwise (k=3, groups=C) convs are im2col with a BLOCK-DIAGONAL weight
#       (each Cout sees only its own Cin block) -- still a GEMM; the on-device hot
#       path for these is `sliding_mul` (task A1), here we only need the chain.
#       Gate: f64 im2col-GEMM chain == reference subsample()  (rel < 1e-6).
#   (2) END-TO-END vs the host reference target block_in [Tp,1024]:
#       f64 chain + out-projection rel vs block_in (matches the reference oracle).
#   (3) ON-DEVICE PRECISION: the same chain with bf16 inputs/weights and an f32
#       mmul accumulator (what the AIE2P mmul brick computes) -> rel-L2 vs block_in,
#       gated <= 0.08 (the node golden gate). This is the number the kernel honors.
#
# Reference: scripts/parakeet_ref_encoder.py (subsample / subsample_flatten),
#            artifacts/parakeet/encoder/{pre_encode,refs}.

import os
import numpy as np

try:
    from ml_dtypes import bfloat16
    HAVE_BF16 = True
except Exception:
    HAVE_BF16 = False

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENC = os.path.join(REPO, "artifacts", "parakeet", "encoder")
TOL = 0.08  # node golden gate (rel-L2)


def PE(name):
    return np.load(f"{ENC}/pre_encode/{name}.npy").astype(np.float64)


def REF(name):
    return np.load(f"{ENC}/refs/{name}.npy").astype(np.float64)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a.ravel() - b.ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def bf16(x):
    """Round to bf16 then back to f64 (simulate the AIE2P bf16 operand format)."""
    if not HAVE_BF16:
        return np.asarray(x, np.float64)
    return np.asarray(x, np.float64).astype(bfloat16).astype(np.float64)


# ---------------- reference conv2d (copied from parakeet_ref_encoder) ----------------
def conv2d_ref(x, w, b, stride, pad, groups):
    Ci, Hin, Win = x.shape
    Co, Cig, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hout = (Hin + 2 * pad - kh) // stride + 1
    Wout = (Win + 2 * pad - kw) // stride + 1
    out = np.zeros((Co, Hout, Wout), np.float64)
    gci = Ci // groups; gco = Co // groups
    for g in range(groups):
        for oc in range(gco):
            co = g * gco + oc
            acc = np.full((Hout, Wout), b[co], np.float64)
            for ic in range(Cig):
                ci = g * gci + ic
                ker = w[co, ic]
                for i in range(kh):
                    for j in range(kw):
                        sub = xp[ci, i:i + stride * Hout:stride, j:j + stride * Wout:stride]
                        acc += ker[i, j] * sub
            out[co] = acc
    return out


# ---------------- im2col -> GEMM (patch-embed idiom) ----------------
def im2col(x, kh, kw, stride, pad):
    """x [Ci,Hin,Win] -> A [Hout*Wout, Ci*kh*kw] (row-major patches, Cin-major)."""
    Ci, Hin, Win = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hout = (Hin + 2 * pad - kh) // stride + 1
    Wout = (Win + 2 * pad - kw) // stride + 1
    cols = np.empty((Hout * Wout, Ci * kh * kw), np.float64)
    for oh in range(Hout):
        for ow in range(Wout):
            patch = xp[:, oh * stride:oh * stride + kh, ow * stride:ow * stride + kw]  # [Ci,kh,kw]
            cols[oh * Wout + ow] = patch.reshape(-1)
    return cols, Hout, Wout


def conv2d_gemm(x, w, b, stride, pad, groups, relu=False, cast=None):
    """conv2d expressed as im2col -> GEMM with optional fused ReLU.

    cast: None (f64) or a fn applied to A and B to simulate operand precision
          (the f32 mmul accumulate is kept in f64 -- bf16 only narrows operands).
    Returns [Co, Hout, Wout] to match conv2d_ref.
    """
    Ci, Hin, Win = x.shape
    Co, Cig, kh, kw = w.shape
    A, Hout, Wout = im2col(x, kh, kw, stride, pad)          # [M, Ci*kh*kw]
    K = Ci * kh * kw
    if groups == 1:
        # dense patch-embed weight: [Co, Ci, kh, kw] -> [Ci*kh*kw, Co]
        B = w.reshape(Co, K).T.copy()                       # [K, Co]
    else:
        # depthwise: block-diagonal weight; column co only touches its own Cin block.
        # A is laid out Cin-major (im2col groups the kh*kw of each channel together),
        # so channel c occupies rows [c*kh*kw : (c+1)*kh*kw] of the K axis.
        assert groups == Ci == Co and Cig == 1
        B = np.zeros((K, Co), np.float64)
        ksz = kh * kw
        for c in range(Co):
            B[c * ksz:(c + 1) * ksz, c] = w[c, 0].reshape(-1)
    if cast is not None:
        A = cast(A); B = cast(B)
    C = A @ B + b[None, :]                                  # [M, Co]  (f32 accumulate)
    if relu:
        C = np.maximum(C, 0.0)                              # fused activation epilogue
    # [M=Hout*Wout, Co] -> [Co, Hout, Wout]
    return C.reshape(Hout, Wout, Co).transpose(2, 0, 1)


# ---------------- full subsample as a chain of im2col->GEMM ----------------
def subsample_gemm(audio, cast=None):
    x = audio[0].T[None]  # [1, T(time), 128(freq)]
    x = conv2d_gemm(x, PE("conv.0.weight"), PE("conv.0.bias"), 2, 1, 1, relu=True, cast=cast)
    x = conv2d_gemm(x, PE("conv.2.weight"), PE("conv.2.bias"), 2, 1, 256, relu=False, cast=cast)
    x = conv2d_gemm(x, PE("conv.3.weight"), PE("conv.3.bias"), 1, 0, 1, relu=True, cast=cast)
    x = conv2d_gemm(x, PE("conv.5.weight"), PE("conv.5.bias"), 2, 1, 256, relu=False, cast=cast)
    x = conv2d_gemm(x, PE("conv.6.weight"), PE("conv.6.bias"), 1, 0, 1, relu=True, cast=cast)
    return x  # [256, Tp, 16]


def out_projection(x, cast=None):
    # x [C=256, H=time=Tp, W=freq=16]; ONNX Transpose [0,2,1,3] -> [H, C, W] -> reshape [Tp, 4096]
    C, Hh, Wf = x.shape
    flat = np.transpose(x, (1, 0, 2)).reshape(Hh, C * Wf)   # [Tp, 4096]
    Wout = PE("out.weight"); Bout = PE("out.bias")          # [4096,1024], [1024]
    if cast is not None:
        flat = cast(flat); Wout = cast(Wout)
    return flat @ Wout + Bout[None, :]                      # [Tp, 1024]


def subsample_ref_chain(audio):
    x = audio[0].T[None]
    x = conv2d_ref(x, PE("conv.0.weight"), PE("conv.0.bias"), 2, 1, 1); x = np.maximum(x, 0)
    x = conv2d_ref(x, PE("conv.2.weight"), PE("conv.2.bias"), 2, 1, 256)
    x = conv2d_ref(x, PE("conv.3.weight"), PE("conv.3.bias"), 1, 0, 1); x = np.maximum(x, 0)
    x = conv2d_ref(x, PE("conv.5.weight"), PE("conv.5.bias"), 2, 1, 256)
    x = conv2d_ref(x, PE("conv.6.weight"), PE("conv.6.bias"), 1, 0, 1); x = np.maximum(x, 0)
    return x


def main():
    audio = REF("audio_signal")          # [1, 128, T]
    block_in = REF("block_in")[0]        # [Tp, 1024]
    fails = []

    # --- gate (1): im2col->GEMM formulation == reference conv2d, layer by layer ---
    ref = subsample_ref_chain(audio)
    gem = subsample_gemm(audio, cast=None)
    r_form = rel(gem, ref)
    ok1 = r_form < 1e-6
    print(f"[gate1] im2col->GEMM chain vs reference conv2d (f64): rel={r_form:.2e}  "
          f"{'OK' if ok1 else 'FAIL'}")
    if not ok1:
        fails.append("formulation")

    # --- gate (2): f64 end-to-end (incl out-proj) vs block_in ---
    out_f64 = out_projection(gem, cast=None)
    r_f64 = rel(out_f64, block_in)
    ok2 = r_f64 <= TOL
    print(f"[gate2] f64 im2col->GEMM + out-proj vs block_in: rel={r_f64:.2e}  "
          f"{'OK' if ok2 else 'FAIL'} (tol {TOL})")
    if not ok2:
        fails.append("e2e-f64")

    # --- gate (3): bf16-operand / f32-accumulate (the AIE2P mmul brick) vs block_in ---
    if HAVE_BF16:
        gem_bf = subsample_gemm(audio, cast=bf16)
        out_bf = out_projection(gem_bf, cast=bf16)
        r_bf = rel(out_bf, block_in)
        ok3 = r_bf <= TOL
        print(f"[gate3] bf16-mmul (operand bf16, acc f32) + out-proj vs block_in: "
              f"rel={r_bf:.2e}  {'OK' if ok3 else 'FAIL'} (tol {TOL})")
        if not ok3:
            fails.append("e2e-bf16")
    else:
        print("[gate3] SKIP (ml_dtypes not available)")

    print(f"shapes: subsample={gem.shape} out={out_f64.shape} block_in={block_in.shape}")
    if fails:
        print(f"RESULT: FAIL ({', '.join(fails)})")
        raise SystemExit(1)
    print("RESULT: PASS (im2col->mmul subsample golden gated <= 0.08)")


if __name__ == "__main__":
    main()
