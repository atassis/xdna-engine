#!/usr/bin/env python3
"""rel-L2 gate for the K=768 GELU resident-FFN rail (BERT / Whisper-small / ESM-2).

Runs the whole five-brick rail on device, each brick fed the PREVIOUS brick's device output:

  x f32[512,768] -> cast@768 -> fc1 gelu(x@W1+b1) -> cast@3072 -> fc2 (h@W2) -> +b2 -> resadd(x, .)
     cast_512x768      512x800x3072 modalgelu    cast_512x3072   512x3072x768 modalid   resadd_512x768_s100

Every brick carries a control that must be shown to FIRE, because a residual alone cannot tell a
correct brick from the wrong one:
  * casts  -- bit-exact vs numpy bf16 rounding (an exactness claim, not a tolerance).
  * GEMMs  -- scored against host truth under EVERY epilogue mode (identity / silu / gelu). The mode
              lives in the instruction stream, not the xclbin, so "gelu won by Nx over silu" is the
              only direct evidence the intended stream is the one that ran.
  * resadd -- scored against a + s*b for s in {0.5, 1.0, 2.0}; s100 must win.

Run on a freed NPU:  .venv-iron/bin/python scripts/verify_k768_gelu_rail.py
"""
import os, sys, time
import numpy as np
from ml_dtypes import bfloat16

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
LN = "mlir-aie/programming_examples/ml/layernorm/build"
PAD_M, KRES, DFF = 512, 768, 3072
f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)


def dispatch(xclbin, insts, ins, out_shape, out_dtype):
    """One NPU dispatch. `ins` are already-typed arrays in runtime-sequence arg order."""
    import pyxrt

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    instr = np.fromfile(insts, np.uint32)
    xb = pyxrt.xclbin(xclbin)
    d = pyxrt.device(0)
    d.register_xclbin(xb)
    kk = pyxrt.kernel(pyxrt.hw_context(d, xb.get_uuid()), xb.get_kernels()[0].get_name())
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    nout = int(np.prod(out_shape)) * np.dtype(out_dtype).itemsize
    bi = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bufs = []
    for i, a in enumerate(ins):
        raw = np.ascontiguousarray(a)
        raw = raw.view(np.uint16) if raw.dtype == bfloat16 else raw
        b = pyxrt.bo(d, raw.nbytes, pyxrt.bo.host_only, kk.group_id(3 + i))
        b.write(raw.tobytes(), 0); b.sync(TO)
        bufs.append(b)
    bc = pyxrt.bo(d, nout, pyxrt.bo.host_only, kk.group_id(3 + len(ins)))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    t0 = time.perf_counter()
    kk(3, bi, instr.size, *bufs, bc).wait()
    dt = time.perf_counter() - t0
    bc.sync(FROM)
    raw = bc.read(nout, 0)
    view = np.frombuffer(raw, np.uint16).view(bfloat16) if out_dtype == bfloat16 \
        else np.frombuffer(raw, out_dtype)
    return view.reshape(out_shape).copy(), dt


def wa_modal(suffix, M, K, N, A_bf, B_bf):
    """whole_array modal GEMM: bf16 A[M,K] @ bf16 B[K,N] -> f32 C[M,N]."""
    return dispatch(f"{WA}/final_{suffix}.xclbin", f"{WA}/insts_{suffix}.txt",
                    [A_bf, B_bf], (M, N), np.float32)


def ln_cast(rows, cols, x_f32):
    """layernorm-family cast brick: f32 -> bf16, elementwise."""
    return dispatch(f"{LN}/final_cast_{rows}x{cols}.xclbin", f"{LN}/insts_cast_{rows}x{cols}.txt",
                    [f32(x_f32)], (rows, cols), bfloat16)


def ln_resadd(rows, cols, stag, a_f32, b_f32):
    """residual-add brick: out = a + scale*b, f32; scale baked per xclbin (stag 100 = 1.0)."""
    return dispatch(f"{LN}/final_resadd_{rows}x{cols}_s{stag}.xclbin",
                    f"{LN}/insts_resadd_{rows}x{cols}_s{stag}.txt",
                    [f32(a_f32), f32(b_f32)], (rows, cols), np.float32)


def rel_l2(a, b):
    a, b = f32(a), f32(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


GELU = lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
SILU = lambda x: x / (1.0 + np.exp(-np.clip(x, -80, 80)))
MODES = {"identity": lambda x: x, "silu": SILU, "gelu": GELU}


def score(name, dev, cands, expect):
    """Score a device output against every candidate host truth; `expect` must win."""
    r = {k: rel_l2(dev, v) for k, v in cands.items()}
    best = min(r, key=r.get)
    runner = min((k for k in r if k != expect), key=lambda k: r[k])
    sep = r[runner] / max(r[expect], 1e-30)
    print(f"  {name}: " + "  ".join(f"{k}={r[k]:.3e}" for k in cands))
    print(f"    -> best={best} (expected {expect}) {'OK' if best == expect else 'MISMATCH'}, "
          f"{sep:.1f}x clear of {runner}")
    return best == expect, r[expect], sep


def main():
    rng = np.random.default_rng(20260825)
    # unit-scale inputs: a transformer FFN sees LN'd activations, so this is the real regime
    x = f32(rng.standard_normal((PAD_M, KRES)))
    W1 = f32(rng.standard_normal((KRES, DFF)) / np.sqrt(KRES))
    b1 = f32(rng.standard_normal(DFF) * 0.1)
    W2 = f32(rng.standard_normal((DFF, KRES)) / np.sqrt(DFF))
    b2 = f32(rng.standard_normal(KRES) * 0.1)

    ok, T = [], {}

    # --- cast@768: f32 activations -> bf16 for fc1's A input ---
    xb_dev, T["cast768"] = ln_cast(PAD_M, KRES, x)
    exact768 = np.array_equal(np.asarray(xb_dev).view(np.uint16), np.asarray(bf(x)).view(np.uint16))
    print(f"  cast@768   : bit-exact vs numpy bf16 = {exact768}")
    ok.append(exact768)

    # --- fc1: modalgelu, bias folded via K-aug (K_real=768 + one k=32 block -> K_aug=800) ---
    Kaug = KRES + 32
    A1 = np.zeros((PAD_M, Kaug), bfloat16); A1[:, :KRES] = xb_dev; A1[:, KRES] = bfloat16(1.0)
    B1 = np.zeros((Kaug, DFF), bfloat16); B1[:KRES, :] = bf(W1); B1[KRES, :] = bf(b1)
    h_dev, T["fc1"] = wa_modal(f"{PAD_M}x{Kaug}x{DFF}_64x32x128_8c_modalgelu", PAD_M, Kaug, DFF, A1, B1)
    pre1 = f32(xb_dev) @ f32(bf(W1)) + f32(bf(b1))
    o, r1, s1 = score("fc1 gelu   ", h_dev, {m: bf(f(pre1)) for m, f in MODES.items()}, "gelu")
    ok.append(o)

    # --- cast@3072: fc1's f32 output -> bf16 for fc2's A input ---
    hb_dev, T["cast3072"] = ln_cast(PAD_M, DFF, h_dev)
    exact3072 = np.array_equal(np.asarray(hb_dev).view(np.uint16),
                               np.asarray(bf(h_dev)).view(np.uint16))
    print(f"  cast@3072  : bit-exact vs numpy bf16 = {exact3072}")
    ok.append(exact3072)

    # --- fc2: K-collapse, identity epilogue; bias rides outside it, added host-side ---
    B2 = np.zeros((DFF, KRES), bfloat16); B2[:] = bf(W2)
    l2_dev, T["fc2"] = wa_modal(f"{PAD_M}x{DFF}x{KRES}_64x32x96_8c_modalid", PAD_M, DFF, KRES, hb_dev, B2)
    pre2 = f32(hb_dev) @ f32(bf(W2))
    o, r2, s2 = score("fc2 identity", l2_dev, {m: bf(f(pre2)) for m, f in MODES.items()}, "identity")
    ok.append(o)

    # --- resadd s100: full residual, out = x + 1.0*(fc2 + b2) ---
    ffn = f32(l2_dev) + b2
    out_dev, T["resadd"] = ln_resadd(PAD_M, KRES, "100", x, ffn)
    o, r3, s3 = score("resadd s100 ", out_dev,
                      {f"s{s}": x + s * ffn for s in (0.5, 1.0, 2.0)}, "s1.0")
    ok.append(o)

    print(f"  device: " + "  ".join(f"{k}={v*1e3:.2f}ms" for k, v in T.items())
          + f"   rail total={sum(T.values())*1e3:.2f}ms")
    print(f"  block out [{PAD_M},{KRES}] finite={np.isfinite(out_dev).all()} "
          f"absmax={np.abs(out_dev).max():.4f} std={out_dev.std():.4f}")

    # bf16 in / f32 accumulate over K=768 and K=3072; the gate is rel-L2, not the chaotic WER
    TOL = 5e-2
    passed = all(ok) and max(r1, r2, r3) < TOL and np.isfinite(out_dev).all()
    print(f"  GATE (5 bricks, rel-L2 < {TOL:g}, every control fired): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
