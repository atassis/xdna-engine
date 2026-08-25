#!/usr/bin/env python3
"""Numerics probes for the K=768 GELU rail: where does its residual actually come from?

Two experiments, both on the ALREADY-BUILT bricks (neither needs a new xclbin):

  ksplit  Is fc2's single K=3072 collapse less accurate than FfnMm2's four K=768
          partials accumulated in f32 on the host? Isolated by re-running the same
          K=3072 brick with three of A's four K-quarters zeroed and summing on host.
          Exact zeros contribute exactly zero and the surviving quarter's k-blocks are
          untouched, so each masked run IS the partial FfnMm2 would dispatch -- same
          kernel, same bfp16 lowering. Only the association order differs.
          Also checks the resadd brick against a sum bf16 cannot represent.

  bfpfit  The modal GEMM deviates from an f64 matmul of its OWN bf16 inputs, which a
          plain-bf16 model says is impossible. Sweep a shared-exponent block-float
          model over (block size x mantissa bits) to name the format it lowers to.

Run on a freed NPU:  scripts/npu_lock.sh run -- .venv-iron/bin/python scripts/probe_rail_numerics.py both
PAD_M=1536 selects the Whisper-small encoder width instead of BERT's 512.
"""
import os, sys
import numpy as np
from ml_dtypes import bfloat16

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
LN = "mlir-aie/programming_examples/ml/layernorm/build"
PAD_M = int(os.environ.get("PAD_M", 1536))
KRES, DFF = 768, 3072

f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)
GELU = lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# One hw_context per xclbin, held for the process: building it per dispatch would put a
# ~1.25 ms context transition inside every measurement (the mistake that withdrew this
# rail's first timing numbers). These probes score residuals, but the habit is the same.
_DEV, _CTX = None, {}


def dispatch(xclbin, insts, ins, out_shape, out_dtype):
    import pyxrt
    global _DEV
    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    if xclbin not in _CTX:
        if _DEV is None:
            _DEV = pyxrt.device(0)
        xb = pyxrt.xclbin(xclbin)
        _DEV.register_xclbin(xb)
        ctx = pyxrt.hw_context(_DEV, xb.get_uuid())
        _CTX[xclbin] = (ctx, pyxrt.kernel(ctx, xb.get_kernels()[0].get_name()))
    kk = _CTX[xclbin][1]
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    instr = np.fromfile(insts, np.uint32)
    nout = int(np.prod(out_shape)) * np.dtype(out_dtype).itemsize
    bi = pyxrt.bo(_DEV, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bufs = []
    for i, a in enumerate(ins):
        raw = np.ascontiguousarray(a)
        raw = raw.view(np.uint16) if raw.dtype == bfloat16 else raw
        b = pyxrt.bo(_DEV, raw.nbytes, pyxrt.bo.host_only, kk.group_id(3 + i))
        b.write(raw.tobytes(), 0); b.sync(TO)
        bufs.append(b)
    bc = pyxrt.bo(_DEV, nout, pyxrt.bo.host_only, kk.group_id(3 + len(ins)))
    bi.write(instr.tobytes(), 0); bi.sync(TO)
    kk(3, bi, instr.size, *bufs, bc).wait()
    bc.sync(FROM)
    raw = bc.read(nout, 0)
    view = np.frombuffer(raw, np.uint16).view(bfloat16) if out_dtype == bfloat16 \
        else np.frombuffer(raw, out_dtype)
    return view.reshape(out_shape).copy()


def fc2(A_bf, B_bf):
    stem = f"{PAD_M}x{DFF}x{KRES}_64x32x96_8c_modalid"
    return dispatch(f"{WA}/final_{stem}.xclbin", f"{WA}/insts_{stem}.txt",
                    [A_bf, B_bf], (PAD_M, KRES), np.float32)


def rail_inputs(rng):
    """fc2's real regime: the rail's own post-GELU intermediate, cast to bf16."""
    x = f32(rng.standard_normal((PAD_M, KRES)))
    W1 = f32(rng.standard_normal((KRES, DFF)) / np.sqrt(KRES))
    b1 = f32(rng.standard_normal(DFF) * 0.1)
    W2 = f32(rng.standard_normal((DFF, KRES)) / np.sqrt(DFF))
    return bf(GELU(f32(bf(x)) @ f32(bf(W1)) + f32(bf(b1)))), bf(W2)


def probe_ksplit():
    h_bf, W2_bf = rail_inputs(np.random.default_rng(20260825))
    single = fc2(h_bf, W2_bf)
    split4, parts = np.zeros((PAD_M, KRES), np.float32), []
    for i in range(4):
        Am = np.zeros_like(h_bf)
        Am[:, i * KRES:(i + 1) * KRES] = h_bf[:, i * KRES:(i + 1) * KRES]
        parts.append(fc2(Am, W2_bf))
        split4 = split4 + parts[-1]          # host f32 accumulate, exactly FfnMm2
    truth = np.float64(f32(h_bf)) @ np.float64(f32(W2_bf))

    print(f"  fc2 = [{PAD_M},{DFF}] @ [{DFF},{KRES}]")
    print(f"  single K=3072 (the rail)     vs f64 truth : {rel_l2(single, truth):.4e}")
    print(f"  4x K=768 host-f32 (FfnMm2)   vs f64 truth : {rel_l2(split4, truth):.4e}")
    print(f"  single vs split4  (the candidate)         : {rel_l2(single, split4):.4e}")
    print(f"  ratio single/split4 : "
          f"{rel_l2(single, truth) / max(rel_l2(split4, truth), 1e-30):.4f}  (needs ~1.4 to explain the rail)")
    # Without this the four "partials" could be four copies of the same full run.
    print("  control: partial norms = " + " ".join(f"{np.linalg.norm(f32(p)):.1f}" for p in parts)
          + f" ; full = {np.linalg.norm(f32(single)):.1f}")

    # resadd: 1.0 + 2^-12 needs 12 mantissa bits; bf16 has 8 and would return 1.0 exactly.
    a = np.full((PAD_M, KRES), 1.0, np.float32)
    b = np.full((PAD_M, KRES), 2.0**-12, np.float32)
    out = dispatch(f"{LN}/final_resadd_{PAD_M}x{KRES}_s100.xclbin",
                   f"{LN}/insts_resadd_{PAD_M}x{KRES}_s100.txt", [a, b], (PAD_M, KRES), np.float32)
    want = np.float32(1.0) + np.float32(2.0**-12)
    print(f"  resadd 1.0 + 2^-12 : device={float(out[0,0])!r} exact_f32={float(want)!r} "
          f"bf16_would_give={float(f32(bf(want)))!r} all_match={bool(np.all(out == want))}")


def bfp(x, blk, mant, along_last=True):
    """Shared-exponent block float: per block of `blk`, one power-of-two scale + `mant`-bit signed mantissa."""
    x = f32(x) if along_last else np.ascontiguousarray(f32(x).T)
    sh = x.shape
    xr = x.reshape(-1, sh[-1] // blk, blk)
    e = np.max(np.abs(xr), axis=-1, keepdims=True)
    scale = np.exp2(np.ceil(np.log2(np.where(e == 0, 1.0, e))))
    q = np.clip(np.round(xr / scale * (2 ** (mant - 1))), -(2 ** (mant - 1)), 2 ** (mant - 1) - 1)
    q = (q / (2 ** (mant - 1)) * scale).reshape(sh)
    return q if along_last else np.ascontiguousarray(q.T)


def probe_bfpfit():
    rng = np.random.default_rng(20260825)
    h = bf(rng.standard_normal((PAD_M, DFF)))
    W2 = bf(rng.standard_normal((DFF, KRES)) / np.sqrt(DFF))
    c = fc2(h, W2)
    # The reference contains every input bit the kernel saw, so a plain-bf16 model of the
    # kernel predicts a deviation of exactly zero. Whatever is left is the internal format.
    truth = np.float64(f32(h)) @ np.float64(f32(W2))
    print(f"  device vs f64 truth on its OWN bf16 inputs : {rel_l2(c, truth):.4e}")
    print(f"  a plain-bf16 model of the kernel predicts  : 0.0000e+00")
    print(f"  bf16-rounded fraction of the f32 C         : {np.mean(f32(bf(c)) == f32(c)):.4f} "
          f"(1.0 would mean a bf16 drain)")
    print(f"  {'blk':>4} {'mant':>5}   rel(device,model)   rel(model,truth)")
    best = None
    for blk in (4, 8, 16, 32):
        for mant in (5, 6, 7, 8, 9):
            m = np.float64(bfp(h, blk, mant)) @ np.float64(bfp(W2, blk, mant, along_last=False))
            r = rel_l2(c, m)
            best = (r, blk, mant) if best is None or r < best[0] else best
            print(f"  {blk:>4} {mant:>5}   {r:.4e}          {rel_l2(m, truth):.4e}")
    print(f"  BEST FIT: blk={best[1]} mant={best[2]} bits, rel(device,model)={best[0]:.4e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("ksplit", "both"):
        print("== fc2 K-collapse vs FfnMm2's host-f32 K-split ==")
        probe_ksplit()
    if which in ("bfpfit", "both"):
        print("== the modal GEMM's internal format ==")
        probe_bfpfit()
