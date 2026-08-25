#!/usr/bin/env python3
"""Isolate the on-chip GELU epilogue's own precision, with the GEMM error cancelled exactly.

The rail's ~1.44x excess residual was attributed to the on-chip GELU (arm B' reproduced it in
full with no rail present). What was never measured is the GELU ALONE: every previous number
scored gelu(A@B) against a host reference, so the bfp16 mmul's error rode along.

The isolation needs no new xclbin. rtp[0] selects the epilogue mode and is baked into the
INSTRUCTION STREAM, not the ELF -- 32 words, one per core. Patch them 2 (gelu) -> 0 (identity)
and the same xclbin, same A, same B, same mmul emits its raw f32 accumulator. Identity mode is a
pure f32 copy (mm_identity_epilogue_f32o), so that output IS bit-exactly the `pC_in` the GELU
epilogue reads. Scoring the gelu run against a host GELU *of that array* cancels the GEMM
completely, leaving only the epilogue.

Then a host replay of the kernel's exact chain (mm_silu_epilogue.cc mm_gelu_epilogue_f32o)
splits the residual into "the bf16 chain, as designed" vs "aie::tanh's own deviation", and a
per-narrow ablation names which single narrow costs the most -- i.e. what a fix has to change.

Run on a freed NPU:
  scripts/npu_lock.sh run -- .venv-iron/bin/python scripts/probe_gelu_epilogue.py
PAD_M selects the width (1536 = Whisper-small encoder, 512 = BERT).
"""
import os
import sys
import tempfile

import numpy as np
from ml_dtypes import bfloat16

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
PAD_M = int(os.environ.get("PAD_M", 1536))
K_REAL, K_AUG, DFF = 768, 800, 3072

f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)
bfr = lambda x: f32(bf(x))  # round-to-nearest-even through bf16, stay in f32 arithmetic
# Round-toward-zero to bf16: drop the low 16 mantissa bits. IEEE is sign-magnitude, so masking
# truncates the MAGNITUDE for both signs. This is the AIE hardware DEFAULT -- mm_silu_epilogue.cc
# only asks for conv_even under MODAL_ROUND_EVEN, and the f32-out GELU epilogue never does.
bft = lambda x: (f32(x).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)

# The kernel's constants are bf16 broadcasts, so the host replay must round them too.
C0 = float(bfr(0.7978845608))  # sqrt(2/pi)
C1 = float(bfr(0.044715))

GELU_F32 = lambda x: 0.5 * x * (1.0 + np.tanh(C0_EXACT * (x + 0.044715 * x**3)))
C0_EXACT = np.float64(0.7978845608028654)


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --- device ------------------------------------------------------------------------------

_DEV, _CTX = None, {}


def dispatch(xclbin, insts, ins, out_shape, out_dtype):
    """One hw_context per xclbin, held for the process (see probe_rail_numerics.dispatch)."""
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


# rtp_write lowers to one 6-word transaction per core: [.., addr, 0, VALUE, 24, ..]. The 32 mode
# words are the leading run, at stride 6 from word 8. Every field is asserted because a silently
# mis-located patch would still run and would read as a finished result.
RTP_IDX = np.arange(8, 8 + 6 * 32, 6)


def patched_insts(path, mode):
    w = np.fromfile(path, np.uint32)
    if not (np.all(w[RTP_IDX] == 2) and np.all(w[RTP_IDX + 1] == 24)
            and len(set(w[RTP_IDX - 2].tolist())) == 32):
        sys.exit(f"rtp mode words not where expected in {path} -- refusing to patch blind")
    w[RTP_IDX] = mode
    fd, out = tempfile.mkstemp(suffix=".txt", prefix=f"insts_mode{mode}_")
    os.close(fd)
    w.tofile(out)
    return out


def fc1(A_bf, B_bf, mode):
    stem = f"{PAD_M}x{K_AUG}x{DFF}_64x32x128_8c_modalgelu"
    insts = f"{WA}/insts_{stem}.txt"
    if mode != 2:
        insts = patched_insts(insts, mode)
    return dispatch(f"{WA}/final_{stem}.xclbin", insts, [A_bf, B_bf], (PAD_M, DFF), np.float32)


# --- host replay of mm_gelu_epilogue_f32o ------------------------------------------------

def gelu_chain(accf, *, narrow_x=True, cube=True, tanh_out=True, tail=True, rnd=None):
    """The kernel's exact chain. Each flag=False lifts ONE narrow to f32 (the ablation)."""
    bfr = rnd or globals()["bfr"]
    r = bfr if narrow_x else f32
    xv = r(accf)
    rc = bfr if cube else f32
    x2 = rc(xv * xv)
    x3 = rc(x2 * xv)
    c1x3 = rc(C1 * x3)
    inner_b = rc(xv + c1x3)
    # aie::mul(c0, inner_b) yields an accum consumed as f32: the tanh ARGUMENT is not narrowed.
    inner = f32(C0) * f32(inner_b)
    t = np.tanh(np.float64(inner))
    t = bfr(t) if tanh_out else f32(t)
    rt = bfr if tail else f32
    return f32(rt(0.5 * rt(xv * rt(t + 1.0))))


def gelu_sigmoid_form(accf):
    """1 + tanh(z) == 2*sigmoid(2z). Same cube, same one transcendental, no (1+t) cancellation:
    sigmoid underflows toward 0 with full relative precision where 1+tanh loses all of it."""
    xv = bfr(accf)
    x3 = bfr(bfr(xv * xv) * xv)
    inner_b = bfr(xv + bfr(C1 * x3))
    inner = f32(C0) * f32(inner_b)
    s = 1.0 / (1.0 + np.exp(-2.0 * np.float64(inner)))
    return f32(bfr(xv * bfr(s)))


def main():
    rng = np.random.default_rng(20260825)
    # A@B ~ N(0,1) puts x across GELU's whole interesting range, where both tails matter.
    A = np.zeros((PAD_M, K_AUG), np.float32)
    A[:, :K_REAL] = rng.standard_normal((PAD_M, K_REAL))
    B = np.zeros((K_AUG, DFF), np.float32)
    B[:K_REAL, :] = rng.standard_normal((K_REAL, DFF)) / np.sqrt(K_REAL)
    A_bf, B_bf = bf(A), bf(B)

    cache = os.environ.get("GELU_CACHE", f"/tmp/gelu_epilogue_{PAD_M}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        c_gelu, c_id = z["c_gelu"], z["c_id"]
        print(f"[cache] device arrays from {cache} (rm it to re-dispatch)")
    else:
        c_gelu = fc1(A_bf, B_bf, 2)
        c_id = fc1(A_bf, B_bf, 0)
        np.savez(cache, c_gelu=c_gelu, c_id=c_id)

    x = f32(c_id)
    want = GELU_F32(np.float64(x))

    print(f"== on-chip GELU epilogue, isolated (PAD_M={PAD_M}, N={DFF}) ==")
    # Controls: identity must BE the matmul, and gelu must not be identity. Without both, a
    # failed patch and a working one look the same.
    host_mm = np.float64(f32(A_bf)) @ np.float64(f32(B_bf))
    print(f"  control: identity-mode vs host f64 matmul : {rel_l2(c_id, host_mm):.4e}  (bfp16 mmul)")
    print(f"  control: gelu-mode vs identity-mode       : {rel_l2(c_gelu, c_id):.4e}  (must be O(1))")
    print(f"  control: x range [{x.min():.2f}, {x.max():.2f}], frac x<0 = {np.mean(x < 0):.3f}")

    dev = rel_l2(c_gelu, want)
    print(f"\n  DEVICE gelu vs f32 GELU of its own input  : {dev:.4e}   <-- the epilogue alone")

    model = gelu_chain(x)
    print(f"  host replay of the kernel chain           : {rel_l2(model, want):.4e}")
    print(f"  device vs that replay                     : {rel_l2(c_gelu, model):.4e}   <-- aie::tanh's own")

    print("\n  ablation: lift ONE narrow to f32 (rel-L2 vs f32 GELU, lower is better)")
    for name, kw in (("full bf16 chain (as shipped)", {}),
                     ("  x input kept f32", dict(narrow_x=False)),
                     ("  cube chain kept f32", dict(cube=False)),
                     ("  tanh OUTPUT kept f32", dict(tanh_out=False)),
                     ("  tail (1+t, x*, 0.5*) f32", dict(tail=False)),
                     ("  everything f32 but x narrow", dict(cube=False, tanh_out=False, tail=False))):
        print(f"    {name:<32} {rel_l2(gelu_chain(x, **kw), want):.4e}")
    print(f"    {'  sigmoid form, still bf16':<32} {rel_l2(gelu_sigmoid_form(x), want):.4e}")

    # The chain under each bf16 rounding mode, scored against the DEVICE. Whichever the silicon
    # uses should sit near the device; the other should not. This is what separates "aie::tanh is
    # inaccurate" from "every narrow truncates".
    print("\n  which rounding mode is the silicon using? (rel-L2 vs DEVICE, lower = that one)")
    for name, rnd in (("round-to-nearest-even", bfr), ("truncate toward zero", bft)):
        print(f"    {name:<32} {rel_l2(gelu_chain(x, rnd=rnd), c_gelu):.4e}"
              f"   (vs f32 GELU: {rel_l2(gelu_chain(x, rnd=rnd), want):.4e})")

    # Where the error lives. If it is the (1+t) cancellation, it concentrates in the negative
    # tail and scales with |x|; a uniform profile would mean tanh's approximation instead.
    print("\n  error by input band (mean |device - f32 GELU|):")
    edges = [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        e = np.abs(np.float64(c_gelu)[m] - want[m])
        print(f"    x in [{lo:>5},{hi:>5})  n={m.sum():>9}  mean|err|={e.mean():.3e}  max={e.max():.3e}")

    probe_tanh(x, c_gelu)




def probe_tanh(x, c_gelu):
    """Recover aie::tanh's actual output from the device result and score it against exact tanh.

    The tail is invertible: 0.5*xt is exact in bf16 (a power of two), so xt == 2*gx exactly, and
    xt = bf16(xv * t_p1) recovers t_p1 to within one bf16 rounding of xt. t_p1 is itself bf16, so
    rounding the quotient back to bf16 lands on the value the kernel actually held.
    """
    xv = bfr(x)
    inner = f32(C0) * f32(bfr(xv + bfr(C1 * bfr(bfr(xv * xv) * xv))))
    t_exact = np.tanh(np.float64(inner))
    t_p1_want = bfr(t_exact + 1.0)           # what a correctly-rounded bf16 tanh would give
    t_p1_dev = bfr(2.0 * np.float64(c_gelu) / np.where(xv == 0, np.nan, xv))

    ok = np.isfinite(t_p1_dev) & (np.abs(xv) > 0.25)   # small |x| makes the division ill-conditioned
    print("\n  aie::tanh recovered from the device (1+t, binned by tanh argument):")
    print(f"    {'inner':>14}  {'n':>9}  {'mean(1+t) dev':>14} {'exact':>10} {'delta':>10}  {'sat@2':>7}")
    edges = [-np.inf, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (inner >= lo) & (inner < hi)
        if m.sum() < 50:
            continue
        d, w = t_p1_dev[m], (t_exact + 1.0)[m]
        print(f"    [{lo:>5},{hi:>5})  {m.sum():>9}  {d.mean():>14.5f} {w.mean():>10.5f} "
              f"{(d - w).mean():>10.2e}  {np.mean(d >= 2.0):>7.3f}")

    sat = ok & (t_p1_dev >= 2.0)
    print(f"    saturated to exactly 1+t=2: {np.mean(sat[ok]):.4f} of samples; "
          f"of those, exact 1+t < 1.999 in {np.mean((t_exact + 1.0)[sat] < 1.999):.4f}")
    print(f"    rel-L2 of recovered (1+t) vs exact : {rel_l2(t_p1_dev[ok], (t_exact + 1.0)[ok]):.4e}")
    print(f"    rel-L2 of correctly-rounded bf16   : {rel_l2(t_p1_want[ok], (t_exact + 1.0)[ok]):.4e}")


if __name__ == "__main__":
    main()
