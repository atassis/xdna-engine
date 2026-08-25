#!/usr/bin/env python3
"""On-device A/B: the hw aie::tanh<bfloat16> SFU LUT vs a sw tanh on exp2f_vec.

After the conv_even fix, mm_gelu_epilogue_f32o's residual is dominated by aie::tanh
itself (recovered (1+t) 1.126e-2 vs 1.422e-3 for a correctly-rounded bf16). The
sibling SFU op measured the same way -- aie::exp2<bfloat16> -- turned out to be a
coarse piecewise-linear interpolator 720x-5771x worse than the same poly, so this
asks whether tanh is one mechanism with it, in isolation rather than through the
encoder, and whether the software form runs at all in an epilogue-shaped loop.

Scores five columns per domain: hwtanh/swtanh against float64 np.tanh, and the two
GELU chains against the float64 tanh-approx GELU they both target. gelu_hw is the
control -- it must reproduce the shipped epilogue's own error, or the comparison is
measuring this probe rather than the kernel.

Pattern (pyxrt buffer wiring) copied from route_b_kernels/exp2_ab/run_exp2_ab.py.

Usage (the NPU is single-tenant -- serialize against any other on-device work):
  scripts/npu_lock.sh run -- .venv-iron/bin/python route_b_kernels/tanh_ab/run_tanh_ab.py
"""
import argparse, os, sys
import numpy as np

N = 512  # must match -DTANHAB_N (Makefile) and tanh_ab_iron.py's N

# tanh's own domains, plus the one the GELU epilogue actually lives on. The kernel's
# `inner` = c0*(x + c1*x^3) is what reaches tanh, and the recovered-(1+t) error was
# reported as concentrated near inner ~ +/-0.5, so [-1,1] is the band that matters.
DOMAINS = [
    ("[-8,8]", -8.0, 8.0),
    ("[-3,3]", -3.0, 3.0),
    ("[-1,1]", -1.0, 1.0),
    ("[-0.5,0.5]", -0.5, 0.5),
]

C0_EXACT = np.float64(0.7978845608028654)


def gelu_ref(x64):
    """float64 tanh-approx GELU -- the function BOTH chains are approximating."""
    return 0.5 * x64 * (1.0 + np.tanh(C0_EXACT * (x64 + 0.044715 * x64**3)))


def metrics(approx_f64, ref_f64, x_f64, floor):
    """Relative error, with |ref| floored so a zero crossing does not dominate."""
    denom = np.maximum(np.abs(ref_f64), floor)
    rel = np.abs(approx_f64 - ref_f64) / denom
    imax = int(np.argmax(rel))
    l2 = float(np.linalg.norm(approx_f64 - ref_f64) / (np.linalg.norm(ref_f64) + 1e-30))
    return {"max_rel": float(rel[imax]), "argmax_x": float(x_f64[imax]),
            "mean_rel": float(rel.mean()), "rel_l2": l2}


def run_once(kk, bo_instr, instr, bo_in, bo_out, X):
    import pyxrt
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_in.write(X.tobytes(), 0)
    bo_in.sync(TO)
    r = kk(3, bo_instr, instr.size, bo_in, bo_out)
    r.wait()
    bo_out.sync(FROM)
    raw = np.frombuffer(bo_out.read(5 * N * 4, 0), dtype=np.float32)
    return [raw[i * N:(i + 1) * N].copy() for i in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default="mlir-aie/programming_examples/ml/tanh_ab/build/final.xclbin")
    ap.add_argument("--insts", default="mlir-aie/programming_examples/ml/tanh_ab/build/insts.bin")
    ap.add_argument("--dump", default=None, help="save per-domain arrays to this .npz")
    a = ap.parse_args()

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build first (make NPU2=1 in the synced sandbox dir)")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size} N={N}")

    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    kk = pyxrt.kernel(ctx, kname)

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
    bo_in = pyxrt.bo(d, N * 4, pyxrt.bo.host_only, kk.group_id(3))
    bo_out = pyxrt.bo(d, 5 * N * 4, pyxrt.bo.host_only, kk.group_id(4))
    bo_instr.write(instr.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    print(f"\n{'domain':>12}  {'path':>8}  {'max_rel':>10}  {'argmax_x':>9}  {'mean_rel':>10}  {'rel_L2':>10}")
    rows, dump = {}, {}
    for name, lo, hi in DOMAINS:
        x64 = np.linspace(lo, hi, N, dtype=np.float64)
        X = x64.astype(np.float32)
        raw, hwtanh, swtanh, gelu_hw, gelu_sw = run_once(kk, bo_instr, instr, bo_in, bo_out, X)

        if not np.array_equal(raw, X):
            print(f"[WARN] {name}: raw passthrough mismatch on {int((raw != X).sum())}/{N} lanes")

        ref_t, ref_g = np.tanh(x64), gelu_ref(x64)
        # tanh is odd and GELU crosses zero, so floor the denominator at the domain's
        # own scale rather than letting a lane near the crossing set max_rel.
        ft, fg = float(np.abs(ref_t).mean()), float(np.abs(ref_g).mean())
        m = {
            "hwtanh": metrics(hwtanh.astype(np.float64), ref_t, x64, ft),
            "swtanh": metrics(swtanh.astype(np.float64), ref_t, x64, ft),
            "gelu_hw": metrics(gelu_hw.astype(np.float64), ref_g, x64, fg),
            "gelu_sw": metrics(gelu_sw.astype(np.float64), ref_g, x64, fg),
        }
        for tag in ("hwtanh", "swtanh", "gelu_hw", "gelu_sw"):
            v = m[tag]
            print(f"{name:>12}  {tag:>8}  {v['max_rel']:>10.4e}  {v['argmax_x']:>9.4f}"
                  f"  {v['mean_rel']:>10.4e}  {v['rel_l2']:>10.4e}")
        rows[name] = m

        if a.dump:
            key = name.strip("[]").replace(",", "_").replace(".", "p")
            for tag, arr in (("x64", x64), ("raw", raw), ("hwtanh", hwtanh),
                             ("swtanh", swtanh), ("gelu_hw", gelu_hw), ("gelu_sw", gelu_sw)):
                dump[f"{key}_{tag}"] = arr

    if a.dump:
        np.savez(a.dump, **dump)
        print(f"\n[dump] wrote {a.dump}")

    print("\n[verdict] hw SFU tanh vs the sw poly form, and what that buys the GELU epilogue:")
    for name, _, _ in DOMAINS:
        m = rows[name]
        rt = m["hwtanh"]["rel_l2"] / max(m["swtanh"]["rel_l2"], 1e-300)
        rg = m["gelu_hw"]["rel_l2"] / max(m["gelu_sw"]["rel_l2"], 1e-300)
        print(f"  {name:>12}: tanh hw/sw = {rt:>8.1f}x   gelu hw/sw = {rg:>6.2f}x"
              f"   (gelu_hw {m['gelu_hw']['rel_l2']:.3e} -> gelu_sw {m['gelu_sw']['rel_l2']:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
