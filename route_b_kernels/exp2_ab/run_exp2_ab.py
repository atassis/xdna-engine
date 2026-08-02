#!/usr/bin/env python3
"""On-device A/B: hw aie::exp2<bfloat16> LUT vs the relpos_mha sw f32 poly (exp2f_vec).

Re-measures the claim in route_b_kernels/relpos_mha/relpos_mha.cc:117-122 ("hw LUT
~2-4% inaccurate, sw poly ~1e-4 accurate") for an upstream mlir-aie PR. Drives the
exp2_ab xclbin (route_b_kernels/exp2_ab/exp2_ab.cc) once per input domain via
pyxrt, and scores hw2x/sw2x against a float64 numpy 2**x reference (+ the bonus
hwex column, bf16_exp.cc's literal e^x pipeline, against float64 np.exp(x)).

Pattern (pyxrt buffer wiring) copied from scripts/run_npu_relpos_scores.py, reduced
to the exp2_ab probe's 1-in/1-out ABI (group_id(3)=in, group_id(4)=out; group_id(1)
=cacheable instr bo, same as every other route_b probe runner in this repo).

Usage (the NPU is single-tenant -- serialize against any other on-device work):
  .venv-iron/bin/python route_b_kernels/exp2_ab/run_exp2_ab.py \\
      --xclbin mlir-aie/programming_examples/ml/exp2_ab/build/final.xclbin \\
      --insts  mlir-aie/programming_examples/ml/exp2_ab/build/insts.bin
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

N = 1024  # must match -DEXP2AB_N (Makefile) and exp2_ab_iron.py's N

DOMAINS = [
    ("[-100,0]", -100.0, 0.0),
    ("[-10,0]", -10.0, 0.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,10]", 0.0, 10.0),
]


def metrics(approx_f64, ref_f64, x_f64):
    rel = np.abs(approx_f64 - ref_f64) / np.abs(ref_f64)
    imax = int(np.argmax(rel))
    return {
        "max_rel": float(rel[imax]),
        "argmax_x": float(x_f64[imax]),
        "mean_rel": float(rel.mean()),
    }


def run_once(kk, bo_instr, instr, bo_in, bo_out, X):
    TO = __import__("pyxrt").xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = __import__("pyxrt").xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_in.write(X.tobytes(), 0)
    bo_in.sync(TO)
    r = kk(3, bo_instr, instr.size, bo_in, bo_out)
    r.wait()
    bo_out.sync(FROM)
    raw = np.frombuffer(bo_out.read(4 * N * 4, 0), dtype=np.float32)
    return raw[0 * N:1 * N].copy(), raw[1 * N:2 * N].copy(), raw[2 * N:3 * N].copy(), raw[3 * N:4 * N].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default="mlir-aie/programming_examples/ml/exp2_ab/build/final.xclbin")
    ap.add_argument("--insts", default="mlir-aie/programming_examples/ml/exp2_ab/build/insts.bin")
    ap.add_argument("--dump", default=None, help="save per-domain raw/hw2x/sw2x/hwex/x64 arrays to this .npz")
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
    bo_out = pyxrt.bo(d, 4 * N * 4, pyxrt.bo.host_only, kk.group_id(4))
    bo_instr.write(instr.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    print(f"\n{'domain':>10}  {'path':>6}  {'max_rel':>10}  {'argmax_x':>10}  {'mean_rel':>10}")
    all_rows = []
    dump = {}
    for name, lo, hi in DOMAINS:
        x64 = np.linspace(lo, hi, N, dtype=np.float64)
        X = x64.astype(np.float32)

        raw, hw2x, sw2x, hwex = run_once(kk, bo_instr, instr, bo_in, bo_out, X)

        # sanity: raw passthrough must match the input exactly (f32, bit-exact)
        if not np.array_equal(raw, X):
            n_bad = int((raw != X).sum())
            print(f"[WARN] {name}: raw passthrough mismatch on {n_bad}/{N} lanes -- alignment bug?")

        ref2x = np.power(2.0, x64)          # float64 2**x reference
        refex = np.exp(x64)                 # float64 e**x reference (bonus hwex col)

        m_hw = metrics(hw2x.astype(np.float64), ref2x, x64)
        m_sw = metrics(sw2x.astype(np.float64), ref2x, x64)
        m_hwex = metrics(hwex.astype(np.float64), refex, x64)

        for tag, m in (("hw2x", m_hw), ("sw2x", m_sw)):
            print(f"{name:>10}  {tag:>6}  {m['max_rel']:>10.4e}  {m['argmax_x']:>10.4f}  {m['mean_rel']:>10.4e}")
            all_rows.append((name, tag, m))
        print(f"{name:>10}  {'hwex*':>6}  {m_hwex['max_rel']:>10.4e}  {m_hwex['argmax_x']:>10.4f}  {m_hwex['mean_rel']:>10.4e}   (* vs float64 np.exp(x), bonus col)")

        if a.dump:
            key = name.strip("[]").replace(",", "_")
            dump[f"{key}_x64"] = x64
            dump[f"{key}_raw"] = raw
            dump[f"{key}_hw2x"] = hw2x
            dump[f"{key}_sw2x"] = sw2x
            dump[f"{key}_hwex"] = hwex

    if a.dump:
        np.savez(a.dump, **dump)
        print(f"\n[dump] wrote {a.dump}")

    print("\n[verdict] per-domain hw2x vs sw2x max relative error (both vs float64 2**x):")
    for name, _, _ in DOMAINS:
        rows = [r for r in all_rows if r[0] == name]
        hw = [r[2] for r in rows if r[1] == "hw2x"][0]
        sw = [r[2] for r in rows if r[1] == "sw2x"][0]
        ratio = hw["max_rel"] / max(sw["max_rel"], 1e-300)
        print(f"  {name:>10}: hw_max_rel={hw['max_rel']:.4e}  sw_max_rel={sw['max_rel']:.4e}  hw/sw={ratio:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
