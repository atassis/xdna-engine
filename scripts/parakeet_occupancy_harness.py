#!/usr/bin/env python3
"""Parakeet encoder per-op occupancy harness -- on-device A/B (brick #8).

THE MEASURE-FIRST GATE (Phase 0 of the Parakeet brick-honoring rebuild, spec
2026-06-28-parakeet-tdt-full-npu-brick-honoring.md). Ranks which encoder GEMM
shapes are COMPUTE-bound (-> mmul/bfp16/int8 tiles pay) vs MOVEMENT/dispatch-bound
(-> the FORMAT/COMPUTE bricks do NOT pay; attack with MOVEMENT bricks instead),
so the rebuild order is data-driven, not guessed.

METHOD = -DDATA_MOVEMENT_ONLY kernel-stub A/B (NOT trace-events; the production
whole_array_iron.py accepts --trace_size but never wires enable_trace, and the
8-col cascade-C-join can't route a trace packet flow -- see perop_trace_measure.py
notes -- so a clean trace path would need editing the shared checkout). The A/B
diffs two byte-identical-dataflow xclbins:
  FULL = the production resident kernel (real bfp16/bf16 MAC datapath)
  STUB = the same xclbin relinked with route_b_kernels/occupancy/mm_movement_stub.cc
         (matmul body elided; objectFIFO DMA + locks + BD chains unchanged)
For each (M=512, K=1024, N) dispatch:
  t_full = movement + dispatch + stall + COMPUTE ;  t_stub = movement+dispatch+stall
  compute_us   = t_full - t_stub
  compute_frac = compute_us / t_full   (near 1 = compute-bound; near 0 = movement-bound)

Goldens (CPU, no NPU; run scripts/parakeet_occupancy_golden.py first) gate the FULL
dispatch at rel-L2 <= 0.08 (proves the timing twin runs a CORRECT GEMM) and assert
the STUB returns ~0 (proves compute was actually elided).

DEVICE discipline: single-tenant NPU; free it first (stop npu-asr/voxd), per the
cascade-FFN runbook. This script only RUNS; the stub xclbin is built CPU-side by
scripts/build_parakeet_occupancy_stub.sh (called by run_parakeet_occupancy.sh).

Run (NPU; after building production + stub xclbins):
  .venv-iron/bin/python scripts/parakeet_occupancy_harness.py [--tile 64x32x128] [--iters 50]
Outputs: artifacts/parakeet/occupancy/occupancy_results.json (ranked table).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
OUTDIR = "artifacts/parakeet/occupancy"
M, K = 512, 1024
N_SHAPES = [1024, 2048, 4096]
GATE_REL = 0.08


def load_golden(n):
    p = f"{OUTDIR}/golden_{M}x{K}x{n}.npz"
    if not os.path.exists(p):
        sys.exit(f"missing golden {p} -- run scripts/parakeet_occupancy_golden.py first")
    d = np.load(p)
    return d["A"], d["B"], d["ref"]  # A,B are uint16 (bf16 bit-views), ref f32


def measure_one(pyxrt, dev, xclbin_path, insts_path, A_u16, B_u16, n, iters):
    """Dispatch one (M,K,N) GEMM `iters` times on `xclbin_path`; return (median_us,
    C[f32]). Mirrors the run_matmul8 ABI (opcode 3) used by src/npu.rs / test.cpp."""
    if not os.path.exists(xclbin_path):
        sys.exit(f"missing xclbin {xclbin_path}")
    if not os.path.exists(insts_path):
        sys.exit(f"missing insts {insts_path}")
    instr = np.fromfile(insts_path, dtype=np.uint32)
    xb = pyxrt.xclbin(xclbin_path)
    kname = xb.get_kernels()[0].get_name()
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_a = pyxrt.bo(dev, A_u16.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_b = pyxrt.bo(dev, B_u16.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_c = pyxrt.bo(dev, M * n * 4, pyxrt.bo.host_only, k.group_id(5))
    bo_tmp = pyxrt.bo(dev, 1, pyxrt.bo.host_only, k.group_id(6))
    bo_tr = pyxrt.bo(dev, 4, pyxrt.bo.host_only, k.group_id(7))
    bo_i.write(instr.tobytes(), 0); bo_i.sync(TO)
    bo_a.write(np.ascontiguousarray(A_u16).tobytes(), 0); bo_a.sync(TO)
    bo_b.write(np.ascontiguousarray(B_u16).tobytes(), 0); bo_b.sync(TO)

    def once():
        k(3, bo_i, instr.size, bo_a, bo_b, bo_c, bo_tmp, bo_tr).wait()

    once()  # warmup
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    ts.sort()
    bo_c.sync(FROM)
    C = np.frombuffer(bo_c.read(M * n * 4, 0), np.float32).reshape(M, n)
    return ts[len(ts) // 2], ts[0], C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="64x32x128",
                    help="64x32x128 (fast BFP16, production default) or 32x32x32 (native bf16)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--shapes", type=int, nargs="*", default=N_SHAPES)
    a = ap.parse_args()
    tile = a.tile
    full_xclbin = f"{WA}/final_512x1024x4096_{tile}_8c.xclbin"
    stub_xclbin = f"{WA}/final_512x1024x4096_{tile}_8c_STUB.xclbin"
    # native tile is accurate (golden-gated); fast bfp16 is lossier per-GEMM (reported only)
    gate = (tile == "32x32x32")

    try:
        import pyxrt
    except Exception as e:
        sys.exit(f"pyxrt import failed ({e}); run inside .venv-iron with the NPU free")
    dev = pyxrt.device(0)

    os.makedirs(OUTDIR, exist_ok=True)
    results = {"tile": tile, "iters": a.iters, "M": M, "K": K,
               "gate_rel": GATE_REL, "golden_gated": gate, "shapes": {}}
    rows = []
    for n in a.shapes:
        A_u16, B_u16, ref = load_golden(n)
        insts = f"{WA}/insts_512x1024x{n}_{tile}_8c.txt"
        print(f"\n=== {M}x{K}x{n} (tile {tile}) ===", flush=True)
        t_full, full_min, C_full = measure_one(pyxrt, dev, full_xclbin, insts, A_u16, B_u16, n, a.iters)
        rel = float(np.linalg.norm(C_full - ref) / (np.linalg.norm(ref) + 1e-9))
        t_stub, stub_min, C_stub = measure_one(pyxrt, dev, stub_xclbin, insts, A_u16, B_u16, n, a.iters)
        stub_absmax = float(np.abs(C_stub).max())

        compute_us = t_full - t_stub
        compute_frac = compute_us / t_full if t_full > 0 else float("nan")
        bound = "compute" if compute_frac >= 0.5 else "movement"
        macs = M * K * n
        gflops = macs / (t_full * 1e-6) / 1e9 if t_full > 0 else 0.0
        ok_full = (rel <= GATE_REL) if gate else True
        ok_stub = stub_absmax < 1e-3  # stub must NOT compute -> C ~ 0
        rec = {
            "M": M, "K": K, "N": n,
            "t_full_us_median": round(t_full, 2), "t_full_us_min": round(full_min, 2),
            "t_stub_us_median": round(t_stub, 2), "t_stub_us_min": round(stub_min, 2),
            "compute_us": round(compute_us, 2),
            "compute_frac": round(compute_frac, 4),
            "movement_frac": round(1 - compute_frac, 4),
            "bound": bound,
            "full_gflops": round(gflops, 1),
            "full_rel_vs_golden": round(rel, 4),
            "full_golden_pass": ok_full,
            "stub_C_absmax": round(stub_absmax, 6),
            "stub_compute_elided": ok_stub,
        }
        results["shapes"][f"{M}x{K}x{n}"] = rec
        rows.append(rec)
        print(f"  full={t_full:.1f}us  stub={t_stub:.1f}us  compute={compute_us:.1f}us "
              f"frac={compute_frac:.2%} -> {bound}  rel={rel:.3e} "
              f"{'PASS' if ok_full else 'FAIL(gate)'}  stub|C|max={stub_absmax:.2e} "
              f"{'OK' if ok_stub else 'WARN(stub computed!)'}", flush=True)

    rows.sort(key=lambda r: r["compute_frac"], reverse=True)
    results["ranked_by_compute_frac"] = [
        {"shape": f"{r['M']}x{r['K']}x{r['N']}", "compute_frac": r["compute_frac"], "bound": r["bound"]}
        for r in rows]
    with open(f"{OUTDIR}/occupancy_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RANKED per-op occupancy (most compute-bound first) ===")
    for r in rows:
        print(f"  {r['M']}x{r['K']}x{r['N']}: compute_frac={r['compute_frac']:.2%} "
              f"-> {r['bound']:8s} ({r['full_gflops']} GFLOP/s, rel={r['full_rel_vs_golden']})")
    bad = [r for r in rows if not r["full_golden_pass"] or not r["stub_compute_elided"]]
    print(f"\nWrote {OUTDIR}/occupancy_results.json")
    if bad:
        print(f"WARNING: {len(bad)} shape(s) failed golden/stub sanity -- see json")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
