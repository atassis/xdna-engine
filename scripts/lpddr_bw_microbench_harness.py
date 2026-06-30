#!/usr/bin/env python3
"""LPDDR bandwidth microbenchmark harness -- dispatch the pure-DMA xclbins built from
route_b_kernels/lpddr_bw/lpddr_bw_microbench.py over a transfer-size sweep, time the
median, and regress  t = c0_fixed + bytes / BW_achievable  to extract the silicon's
achievable LPDDR bandwidth (the number the KB has only as a ~120 GB/s datasheet figure;
optimization-map open gap #2, hw-envelope). See the design header in the generator and
the launcher internal notes.

WHAT IT MEASURES (per mode, from the generator):
  read  : pure L3->L1.  moved_bytes = bytes
  write : pure L1->L3.  moved_bytes = bytes
  rdwr  : L3->L1->L3 round-trip (concurrent). moved_bytes = 2*bytes (aggregate)
The regression slope (us per moved-MB) inverts to GB/s; the intercept is the byte-
independent dispatch/launch floor (compare to the GEMM's ~91 us fixed floor in
[[encoder-dma-occupancy]]).

ABI (matches the upstream passthrough_dmas test.cpp + the occupancy harness measure_one):
  opcode 3; kernel(3, instr_bo@gid1, instr_size, buf0@gid3, buf1@gid4, ...). The number of
  data buffers equals the runtime_sequence arg count, which this harness derives from
  (mode, cols): read/write -> cols buffers; rdwr -> 2*cols (cols inputs then cols outputs).

MODES:
  --analyze-only RESULTS.json   regress an existing sweep (NO NPU).
  (default)                     RUN the sweep on the NPU (needs a window + built xclbins);
                                writes <outdir>/lpddr_bw_results.json then regresses.
                                Mirrors the occupancy harness device discipline -- wrap with
                                scripts/run_lpddr_bw_microbench.sh (service-stop/fuser/restart).

xclbin/insts naming (produced by scripts/build_lpddr_bw_microbench.sh):
  <build>/lpddr_{mode}_c{cols}_{bytes}.xclbin
  <build>/lpddr_{mode}_c{cols}_{bytes}.insts.bin
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.environ.get("PARAKEET_TOOLROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BUILD = os.path.join(REPO, "artifacts/parakeet/lpddr_bw")
OUTDIR = os.path.join(REPO, "artifacts/parakeet/lpddr_bw")


def n_buffers(mode, cols):
    return 2 * cols if mode == "rdwr" else cols


def moved_bytes(mode, total_bytes):
    """Total LPDDR bytes that actually cross the DRAM interface for one dispatch."""
    return 2 * total_bytes if mode == "rdwr" else total_bytes


def artifact_paths(build_dir, mode, cols, line, depth, total_bytes):
    base = os.path.join(build_dir,
                        f"lpddr_{mode}_c{cols}_l{line}_d{depth}_{total_bytes}")
    return base + ".xclbin", base + ".insts.bin"


def measure_one(pyxrt, dev, xclbin_path, insts_path, mode, cols, total_bytes, iters):
    """Dispatch one (mode, cols, bytes) pure-DMA xclbin `iters` times; return median us.
    Allocates n_buffers host_only BOs of per-column size; inputs are synced TO_DEVICE,
    outputs left as-is (contents irrelevant -- we measure movement, not correctness)."""
    if not os.path.exists(xclbin_path):
        sys.exit(f"missing xclbin {xclbin_path} -- build via build_lpddr_bw_microbench.sh")
    if not os.path.exists(insts_path):
        sys.exit(f"missing insts {insts_path}")
    instr = np.fromfile(insts_path, dtype=np.uint32)
    xb = pyxrt.xclbin(xclbin_path)
    kname = xb.get_kernels()[0].get_name()
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE

    nb = n_buffers(mode, cols)
    per_col_bytes = total_bytes // cols
    bo_i = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_i.write(instr.tobytes(), 0)
    bo_i.sync(TO)
    # rdwr arg order = cols inputs then cols outputs; read/write = cols of that one role.
    n_inputs = cols if mode in ("read", "rdwr") else 0
    bufs = []
    payload = np.zeros(per_col_bytes // 4, np.int32)
    for i in range(nb):
        bo = pyxrt.bo(dev, per_col_bytes, pyxrt.bo.host_only, k.group_id(3 + i))
        if i < n_inputs:  # input region: prime it so the read DMA has real data to move
            bo.write(payload.tobytes(), 0)
            bo.sync(TO)
        bufs.append(bo)

    def once():
        k(3, bo_i, instr.size, *bufs).wait()

    once()  # warmup (xclbin load, first-touch)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        once()
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def regress(points):
    """points: [{moved_MB, t_us}] -> {c0_fixed_us, c2_us_per_MB, BW_GB_s, rms}."""
    y = np.array([p["t_us"] for p in points], float)
    X = np.vstack([np.ones(len(points)),
                   np.array([p["moved_MB"] for p in points], float)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    c0, c2 = float(coef[0]), float(coef[1])
    return {
        "c0_fixed_us": c0,
        "c2_us_per_MB": c2,
        "BW_GB_s": (1e6 / c2 / 1e3) if c2 > 0 else float("inf"),
        "rms_resid_us": float(np.sqrt(np.mean((y - pred) ** 2))),
        "npoints": len(points),
    }


def report_config(cfg, mode):
    """Print one (line,depth) config's bytes-sweep + regression; return its regression dict."""
    print(f"\n  -- line={cfg['line']}B depth={cfg['depth']} --")
    print(f"  {'bytes':>10} {'moved_MB':>9} {'t_med_us':>9} {'t_min_us':>9} {'inst_GB/s':>10}")
    pts = []
    for r in cfg["points"]:
        moved_MB = moved_bytes(mode, r["bytes"]) / 1e6
        inst = moved_MB / (r["t_med_us"] / 1e6) / 1e3  # naive per-point (no overhead sub)
        print(f"  {r['bytes']:>10} {moved_MB:9.2f} {r['t_med_us']:9.1f} {r['t_min_us']:9.1f} {inst:10.1f}")
        pts.append({"moved_MB": moved_MB, "t_us": r["t_med_us"]})
    if len(pts) >= 2:
        c = regress(pts)
        print(f"    regression: c0={c['c0_fixed_us']:.1f}us  BW={c['BW_GB_s']:.1f} GB/s  "
              f"(rms={c['rms_resid_us']:.1f}us, n={c['npoints']})")
        cfg["regression"] = c
        return c
    print("    (need >=2 sweep points to regress)")
    return None


def report(results):
    print(f"\n=== LPDDR bandwidth microbench: mode={results['mode']} cols={results['cols']} "
          f"iters={results['iters']} ===")
    best = None
    for cfg in results["configs"]:
        c = report_config(cfg, results["mode"])
        if c and (best is None or c["BW_GB_s"] > best[0]["BW_GB_s"]):
            best = (c, cfg["line"], cfg["depth"])
    if best:
        c, line, depth = best
        results["peak"] = {"BW_GB_s": c["BW_GB_s"], "c0_fixed_us": c["c0_fixed_us"],
                           "line": line, "depth": depth}
        print(f"\n  ===> PEAK ACHIEVABLE LPDDR BW = {c['BW_GB_s']:.1f} GB/s "
              f"@ line={line}B depth={depth}   (datasheet ~120; encoder-GEMM effective ~57)")
        print(f"       fixed dispatch floor (best config) = {c['c0_fixed_us']:.1f} us")
        print(f"       NOTE: peak over the line x depth sweep -- a single low config would")
        print(f"             under-report and falsely impugn the 120 GB/s datasheet.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", metavar="RESULTS.json", default=None)
    ap.add_argument("--mode", choices=["read", "write", "rdwr"], default="rdwr")
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--sweep-line", type=int, nargs="*", default=[1024, 4096, 16384],
                    help="objectFIFO BD/transfer granularities in bytes to sweep")
    ap.add_argument("--sweep-depth", type=int, nargs="*", default=[2, 4],
                    help="objectFIFO depths to sweep")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--build-dir", default=DEFAULT_BUILD)
    ap.add_argument("--sweep-bytes", type=int, nargs="*",
                    default=[65536, 262144, 1048576, 4194304, 16777216, 67108864],
                    help="total bytes per direction (default 64KB..64MB x4 steps)")
    a = ap.parse_args()

    if a.analyze_only:
        report(json.load(open(a.analyze_only)))
        return 0

    try:
        import pyxrt
    except Exception as e:
        sys.exit(f"pyxrt import failed ({e}); run inside .venv-iron with the NPU free")
    dev = pyxrt.device(0)
    os.makedirs(OUTDIR, exist_ok=True)
    results = {"mode": a.mode, "cols": a.cols, "iters": a.iters, "configs": []}
    for line in a.sweep_line:
        for depth in a.sweep_depth:
            cfg = {"line": line, "depth": depth, "points": []}
            for tb in a.sweep_bytes:
                if tb % (a.cols * line) != 0:
                    continue
                xclbin, insts = artifact_paths(a.build_dir, a.mode, a.cols, line, depth, tb)
                if not (os.path.exists(xclbin) and os.path.exists(insts)):
                    print(f"  skip l{line} d{depth} bytes={tb}: missing "
                          f"{os.path.basename(xclbin)} -- build it first")
                    continue
                t_med, t_min = measure_one(pyxrt, dev, xclbin, insts, a.mode, a.cols, tb, a.iters)
                cfg["points"].append({"bytes": tb, "t_med_us": round(t_med, 2),
                                      "t_min_us": round(t_min, 2)})
                print(f"  l{line} d{depth} bytes={tb}: t_med={t_med:.1f}us t_min={t_min:.1f}us")
            if cfg["points"]:
                results["configs"].append(cfg)
    report(results)
    outp = os.path.join(OUTDIR, f"lpddr_bw_{a.mode}_c{a.cols}_results.json")
    json.dump(results, open(outp, "w"), indent=2)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
