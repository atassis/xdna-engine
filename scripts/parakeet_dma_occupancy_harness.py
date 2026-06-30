#!/usr/bin/env python3
"""Parakeet encoder DMA-occupancy harness -- split DMA-wait from dispatch/BD/lock/stall
inside the 91-95% "movement" window the per-op occupancy A/B lumps together
([[parakeet-occupancy-measured-ab]]). Gates the byte-dedup lever from
[[encoder-lpddr-traffic-map]]: how much of "movement" is actually LPDDR DMA (=on the
critical path, dedup-able) vs fixed dispatch overhead (NOT byte-reducible)?

WHY a sweep, not a single A/B: the existing FULL-vs-STUB twin isolates COMPUTE
(full-stub). The STUB time (movement+dispatch+stall, no compute) still mixes per-byte
DMA with per-transfer BD/lock overhead and a fixed dispatch floor. We separate them by
SWEEPING the byte volume + transfer count and regressing:

    t_stub(shape) ~= c0  +  c1 * n_transfers  +  c2 * bytes_MB
                     |--fixed--| |--per-BD/lock--|  |--DMA, 1/c2 = eff GB/s--|

- N-sweep on the resident K=1024 xclbin (cheap: insts-only, no rebuild) varies bytes AND
  transfers ~collinearly -> gives the COMBINED movement-marginal rate (an UPPER bound on
  per-byte DMA; the 2-param [1,bytes] fit).
- Adding a K-sweep and/or tile-size (m,k,n) sweep breaks the collinearity (bigger tiles =
  more bytes per BD, fewer BDs) -> the 3-param fit cleanly separates c2 (DMA) from c1
  (per-transfer). These points need CPU-side rebuilds (scripts/build_parakeet_dma_sweep.sh).

HONEST LIMIT: a perfectly clean DMA/BD split wants trace PORT_RUNNING counters, which the
8-col cascade-C-join can't route (see parakeet_occupancy_harness notes). This latency
regression gives a BOUNDED decomposition, not a counter-exact one -- but it is enough to
answer "is the duplicate re-read on the critical path or hidden", which is the gate.

MODES:
  --analyze-only RESULTS.json   regress an existing sweep (NO NPU). Works today on
                                artifacts/parakeet/occupancy/occupancy_results.json
                                (the 3 N points) for a provisional 2-param fit.
  (default)                     RUN the sweep on the NPU (needs a window + built xclbins);
                                writes artifacts/parakeet/occupancy/dma_sweep_results.json
                                then regresses. Mirrors the occupancy harness device
                                discipline (stop services / fuser / restart via the runner).
"""
import argparse, importlib.util, json, os, sys
import numpy as np

# Toolchain root (.venv-iron, mlir-aie/build, artifacts/) -- lives in the MAIN checkout; a
# lever-worktree sets PARAKEET_TOOLROOT to point here while running worktree-edited code.
REPO = os.environ.get("PARAKEET_TOOLROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WA = os.path.join(REPO, "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build")
OUTDIR = os.path.join(REPO, "artifacts/parakeet/occupancy")
GEN = os.path.join(REPO, "route_b_kernels/whole_array_fused/whole_array_modal_iron.py")
BIN, BOUT = 2, 4  # bf16 in, f32 out


def _gen():
    spec = importlib.util.spec_from_file_location("wa_gen", GEN)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def byte_transfer_model(M, K, N, m=64, k=32, n=128, cols=8):
    """Host-only (no NPU): exact LPDDR bytes + transfer (fill/drain) count for one
    dispatch, from --generate-taps. Same enumeration as the encoder-lpddr-traffic-map."""
    wa = _gen()
    A_seq, B_seq, C_seq = wa.my_matmul("npu2", M, K, N, m, k, n, cols,
                                       b_col_maj=0, do_silu=True,
                                       emulate_bf16_mmul_with_bfp16=True,
                                       trace_size=0, generate_taps=True, do_gelu=False)
    def elems(seq):
        return sum(int(np.prod([int(x) for x in t.sizes])) for t in seq)
    def ntap(seq):
        return sum(1 for _ in seq)
    a_el, b_el, c_el = elems(A_seq), elems(B_seq), elems(C_seq)
    bytes_MB = (a_el * BIN + b_el * BIN + c_el * BOUT) / 1e6
    n_transfers = ntap(A_seq) + ntap(B_seq) + ntap(C_seq)
    return dict(M=M, K=K, N=N, bytes_MB=bytes_MB, n_transfers=n_transfers,
                A_MB=a_el * BIN / 1e6, B_MB=b_el * BIN / 1e6, C_MB=c_el * BOUT / 1e6,
                A_rr=a_el / (M * K), B_rr=b_el / (K * N))


def regress(points, use_transfers):
    """points: list of dicts with bytes_MB, n_transfers, t_stub_us. Returns coeffs."""
    y = np.array([p["t_stub_us"] for p in points], float)
    cols = [np.ones(len(points))]
    names = ["c0_fixed_us"]
    if use_transfers:
        cols.append(np.array([p["n_transfers"] for p in points], float)); names.append("c1_us_per_transfer")
    cols.append(np.array([p["bytes_MB"] for p in points], float)); names.append("c2_us_per_MB")
    X = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    out = dict(zip(names, [float(c) for c in coef]))
    out["eff_GB_s"] = float(1e6 / out["c2_us_per_MB"] / 1e3) if out["c2_us_per_MB"] > 0 else float("inf")
    out["rms_resid_us"] = float(np.sqrt(np.mean(resid ** 2)))
    out["npoints"] = len(points)
    out["model"] = "t = c0 + c1*transfers + c2*bytes" if use_transfers else "t = c0 + c2*bytes"
    return out, pred


def analyze(results_path, dup_bytes_by_N):
    """Regress + report DMA fraction per shape and realizable byte-dedup latency."""
    d = json.load(open(results_path))
    shapes = d["shapes"] if "shapes" in d else d
    pts = []
    for key, v in shapes.items():
        M, K, N = (int(x) for x in key.split("x"))
        bt = byte_transfer_model(M, K, N)
        pts.append(dict(N=N, bytes_MB=bt["bytes_MB"], n_transfers=bt["n_transfers"],
                        t_stub_us=v["t_stub_us_median"], t_full_us=v["t_full_us_median"]))
    pts.sort(key=lambda p: p["bytes_MB"])
    use_t = len(pts) >= 4  # need >3 points for the 3-param fit; else 2-param (DMA-only)
    coef, pred = regress(pts, use_t)
    print(f"\n=== DMA-occupancy regression ({coef['model']}, n={coef['npoints']}) ===")
    for k_, v_ in coef.items():
        if k_ not in ("model",):
            print(f"  {k_:20s} = {v_:.3f}")
    print(f"\n  per-shape decomposition (stub = movement+dispatch+stall, compute already removed):")
    print(f"  {'N':>5} {'bytes_MB':>8} {'t_stub':>7} {'DMA_us':>7} {'DMA%':>5} {'fixed_us':>8} | {'dup_MB':>6} {'dedup_us':>8} {'dedup%disp':>10}")
    for p in pts:
        dma_us = coef["c2_us_per_MB"] * p["bytes_MB"]
        dma_frac = dma_us / p["t_stub_us"]
        fixed = p["t_stub_us"] - dma_us
        dup = dup_bytes_by_N.get(p["N"], 0.0)
        dedup_us = coef["c2_us_per_MB"] * dup
        dedup_disp = dedup_us / p["t_full_us"]
        print(f"  {p['N']:>5} {p['bytes_MB']:8.2f} {p['t_stub_us']:7.1f} {dma_us:7.1f} {dma_frac:5.0%} {fixed:8.1f} |"
              f" {dup:6.2f} {dedup_us:8.1f} {dedup_disp:10.0%}")
    print(f"\n  -> effective DMA bandwidth ~= {coef['eff_GB_s']:.0f} GB/s "
          f"(spec 120; cf decode fused-decode-effective-ddr-bw 4.77)")
    print(f"  -> fixed dispatch floor ~= {coef['c0_fixed_us']:.0f} us (byte-INDEPENDENT; not dedup-able)")
    if not use_t:
        print("  NOTE: 2-param fit (only 3 N points). c2 here is the COMBINED movement-marginal")
        print("        rate (per-byte DMA + per-transfer BD/lock) = an UPPER bound on pure DMA.")
        print("        Run the K/tile sweep (build_parakeet_dma_sweep.sh) for the 3-param split.")
    return coef


# --- duplicate bytes per N from the encoder-lpddr-traffic-map (read-once ideal delta) ---
def dup_bytes():
    out = {}
    for N in (1024, 2048, 3072, 4096):
        bt = byte_transfer_model(512, 1024, N)
        A_log, B_log = 512 * 1024 * BIN / 1e6, 1024 * N * BIN / 1e6
        out[N] = (bt["A_MB"] - A_log) + (bt["B_MB"] - B_log)  # extra bytes beyond read-once
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", metavar="RESULTS.json", default=None,
                    help="regress an existing sweep json (NO NPU)")
    ap.add_argument("--tile", default="64x32x128")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--sweep-N", type=int, nargs="*", default=[1024, 2048, 3072, 4096])
    a = ap.parse_args()

    dupes = dup_bytes()
    if a.analyze_only:
        analyze(a.analyze_only, dupes)
        return 0

    # RUN mode: device sweep over the STUB (+FULL for compute-subtraction). Reuses the
    # occupancy harness measure_one ABI. Needs a free NPU + built insts/xclbins.
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from parakeet_occupancy_harness import measure_one  # same opcode-3 run_matmul8 ABI
    try:
        import pyxrt
    except Exception as e:
        sys.exit(f"pyxrt import failed ({e}); run inside .venv-iron with the NPU free")
    dev = pyxrt.device(0)
    tile = a.tile
    full_x = f"{WA}/final_512x1024x4096_{tile}_8c.xclbin"
    stub_x = f"{WA}/final_512x1024x4096_{tile}_8c_STUB.xclbin"
    M, K = 512, 1024
    os.makedirs(OUTDIR, exist_ok=True)
    res = {"tile": tile, "iters": a.iters, "M": M, "K": K, "shapes": {}}
    for N in a.sweep_N:
        insts = f"{WA}/insts_512x1024x{N}_{tile}_8c.txt"
        g = f"{OUTDIR}/golden_{M}x{K}x{N}.npz"
        if not (os.path.exists(insts) and os.path.exists(g)):
            print(f"  skip N={N}: missing insts/golden ({insts}) -- build via build_parakeet_dma_sweep.sh")
            continue
        gd = np.load(g); A_u16, B_u16 = gd["A"], gd["B"]
        t_full, _, _ = measure_one(pyxrt, dev, full_x, insts, A_u16, B_u16, N, a.iters)
        t_stub, _, _ = measure_one(pyxrt, dev, stub_x, insts, A_u16, B_u16, N, a.iters)
        res["shapes"][f"{M}x{K}x{N}"] = {"t_full_us_median": round(t_full, 2),
                                         "t_stub_us_median": round(t_stub, 2)}
        print(f"  N={N}: full={t_full:.1f} stub={t_stub:.1f} compute={t_full-t_stub:.1f}")
    outp = f"{OUTDIR}/dma_sweep_results.json"
    json.dump(res, open(outp, "w"), indent=2)
    print(f"wrote {outp}")
    analyze(outp, dupes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
