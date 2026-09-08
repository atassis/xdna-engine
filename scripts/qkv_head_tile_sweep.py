#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the fused QKV head's transport deficit come from TRANSFER SIZE? Sweep tile_size_input.

A depth sweep of the fused decode (bench_layer_arms.py, 5 arms, S=512, residuals +/-0.015 ms) reads

    t = 1.4601 ms/layer * L + 5.886 ms      bytes = 34.7737 MB/layer * L + 311.49 MB

so the per-TOKEN term (one lm_head GEMV) moves 311.49 MB at 52.92 GB/s -- the measured 52.69 GB/s
pure-read fabric ceiling -- while the per-LAYER body moves 34.77 MB at 23.82 GB/s, 45% of it. Same
token, same dispatch, same silicon. The difference between those two is not bytes and not
configures; both are counted. What is left is DMA transfer LATENCY rather than issue cost.

`tile_size_input` is the knob that tests it directly and at CONSTANT BYTES: it is the number of
D-wide weight rows per ObjectFifo fill, so N_W_TILES = HD/tsi fills of tsi*D*2 bytes each. Every
arm reads the same 8.389 MB of Wqkv; only the transfer SHAPE changes.

  tsi   fills/head   bytes/fill
    1      128          2 KB
    2       64          4 KB
    4       32          8 KB     <- shipped
    8       16         16 KB
   16        8         32 KB

If achieved bandwidth is flat in tsi, per-transfer cost is not the mechanism and the deficit is
elsewhere (lock round-trips, serialisation between cores). If it climbs, the shipped tsi=4 is
leaving transport on the table and the whole layer body is mis-shaped the same way.

Arms are ALTERNATED round-robin so drift lands on every arm equally (the box has been measured
drifting 33% in four hours). Reports a CYCLE-free achieved GB/s plus the raw latency.

  python scripts/qkv_head_tile_sweep.py --arms 1 2 4 8 16 --rounds 5 --iters 20
Needs the NPU free. Do NOT stop npu-vox (owner's dictation).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.qkv_head_dp.op import QKVHeadDataParallel
from iron.operators.qkv_head_dp.reference import reference


def build(tsi, cols, D, HD, Hq, Hkv, S, spread=False):
    """`spread` passes aiecc --cores-per-col 1, which is what the DECODE GRAPH already does
    (DECODE_PLACER_FLAGS_DEFAULT) and what a bare operator build does NOT. Without it the default
    placer column-major-fills 4 rows before moving on, so N=8 lands on physical columns 0-1 -- and
    LPDDR read bandwidth scales with COLUMNS (npu-lpddr-read-scaling-and-peak measures 52.7 GB/s
    scaling to 8 of them). A standalone measurement of a 2-column build is therefore not a
    measurement of what ships."""
    tag = f"tsi{tsi}_S{S}_n{cols}" + ("_spread" if spread else "")
    bd = Path(__file__).resolve().parents[2] / "build" / f"qkv_head_dp_{tag}_sweep"
    op = QKVHeadDataParallel(D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=cols,
                             tile_size_input=tsi, kv_offset_parameter=None,
                             context=AIEContext(build_dir=bd))
    op.set_up_artifacts()
    if spread:
        op.xclbin_artifact.extra_flags = list(op.xclbin_artifact.extra_flags) + \
            ["--cores-per-col", "1"]
    op.compile()
    return op


def placed_columns(op):
    d = Path(op.xclbin_artifact.mlir_input.filename)
    sub = d.parent / f"{d.stem}.mlir.d"
    tiles = [tuple(int(x) for x in p.name.removeprefix("elfs_main_core_").split("_"))
             for p in sub.glob("elfs_main_core_*") if p.is_dir()]
    return sorted(set(c for c, _ in tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["4", "4s"],
                    help="arm spec TSI[s][@CORES]: tile_size_input, 's' = --cores-per-col 1, "
                         "'@N' = num_aie_columns (cores). Bytes are identical in every arm; "
                         "@N changes only how many cores CONSUME them, which is what separates "
                         "a fabric limit from a per-core one.")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    D, HD, Hq, Hkv, S = 1024, 128, 16, 8, 512
    QD, KVD = Hq * HD, Hkv * HD
    WQKV_BYTES = (QD + 2 * KVD) * D * 2          # the stream under test, identical in every arm

    torch.manual_seed(0)
    cur, n_in = torch.randn(D, dtype=torch.bfloat16), torch.randn(D, dtype=torch.bfloat16)
    wqkv = torch.randn(QD + 2 * KVD, D, dtype=torch.bfloat16) * 0.05
    n_qn = torch.randn(HD, dtype=torch.bfloat16)
    n_kn = torch.randn(HD, dtype=torch.bfloat16)
    ang = torch.randn(HD, dtype=torch.bfloat16)
    golden = reference(cur, n_in, wqkv.reshape(-1), n_qn, n_kn, ang, D, HD, Hq, Hkv, 1e-6)
    kc = torch.zeros(Hkv * S * HD, dtype=torch.bfloat16)
    vc = torch.zeros(Hkv * S * HD, dtype=torch.bfloat16)
    for h in range(Hkv):
        kc[h * S * HD: h * S * HD + HD] = golden[QD + h * HD: QD + (h + 1) * HD]
        vc[h * S * HD: h * S * HD + HD] = golden[QD + KVD + h * HD: QD + KVD + (h + 1) * HD]
    ins = {"cur": cur, "n_in": n_in, "wqkv": wqkv.reshape(-1), "n_qn": n_qn, "n_kn": n_kn,
           "ang": ang, "kc": torch.zeros_like(kc), "vc": torch.zeros_like(vc)}
    outs = {"q": golden[:QD], "kc": kc, "vc": vc}

    ops, skipped, cols_of = {}, {}, {}
    for spec in a.arms:
        body, _, ncores = spec.partition("@")
        cores = int(ncores) if ncores else a.cols
        spread = body.endswith("s")
        tsi = int(body[:-1] if spread else body)
        try:
            ops[spec] = build(tsi, cores, D, HD, Hq, Hkv, S, spread=spread)
            cols_of[spec] = placed_columns(ops[spec])
            print(f"[sweep] built {spec}: tsi={tsi} cores={cores} spread={spread} "
                  f"columns={cols_of[spec]}", flush=True)
        except Exception as e:
            skipped[spec] = str(e).strip().splitlines()[-1][:160]
            print(f"[sweep] {spec} DID NOT BUILD: {skipped[spec]}", flush=True)
    if not ops:
        raise SystemExit("no arm built")

    samples = {t: [] for t in ops}
    for r in range(a.rounds):
        order = sorted(ops) if r % 2 == 0 else sorted(ops, reverse=True)   # ABBA
        for spec in order:
            errors, lat_us, _ = run_test(ops[spec], ins, outs, rel_tol=0.05, abs_tol=0.5,
                                         warmup_iters=3, timed_iters=a.iters)
            assert not errors, f"{spec} MISMATCH: {list(errors)[:3]}"
            samples[spec].append(lat_us)
        print(f"[sweep] round {r} done", flush=True)

    print(f"\n{'arm':>5} {'cores':>5} {'B/fill':>8} {'median_us':>10} {'min_us':>9} "
          f"{'GB/s(med)':>10} {'GB/s(min)':>10} {'% of 52.69':>11}")
    report = {}
    for spec in sorted(ops):
        body, _, ncores = spec.partition("@")
        cores = int(ncores) if ncores else a.cols
        spread = body.endswith("s")
        tsi = int(body[:-1] if spread else body)
        xs = samples[spec]
        med, mn = statistics.median(xs), min(xs)
        gbps, gbps_mn = WQKV_BYTES / (med * 1e-6) / 1e9, WQKV_BYTES / (mn * 1e-6) / 1e9
        report[spec] = {"reps_us": xs, "median_us": med, "min_us": mn, "gbps_median": gbps,
                        "gbps_min": gbps_mn, "columns": cols_of[spec], "tsi": tsi,
                        "cores": cores, "bytes_per_fill": tsi * D * 2}
        print(f"{spec:>5} {cores:5} {tsi*D*2:8} {med:10.1f} {mn:9.1f} "
              f"{gbps:10.2f} {gbps_mn:10.2f} {100*gbps/52.69:10.1f}%")
    if skipped:
        # No silent caps: an arm that did not build is reported, not dropped.
        print("\nnot built:")
        for tsi, why in skipped.items():
            print(f"  tsi={tsi}: {why}")
    if a.out_json:
        json.dump({"wqkv_bytes": WQKV_BYTES, "cols": a.cols, "arms": report, "skipped": skipped},
                  open(a.out_json, "w"), indent=2)
        print(f"\n[sweep] wrote {a.out_json}")


if __name__ == "__main__":
    sys.exit(main())
