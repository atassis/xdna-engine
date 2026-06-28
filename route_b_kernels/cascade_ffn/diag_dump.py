#!/usr/bin/env python3
"""Per-row localization harness for the M-tile cascade FFN (device vs host).

Compiles + runs the cascade-FFN ELF once and compares the device output to the
matching host intermediate PER ROW -- it tells you exactly which row and which
stage first diverges. This is how the M_TILE>1 weight-stream-overrun bug was
isolated (see internal notes). Run it from a dir
holding a fresh mv_bf16_gelu.o (XRTRunner links link_with from CWD), with
`source scripts/air_env.sh` + ironenv on PATH (torch); single-tenant NPU.

  --mode full   : compares the FULL FFN output o2[MT,768] vs the golden -- WORKS
                  AS-IS (this is the production correctness check, == test_redispatch).

  --mode ln|xpw|fc1|h|partial|w1 : per-STAGE dumps. These need the matching
  FFN_DUMP_* env hooks IN ffn_cascade.py, which were STRIPPED after the fix. To
  re-enable for Phase-2 debugging, re-add an env-gated block that (HEAD core only)
  DMAs the intermediate to the out BO then `return`, skipping downstream stages
  WITHOUT leaving unpaired channels (skip the matching weight relay). The bisection
  set was: LN=xnorm (skip weights) | XPW=xnorm after the w1 stream ran | FC1=raw
  fc1 before bias/gelu | H=post fc1+bias+gelu | PARTIAL=core0 pre-cascade fc2 |
  W1=the first w1 tile core0 receives (weight-delivery check). Plus FFN_NOBIAS
  (pure weight stream, no bias) and FFN_NW1=N (stream+process only N w1 tiles) were
  the decisive isolators. The host refs below already match the K-aug Wfc1 [FF,800].

  FFN_DUMP_FC1=1 python diag_dump.py --m-tile 4 --mode fc1   (after re-adding the hook)
  (no env)       python diag_dump.py --m-tile 4 --mode full
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "route_b_kernels", "cascade_ffn"))

from ffn_cascade import build_module  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402
import filelock, tempfile  # noqa: E402

BF16 = ml_dtypes.bfloat16
D, FF, NCORES, M_INPUT, EPS = 768, 3072, 8, 8, 1e-5
M_SLAB = FF // NCORES  # 384


def rel_l2(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (n if n else 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tile", type=int, default=4, dest="MT")
    ap.add_argument("--mode", required=True,
                    choices=["ln", "xpw", "fc1", "h", "partial", "full", "w1"])
    ap.add_argument("--buffers", default=None)
    ap.add_argument("--ncols", type=int, default=0, help="override compared col count")
    args = ap.parse_args()
    MT = args.MT
    buf = args.buffers or f"{REPO}/artifacts/cascade_ffn_mtile/iron_baseline/buffers"

    def rd(name, shape):
        raw = np.fromfile(f"{buf}/{name}.bin", dtype=np.uint16)
        return raw.view(BF16).reshape(shape)

    x = rd("x", (MT, D))
    # K-aug layout (gen_golden): Wfc1 is [FF, D+32] -- cols 0:D = weights, col D =
    # bias_fc1 (cols D+1:D+32 = 0). The device fc1 folds the bias col; here we split
    # it back out for the host reference.
    FC1_WROW = D + 32
    w1full = rd("Wfc1", (FF, FC1_WROW))      # the ELF's Wfc1 BO (K-aug)
    w1 = w1full[:, :D]                       # weights (host ref)
    bias_fc1 = w1full[:, D]                  # baked bias column (host ref)
    bfc1 = np.fromfile(f"{buf}/bfc1.bin", dtype=np.uint16)
    bfc2_raw = np.fromfile(f"{buf}/bfc2.bin", dtype=np.uint16)
    biases = np.concatenate([bfc1, bfc2_raw]).view(BF16)  # [FF+D] ELF BO (device reads only b_fc2)
    bfc2 = bfc2_raw.view(BF16)
    w2 = rd("Wfc2", (D, FF))

    # --- host references (device-faithful bf16 at each stage), per gen_golden ---
    def b(a): return np.asarray(a).astype(BF16)
    import torch
    n_hw = b(torch.nn.functional.layer_norm(
        torch.from_numpy(x.astype(np.float32)), (D,)).numpy())          # [MT,D]
    h1 = b(n_hw.astype(np.float32) @ w1.astype(np.float32).T)            # [MT,FF] raw fc1 (no bias)
    h2 = b(h1.astype(np.float32) + bias_fc1.astype(np.float32))
    h3 = b(torch.nn.functional.gelu(
        torch.from_numpy(h2.astype(np.float32)), approximate="tanh").numpy())  # [MT,FF]
    # core0 fc2 partial: h3[:, 0:M_SLAB] @ w2[:, 0:M_SLAB].T  (pre-cascade, no b_fc2)
    part0 = b(h3[:, 0:M_SLAB].astype(np.float32) @ w2[:, 0:M_SLAB].astype(np.float32).T)  # [MT,D]
    o1 = b(h3.astype(np.float32) @ w2.astype(np.float32).T)
    o2 = b(o1.astype(np.float32) + bfc2.astype(np.float32))  # [MT,D] full

    ref_map = {
        "ln": (n_hw, D), "xpw": (n_hw, D),
        "fc1": (h1[:, :M_SLAB], M_SLAB),
        "h": (h3[:, :M_SLAB], M_SLAB),
        "partial": (part0, D),
        "full": (o2, D),
        "w1": (w1[:MT, :], D),   # first MT weight rows core0 receives (tile 0)
    }
    ref, ncol = ref_map[args.mode]
    if args.ncols:
        ncol = args.ncols
        ref = ref[:, :ncol]

    # --- compile + run the ELF once, capture raw output [MT, D] ---
    module = build_module(D, FF, NCORES, M_INPUT, EPS, MT)
    backend = XRTBackend(output_format="elf", instance_name="ffn_cascade",
                         use_lock_race_condition_fix=False,
                         stack_size=4096, n_perf_iters=1, n_warmup_iters=0, verbose=False)
    out_ph = np.zeros((MT, D), dtype=BF16)
    compiled = backend.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        slots = fn(x, w1full, biases, w2, out_ph)
    backend.unload()
    dev_full = np.asarray(slots[4]).view(BF16).reshape(MT, D)
    dev = dev_full[:, :ncol]

    print(f"\n=== mode={args.mode} M_TILE={MT}: device vs host, per row (ncol={ncol}) ===")
    for r in range(MT):
        print(f"  row {r}: rel-L2 = {rel_l2(dev[r], ref[r]):.4f}")
    print(f"  ALL : rel-L2 = {rel_l2(dev, ref):.4f}")
    # also: are device rows identical to each other? (input rows differ, so should differ)
    print(f"  device row0[:6]={dev[0,:6].astype(np.float32)}")
    print(f"  host   row0[:6]={ref[0,:6].astype(np.float32)}")
    if MT > 1:
        print(f"  device row1[:6]={dev[1,:6].astype(np.float32)}")
        print(f"  host   row1[:6]={ref[1,:6].astype(np.float32)}")


if __name__ == "__main__":
    main()
