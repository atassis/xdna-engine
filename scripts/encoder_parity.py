#!/usr/bin/env python3
"""Numerical-parity gate for device encoder changes (replaces the chaotic 17-clip greedy WER for
validating numerically-equivalent changes).

The 17-clip greedy-decode WER is CHAOTIC at ~1e-5: perturbing the SHIPPED encoder by a meaningless
+-1e-5 per-element noise swings its WER across an ~8.2-9.2 band. So the greedy WER on a small clip set
CANNOT tell a correct device change from the shipped path. This gate instead measures each path's
distance from the TRUE f32 encoder and asks: is the candidate any further from truth than the shipped
baseline? (both device paths are ~8.5% off f32 -- the bf16 matmul noise floor.)

Usage:
  # 1. capture the three encode dirs (17-clip mels):
  cargo run --features npu --release --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_ref --cpu
  NPU_XCLBIN_ROOT=$PWD cargo run ... --bin parakeet_encode_npu -- artifacts/wer_mels /tmp/enc_ship
  PARAKEET_FUSED_BLOCK=1 NPU_XCLBIN_ROOT=$PWD cargo run ... -- artifacts/wer_mels /tmp/enc_fused
  # 2. gate:
  scripts/encoder_parity.py /tmp/enc_ref /tmp/enc_ship /tmp/enc_fused [--margin 0.15]

PASS iff  mean rel-L2(candidate, f32-truth)  <=  mean rel-L2(baseline, f32-truth) * (1 + margin).
"""
import sys, glob, os
import numpy as np


def rl2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    margin = 0.15
    for i, a in enumerate(sys.argv):
        if a == "--margin":
            margin = float(sys.argv[i + 1])
    if len(args) != 3:
        print("usage: encoder_parity.py <f32_ref_dir> <baseline_dir> <candidate_dir> [--margin 0.15]")
        sys.exit(2)
    ref_dir, base_dir, cand_dir = args
    base_e, cand_e, bc = [], [], []
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ref_dir, "*.npy")))
    if not names:
        print(f"no .npy in {ref_dir}"); sys.exit(2)
    for n in names:
        ref = np.load(os.path.join(ref_dir, n))
        b = np.load(os.path.join(base_dir, n))
        c = np.load(os.path.join(cand_dir, n))
        base_e.append(rl2(ref, b)); cand_e.append(rl2(ref, c)); bc.append(rl2(b, c))
    mb, mc = float(np.mean(base_e)), float(np.mean(cand_e))
    print(f"clips={len(names)}")
    print(f"baseline  vs f32-truth : mean rel-L2 = {mb:.4f}  (max {np.max(base_e):.4f})")
    print(f"candidate vs f32-truth : mean rel-L2 = {mc:.4f}  (max {np.max(cand_e):.4f})")
    print(f"candidate vs baseline  : mean rel-L2 = {np.mean(bc):.4f}")
    thr = mb * (1.0 + margin)
    ok = mc <= thr
    print(f"GATE: candidate {mc:.4f} {'<=' if ok else '>'} baseline*(1+{margin}) {thr:.4f}  -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
