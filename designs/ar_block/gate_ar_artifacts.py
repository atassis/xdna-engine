#!/usr/bin/env python3
"""Device gate for the six designs `export_ar_artifacts.py` emits.

Writes raw little-endian f32 fixtures, then drives `npu-dev s2-design` (rust/npu-s2) over each
design directory. The probe dispatches twice and reports rel-L2 plus run-to-run bit-identity;
per this project's convention the BLOCKING check is 1:1 determinism and rel-L2 is a note
(error-metrics-are-notes-not-gates), so a rel-L2 regression here is a signal to investigate,
not a build break.

Every reference comes from the BRICK'S OWN golden.py -- never a fixture authored here. That is
the rule npu-s2 already paid for twice: its original fixture tests all passed while six field
names were wrong, and a later defect survived because a Rust probe and a numpy cross-check
shared a wrong layout premise and agreed with each other.

MEASURED 2026-09-02, first device run of all six, every one run2run bit-identical:
    rmsnorm             7.634e-08      qk_norm_q  6.147e-08      qk_norm_k  6.303e-08
    rope_interleaved_q  5.912e-03      rope_interleaved_k  5.625e-03
    swiglu              1.026e-02
The two magnitudes are expected, not a discrepancy: rmsnorm/qk-norm compute in f32 end to end,
while rope-interleaved and swiglu carry bf16 internals, so ~1e-3..1e-2 IS their f32-boundary
noise floor. rope-interleaved is the one design here with no prior device precedent for the
exact code path -- its f32<->bf16 wrapper shim is new -- which is why it is gated at both roles
rather than sampled.

Usage:  gate_ar_artifacts.py <export_dir> <npu-dev_path> [--fixtures DIR]
The NPU is single-tenant: run this under scripts/npu_lock.sh, not bare.
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

BRICKS = Path(__file__).resolve().parent.parent / "bricks"

# Shapes are the exporter's, not independent guesses -- see export_ar_artifacts.py's spec
# functions for each one's derivation from the GGUF hparams.
EPS, HEAD_DIM, HIDDEN, FF, ROPE_BASE = 1e-6, 128, 2560, 9728, 1.0e6
SWIGLU_TILES = 8  # forced: a flat 19456-elem tile exceeds aie.dma_bd's 16383-word cap.


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_fixtures(out: Path, seed: int = 0) -> None:
    g_rms = _load("_g_rms", BRICKS / "rmsnorm/golden.py")
    g_qk = _load("_g_qk", BRICKS / "qk-norm/golden.py")
    g_rope = _load("_g_rope", BRICKS / "rope-interleaved/golden.py")
    g_sw = _load("_g_sw", BRICKS / "swiglu/golden.py")

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    put = lambda n, a: (out / n).write_bytes(np.ascontiguousarray(a, np.float32).tobytes())

    x = rng.standard_normal((1, HIDDEN), np.float32)
    gamma = rng.standard_normal(HIDDEN).astype(np.float32)
    put("rmsnorm.in.bin", x)
    put("rmsnorm.res.bin", gamma)
    put("rmsnorm.exp.bin", g_rms.rmsnorm_ref(x, gamma, eps=EPS))

    for role, rows in (("q", 32), ("k", 8)):  # head_count / head_count_kv
        x = rng.standard_normal((rows, HEAD_DIM), np.float32)
        gamma = rng.standard_normal(HEAD_DIM).astype(np.float32)
        put(f"qk_norm_{role}.in.bin", x)
        put(f"qk_norm_{role}.res.bin", gamma)
        put(f"qk_norm_{role}.exp.bin", g_qk.qk_norm(x, gamma, eps=EPS).astype(np.float32))

    for role, rows in (("q", 32), ("k", 8)):
        # ONE cossin row: at a decode step every head shares the token's position, which is why
        # the exported design's resident_len is D and not rows*D.
        pos = 37
        x = rng.standard_normal((rows, HEAD_DIM), np.float32)
        cossin = g_rope.build_cossin_resident(np.array([pos]), HEAD_DIM, base=ROPE_BASE)
        put(f"rope_interleaved_{role}.in.bin", x)
        put(f"rope_interleaved_{role}.res.bin", cossin.reshape(-1))
        put(f"rope_interleaved_{role}.exp.bin",
            g_rope.rope_interleaved_ref(x, np.full(rows, pos), HEAD_DIM, base=ROPE_BASE))

    chunk = FF // SWIGLU_TILES
    gate = rng.standard_normal(FF).astype(np.float32)
    up = rng.standard_normal(FF).astype(np.float32)
    put("swiglu.in.bin", np.concatenate(
        [np.concatenate([gate[c * chunk:(c + 1) * chunk], up[c * chunk:(c + 1) * chunk]])
         for c in range(SWIGLU_TILES)]))
    put("swiglu.exp.bin", g_sw.swiglu_ref(gate, up).astype(np.float32))


DESIGNS = ("rmsnorm", "qk_norm_q", "qk_norm_k",
           "rope_interleaved_q", "rope_interleaved_k", "swiglu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("probe", type=Path, help="path to the built npu-dev binary")
    ap.add_argument("--fixtures", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()

    fix = opts.fixtures or (opts.export_dir / "_fixtures")
    write_fixtures(fix, opts.seed)

    failed = []
    for name in DESIGNS:
        res = fix / f"{name}.res.bin"
        rc = subprocess.run(
            [str(opts.probe), "s2-design", str(opts.export_dir / name), str(fix / f"{name}.in.bin"),
             str(res) if res.exists() else "-", str(fix / f"{name}.exp.bin")]).returncode
        print(f"  {name}: {'PASS' if rc == 0 else f'FAIL rc={rc}'}")
        if rc:
            failed.append(name)
    print(f"{len(DESIGNS) - len(failed)}/{len(DESIGNS)} designs pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
