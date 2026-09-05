#!/usr/bin/env python3
"""Dump in/resident/expected .bin for one exported design, for the Rust probe to gate against.

The expected output is produced by running the design through the PYTHON path -- the same
bricklib rail every codec gate already uses -- so the Rust probe is compared against a
device-verified result, not a host model. If Rust matches, the exported artifact plus
npu-s2's buffer wiring reproduce the Python dispatch exactly, which is what the slice is for.

The specific thing under test is the arg-index-3.. convention: npu-s2 INFERRED it from the
fixed-arity kernels (run_mha's Q@3 K@4 V@5 O@6) and it has never been confirmed against a
_build_streamed design.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_driver as wd
import bricklib

art = Path(sys.argv[1])
meta = json.loads((art / "meta.json").read_text())
n_tiles, in_tile = meta["n_tiles"], meta["in_tile"]
out_numel, resident_len = meta["out_numel"], meta["resident_len"]
sym, flags = meta["symbol"], meta["compile_flags"] or None
depth = meta["resident_depth"]

rng = np.random.default_rng(1234)
tiles = (rng.standard_normal((n_tiles, in_tile)) * 0.05).astype(np.float32)
resident = rng.standard_normal(resident_len).astype(np.float32) if resident_len else None

# rebuild the same design from the shim the exporter saved beside the artifact
shim = art / "shim.cc"
design = bricklib._build_streamed(sym, shim, n_tiles, in_tile, out_numel, resident_len, flags,
                                  np.float32, np.float32,
                                  (np.float32 if resident_len else None), depth)
in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
out_t = bricklib.iron.zeros((n_tiles * out_numel,), dtype=np.float32, device="npu")
if resident_len:
    r_t = bricklib.iron.tensor(resident, dtype=np.float32, device="npu")
    design(in_t, r_t, out_t)
else:
    design(in_t, out_t)
got = out_t.numpy().copy().reshape(n_tiles, out_numel)

tiles.tofile(art / "probe_in.bin")
if resident_len:
    resident.tofile(art / "probe_resident.bin")
got.astype(np.float32).tofile(art / "probe_expected.bin")
print(f"{meta['design_name']}: tiles{tiles.shape} resident={resident_len} out{got.shape} "
      f"nz={np.abs(got).sum():.6e}")
print(f"wrote probe_in.bin / probe_resident.bin / probe_expected.bin into {art}")
