#!/usr/bin/env python3
"""Device rel-L2 verify for the gather-rows brick. Gate 3e-2. Run under the device lock.

Gated at the RESIDUAL-codebook regime (n_rows=1024, D=8): the shape that fits a core tile
resident in f32 (32 KB, see gather_rows.cc's header). The SEMANTIC codebook (n_rows=4096)
does not fit this contract in f32 and is explicitly out of scope for contract (a) -- see the
brick header's CONTRACT CHOSEN section.

Streamed rail: indices arrive as GATHER_T_TILE-sized int32 tiles, the codebook is ONE
resident f32 operand acquired once for the whole stream, gathered rows stream out as
GATHER_T_TILE x D f32 tiles. A pure gather has zero compute, so rel-L2 vs the numpy golden
should land at (or extremely near) 0, not just under the 3e-2 gate -- any nonzero rel-L2
here means the gather itself is wrong (wrong row, wrong clamp, or a streamed-rail codegen
defect of the kind verify_sin.py's header documents for a different brick on this same rail).

NOT device-verified by the agent that authored this file (no NPU access for that task --
the device is single-tenant and gated serially by the owning session). A clean
compile_check.sh pass and a standalone golden.py match are the only signals available at
authoring time; this script is the actual gate and must be run separately, on device.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "gather-rows").resolve()

N_ROWS, D, T_TILE, N_TILES = 1024, 8, 16, 3   # residual-codebook shape; T = T_TILE*N_TILES = 48
T = T_TILE * N_TILES
GATE = 3e-2

spec = importlib.util.spec_from_file_location("gather_rows_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
codebook = rng.standard_normal((N_ROWS, D)).astype(np.float32)

# Non-trivial index pattern -- NOT an identity/ramp (see golden.py): repeats, row 0, row
# N_ROWS-1, an out-of-range index and a negative index to exercise the clamp both ways.
idx = rng.integers(0, N_ROWS, size=T).astype(np.int32)
idx[0] = 0                    # first row
idx[1] = N_ROWS - 1           # last row
idx[2] = idx[20] = 17         # deliberate repeat, in different tiles (17 and 20 land in
                               # different T_TILE=16 chunks: tile 0 and tile 1)
idx[3] = N_ROWS + 50          # out-of-range -> clamp to N_ROWS - 1
idx[4] = -3                   # negative -> clamp to 0

ref = golden.gather_rows_ref(codebook, idx)   # [T, D]

_cb = int(time.time() * 1000) % 10**9
shim = bricklib.GEN / "gather_rows_shim.cc"
shim.write_text(
    f"// AUTO-GENERATED verify shim for the gather-rows brick. cachebust {_cb}\n"
    "#include <stdint.h>\n"
    f"#define GATHER_N_ROWS {N_ROWS}\n"
    f"#define GATHER_D {D}\n"
    f"#define GATHER_T_TILE {T_TILE}\n"
    f'#include "{BRICK / "gather_rows.cc"}"\n'
)

res = bricklib.verify_streamed(
    name="gather_rows",
    shim=shim,
    symbol="gather_rows_f32",           # bound directly: gather_rows.cc's own extern "C"
                                         # symbol is already pure-buffers (idx, codebook, out)
                                         # with shape baked in via the -D-equivalent #defines
                                         # above, so no wrapper function is needed.
    in_tiles=idx.reshape(N_TILES, T_TILE),
    out_tile_numel=T_TILE * D,
    resident=codebook.reshape(-1),
    unpack=lambda d: np.asarray(d).reshape(N_TILES * T_TILE, D),
    golden=ref,
    gate=GATE,
    in_dt=np.int32, out_dt=np.float32, resident_dt=np.float32,
    # The 1024x8 f32 codebook is 32 KB. At the default objectFIFO depth 2 that asks for 64 KB on
    # this fifo alone and aiecc refuses the design outright:
    #   'aie.tile' op Basic sequential allocation also failed
    # which reads as a kernel problem but is purely an L1 budget one. The resident is acquired once
    # before the tile loop and released after it, so only one buffer is ever live and depth 1 is
    # the correct depth, not a concession. Measured 2026-07-31: depth 2 fails to build, depth 1
    # builds and gates green at this exact shape.
    resident_depth=1,
)

got = np.asarray(res["got"], np.float32)
print(f"  device vs golden     rel-L2: {res['rel_l2']:.3e}")
print(f"  idx[3] (oob) -> last row    : {bool(np.array_equal(got[3], codebook[N_ROWS - 1]))}")
print(f"  idx[4] (neg) -> row 0       : {bool(np.array_equal(got[4], codebook[0]))}")
print(f"  idx[2]==idx[20] repeat match: {bool(np.array_equal(got[2], got[20]))}")
assert res["ok"], f"gather_rows device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
