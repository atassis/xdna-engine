#!/usr/bin/env python3
"""Device rel-L2 verify for the gather-rows brick. Gate 3e-2. Run under the device lock.

Gated at the RESIDUAL-codebook regime (n_rows=1024, D=8): the shape that fits a core tile
resident in f32 (32 KB, see gather_rows.cc's header). The SEMANTIC codebook (n_rows=4096)
does not fit this contract in f32 and is explicitly out of scope for contract (a) -- see the
brick header's CONTRACT CHOSEN section.

SECOND, ADDITIONAL gate below: the SEMANTIC codebook (n_rows=4096) served as 4 resident
chunks of 1024 rows each -- exactly the shape gated above, unmodified. gather_rows.cc is not
touched: the driver runs the SAME gather_rows_f32(N_ROWS=1024,...) kernel 4 times, once per
chunk, with the SAME local-index tile every time (idx % 1024) and only the resident codebook
DATA differing (rows [c*1024:(c+1)*1024)); the host then does a disjoint SELECT -- for output
row t, chunk_of[t] = idx_clamped[t] // 1024 is a single value in [0,3], so exactly one chunk's
row t is kept and the other 3 are discarded, never summed. This is exact by construction (the
4 row-ranges partition [0,4096) with no overlap), so this gate must be bit-exact too, same as
the 1024-row gate above.

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

# ============================================================================================
# SEMANTIC codebook (n_rows=4096), served as 4x already-gated 1024-row chunks. See the module
# docstring. gather_rows.cc / the shim above are REUSED VERBATIM -- N_ROWS=1024, D=8,
# T_TILE=16 are identical to the residual-codebook gate, so this is the same compiled kernel
# shape, only dispatched 4 times against different resident data.
N_CHUNKS = 4
N_ROWS_SEM = N_CHUNKS * N_ROWS      # 4096
assert N_ROWS_SEM == 4096 and N_ROWS_SEM % N_CHUNKS == 0

rng_sem = np.random.default_rng(1)   # distinct stream from the residual codebook's rng above
codebook_sem = rng_sem.standard_normal((N_ROWS_SEM, D)).astype(np.float32)

# Same edge-case discipline as the residual gate, PLUS every chunk boundary explicitly (1023/
# 1024, 2047/2048, 3071/3072): a disjoint-select combine is exactly where an off-by-one in
# chunk_of/local_idx would silently read the wrong chunk's row.
idx_sem = rng_sem.integers(0, N_ROWS_SEM, size=T).astype(np.int32)
idx_sem[0] = 0                        # first row, chunk 0
idx_sem[1] = N_ROWS_SEM - 1           # last row, chunk 3
idx_sem[2] = idx_sem[20] = 777        # deliberate repeat, different tiles, chunk 0
idx_sem[3] = N_ROWS_SEM + 50          # out-of-range -> clamp to 4095 -> chunk 3
idx_sem[4] = -3                       # negative -> clamp to 0 -> chunk 0
idx_sem[5] = N_ROWS - 1               # 1023, last of chunk 0
idx_sem[6] = N_ROWS                   # 1024, first of chunk 1
idx_sem[7] = 2 * N_ROWS - 1           # 2047, last of chunk 1
idx_sem[8] = 2 * N_ROWS               # 2048, first of chunk 2
idx_sem[9] = 3 * N_ROWS - 1           # 3071, last of chunk 2
idx_sem[10] = 3 * N_ROWS              # 3072, first of chunk 3

ref_sem = golden.gather_rows_ref(codebook_sem, idx_sem)   # [T, D], true 4096-row golden

# idx_clamped/local_idx/chunk_of are HOST arithmetic, matching rvq_lookup's own
# np.clip(idx, 0, size-1) exactly -- computed once, fed identically to all 4 dispatches.
idx_clamped = np.clip(idx_sem.astype(np.int64), 0, N_ROWS_SEM - 1)
local_idx = (idx_clamped % N_ROWS).astype(np.int32)   # always in [0, N_ROWS-1]: kernel's own
                                                        # internal clamp never trips
chunk_of = (idx_clamped // N_ROWS).astype(np.int64)    # always in [0, N_CHUNKS-1]

chunk_got = []
for c in range(N_CHUNKS):
    sub_codebook = codebook_sem[c * N_ROWS:(c + 1) * N_ROWS]
    sub_ref = golden.gather_rows_ref(sub_codebook, local_idx)   # what THIS chunk's raw
                                                                  # gather must produce,
                                                                  # regardless of ownership
    res_c = bricklib.verify_streamed(
        name=f"gather_rows_sem_chunk{c}",
        shim=shim,                          # same generated shim, N_ROWS=1024/D=8/T_TILE=16
        symbol="gather_rows_f32",
        in_tiles=local_idx.reshape(N_TILES, T_TILE),   # identical on every chunk dispatch
        out_tile_numel=T_TILE * D,
        resident=sub_codebook.reshape(-1),
        unpack=lambda d: np.asarray(d).reshape(N_TILES * T_TILE, D),
        golden=sub_ref,
        gate=GATE,
        in_dt=np.int32, out_dt=np.float32, resident_dt=np.float32,
        resident_depth=1,
    )
    assert res_c["ok"], f"chunk {c} gate failed: {res_c['status']} rel_l2={res_c['rel_l2']}"
    chunk_got.append(np.asarray(res_c["got"], np.float32))

# Host disjoint SELECT: for row t, exactly one chunk owns it (chunk_of[t]); the other 3
# chunks' row t is computed but never read here -- a select, not a sum.
got_sem = np.empty((T, D), np.float32)
for t in range(T):
    got_sem[t] = chunk_got[chunk_of[t]][t]

rel_l2_sem = golden.rel_l2(got_sem, ref_sem)
print(f"  [semantic 4x1024] device vs golden rel-L2: {rel_l2_sem:.3e}")
print(f"  idx_sem[3] (oob) -> last row : {bool(np.array_equal(got_sem[3], codebook_sem[N_ROWS_SEM - 1]))}")
print(f"  idx_sem[4] (neg) -> row 0    : {bool(np.array_equal(got_sem[4], codebook_sem[0]))}")
print(f"  idx_sem[2]==idx_sem[20] match: {bool(np.array_equal(got_sem[2], got_sem[20]))}")
for t in (5, 6, 7, 8, 9, 10):
    ok_b = bool(np.array_equal(got_sem[t], codebook_sem[int(idx_clamped[t])]))
    print(f"  idx_sem[{t}]={int(idx_sem[t]):5d} chunk boundary row match: {ok_b}")
    assert ok_b, f"chunk-boundary row mismatch at t={t}, idx={int(idx_sem[t])}"
assert np.array_equal(got_sem, ref_sem), (
    f"semantic chunked-gather must be BIT-EXACT vs the 4096-row golden, rel_l2={rel_l2_sem:.3e}")
print("PASS (semantic 4x1024 chunked)")
