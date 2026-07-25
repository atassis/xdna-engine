#!/usr/bin/env python3
"""Gate the argmax_slice kernel at GEMMA's chunk size on the RELIABLE brick rail.

WHY THIS EXISTS (separating two questions that got tangled).

The on-device argmax for an LLM lm-head has to answer two independent things:
  (1) is the KERNEL correct when a column's slice is tiled to fit L1?
  (2) does the FusedMLIROperator buffer plumbing deliver/return it?

Trying to answer both at once in `xdna-engine/route_b_kernels/decode_fused/probe_argmax_vocab.py`
went badly: that path returns all-zero reads most of the time. It DID once produce an exactly
correct global index over the full 262144 vocab (dev 100929 == ref 100929, value bit-equal),
which is not something a broken tiling produces by luck -- but a later run gave all-zeros
across 60 consecutive dispatches, so it cannot be gated there.

Meanwhile the brick rail in this directory ran 22 bricks the same day with run2run=0.00e+00
every time. So: gate the KERNEL here, where reads are trustworthy, and leave the fusion
plumbing as its own separate problem.

Shape mapping -- the rowwise rail already IS the chunked argmax:
  m        = COLS * NCHUNKS = 64   tiles (what a chunked design streams)
  in_cols  = CHUNK = 4096          one tile of a column's slice
  out_cols = 4 bf16 = 8 bytes      the packed [max_val:f32 | local_idx:i32]

The gate is INDEX-EXACTNESS per tile plus an exact global reduce, never rel-L2 -- a
near-miss index is a wrong token.
"""
import numpy as np
import ml_dtypes

import bricklib

BF16 = ml_dtypes.bfloat16
ARGMAX_CC = (""
             "route_b_kernels/decode_fused/argmax_slice.cc")

COLS, CHUNK = 8, 4096
NCHUNKS = 8                      # 32768-element slice / 4096 = what Gemma's vocab needs
M = COLS * NCHUNKS               # 64 tiles
N = COLS * NCHUNKS * CHUNK       # 262144
PACK = 4                         # bf16 slots = 8 bytes


def _decode(dev_rows):
    """(m, PACK) bf16 rows -> [(val, local_idx)] per tile."""
    out = []
    for r in range(dev_rows.shape[0]):
        b = np.asarray(dev_rows[r], BF16).tobytes()
        out.append((float(np.frombuffer(b[0:4], np.float32)[0]),
                    int(np.frombuffer(b[4:8], np.int32)[0])))
    return out


def _case(name, rng):
    x = (rng.standard_normal(N) * 0.5).astype(np.float32)
    if name == "max_in_first_tile":
        x[17] = 50.0
    elif name == "max_in_last_tile":
        x[N - 3] = 50.0
    elif name == "max_at_tile_boundary":
        x[CHUNK - 1] = 50.0
    elif name == "duplicate_max":
        x[100] = 50.0
        x[N - 100] = 50.0
    return np.asarray(x, BF16)


def _run(name, x):
    """Stream the N-element vector as M tiles of CHUNK through argmax_slice_bf16."""
    sym = "argmax_slice_probe"
    shim = (f'extern "C" void {sym}(bfloat16* in, bfloat16* out) {{ '
            f'argmax_slice_bf16(in, out, {CHUNK}); }}')
    # expected is only used for the rail's rel-L2 print; the real gate is below.
    exp = np.zeros((M, PACK), np.float32)
    r = bricklib.verify_rowwise(
        f"argmax-slice-{name}", ARGMAX_CC, shim, sym,
        M, CHUNK, PACK, x.reshape(M, CHUNK), exp, gate=1e9,
        in_dt=BF16, out_dt=BF16)
    return r["got"]


def do_argmax_slice_gemma_chunk():
    rng = np.random.default_rng(7)
    cases = ["random", "max_in_first_tile", "max_in_last_tile",
             "max_at_tile_boundary", "duplicate_max"]
    npass = 0
    for nm in cases:
        x = _case(nm, rng)
        xf = x.astype(np.float32)
        ref = int(np.argmax(xf))
        got_rows = _run(nm, x)
        parts = _decode(np.asarray(got_rows).reshape(M, PACK))

        # per-tile exactness: each tile's reported local index must be that tile's argmax
        bad_tiles = []
        for t, (v, li) in enumerate(parts):
            tile = xf[t * CHUNK:(t + 1) * CHUNK]
            if li != int(np.argmax(tile)):
                bad_tiles.append((t, li, int(np.argmax(tile))))
        # global reduce, first-max-wins (matches kernel's strict > and numpy)
        best_v, best_g = -np.inf, -1
        for t, (v, li) in enumerate(parts):
            if v > best_v:
                best_v, best_g = v, t * CHUNK + li

        ok = (best_g == ref) and not bad_tiles
        npass += ok
        print(f"[argmax-slice {nm:20s}] dev_idx={best_g:7d} ref_idx={ref:7d} "
              f"bad_tiles={len(bad_tiles)} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if bad_tiles[:3]:
            print(f"    first bad tiles (tile, dev_local, ref_local): {bad_tiles[:3]}", flush=True)

    status = "PASS" if npass == len(cases) else "FAIL"
    print(f"[argmax-slice] {npass}/{len(cases)} index-exact -> {status}", flush=True)
    return dict(name="argmax-slice-gemma-chunk", rel_l2=0.0 if npass == len(cases) else 1.0,
                ok=npass == len(cases), status=status)


do_argmax_slice_gemma_chunk.brick_name = "argmax-slice-gemma-chunk"

if __name__ == "__main__":
    do_argmax_slice_gemma_chunk()
