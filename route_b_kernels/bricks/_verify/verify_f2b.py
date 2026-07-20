#!/usr/bin/env python3
"""F2b device-verify: int4 GEMM/GEMV (mmul_8_4 = mmul<4,16,16>).

HISTORY: root cause cracked 2026-07-20 (identity + stride probes) -- the kernel strided B
blocks by MMUL::size_B(256) on an int4* whose arithmetic advances 1 byte/int4, so each 16x16
B block landed on a 256-BYTE slot (128 packed + 128 pad). The device-verify harness masked
that with a padded pack.

UPDATE 2026-07-20 (int4-gemm-kernel-stride-fix): the KERNELS are now fixed to stride by
size_B/2 = 128 bytes, so a CONTIGUOUS packed weight (128 bytes/block, no gaps -- what a real
consumer emits, half the LPDDR bytes) is correct. `pack_int4_blocks` below now produces that
contiguous layout; the harness padding is retired. A tiling + C tiling are plain 4x16 blocks.
Device re-verify (rel-L2 = 0 with contiguous pack) is the deferred gate that fully closes both
int4-gemm-kernel-stride-fix and the gemm/gemv-int8xint4 bricks.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path
import numpy as np

import ml_dtypes
import bricklib
from verify_f2 import tile_pack, tile_unpack

BRICKS = Path(__file__).parent.parent
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)


def golden_mod(brick, fname):
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick.replace('-', '_')}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / fname)


def pack_int4(x):
    """standard aie_api sub-byte pack: 2 signed int4/byte, low nibble=even, high=odd."""
    flat = x.reshape(-1)
    lo = (flat[0::2].astype(np.uint8) & 0x0F)
    hi = (flat[1::2].astype(np.uint8) & 0x0F)
    return (lo | (hi << 4)).astype(np.uint8)


def pack_int4_blocks(b, K, N, pack_fn):
    """int4 B -> (ki,ni) 16x16 blocks in (ki-major, ni-minor) order, each pack_int4'd to a
    CONTIGUOUS 128-byte slot (no padding). Matches the fixed kernel stride size_B/2 = 128
    (int4-gemm-kernel-stride-fix). Returns int8 byte buffer of size Kb*Nb*128."""
    Kb, Nb = K // 16, N // 16
    blocks = b.reshape(Kb, 16, Nb, 16).transpose(0, 2, 1, 3)  # (ki,ni,16,16)
    out = []
    for ki in range(Kb):
        for ni in range(Nb):
            out.append(pack_fn(blocks[ki, ni].reshape(-1)).astype(np.int8))  # 128 bytes, no pad
    return np.concatenate(out)


results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        print(f"[{fn.brick_name:22s}] ERROR: {e}", flush=True)
        traceback.print_exc()
        results.append(dict(name=fn.brick_name, status="ERROR", ok=False, err=str(e)))


def _run(sym, shim_body, a_bytes, b_bytes, out_numel, cflags, a_np, b_padded, out_dt=np.int32):
    shim = GEN / f"{sym}_shim.cc"
    shim.write_text(shim_body)
    design = bricklib._build_oneshot(sym, shim, [a_bytes, b_bytes], out_numel,
                                     [np.int8, np.int8], out_dt, cflags)
    import aie.iron as iron
    a_t = iron.tensor(np.ascontiguousarray(a_np), dtype=np.int8, device="npu")
    b_t = iron.tensor(np.ascontiguousarray(b_padded), dtype=np.int8, device="npu")
    c_t = iron.zeros((out_numel,), dtype=out_dt, device="npu")
    design(a_t, b_t, c_t)
    return np.array(c_t.numpy().copy(), copy=True)


def do_gemm_int8xint4():
    g, cc = golden_mod("gemm-int8xint4", "gemm_int8xint4.cc")
    M, K, N = 64, 64, 64
    rng = np.random.default_rng(31)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    ref = g.gemm_int8xint4_ref(a, b)
    sym = "gemm_int8xint4_64x64x64"
    (GEN / f"{sym}_shim.cc").write_text(f'#include <stdint.h>\n#include "{cc}"\n')
    b_pad = pack_int4_blocks(b, K, N, pack_int4)
    dev = _run(sym, f'#include <stdint.h>\n#include "{cc}"\n',
               M * K, b_pad.size, M * N, [],
               tile_pack(a, 4, 16), b_pad)
    got = tile_unpack(dev, M, N, 4, 16)
    rl2 = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))
    ok = rl2 <= 3e-2
    print(f"[gemm-int8xint4       ] rel_l2={rl2:.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(name="gemm-int8xint4", rel_l2=rl2, ok=ok, status="PASS" if ok else "FAIL")
do_gemm_int8xint4.brick_name = "gemm-int8xint4"


def do_gemv_int8xint4():
    g, cc = golden_mod("gemv-int8xint4", "gemv_int8xint4.cc")
    M, K, N = 4, 64, 64
    rng = np.random.default_rng(9)
    a_row = rng.integers(-127, 128, (K,), dtype=np.int64).astype(np.int8)
    a = np.zeros((M, K), dtype=np.int8)
    a[0] = a_row
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    ref_row = g.gemv_int8xint4_ref(a_row, b)                 # [N] int32, row 0
    sym = "gemv_int8xint4"
    # gemv stages A itself from PLAIN ROW-MAJOR [M,K] (not block-tiled, unlike gemm).
    # B = (kt,nt) blocks with the padded 256-byte int4 stride; C = (nt) 4x16 blocks.
    b_pad = pack_int4_blocks(b, K, N, pack_int4)
    dev = _run(sym, f'#include <stdint.h>\n#include "{cc}"\n',
               M * K, b_pad.size, M * N,
               [f"-DGEMV_K={K}", f"-DGEMV_N={N}"],
               a.reshape(-1), b_pad)
    got = tile_unpack(dev, M, N, 4, 16)[0]                   # row 0
    rl2 = float(np.linalg.norm((got - ref_row).ravel()) / (np.linalg.norm(ref_row.ravel()) + 1e-12))
    ok = rl2 <= 3e-2
    print(f"[gemv-int8xint4       ] rel_l2={rl2:.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(name="gemv-int8xint4", rel_l2=rl2, ok=ok, status="PASS" if ok else "FAIL")
do_gemv_int8xint4.brick_name = "gemv-int8xint4"


def do_gemm_int8xint4_dequant():
    g, cc = golden_mod("gemm-int8xint4-dequant", "gemm_int8xint4_dequant.cc")
    M, K, N, G = 8, 64, 64, 64
    ngrp = K // G
    rng = np.random.default_rng(17)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)   # AWQ-ish [K/G,N]
    _, ref_bf16 = g.dequant_gemm_ref(a, b, scale, G)                 # [M,N] f32(bf16-rounded)
    b_pad = pack_int4_blocks(b, K, N, pack_int4)                     # 256B/block padded
    b_bytes = b_pad.size
    # pack scale (row-major [K/G,N] f32) into the weight buffer after B; shim splits.
    wbuf = np.concatenate([b_pad, scale.reshape(-1).view(np.int8)])
    sym = "gemm_int8xint4_dequant_8x64x64_g64"
    shim = ('#include <stdint.h>\n'
            f'#include "{cc}"\n'
            'extern "C" void gi4dq_verify(const int8_t*a,const int8_t*wbuf,bfloat16*c){'
            'const int4*b=(const int4*)wbuf;'
            f'const float*s=(const float*)(wbuf+{b_bytes});'
            'gemm_int8xint4_dequant_tile<8,64,64,64>(a,b,s,c);}')
    (GEN / f"{sym}_shim.cc").write_text(shim)
    design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                     [M * K, wbuf.size], M * N,
                                     [np.int8, np.int8], ml_dtypes.bfloat16, [])
    import aie.iron as iron
    a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    w_t = iron.tensor(np.ascontiguousarray(wbuf), dtype=np.int8, device="npu")
    c_t = iron.zeros((M * N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(a_t, w_t, c_t)
    dev = np.array(c_t.numpy().copy(), copy=True)
    got = tile_unpack(dev, M, N, 4, 16).astype(np.float32)
    ref = ref_bf16.astype(np.float32)
    print(f"  [dbg] dev nz-sum={np.abs(dev.astype(np.float32)).sum():.3e} "
          f"dev[:4]={dev.astype(np.float32).reshape(-1)[:4]} ref[0,:4]={ref[0,:4]}", flush=True)
    rl2 = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))
    ok = rl2 <= 2e-2
    print(f"[gemm-int8xint4-dequant] rel_l2={rl2:.3e} gate=2e-2 {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(name="gemm-int8xint4-dequant", rel_l2=rl2, ok=ok, status="PASS" if ok else "FAIL")
do_gemm_int8xint4_dequant.brick_name = "gemm-int8xint4-dequant"


if __name__ == "__main__":
    for fn in (do_gemm_int8xint4, do_gemv_int8xint4, do_gemm_int8xint4_dequant):
        guard(fn)
    print("\n==== F2b (int4, padded stride) SUMMARY ====", flush=True)
    for r in results:
        print(f"  {r['name']:22s} {r['status']:10s} rel_l2={r.get('rel_l2', float('nan')):.3e}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"F2b: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps(results))
    import os
    os._exit(0)
