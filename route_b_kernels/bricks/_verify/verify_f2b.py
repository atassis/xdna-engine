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
    # The BOUND SYMBOL must be the 3-arg shim, NOT the brick's own exported wrapper.
    # `gemm_int8xint4_dequant_8x64x64_g64` is declared in the .cc we #include and takes
    # FOUR pointers (a, b, scale, c). A core tile only has 2 input DMA channels, so the
    # harness can deliver just 3 buffers (A, wbuf, C) -- binding that 4-arg symbol called
    # it with 3, landing `scale` on the zero-initialised OUTPUT buffer (the long-recorded
    # "scale reads as EXACTLY 0") and leaving the real `c` a dangling register, so every
    # store missed the output and C came back exactly zero. The shim below is the thing
    # that packs scale out of wbuf, and it is what must be invoked.
    fname = "gemm_int8xint4_dequant_8x64x64_g64"   # file stem only
    sym = "gi4dq_verify"                            # the symbol actually called
    shim = ('#include <stdint.h>\n'
            f'#include "{cc}"\n'
            f'extern "C" void {sym}(const int8_t*a,const int8_t*wbuf,bfloat16*c){{'
            'const int4*b=(const int4*)wbuf;'
            f'const float*s=(const float*)(wbuf+{b_bytes});'
            'gemm_int8xint4_dequant_tile<8,64,64,64>(a,b,s,c);}')
    (GEN / f"{fname}_shim.cc").write_text(shim)
    design = bricklib._build_oneshot(sym, GEN / f"{fname}_shim.cc",
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


def _dequant_shape(M, K, N, G, seed):
    """Gate gemm-int8xint4-dequant at an arbitrary exported shape.

    Generalised from the 8x64x64_g64 case so the brick can be pushed toward shapes a real
    model actually uses. NOTE the symbol discipline: the .cc exports a FOUR-pointer wrapper
    (a, b, scale, c) per shape, but a core tile has only 2 input DMA channels so the harness
    delivers 3 buffers (A, wbuf, C). Binding that wrapper silently lands `scale` on the
    zero-initialised output buffer and dangles the real output pointer. So bind the 3-arg
    shim below, never the exported name. (bricklib._check_symbol_arity now enforces this.)
    """
    g, cc = golden_mod("gemm-int8xint4-dequant", "gemm_int8xint4_dequant.cc")
    ngrp = K // G
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)
    _, ref_bf16 = g.dequant_gemm_ref(a, b, scale, G)
    b_pad = pack_int4_blocks(b, K, N, pack_int4)
    b_bytes = int(b_pad.size)
    wbuf = np.concatenate([b_pad, scale.reshape(-1).view(np.int8)])

    fname = f"gemm_int8xint4_dequant_{M}x{K}x{N}_g{G}"
    sym = f"gi4dq_verify_{M}x{K}x{N}_g{G}"
    shim = ('#include <stdint.h>\n'
            f'#include "{cc}"\n'
            f'extern "C" void {sym}(const int8_t*a,const int8_t*wbuf,bfloat16*c){{'
            'const int4*b=(const int4*)wbuf;'
            f'const float*s=(const float*)(wbuf+{b_bytes});'
            f'gemm_int8xint4_dequant_tile<{M},{K},{N},{G}>(a,b,s,c);}}')
    (GEN / f"{fname}_shim.cc").write_text(shim)
    design = bricklib._build_oneshot(sym, GEN / f"{fname}_shim.cc",
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
    rl2 = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))
    ok = rl2 <= 2e-2
    nm = f"gemm-int8xint4-dequant-{M}x{K}x{N}_g{G}"
    print(f"[{nm:34s}] rel_l2={rl2:.3e} gate=2e-2 {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(name=nm, rel_l2=rl2, ok=ok, status="PASS" if ok else "FAIL")


def _dequant_shape_streamed(M, K, N, G, Nt, seed):
    """Gate gemm-int8xint4-dequant at ANY shape, by streaming the weight in N-tiles.

    The one-shot rail stages A, the whole weight+scale buffer and the whole C into L1, so it
    stops at the first shape whose working set passes 64 KB (`_dequant_shape` at 64x128x128
    already fails to BUILD). Tiling the OUTPUT COLUMNS fixes that without touching the kernel:

      C[:, n0:n0+Nt] = f(A, B[:, n0:n0+Nt], scale[:, n0:n0+Nt])

    Every n-tile is an independent M x K x Nt GEMM over the FULL K, so no cross-tile
    accumulation is needed and the kernel template is simply instantiated at N=Nt. A is the
    same for every tile -- it is the RESIDENT operand, acquired once for the whole stream --
    while the weight, which is the operand that actually grows with the model, arrives one
    tile at a time. L1 then holds A + one weight tile + one C tile instead of the whole
    weight, and the shape the rail can reach is set by Nt rather than by N.

    Nt tunes the L1 footprint: a tile costs K*Nt/2 weight bytes + (K/G)*Nt*4 scale bytes.
    """
    g, cc = golden_mod("gemm-int8xint4-dequant", "gemm_int8xint4_dequant.cc")
    assert N % Nt == 0 and Nt % 16 == 0, f"Nt={Nt} must be a multiple of 16 dividing N={N}"
    ngrp = K // G
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)
    _, ref_bf16 = g.dequant_gemm_ref(a, b, scale, G)

    n_tiles = N // Nt
    tiles = []
    for t in range(n_tiles):
        sl = slice(t * Nt, (t + 1) * Nt)
        # Each tile is packed as its OWN M x K x Nt problem: (ki, ni) 16x16 int4 blocks for
        # the slice, then that slice's [K/G, Nt] f32 scales. The kernel indexes both with
        # nTiles = Nt/16, so it needs the slice's layout, not a view into the full weight.
        b_t = pack_int4_blocks(b[:, sl], K, Nt, pack_int4)
        s_t = np.ascontiguousarray(scale[:, sl]).reshape(-1).view(np.int8)
        tiles.append(np.concatenate([b_t, s_t]))
    b_tile_bytes = int(pack_int4_blocks(b[:, :Nt], K, Nt, pack_int4).size)
    in_tiles = np.stack(tiles)

    fname = f"gemm_int8xint4_dequant_{M}x{K}x{N}_g{G}_nt{Nt}_streamed"
    sym = f"gi4dq_stream_{M}x{K}x{N}_g{G}_nt{Nt}"
    # Argument order is the streamed-rail contract: (tile_in, resident, tile_out).
    shim = ('#include <stdint.h>\n'
            f'#include "{cc}"\n'
            f'extern "C" void {sym}(const int8_t*wtile,const int8_t*a,bfloat16*c){{'
            'const int4*b=(const int4*)wtile;'
            f'const float*s=(const float*)(wtile+{b_tile_bytes});'
            f'gemm_int8xint4_dequant_tile<{M},{K},{Nt},{G}>(a,b,s,c);}}')
    (GEN / f"{fname}_shim.cc").write_text(shim)

    def unpack(dev):
        # dev is (n_tiles, M*Nt) bf16, each tile [mTiles][Nt/16][4x16] -> (M, Nt); the tiles
        # are consecutive column slices, so the full C is their horizontal concatenation.
        cols = [tile_unpack(dev[t].reshape(-1), M, Nt, 4, 16) for t in range(n_tiles)]
        return np.concatenate(cols, axis=1).astype(np.float32)

    nm = f"gi4dq-streamed-{M}x{K}x{N}_g{G}_nt{Nt}"
    return bricklib.verify_streamed(
        nm, GEN / f"{fname}_shim.cc", sym,
        in_tiles=in_tiles, out_tile_numel=M * Nt,
        resident=tile_pack(a, 4, 16),
        unpack=unpack, golden=ref_bf16.astype(np.float32), gate=2e-2,
        in_dt=np.int8, out_dt=ml_dtypes.bfloat16, resident_dt=np.int8)


def do_gemm_int8xint4_dequant_64x128x128_streamed():
    """The shape the one-shot rail cannot BUILD, now reachable by streaming the weight."""
    return _dequant_shape_streamed(64, 128, 128, 64, Nt=32, seed=23)
do_gemm_int8xint4_dequant_64x128x128_streamed.brick_name = "gi4dq-streamed-64x128x128"


def do_gemm_int8xint4_dequant_tallk_streamed():
    """The tall-K decode shape -- the reason int4 is a rails-grade lever at all.

    B alone is ~2.1 MB, roughly 33x a core tile's L1, so this is unreachable one-shot by two
    orders of magnitude. At Nt=16 a tile is 1024*16/2 + 8*16*4 = 8.7 KB.
    """
    return _dequant_shape_streamed(4, 1024, 4096, 128, Nt=16, seed=29)
do_gemm_int8xint4_dequant_tallk_streamed.brick_name = "gi4dq-streamed-4x1024x4096"


def do_gemm_int8xint4_dequant_64x128x128():
    """The next exported shape up from the gated 8x64x64.

    This is deliberately a load-bearing probe of the RAIL, not just the kernel: bricklib's
    one-shot builder moves whole operands into L1, and by arithmetic this shape needs
    roughly A 8 KB + wbuf 9 KB + C 16 KB ~= 33 KB, which at objectFIFO depth 2 is ~66 KB
    against a 64 KB L1. So it sits right at (or just past) the one-shot ceiling. Either it
    passes -- int4 extends to a second, larger shape -- or it fails on allocation, which
    confirms on DEVICE that reaching real model shapes needs a streaming/tiled-operand rail
    rather than more one-shot shapes.
    """
    return _dequant_shape(64, 128, 128, 64, seed=23)
do_gemm_int8xint4_dequant_64x128x128.brick_name = "gemm-int8xint4-dequant-64x128x128"


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
