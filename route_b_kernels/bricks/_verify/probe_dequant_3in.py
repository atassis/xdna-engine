#!/usr/bin/env python3
"""int4-dequant blocker probe: feed `scale` as a REAL 3rd input instead of packing it
into the weight buffer.

Background: the fused gemm-int8xint4-dequant brick
delivers `scale` as EXACTLY 0 on device, so C=0 (rel_l2 1.0). The scale was appended to
the padded-B weight buffer (`wbuf=[B_pad|scale]`) and recovered in the shim as
`(const float*)(wbuf + b_bytes)`, to stay inside what was believed to be a 2-input DMA
budget. Baking `scale=1.0` locally in the shim yields NONZERO output, so the kernel body,
the mmul and the bf16 store all work -- only the scale delivery is broken.

`bricklib._build_oneshot` already supports up to 4 inputs, so the open question ("does the
placer allow 3-in?") is directly testable. This probe answers exactly that and nothing else:

  A) scale as a 3rd ObjectFifo input        -> does it place/route, and is scale nonzero?
  B) the same shapes with scale baked to 1  -> the known-good control, to separate
                                               "scale delivery" from the f32-epilogue bug.

Run:  DRAIN_MODS=probe_dequant_3in ./run.sh drain_mods.py
"""
import numpy as np
import ml_dtypes

import bricklib
from bricklib import GEN
from verify_f2b import golden_mod, pack_int4_blocks, pack_int4, tile_pack, tile_unpack

M, K, N, G = 8, 64, 64, 64


def _case(bake_scale_one):
    """Build + run the fused dequant brick. bake_scale_one=True is the control."""
    g, cc = golden_mod("gemm-int8xint4-dequant", "gemm_int8xint4_dequant.cc")
    ngrp = K // G
    rng = np.random.default_rng(17)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)
    if bake_scale_one:
        scale = np.ones((ngrp, N), dtype=np.float32)
    _, ref_bf16 = g.dequant_gemm_ref(a, b, scale, G)
    b_pad = pack_int4_blocks(b, K, N, pack_int4)

    tag = "ones" if bake_scale_one else "real"
    sym = f"gi4dq3_{tag}_{M}x{K}x{N}_g{G}"
    shim = ('#include <stdint.h>\n'
            f'#include "{cc}"\n'
            f'extern "C" void {sym}(const int8_t*a,const int8_t*bb,const float*s,bfloat16*c){{'
            'const int4*b=(const int4*)bb;'
            f'gemm_int8xint4_dequant_tile<{M},{K},{N},{G}>(a,b,s,c);}}')
    shim_p = GEN / f"{sym}_shim.cc"
    shim_p.write_text(shim)

    design = bricklib._build_oneshot(
        sym, shim_p,
        [M * K, b_pad.size, ngrp * N], M * N,
        [np.int8, np.int8, np.float32], ml_dtypes.bfloat16, [])

    import aie.iron as iron
    a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    b_t = iron.tensor(np.ascontiguousarray(b_pad), dtype=np.int8, device="npu")
    s_t = iron.tensor(np.ascontiguousarray(scale.reshape(-1)), dtype=np.float32, device="npu")
    c_t = iron.zeros((M * N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(a_t, b_t, s_t, c_t)
    dev = np.array(c_t.numpy().copy(), copy=True)

    got = tile_unpack(dev, M, N, 4, 16).astype(np.float32)
    ref = ref_bf16.astype(np.float32)
    nz = float(np.abs(got).sum())
    rl2 = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))

    # Per-4x16-tile error map: the f32-epilogue bug shows up as exactly ONE bad tile.
    bad = []
    for mi in range(M // 4):
        for ni in range(N // 16):
            sub_g = got[mi * 4:(mi + 1) * 4, ni * 16:(ni + 1) * 16]
            sub_r = ref[mi * 4:(mi + 1) * 4, ni * 16:(ni + 1) * 16]
            t = float(np.linalg.norm((sub_g - sub_r).ravel()) / (np.linalg.norm(sub_r.ravel()) + 1e-12))
            if t > 1e-2:
                bad.append((mi, ni, round(t, 4)))

    ok = nz > 0.0 and rl2 <= 2e-2
    print(f"[gi4dq-3in scale={tag:4s}] rel_l2={rl2:.3e} nz={nz:.3e} "
          f"bad_tiles={bad} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(name=f"gemm-int8xint4-dequant-3in-{tag}", rel_l2=rl2, ok=ok,
                status="PASS" if ok else "FAIL", bad_tiles=bad, nonzero=nz)


def do_gi4dq_3in_ones():
    """Control: scale==1 delivered over the 3rd fifo. Isolates the f32-epilogue bug."""
    return _case(bake_scale_one=True)


def do_gi4dq_3in_real():
    """The real gate: per-group scale delivered over the 3rd fifo."""
    return _case(bake_scale_one=False)
