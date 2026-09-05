#!/usr/bin/env python3
"""F2c device-verify: the bf16 / bfp16 GEMMs (gemm-bfp16-ebs8, gemm-bf16xbfp16).

Both bricks run the resident stream as bf16 in/out; the bfp16ebs8 block-float
conversion happens ON-CHIP inside the kernel (gemm-bfp16-ebs8 converts BOTH
operands; gemm-bf16xbfp16 keeps the activation bf16 and compresses the weight).

Authoring note (why NO host-side bfp16ebs8 packing here): the bfp16ebs8 memory
byte layout is DEVICE-DEFINED (per mlir-aie ml/block_datatypes/gemm_asymmetric_
tile_buffering/gemm_atb_layout.h it is a column-major 8x8 sub-block, 1x2 super-
block pre-shuffle) and is NOT confirmable CPU-only. So both configs feed the
kernel plain bf16 and let aie_api's own block_vector store/load own the layout:
- gemm-bfp16-ebs8: the kernel's public entry already takes bf16 A + bf16 B.
- gemm-bf16xbfp16: whose real entry takes a bfp16ebs8* weight, is verified via
  an ON-CHIP-CONVERT shim -- the shim streams bf16 B, converts each 8x8 block to
  block_vector<bfp16ebs8,64> with the SAME to_v64bfp16ebs8 the kernel uses, and
  writes it through a block_vector_output_buffer_stream into a scratch the core's
  block_vector_input_buffer_stream reads back. Layout is internally consistent by
  construction; the one residual device-gated detail is the within-8x8-block
  orientation the mmul's op_transpose(B) expects (see do_gemm_bf16xbfp16).

CPU gate (this file, no device): the shim bodies compile clean under standalone
Peano (checked separately) and the golden references are green. The rel-L2 gate
is the deferred device step (run do_*() on aie2p under scripts/npu_lock.sh).
"""
import importlib.util
import json
import traceback
from pathlib import Path
import numpy as np

import ml_dtypes
import bricklib
from verify_f2 import tile_pack, tile_unpack

BRICKS = Path(__file__).parent.parent
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

BF16 = ml_dtypes.bfloat16


def golden_mod(brick, fname):
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick.replace('-', '_')}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / fname)


def pack_B_nk_blocks(b, K, N):
    """bf16 weight B[K,N] -> the core's (n_block, k_block) block order, each 8x8
    block transposed to (n_local, k_local) so the mmul's op_transpose(B) recovers
    B[k,n] (device-gated orientation -- see do_gemm_bf16xbfp16). Returns a flat
    bf16 array of length K*N, block (n,k) at offset (n*(K//8)+k)*64."""
    Kb, Nb = K // 8, N // 8
    out = np.empty((Nb, Kb, 8, 8), dtype=b.dtype)
    for n in range(Nb):
        for k in range(Kb):
            blk = b[k * 8:(k + 1) * 8, n * 8:(n + 1) * 8]  # (k_local, n_local)
            out[n, k] = blk.T                              # (n_local, k_local)
    return np.ascontiguousarray(out.reshape(-1))


results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        print(f"[{fn.brick_name:22s}] ERROR: {e}", flush=True)
        traceback.print_exc()
        results.append(dict(name=fn.brick_name, status="ERROR", ok=False, err=str(e)))


def do_gemm_bfp16_ebs8():
    """bf16 A[M,K] @ bf16 B[K,N] -> bf16 C, both operands block-quantized to
    bfp16ebs8 ON-CHIP. Isomorphic to do_gemm_int8 but bf16 dtype + the bf16
    golden. A block-tiled (mi,ki) 8x8, B block-tiled (ki,ni) 8x8, C (mi,ni)."""
    g, cc = golden_mod("gemm-bfp16-ebs8", "gemm_bfp16_ebs8.cc")
    M, K, N = 64, 64, 64                                  # 64x64 bf16 = 8KB/buf, fits 64KB L1
    rng = np.random.default_rng(41)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref = np.asarray(g.gemm_bfp16_ebs8_ref(a, b), np.float32)   # [M,N] bf16-rounded-f32
    a_bf = g.to_bf16(a).astype(BF16)
    b_bf = g.to_bf16(b).astype(BF16)
    # The kernel accumulates with mac_8x8_8x8T, i.e. A @ B^T per 8x8 block, so each
    # B block must be pre-transposed (ki-major/ni-minor block order unchanged, only
    # the within-block [ki,ni] -> [ni,ki]). tile_pack alone left B untransposed,
    # which is the 1.306 rel-L2. transpose(0,2,3,1) = tile_pack's (0,2,1,3) with the
    # inner two axes swapped.
    b_packed = np.ascontiguousarray(
        b_bf.reshape(K // 8, 8, N // 8, 8).transpose(0, 2, 3, 1).reshape(-1))
    return bricklib.verify_oneshot(
        "gemm-bfp16-ebs8", cc, "", "gemm_bfp16_ebs8_64x64x64",
        inputs=[(tile_pack(a_bf, 8, 8), BF16), (b_packed, BF16)],
        out_numel=M * N, out_shape=(M, N),
        unpack=lambda flat: tile_unpack(np.asarray(flat, np.float32), M, N, 8, 8),
        golden=ref, gate=3e-2, out_dt=BF16)
do_gemm_bfp16_ebs8.brick_name = "gemm-bfp16-ebs8"


def do_gemm_bf16xbfp16():
    """bf16 A[M,K] @ bfp16ebs8-weight B[K,N] -> bf16 C. Verified via the on-chip-
    convert shim: host feeds bf16 A (block-tiled (m,k)) + bf16 B (packed in the
    core's (n,k) block order); the shim converts B->bfp16ebs8 on-chip and calls
    gemm_bf16xbfp16_core. DEVICE-GATED: the within-8x8-block orientation
    (pack_B_nk_blocks transposes each block for op_transpose(B)); if the device
    rel-L2 lands near the vs-fp32 format bound instead of ~0, try the
    non-transposed block (drop the .T in pack_B_nk_blocks)."""
    g, cc = golden_mod("gemm-bf16xbfp16", "gemm_bf16xbfp16.cc")
    M, K, N = 64, 64, 64
    rng = np.random.default_rng(43)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref = np.asarray(g.gemm_bf16xbfp16(a, b), np.float32)       # [M,N] fp32-accum
    a_bf = g.to_bf16(a).astype(BF16)
    b_bf = g.to_bf16(b).astype(BF16)
    KB, NB = K // 8, N // 8
    # NOTE: verify_oneshot's shim wrapper already emits `#include <stdint.h>` +
    # `#include "{brick_cc}"`, so this body must NOT re-include the brick .cc
    # (it has no include guard -> double-include = redefinition compile error).
    shim = (
        'extern "C" void gemm_bf16xbfp16_verify(const bfloat16*A,const bfloat16*Bbf16,bfloat16*C){'
        f'constexpr unsigned M={M},K={K},N={N},KB=K/8,NB=N/8,NBLK=KB*NB;'
        'using BV=aie::block_vector<bfp16ebs8,64>;'
        'aie::set_rounding(aie::rounding_mode::conv_even);'
        'alignas(64) static uint8_t Bq_raw[NBLK*BV::memory_bytes()];'
        'bfp16ebs8*Bq=reinterpret_cast<bfp16ebs8*>(Bq_raw);'
        'aie::block_vector_output_buffer_stream<bfp16ebs8,64> bout(Bq);'
        'for(unsigned blk=0;blk<NBLK;++blk){'
        'aie::vector<bfloat16,64> v=aie::load_v<64>(Bbf16+blk*64);'
        'aie::accum<accfloat,64> acc(v);bout.push(::to_v64bfp16ebs8(acc));}'
        'gemm_bf16xbfp16_core<8,8,8,M/8,K/8,N/8>(A,Bq,C);}')
    return bricklib.verify_oneshot(
        "gemm-bf16xbfp16", cc, shim, "gemm_bf16xbfp16_verify",
        inputs=[(tile_pack(a_bf, 8, 8), BF16), (pack_B_nk_blocks(b_bf, K, N), BF16)],
        out_numel=M * N, out_shape=(M, N),
        unpack=lambda flat: tile_unpack(np.asarray(flat, np.float32), M, N, 8, 8),
        golden=ref, gate=3e-2, out_dt=BF16)
do_gemm_bf16xbfp16.brick_name = "gemm-bf16xbfp16"


def _cpu_selfcheck():
    """CPU-only golden cross-check (no device): recompute both references and
    confirm they are finite, correctly shaped, and within their own gate vs the
    fp32 GEMM -- the numpy half of the CPU gate. The do_*() configs above are the
    deferred DEVICE step."""
    ok = True
    # gemm-bfp16-ebs8
    g, _ = golden_mod("gemm-bfp16-ebs8", "gemm_bfp16_ebs8.cc")
    rng = np.random.default_rng(41)
    a = rng.standard_normal((64, 64)).astype(np.float32)
    b = (rng.standard_normal((64, 64)).astype(np.float32) * (1.0 / 8.0))
    ref = np.asarray(g.gemm_bfp16_ebs8_ref(a, b), np.float32)
    fp32 = a @ b
    r = float(np.linalg.norm((ref - fp32).ravel()) / (np.linalg.norm(fp32.ravel()) + 1e-12))
    pb = np.asarray(tile_pack(g.to_bf16(a).astype(BF16), 8, 8))
    good = ref.shape == (64, 64) and np.isfinite(ref).all() and pb.size == 64 * 64 and r <= 0.08
    print(f"[gemm-bfp16-ebs8 ] ref{ref.shape} finite={np.isfinite(ref).all()} "
          f"rel-L2(vs fp32)={r:.4f} tile_pack={pb.size} -> {'PASS' if good else 'FAIL'}")
    ok &= good
    # gemm-bf16xbfp16
    g2, _ = golden_mod("gemm-bf16xbfp16", "gemm_bf16xbfp16.cc")
    rng = np.random.default_rng(43)
    a2 = rng.standard_normal((64, 64)).astype(np.float32)
    b2 = (rng.standard_normal((64, 64)).astype(np.float32) * (1.0 / 8.0))
    ref2 = np.asarray(g2.gemm_bf16xbfp16(a2, b2), np.float32)
    fp32b = a2 @ b2
    r2 = float(np.linalg.norm((ref2 - fp32b).ravel()) / (np.linalg.norm(fp32b.ravel()) + 1e-12))
    packB = pack_B_nk_blocks(g2.to_bf16(b2).astype(BF16), 64, 64)
    good2 = ref2.shape == (64, 64) and np.isfinite(ref2).all() and packB.size == 64 * 64 and r2 <= 0.03
    print(f"[gemm-bf16xbfp16 ] ref{ref2.shape} finite={np.isfinite(ref2).all()} "
          f"rel-L2(vs fp32)={r2:.4f} packB={packB.size} -> {'PASS' if good2 else 'FAIL'}")
    ok &= good2
    return ok


if __name__ == "__main__":
    import sys
    print("verify_bfp16 CPU golden cross-check (device rel-L2 gate is deferred):")
    sys.exit(0 if _cpu_selfcheck() else 1)
