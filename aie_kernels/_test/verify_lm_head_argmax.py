#!/usr/bin/env python3
"""F3 lm-head-argmax device-verify: fused vocab-projection GEMM + argmax.

    logits[VOCAB] = hidden[HIDDEN] @ W[HIDDEN, VOCAB]   (bf16 in, f32 acc)
    (best_value, best_index) = argmax_v(logits)

The kernel co-produces TWO outputs (best_value f32 + best_index i32) but the
verify rail (bricklib.verify_oneshot) has ONE output buffer, so the shim packs
both into a single int32[2]:  out[0] = float_bits(best_value); out[1] = index.

GATE SEMANTICS (the load-bearing bit):
  rel-L2 over the RAW packed [value_bits, index] buffer is MEANINGLESS -- the
  value bits reinterpret to ~1e9 while the index is < VOCAB, so a naive rel-L2
  would be dominated by garbage. The PRIMARY gate here is INDEX-EXACT: the
  device argmax MUST land on the same token id as the golden argmax. The value
  rel-L2 (<= 3e-2, the usual bf16-GEMM tolerance) is only a SECONDARY sanity
  check on the winning logit. We express both through verify_oneshot's single
  rel-L2: `unpack` splits the buffer and, on any index mismatch, POISONS the
  returned value so rel-L2 blows past the gate -- i.e. a wrong argmax can NEVER
  report PASS. When the index matches, rel-L2 collapses to the value rel-L2.

CPU-ONLY entrypoint (`__main__`) does a pure-numpy golden + packing cross-check
and never touches the device. `do_lm_head_argmax()` is DEVICE-run only (it calls
verify_oneshot -> iron.jit); run it under the NPU lock via ./run.sh.
"""
import importlib.util
from pathlib import Path
import numpy as np
import ml_dtypes

import bricklib

BRICKS = Path(__file__).parent.parent
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

# --- on-chip shape ---------------------------------------------------------
# The .cc default HIDDEN=256, VOCAB=512 gives a 256*512*2 = 256KB bf16 weight,
# which blows the core tile's 64KB L1. Shrink via -D overrides so the resident
# weight fits: 64x128 bf16 = 16KB weight + 8*64*2 = 1KB hidden, well under 64KB.
HIDDEN, VOCAB, M_PAD = 64, 128, 8
KT = NT = 8  # mmul<8,8,8> tile widths -- MUST match lm_head_argmax.cc kKTile/kNTile

SYMBOL = "lm_head_argmax_verify_64x128"
COMPILE_FLAGS = [f"-DLMHEAD_HIDDEN={HIDDEN}", f"-DLMHEAD_VOCAB={VOCAB}",
                 f"-DLMHEAD_M_PAD={M_PAD}"]

# Pure-buffer verify shim: #include's the brick .cc (added by verify_oneshot),
# calls the two-output kernel, packs [float_bits(best_value), best_index] -> int32[2].
SHIM_BODY = (
    'extern "C" void %s(const bfloat16* hidden, const bfloat16* weight, int32_t* out){'
    'float best_value; int32_t best_index;'
    'lm_head_argmax(hidden, weight, &best_value, &best_index);'
    'out[0] = __builtin_bit_cast(int32_t, best_value);'
    'out[1] = best_index;}'
) % SYMBOL


def golden_mod(brick, fname):
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick.replace('-', '_')}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / fname)


def pack_hidden(hidden_row_bf16):
    """[HIDDEN] bf16 -> [M_PAD, HIDDEN] zero-padded (row 0 = live query) -> flat
    8x8 A-tiles, kt-major, row-major (M,K) within each block. Matches the kernel's
    A = load_v<M_PAD*KT>(hidden + k*M_PAD) walk (== gemm-int8 tile_pack(_,8,8))."""
    h = np.zeros((M_PAD, HIDDEN), dtype=ml_dtypes.bfloat16)
    h[0] = hidden_row_bf16
    return np.ascontiguousarray(
        h.reshape(M_PAD // 8, 8, HIDDEN // KT, KT).transpose(0, 2, 1, 3).reshape(-1))


def pack_weight(w_bf16):
    """[HIDDEN, VOCAB] bf16 -> nt-major 8x8 blocks, each [K_TILE,N_TILE] row-major.
    Matches the kernel's weight_tile = weight + n*HIDDEN (nt-major) then
    B = load_v<KT*NT>(weight_tile + k*NT) (kt-minor, plain [K,N] block)."""
    Kb, Nb = HIDDEN // KT, VOCAB // NT
    return np.ascontiguousarray(
        w_bf16.reshape(Kb, KT, Nb, NT).transpose(2, 0, 1, 3).reshape(-1))


def do_lm_head_argmax():
    """DEVICE-run verify config. Do NOT call at CPU/import time (triggers iron.jit)."""
    g, cc = golden_mod("lm-head-argmax", "lm_head_argmax.cc")
    # golden reads HIDDEN/VOCAB/M_PAD as module globals -> override to our -D shape.
    g.HIDDEN, g.VOCAB, g.M_PAD = HIDDEN, VOCAB, M_PAD
    hidden_row, weight = g._make_reference_inputs(seed=0)
    ref_value, ref_index = g.lm_head_argmax_golden(hidden_row, weight)
    assert 0 <= ref_index < VOCAB, f"golden index {ref_index} out of [0,{VOCAB})"

    h_bf16 = g.to_bf16(hidden_row).astype(ml_dtypes.bfloat16)
    w_bf16 = g.to_bf16(weight).astype(ml_dtypes.bfloat16)
    hidden_packed = pack_hidden(h_bf16)
    weight_packed = pack_weight(w_bf16)

    def unpack(dev_flat):
        # dev buffer = int32[2] = [float_bits(best_value), best_index].
        dev_flat = np.asarray(dev_flat, np.int32).reshape(-1)
        dev_value = float(dev_flat[0:1].view(np.float32)[0])
        dev_index = int(dev_flat[1])
        print(f"  [lm-head-argmax] dev=(idx={dev_index}, val={dev_value:.6f}) "
              f"ref=(idx={ref_index}, val={ref_value:.6f}) "
              f"index_match={dev_index == ref_index}")
        # PRIMARY gate: index must be exact. Poison the value on mismatch so
        # verify_oneshot's rel-L2 exceeds the gate (wrong argmax => guaranteed FAIL).
        if dev_index != ref_index:
            return np.array([float(ref_value) + 1.0e30], np.float64)
        # SECONDARY gate: value rel-L2 <= 3e-2 (bf16 GEMM tolerance) on the logit.
        return np.array([dev_value], np.float64)

    return bricklib.verify_oneshot(
        "lm-head-argmax", cc, SHIM_BODY, SYMBOL,
        inputs=[(hidden_packed, ml_dtypes.bfloat16),
                (weight_packed, ml_dtypes.bfloat16)],
        out_numel=2, out_shape=(2,),
        unpack=unpack, golden=np.array([float(ref_value)], np.float64),
        gate=3e-2, out_dt=np.int32, compile_flags=COMPILE_FLAGS)
do_lm_head_argmax.brick_name = "lm-head-argmax"


def _reconstruct_logits_from_packed(hidden_packed, weight_packed):
    """CPU replay of the kernel's tiled reads from the PACKED buffers -> row-0
    logits in f64. Validates pack_hidden/pack_weight without a device."""
    hp = np.asarray(hidden_packed, np.float32)
    wp = np.asarray(weight_packed, np.float32)
    Kb, Nb = HIDDEN // KT, VOCAB // NT
    logits = np.zeros(VOCAB, np.float64)
    for nt in range(Nb):
        wtile = wp[nt * (Kb * KT * NT):(nt + 1) * (Kb * KT * NT)]
        acc = np.zeros(NT, np.float64)
        for kt in range(Kb):
            a_row0 = hp[kt * (8 * KT):(kt + 1) * (8 * KT)].reshape(8, KT)[0]  # row 0 = query
            b_blk = wtile[kt * (KT * NT):(kt + 1) * (KT * NT)].reshape(KT, NT)
            acc += a_row0.astype(np.float64) @ b_blk.astype(np.float64)
        logits[nt * NT:(nt + 1) * NT] = acc
    return logits


if __name__ == "__main__":
    # CPU-ONLY: recompute the golden at our small device shape, assert it is a
    # well-formed argmax result, and cross-check the tile-packing helpers. No
    # device / no iron.jit / do NOT call do_lm_head_argmax().
    g, _ = golden_mod("lm-head-argmax", "lm_head_argmax.cc")
    g.HIDDEN, g.VOCAB, g.M_PAD = HIDDEN, VOCAB, M_PAD
    hidden_row, weight = g._make_reference_inputs(seed=0)
    ref_value, ref_index = g.lm_head_argmax_golden(hidden_row, weight)

    assert 0 <= ref_index < VOCAB, f"index {ref_index} out of vocab [0,{VOCAB})"
    assert np.isfinite(ref_value), f"value {ref_value} not finite"

    # Packing cross-check: replaying the kernel's tiled reads over the packed
    # buffers must reproduce the golden argmax (index + logit), proving the
    # host pack_hidden/pack_weight layout matches the .cc load pattern.
    h_bf16 = g.to_bf16(hidden_row).astype(ml_dtypes.bfloat16)
    w_bf16 = g.to_bf16(weight).astype(ml_dtypes.bfloat16)
    logits = _reconstruct_logits_from_packed(pack_hidden(h_bf16), pack_weight(w_bf16))
    pk_index = int(np.argmax(logits))
    pk_value = float(logits[pk_index])
    assert pk_index == ref_index, f"packing argmax {pk_index} != golden {ref_index}"
    val_rl2 = abs(pk_value - ref_value) / (abs(ref_value) + 1e-12)
    assert val_rl2 <= 1e-3, f"packing value rel-L2 {val_rl2:.3e} too large"

    print(f"PASS lm-head-argmax CPU cross-check: HIDDEN={HIDDEN} VOCAB={VOCAB} "
          f"M_PAD={M_PAD} -> best_index={ref_index} best_value={ref_value:.6f} "
          f"(packing index match, value rel-L2={val_rl2:.2e})")
