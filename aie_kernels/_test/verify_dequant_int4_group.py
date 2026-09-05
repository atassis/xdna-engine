#!/usr/bin/env python3
"""Device-verify config for the `dequant-int4-group` brick (brick-wave1-device-verify).

FORMAT-layer primitive: one row of `cols` int4 GROUP-quant values (2 vals/byte, low
nibble = even col, high = odd col, signed two's-complement [-8,7]) -> bf16 dequant
`(q - zp) * scale` at GROUP granularity. Symmetric (HAS_ZP=0, zp folded away) is the
primary case; asymmetric (HAS_ZP=1) is a compile-flag variant.

RAIL SHAPE (mirrors verify_f2b's dequant): a core tile has only 2-in/2-out DMA
channels but this op has THREE logical inputs (packed uint8 / scale f32 / zp int8).
We pack all three into ONE resident input buffer and split it by byte offset inside
the shim -- so exactly 1 input fifo + 1 output fifo cross the tile boundary.

  input buffer layout (always, both sym and asym):
      [ packed : cols/2 bytes | scale : ngroups*4 bytes | zp : ngroups bytes ]
  The zp slot is ALWAYS present (zeros for the symmetric case) so the shim's pointer
  arithmetic stays in-bounds and the two builds differ ONLY by the -DHAS_ZP flag.

The shim calls the TEMPLATE `dequant_int4_group_row<N,GROUP>` directly (not the
extern-C default wrapper) so N/GROUP are explicit and unambiguous. Output is aie
`bfloat16`; host dtype = ml_dtypes.bfloat16. Golden already rounds through the bf16
grid (round-to-nearest-even == conv_even), so it is directly rel-L2 comparable.

The do_* fns DRIVE THE DEVICE (iron.jit build) -- run them only in a device session,
never at import/CPU time. `__main__` here is a CPU-ONLY golden cross-check.
"""
import importlib.util
from pathlib import Path
import numpy as np

BRICK_DIR = Path(__file__).parent.parent / "dequant-int4-group"
BRICK_CC = str(BRICK_DIR / "dequant_int4_group.cc")

# Verify shape. cols=256/GROUP=64 -> 4 groups; N=16 = one bf16 store-vector width
# (matches the brick's default extern-C instantiation dequant_int4_group_row<16,64>).
# 256 cols is tiny vs the 64KB L1 (packed=128B + scale=16B + zp=4B).
COLS, GROUP, N = 256, 64, 16
SEED = 0


def _golden():
    """Load the brick's own golden.py by path (its pack_int4_row is authoritative)."""
    spec = importlib.util.spec_from_file_location(
        "dequant_int4_group_golden", BRICK_DIR / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_case(has_zp):
    """Build one quantized int4 row + per-group scale/zp via the GOLDEN module and its
    reference (symmetric or asymmetric). Returns (golden_mod, packed, scale, zp, ref)."""
    g = _golden()
    rng = np.random.default_rng(SEED)
    ngroups = COLS // GROUP
    q_true = rng.integers(-8, 8, size=COLS, dtype=np.int16)
    packed = g.pack_int4_row(q_true)                                   # uint8[cols/2]
    scale = (0.01 + rng.random(ngroups)).astype(np.float32)           # f32[ngroups] > 0
    zp = rng.integers(-2, 3, size=ngroups, dtype=np.int8)             # int8[ngroups]
    ref = g.dequant_int4_group_golden(packed, scale, (zp if has_zp else None),
                                      COLS, GROUP)                     # f32[cols], bf16-grid
    return g, packed, scale, zp, ref


def _pack_wbuf(packed, scale, zp):
    """[packed uint8 | scale f32-as-bytes | zp int8-as-bytes] -> one uint8 buffer.
    scale starts at cols/2 (=128 here, 4-byte aligned) so `const float*` reads are aligned."""
    return np.concatenate([
        np.asarray(packed, np.uint8),
        np.ascontiguousarray(np.asarray(scale, np.float32)).view(np.uint8),
        np.ascontiguousarray(np.asarray(zp, np.int8)).view(np.uint8),
    ])


def _shim_body(symbol):
    """Pure-buffer verify shim: split the one packed input by byte offset, call the
    brick template, write bf16. `bfloat16` is in scope (brick #includes aie_api)."""
    packed_bytes = COLS // 2
    scale_bytes = (COLS // GROUP) * 4
    return (
        f'extern "C" void {symbol}(uint8_t* wbuf, bfloat16* out) {{\n'
        f'  const uint8_t* packed = wbuf;\n'
        f'  const float*   scale  = (const float*)(wbuf + {packed_bytes});\n'
        f'  const int8_t*  zp     = (const int8_t*)(wbuf + {packed_bytes} + {scale_bytes});\n'
        f'  dequant_int4_group_row<{N}, {GROUP}>(packed, scale, zp, out, {COLS});\n'
        f'}}'
    )


def _verify(has_zp):
    """DEVICE-RUN. Builds the xclbin via iron.jit and gates rel-L2 <= 3e-2 vs golden."""
    import bricklib          # imports aie.iron -> lazy so __main__ stays CPU-only
    import ml_dtypes
    g, packed, scale, zp, ref = _make_case(has_zp)
    wbuf = _pack_wbuf(packed, scale, zp)
    name = "dequant-int4-group-haszp" if has_zp else "dequant-int4-group"
    sym = ("dq_int4_grp_haszp_c%d_g%d" if has_zp else "dq_int4_grp_sym_c%d_g%d") % (COLS, GROUP)
    return bricklib.verify_oneshot(
        name, BRICK_CC, _shim_body(sym), sym,
        inputs=[(wbuf, np.uint8)],
        out_numel=COLS, out_shape=(COLS,),
        unpack=lambda flat: np.asarray(flat).reshape(-1).astype(np.float32),
        golden=ref, gate=3e-2,
        compile_flags=(["-DHAS_ZP=1"] if has_zp else []),
        out_dt=ml_dtypes.bfloat16)


def do_dequant_int4_group():
    """Primary: symmetric int4 group dequant (zp folded away, HAS_ZP=0). DEVICE-RUN."""
    return _verify(has_zp=False)
do_dequant_int4_group.brick_name = "dequant-int4-group"


def do_dequant_int4_group_haszp():
    """Variant: asymmetric int4 group dequant (HAS_ZP=1). DEVICE-RUN."""
    return _verify(has_zp=True)
do_dequant_int4_group_haszp.brick_name = "dequant-int4-group-haszp"


if __name__ == "__main__":
    # CPU-ONLY golden cross-check. Does NOT touch the device / iron.jit.
    g = _golden()
    rng = np.random.default_rng(SEED)
    ngroups = COLS // GROUP

    q_true = rng.integers(-8, 8, size=COLS, dtype=np.int16)
    packed = g.pack_int4_row(q_true)
    q_back = g.unpack_int4_row(packed, COLS)
    assert np.array_equal(q_true, q_back), "pack/unpack round-trip mismatch"

    scale = (0.01 + rng.random(ngroups)).astype(np.float32)
    zp = rng.integers(-2, 3, size=ngroups, dtype=np.int8)

    y_sym = g.dequant_int4_group_golden(packed, scale, None, COLS, GROUP)   # HAS_ZP=0
    y_asym = g.dequant_int4_group_golden(packed, scale, zp, COLS, GROUP)    # HAS_ZP=1

    assert y_sym.shape == (COLS,) and y_asym.shape == (COLS,), "output shape"
    assert y_sym.dtype == np.float32 and y_asym.dtype == np.float32, "golden dtype"
    assert np.all(np.isfinite(y_sym)) and np.all(np.isfinite(y_asym)), "finiteness"
    assert g.rel_l2(y_sym, y_sym) == 0.0, "self rel_l2 must be 0"
    if np.any(zp != 0):
        assert not np.array_equal(y_sym, y_asym), "zp!=0 should change the result"

    # byte-layout of the packed device input buffer (sanity: sizes line up)
    wbuf = _pack_wbuf(packed, scale, zp)
    assert wbuf.size == COLS // 2 + ngroups * 4 + ngroups, "wbuf byte layout"

    print(f"PASS dequant-int4-group CPU golden cross-check "
          f"(cols={COLS} group={GROUP} ngroups={ngroups}; sym+asym finite, "
          f"round-trip ok, wbuf={wbuf.size}B)")
