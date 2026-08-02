#!/usr/bin/env python3
"""Dump the keys rope_lut.cc's VECTOR quantize chain actually produces, at pos=0.

The gather is exonerated (probe_gather_width: 16 distinct keys in, 16 distinct values out, 0
duplicates). So the 2026-07-26 signature -- pos=0, where every key must be 0, yet lanes 8-15 read
LUT index +8 -- cannot come from the gather: a correct gather returns equal values for equal keys.
The keys reaching fetch() were therefore not all zero.

probe_keybuf.py does NOT settle this. It replicates the key computation in SCALAR C, while the
kernel deliberately uses vector ops -- rope_lut.cc's own comment records that the aie2p scalar-f32
path miscompiles (row-invariant key_buf). A scalar probe cannot see a defect in the vector chain.

This replicates the kernel's vector chain OP FOR OP and reads the keys back. Keys are widened to
int32 for the store: `vector<int8,16>` is sub-native and storing it is a KNOWN hazard (it wrote past
its 16 bytes and corrupted following keys -- the bug the fused rewrite removed by not storing at
all). Widening keeps this probe measuring the COMPUTE rather than re-triggering the store bug.

pos=0 makes the expected answer exact and independent of inv_freq: theta = 0*inv_freq = 0 for every
lane, so every key must be 0. Any non-zero lane localises the defect inside the quantize chain, and
the lane pattern says which op.

STATUS 2026-08-01: TRUSTWORTHY AT pos=0 ONLY. Do not read its pos!=0 rows as a kernel finding.

  pos=0 is stable and reproducible across every variant run: all 64 keys are 0, in both arms. That
  result is real and is what this probe was built to test.

  pos!=0 is NOT measuring the kernel. `const int32_t p = pos[m]` is a SCALAR int32 load, and
  rope_lut.cc's own comment records that the aie2p scalar path miscompiles here -- "scalar
  (float)pos[m] / scalar float mul collapse to a constant on device -> key_buf becomes
  row-invariant". Rows 1..3 come back all-zero, i.e. behaving like row 0, which is that miscompile
  reproducing rather than new information.

  Two artifacts were chased and identified before that, both worth knowing:
  (1) At the natural offset of M int32, `inv_freq` was 16-byte aligned while a 16-lane float load_v
      is 64 bytes. Unaligned L1 vector loads SNAP to the aligned base, so the load reread the pos
      words -- bit patterns 0..3 are denormals ~ 0 -- and the keys came back shifted right by 4
      lanes with zeros in front. That looks EXACTLY like a kernel lane-offset defect and is not one.
      Fixed by POS_PAD; kept as a warning.
  (2) Arm B's output changed when arm A's unrelated store was added, so a single-store reading of
      this shim is not safe to trust either.

  To make this probe valid for pos!=0, the row index must reach the vector chain without a scalar
  f32 dependency -- e.g. feed a precomputed posf VECTOR per row from the host, or index a vector
  load of pos. Until then, only the pos=0 row is evidence.
"""
import importlib.util

import numpy as np

from bricklib import GEN, iron, _build_oneshot

spec = importlib.util.spec_from_file_location(
    "g", "route_b_kernels/bricks/rope-lut/golden.py"
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

M, ROT = 4, 128
HALF = ROT // 2
KVEC = 16

inv_freq = g.build_inv_freq(ROT).astype(np.float32)
pos = np.arange(M, dtype=np.int32)
# inv_freq must start on a 64-byte boundary: a 16-lane float load_v is 64 bytes and an UNALIGNED
# L1 vector load SNAPS to the aligned base (see dwconv1d.cc). At the natural offset of M=4 int32
# (16 bytes) the load silently rereads the pos words, whose bit patterns are denormals ~ 0, and the
# key vector comes back shifted right by 4 lanes with zeros in front -- a probe artifact that looks
# exactly like a kernel lane-offset defect.
POS_PAD = 16  # int32 slots before inv_freq => 64-byte aligned
_pos_block = np.zeros(POS_PAD, dtype=np.int32)
_pos_block[:M] = pos
cbuf = np.concatenate([_pos_block, inv_freq.view(np.int32)]).astype(np.int32)

SHIM = f"""#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void key_vec_probe(int32_t *restrict cbuf, int32_t *restrict out) {{
  constexpr int M = {M}, HALF = {HALF}, kVec = {KVEC}, POS_PAD = {POS_PAD};
  constexpr float kPi = 3.14159265358979f, kTwoPi = 2.0f * kPi;
  constexpr float kScaleInv = 1.0f;
  const int32_t *pos = cbuf;
  const float *inv_freq = (const float *)(cbuf + POS_PAD);
  // to_fixed follows the core rounding mode, and rope_lut.cc selects conv_even before its loop.
  // Omitting this makes the probe quantize under a different mode than the kernel it models.
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int m = 0; m < M; ++m) {{
    const int32_t p = pos[m];
    ::aie::vector<float, kVec> posf =
        ::aie::mul(::aie::to_float(::aie::broadcast<int32, kVec>(p)), kScaleInv);
    for (int i = 0; i < HALF; i += kVec) {{
      ::aie::vector<float, kVec> invf = ::aie::load_v<kVec>(inv_freq + i);
      ::aie::vector<float, kVec> theta = ::aie::mul(posf, invf);
      ::aie::vector<float, kVec> kwf = ::aie::mul(theta, 1.0f / kTwoPi);
      ::aie::vector<int32, kVec> k = ::aie::to_fixed<int32>(kwf);
      ::aie::vector<float, kVec> kf = ::aie::to_float(k);
      ::aie::vector<float, kVec> ktwopi = ::aie::mul(kf, kTwoPi);
      ::aie::vector<float, kVec> wrapped = ::aie::sub(theta, ktwopi);
      ::aie::vector<float, kVec> q = ::aie::mul(wrapped, 128.0f / kPi);
      // ARM A -- quantise straight to int32, never touching a sub-native int8 vector.
      ::aie::vector<int32, kVec> direct = ::aie::to_fixed<int32>(q);
      ::aie::store_v(out + m * HALF + i, direct);
      // ARM B -- the kernel's own step: quantise to a sub-native vector<int8,16>, then widen
      // only for the store. A differs from B => the int8 conversion is where lanes move.
      ::aie::vector<int8, kVec> keys = ::aie::to_fixed<int8>(q);
      ::aie::vector<int32, kVec> ks = ::aie::to_fixed<int32>(::aie::to_float(keys));
      ::aie::store_v(out + M * HALF + m * HALF + i, ks);
    }}
  }}
  ::aie::set_rounding(saved_rounding);
}}
"""


def host_keys():
    """Same chain in numpy, as the oracle."""
    out = np.zeros((M, HALF), dtype=np.int64)
    for m in range(M):
        theta = np.float32(pos[m]) * inv_freq
        kk = np.rint(theta / np.float32(2 * np.pi)).astype(np.int32)
        wrapped = theta - kk.astype(np.float32) * np.float32(2 * np.pi)
        q = wrapped * np.float32(128.0 / np.pi)
        out[m] = np.clip(np.rint(q), -128, 127).astype(np.int64)
    return out


def main():
    p = GEN / "key_vec_probe_shim.cc"
    p.write_text(SHIM)
    design = _build_oneshot(
        "key_vec_probe", p, [cbuf.size], 2 * M * HALF, [np.int32], np.int32, []
    )
    ct = iron.tensor(np.ascontiguousarray(cbuf), dtype=np.int32, device="npu")
    ot = iron.zeros((2 * M * HALF,), dtype=np.int32, device="npu")
    design(ct, ot)
    both = ot.numpy().astype(np.int64)
    direct = both[: M * HALF].reshape(M, HALF)
    dev = both[M * HALF :].reshape(M, HALF)
    exp = host_keys()

    print("ARM A (to_fixed<int32> straight from q -- no int8 vector)")
    for m in range(M):
        bad = np.nonzero(direct[m] - exp[m])[0]
        print(f"  m={m} pos={pos[m]}: mismatches {bad.size}/{HALF}")
    print(f"  m=1 first 16: {direct[1][:16].tolist()}")
    print(f"  host  first 16: {exp[1][:16].tolist()}\n")
    print("ARM B (via sub-native vector<int8,16>, as the kernel does)")

    print(f"pos = {pos.tolist()}, HALF = {HALF}, kVec = {KVEC}\n")
    print("row m=0 (pos=0) -- EVERY key must be 0")
    print(f"  device[0][:32] {dev[0][:32].tolist()}")
    nz = np.nonzero(dev[0])[0]
    print(f"  non-zero lanes: {nz.tolist() if nz.size else 'none'}")
    if nz.size:
        print(f"  their values  : {dev[0][nz].tolist()}")
        print(f"  lane mod {KVEC}   : {sorted(set((nz % KVEC).tolist()))}")

    print("\nall rows vs the host chain")
    for m in range(M):
        d = dev[m] - exp[m]
        bad = np.nonzero(d)[0]
        print(
            f"  m={m} pos={pos[m]}: mismatches {bad.size}/{HALF}"
            + (
                f"  |delta| max {np.abs(d).max()}  deltas {sorted(set(d[bad].tolist()))}"
                if bad.size
                else ""
            )
        )
    # A +-1 spread is quantiser boundary disagreement between two roundings of the SAME angle
    # and is benign; anything larger is a different angle, i.e. a real defect.
    dmax = int(np.abs(dev - exp).max())
    print(f"\nworst |device - host| over all rows: {dmax}  -> {'boundary only' if dmax <= 1 else 'STRUCTURAL'}")
    print("m=1 side by side, first 16 lanes")
    print(f"  device {dev[1][:16].tolist()}")
    print(f"  host   {exp[1][:16].tolist()}")


if __name__ == "__main__":
    main()
