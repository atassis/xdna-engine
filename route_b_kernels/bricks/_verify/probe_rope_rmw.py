#!/usr/bin/env python3
"""Do the per-row loads inside rope_lut.cc return the data, or zero?

[[rope-lut-buffers-and-rail-both-exonerated]] narrowed the only-row-0-survives defect by arithmetic:
rows 1+ come back with out1 AND out2 exactly zero, and since out1 = x1*c - x2*s, out2 = x1*s + x2*c
and sin^2+cos^2 = 1, no LUT read can make both vanish. Only x1 = x2 = 0 can. So the loads from
`row + i` should be returning zero for every m >= 1.

This strips the kernel to the claim. Same three-buffer harness signature, same copy, same per-row
pointer arithmetic, same 16-lane load and store -- but NO trig, NO parallel_lookup, NO arithmetic:

    for m: row = qk_out + m*D
      for i: store_v(row + i, load_v<kVec>(row + i))     // read-modify-write IDENTITY

If the loads work this is a no-op and the output equals the input. If rows 1+ come back zero, the
defect is a bare load/store over a row-strided pointer and everything about RoPE is irrelevant --
a five-line repro.

Arm B keeps the two aie::parallel_lookup objects constructed but unused, to separate "the loop is
broken" from "constructing the lookups breaks the loop".
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

M, D = 16, 128
N = M * D
KVEC = 16
CBUF_N = M + 64
BRICKS_ABS = None


def _shim(name, with_lut):
    import pathlib

    tab = pathlib.Path("route_b_kernels/bricks/rope-lut/rope_lut_tables.inc").resolve()
    head = '#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
    if with_lut:
        head += f'#include "{tab}"\n'
    body = f'''extern "C" void {name}(bfloat16 *restrict qk_in, int32_t *restrict cbuf,
                     bfloat16 *restrict qk_out) {{
  (void)cbuf;
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
'''
    if with_lut:
        body += '''  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, 0, 128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, 0, 128);
  (void)sin_look; (void)cos_look;
'''
    body += f'''  for (unsigned m = 0; m < {M}u; ++m) {{
    bfloat16 *row = qk_out + (size_t)m * {D}u;
    for (unsigned i = 0; i < {D}u; i += {KVEC}u)
      ::aie::store_v(row + i, ::aie::load_v<{KVEC}>(row + i));
  }}
}}
'''
    return head + body


def run(label, name, with_lut, src, want):
    p = GEN / f"{name}_shim.cc"
    p.write_text(_shim(name, with_lut))
    d = _build_oneshot(name, p, [N, CBUF_N], N,
                       [ml_dtypes.bfloat16, np.int32], ml_dtypes.bfloat16, [])
    it = iron.tensor(np.ascontiguousarray(src), dtype=ml_dtypes.bfloat16, device="npu")
    cb = iron.tensor(np.zeros(CBUF_N, dtype=np.int32), dtype=np.int32, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    d(it, cb, ot)
    got = ot.numpy().astype(np.float32).reshape(M, D)
    bad = got != want
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    print(f"{label}: exact {N - int(bad.sum())}/{N}   damaged rows {rows if rows else 'NONE'}")
    if rows:
        r = rows[0]
        print(f"    row {r} want[:6] {want[r][:6].tolist()}")
        print(f"    row {r} got [:6] {got[r][:6].tolist()}")


def main():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(N).astype(np.float32).astype(ml_dtypes.bfloat16)
    want = src.astype(np.float32).reshape(M, D)
    print("copy, then a per-row read-modify-write IDENTITY. Output must equal input.\n")
    run("rmw, no lut ", "rmw_plain", False, src, want)
    run("rmw, with lut", "rmw_lut", True, src, want)


if __name__ == "__main__":
    main()
