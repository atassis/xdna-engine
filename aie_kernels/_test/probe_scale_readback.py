#!/usr/bin/env python3
"""What does the fused dequant kernel ACTUALLY read at pScale?

The fused gemm-int8xint4-dequant brick returns C == EXACTLY zero on device, while the same
kernel with a locally-baked scale returns nonzero. Three hypotheses have now been tested
and refuted on device:

  * DMA truncation of the packed tail  -- refuted: probe_wbuf_tail.py showed a packed
    [B|scale] buffer arrives whole, scale bytes decoding correctly as floats.
  * the scalar-store -> vector-load stack array (`float sbuf[64]`) -- replaced with a pure
    `concat(load_v<16>)` replication; C still exactly zero.
  * a false `__restrict` on pB/pScale, which alias the same packed allocation -- restrict
    dropped from both; C still exactly zero.

So stop guessing and read the pointer back. This probe reuses verify_f2b's EXACT packing
(`pack_int4_blocks` -> contiguous 128B blocks, then scale appended) and the same 2-input
shim shape, but instead of computing anything it copies the scale words straight out.

Everything is int32 word copies -- no scalar float arithmetic and no local arrays, both of
which have hung the core in earlier probes.

  out[0:16]  = raw 32-bit words at (wbuf + b_bytes)      <- what the kernel sees as pScale
  out[16:32] = raw 32-bit words at (wbuf + b_bytes) via a SEPARATE second input buffer
               carrying only the scale (control: same bytes, unpacked delivery)
  out[32]    = b_bytes as the kernel was told it
  out[33]    = the first weight byte (liveness canary)

Run:  ./run.sh probe_scale_readback.py
"""
from pathlib import Path
import numpy as np

import aie.iron as iron
import bricklib
from verify_f2b import pack_int4_blocks, pack_int4

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, K, N, G = 8, 64, 64, 64
ngrp = K // G

rng = np.random.default_rng(17)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)

b_pad = pack_int4_blocks(b, K, N, pack_int4)
b_bytes = int(b_pad.size)
wbuf = np.concatenate([b_pad, scale.reshape(-1).view(np.int8)])
print(f"[scale_readback] b_bytes={b_bytes} scale_bytes={scale.size*4} wbuf={wbuf.size}")

OUT_N = 34
sym = "scale_readback_probe"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* wbuf, const int8_t* sonly, int32_t* __restrict out) {{
  const int32_t* s  = (const int32_t*)(wbuf + {b_bytes});
  const int32_t* s2 = (const int32_t*)(sonly);
  for (int i = 0; i < 16; ++i) out[i]      = s[i];
  for (int i = 0; i < 16; ++i) out[16 + i] = s2[i];
  out[32] = {b_bytes};
  out[33] = (int32_t)wbuf[0];
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [wbuf.size, scale.size * 4], OUT_N,
                                 [np.int8, np.int8], np.int32, [])
w_t = iron.tensor(np.ascontiguousarray(wbuf), dtype=np.int8, device="npu")
s_t = iron.tensor(np.ascontiguousarray(scale.reshape(-1).view(np.int8)), dtype=np.int8, device="npu")
o_t = iron.zeros((OUT_N,), dtype=np.int32, device="npu")
design(w_t, s_t, o_t)
got = np.array(o_t.numpy().copy(), copy=True)

packed_f = got[:16].astype(np.int32).tobytes()
packed_f = np.frombuffer(packed_f, dtype=np.float32)
sep_f = np.frombuffer(got[16:32].astype(np.int32).tobytes(), dtype=np.float32)
exp = scale.reshape(-1)[:16]

print(f"\n  b_bytes seen by kernel : {got[32]}  (host packed at {b_bytes})")
print(f"  wbuf[0] canary         : {got[33]}  (host {int(b_pad[0])})")
print(f"\n  expected scale[:6]     : {exp[:6]}")
print(f"  PACKED  (wbuf+b_bytes) : {packed_f[:6]}")
print(f"  SEPARATE (own buffer)  : {sep_f[:6]}")

packed_ok = np.allclose(packed_f, exp, rtol=1e-6, atol=0)
sep_ok = np.allclose(sep_f, exp, rtol=1e-6, atol=0)
print(f"\n  packed matches   : {packed_ok}   (all-zero: {bool((got[:16]==0).all())})")
print(f"  separate matches : {sep_ok}   (all-zero: {bool((got[16:32]==0).all())})")

if packed_ok:
    v = "kernel DOES read the packed scale correctly -- the zero C is downstream of the load"
elif sep_ok:
    v = "packed read is broken but a SEPARATE buffer reads fine -- the packed-offset route is the defect"
else:
    v = "NEITHER route delivers the scale -- delivery itself is broken in this shape"
print(f"\n  VERDICT: {v}")
