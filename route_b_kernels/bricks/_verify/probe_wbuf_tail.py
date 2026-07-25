#!/usr/bin/env python3
"""Does the TAIL of a packed weight buffer actually arrive in L1?

The fused gemm-int8xint4-dequant brick packs `scale` after the padded-B weights in ONE
input buffer (`wbuf=[B_pad|scale]`) because a real 3rd input exceeds the core tile's DMA
fanin -- confirmed by the placer: "tile requires 3 input/1 output DMA channels, but only
2 input/2 output available" / "reduce the LTO's DMA fanin (e.g. via memtile staging)".

On device the kernel reads that scale region as EXACTLY 0, which is the signature of
memory that was never written rather than of a wrong offset (a wrong offset would read
weight bytes as garbage floats, not zeros).

This probe removes every other variable: no mmul, no int4, no epilogue. It just echoes
bytes back from chosen offsets of a single packed input buffer, so we learn exactly how
much of it the DMA delivered.

  sec0: wbuf[0..63]              -- head, known-good control
  sec1: wbuf[b_bytes-64..]       -- last bytes of the B region
  sec2: wbuf[b_bytes..b_bytes+63]-- FIRST bytes of the scale region  <-- the question
  sec3: wbuf[total-64..]         -- very last bytes of the buffer

Host fills the scale region with a known ramp, so "0" and "wrong" are distinguishable.

Run:  ./run.sh probe_wbuf_tail.py
"""
from pathlib import Path
import numpy as np

import aie.iron as iron
import bricklib

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

B_BYTES = 4096                      # same as the 8x64x64 g64 case: 16 blocks x 256B
NSCALE = 64                         # ngrp*N floats
S_BYTES = NSCALE * 4
TOTAL = B_BYTES + S_BYTES
NSEC, SECLEN = 4, 64

# NOTE: no local arrays and no scalar float ops in this shim. Probes that used either
# hung the core outright (ERT_CMD_STATE_TIMEOUT) while a known-good brick passed bit-exact
# immediately after -- so those constructs, not the device, were the problem. The section
# offsets are therefore unrolled as literals rather than held in a local `int offs[]`.
sym = "wbuf_tail_probe"
_copies = "\n".join(
    f'  for (int i = 0; i < {SECLEN}; ++i) out[{s * SECLEN} + i] = wbuf[{off} + i];'
    for s, off in enumerate([0, B_BYTES - SECLEN, B_BYTES, TOTAL - SECLEN]))
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict wbuf, int8_t* __restrict out) {{
{_copies}
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

# B region: a recognisable non-zero pattern. Scale region: a float ramp 1.0, 2.0, ...
b_region = (np.arange(B_BYTES, dtype=np.int64) % 127 - 63).astype(np.int8)
scale = (np.arange(NSCALE, dtype=np.float32) + 1.0)
wbuf = np.concatenate([b_region, scale.view(np.int8)])
assert wbuf.size == TOTAL, (wbuf.size, TOTAL)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [TOTAL], NSEC * SECLEN,
                                 [np.int8], np.int8, [])
w_t = iron.tensor(np.ascontiguousarray(wbuf), dtype=np.int8, device="npu")
o_t = iron.zeros((NSEC * SECLEN,), dtype=np.int8, device="npu")
design(w_t, o_t)
got = np.array(o_t.numpy().copy(), copy=True)

names = ["sec0 head        wbuf[0:64]",
         f"sec1 B tail      wbuf[{B_BYTES-SECLEN}:{B_BYTES}]",
         f"sec2 SCALE head  wbuf[{B_BYTES}:{B_BYTES+SECLEN}]",
         f"sec3 buf tail    wbuf[{TOTAL-SECLEN}:{TOTAL}]"]
offs = [0, B_BYTES - SECLEN, B_BYTES, TOTAL - SECLEN]

print(f"[wbuf_tail] B_BYTES={B_BYTES} S_BYTES={S_BYTES} TOTAL={TOTAL}")
all_ok = True
for s in range(NSEC):
    g = got[s * SECLEN:(s + 1) * SECLEN]
    e = wbuf[offs[s]:offs[s] + SECLEN]
    ok = bool((g == e).all())
    all_ok &= ok
    print(f"  {names[s]}: match={ok} allzero={bool((g==0).all())}")
    if not ok:
        print(f"      exp[:8]={e[:8].tolist()}")
        print(f"      got[:8]={g[:8].tolist()}")

# Decode the scale section as floats -- this is what the brick actually consumes.
sc_got = got[2 * SECLEN:3 * SECLEN].tobytes()
sc_f = np.frombuffer(sc_got, dtype=np.float32)
print(f"\n  scale-as-float got[:8] = {sc_f[:8]}")
print(f"  scale-as-float exp[:8] = {scale[:8]}")
print(f"\n  VERDICT: {'whole buffer arrives (scale delivery is NOT a DMA truncation)' if all_ok else 'BUFFER TRUNCATED / not delivered -- see first failing section above'}")
