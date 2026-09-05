#!/usr/bin/env python3
"""F3 specials device-verify (tractable ones). transpose-dma: pure strided permutation,
   bit-exact. (rope-lut/lm-head-argmax/moe/gdn/dequant-int4 are heavier -- parked.)
"""
import json
import sys
import traceback
from pathlib import Path
import numpy as np

import bricklib

BRICKS = Path(__file__).parent.parent
results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        print(f"[{fn.brick_name:22s}] ERROR: {e}")
        traceback.print_exc()
        results.append(dict(name=fn.brick_name, status="ERROR", ok=False, err=str(e)))


def do_transpose_dma():
    cc = str(BRICKS / "transpose-dma" / "transpose_dma.cc")
    mb, nb = 32, 32  # 32x32 int32 = 4KB/buf; larger overflows the core tile's 64KB L1
    rng = np.random.default_rng(5)
    x = rng.integers(-2_000_000_000, 2_000_000_000, size=(mb, nb), dtype=np.int32)
    ref = x.T.copy()  # [nb, mb]
    shim = ('extern "C" void tdma_verify(uint32_t*in,uint32_t*out){'
            'transpose_dma_2d((TELEM*)in,(TELEM*)out,%d,%d);}' % (mb, nb))
    return bricklib.verify_oneshot(
        "transpose-dma", cc, shim, "tdma_verify",
        inputs=[(x.reshape(-1), np.int32)],
        out_numel=mb * nb, out_shape=(nb, mb),
        unpack=lambda flat: flat.reshape(nb, mb),
        golden=ref, gate=3e-2, out_dt=np.int32,
        compile_flags=["-DTPOSE_i32"])
do_transpose_dma.brick_name = "transpose-dma"


if __name__ == "__main__":
    for fn in (do_transpose_dma,):
        guard(fn)
    print("\n==== F3 SUMMARY ====")
    for r in results:
        print(f"  {r['name']:22s} {r['status']:10s} rel_l2={r.get('rel_l2', float('nan')):.3e}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"F3: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps(results))
    sys.exit(0 if passed == len(results) else 1)
