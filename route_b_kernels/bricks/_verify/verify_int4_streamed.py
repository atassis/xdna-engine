#!/usr/bin/env python3
"""Device gate for the STREAMED (tiled-operand) int4 dequant rail.

Split out of verify_f2b so a run rebuilds only these two xclbins: the point of this rail is
the shapes the one-shot builder cannot stage into L1, and iterating on it while rebuilding
every F2b brick wastes device time.

  gi4dq-streamed-64x128x128   -- the shape `_dequant_shape` fails to BUILD one-shot
  gi4dq-streamed-4x1024x4096  -- the tall-K decode shape (B ~2.1 MB, ~33x a core tile L1)
"""
import json
import traceback

from verify_f2b import (
    do_gemm_int8xint4_dequant_64x128x128_streamed,
    do_gemm_int8xint4_dequant_tallk_streamed,
)

do_streamed_64x128x128 = do_gemm_int8xint4_dequant_64x128x128_streamed
do_streamed_tallk = do_gemm_int8xint4_dequant_tallk_streamed

if __name__ == "__main__":
    results = []
    for fn in (do_streamed_64x128x128, do_streamed_tallk):
        name = getattr(fn, "brick_name", fn.__name__)
        try:
            results.append(fn())
        except Exception as e:
            print(f"[{name:34s}] ERROR: {e}", flush=True)
            traceback.print_exc()
            results.append(dict(name=name, status="ERROR", ok=False, err=str(e)))
    print("\n==== streamed int4 rail SUMMARY ====", flush=True)
    for r in results:
        rl2 = r.get("rel_l2")
        rl2s = f"rel_l2={rl2:.3e}" if isinstance(rl2, float) else "rel_l2=--"
        print(f"  {r['name']:34s} {r['status']:10s} {rl2s}")
    print(f"streamed: {sum(1 for r in results if r.get('ok'))}/{len(results)} PASS")
    print("JSON " + json.dumps([{k: v for k, v in r.items() if k != 'got'} for r in results]))
    import os
    os._exit(0)
