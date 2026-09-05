#!/usr/bin/env python3
"""THE quantizer segment, gated stage by stage. Gate: rel-L2 vs codec_quantizer_ref <= 3e-2 per stage.

    codes [10,T]  ->  rvq_lookup  ->  post_module (8-layer transformer)
                   ->  upsample.0 (conv_transpose + ConvNeXt)  ->  upsample.1  ->  latent [1024,4T]

Same shape as verify_whole_decoder.py: each stage is gated CHAINED (fed the previous stage's own
DEVICE output, the real deployment path -- not fed a fresh slice of the true stream the way
verify_stage123.py's "iso" gates do), against a HOST TRUTH computed by calling
scripts/codec_quantizer_ref.py's own functions directly, in the same sequence quantizer_driver.decode()
uses, so the golden here is exactly the oracle-matching reference, never a reimplementation of it.
Three checks per stage, matching this codebase's own gate discipline -- rel-L2 is a NOTE, and 1:1
determinism against the reference is what actually gates:
  - rel-L2 <= GATE (reported, not blocking on its own)
  - run2run: the STAGE's own device function called TWICE on the same input must agree exactly --
    the CLFLUSH-race guard bricklib already runs per DISPATCH, re-run here per STAGE so a
    non-determinism that only shows up across a stage's many dispatches (not within one) is caught
  - alignment: ALWAYS 0 expected here, unlike the decoder. Every op in quantizer_driver.py either has
    ctx=0 (RVQ out_proj, wqkv/wo/w1/w2/w3, pwconv1/pwconv2, the upsample conv_transpose -- see that
    module's docstring) or computes the WHOLE T in one shot with no dropped context (dwconv, RoPE
    windowing, prefill_attn_chunk's row loop) -- there is no windowing-consumed-context offset for a
    shift-search to paper over. A nonzero shift here means a real index bug, not benign windowing,
    unlike the decoder's window-stitched audio stream.

--limit-frames N caps codes to the first N code frames BEFORE anything runs, so a first device pass
can be cheap (post_module's 8-layer transformer is by far the most dispatch-heavy stage -- see
quantizer_shapes.py's own dispatch-count table -- and run2run doubles every stage's dispatch count on
top of that). Recommended first run: --limit-frames 4 or 8, not the full clip.

    python3 verify_quantizer_segment.py <dump-dir> [--limit-frames N] [--skip-run2run]

dump_dir needs codec_codes.{bin,shape} (see codec_quantizer_ref.load_codes). Run under the device
lock with PYTHONPATH at the pinned toolchain instance, same as verify_whole_decoder.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import gguf_extract as gx  # noqa: E402
import codec_quantizer_ref as cq  # noqa: E402
import quantizer_driver as qd  # noqa: E402
import quantizer_shapes as qs  # noqa: E402

GATE = 3.0e-2


def alignment(cur, truth):
    """Same explicit shift-search verify_whole_decoder.py's own `alignment()` uses, applied to this
    segment's [C,T] layout (shift along axis 1, T) -- see module docstring for why 0 is the only
    result expected here, unlike the decoder's windowed audio stream."""
    V = cur.shape[1]
    if V <= 4:
        return 0
    return min(range(-2, 3),
              key=lambda s_: np.linalg.norm(cur[:, 2:V - 2] - truth[:, 2 + s_:V - 2 + s_]))


def rel_l2(cur, truth):
    num = np.linalg.norm((cur - truth).astype(np.float64))
    den = np.linalg.norm(truth.astype(np.float64))
    return float(num / den) if den != 0.0 else float(num)


def check(label, cur, truth, determ, gate=GATE):
    assert cur.shape == truth.shape, f"{label}: {cur.shape} vs {truth.shape}"
    rl2 = rel_l2(cur, truth)
    shift = alignment(cur, truth)
    print(f"    {label:24s} rel-L2 {rl2:.3e}  align {shift:+d}  run2run {determ:.3e}  "
         f"shape {cur.shape}")
    assert shift == 0, f"{label}: misaligned by {shift} -- a real index bug (see module docstring)"
    assert determ == 0.0, f"{label}: run2run {determ:.3e} != 0 -- non-deterministic device output"
    assert rl2 <= gate, f"{label}: gate failed rel_l2={rl2} > {gate}"
    return rl2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="cap codes to the first N code frames before anything runs")
    ap.add_argument("--skip-run2run", action="store_true",
                    help="skip the double-run determinism check (halves dispatch count; use only "
                        "for a fast first pass, not as the real gate)")
    args = ap.parse_args()

    _cache = {}
    def W(name):
        if name not in _cache:
            _cache[name] = gx.load(qs.GGUF, name).astype(np.float32)
        return _cache[name]

    codes = cq.load_codes(args.dump_dir)
    n_avail = codes.shape[1]
    if args.limit_frames is not None:
        assert 1 <= args.limit_frames <= n_avail, (
            f"--limit-frames {args.limit_frames} out of range 1..{n_avail}")
        codes = codes[:, :args.limit_frames]
    T = codes.shape[1]
    print(f"codes {codes.shape} (of {n_avail} available)")

    qd._selftest_weight_folds(W)

    # ---- host truth: codec_quantizer_ref's own functions, called in decode()'s exact sequence ----
    t0 = time.time()
    host_stage = cq.rvq_lookup(codes, W, cq.QPREFIX)
    host_pm = cq.rvq_transformer(host_stage, W, f"{cq.QPREFIX}.post_module")
    host_up0 = cq.quantizer_stage_up(host_pm, W, f"{cq.QPREFIX}.upsample.0", cq.DOWNSAMPLE_FACTORS[0])
    host_up1 = cq.quantizer_stage_up(host_up0, W, f"{cq.QPREFIX}.upsample.1", cq.DOWNSAMPLE_FACTORS[1])
    print(f"host truth computed in {time.time() - t0:.2f}s  "
         f"(rvq {host_stage.shape}, post_module {host_pm.shape}, "
         f"up0 {host_up0.shape}, up1/latent {host_up1.shape})")

    # ---- device, CHAINED (fed the previous stage's own device output -- the real path) ----
    def run_twice(fn):
        a = fn()
        if args.skip_run2run:
            return a, 0.0
        b = fn()
        d = float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel()))
        return a, d

    qd.reset_stats()
    disp = {"prev": 0}
    def phase(name):
        now = qd.stats()["dispatches"]
        print(f"    [{name}: {now - disp['prev']} dispatches]")
        disp["prev"] = now

    FW = qd.fold_weights(W)
    phase("fold_weights (host-only, 0 dispatches expected)")

    dev_stage, d0 = run_twice(lambda: qd.rvq_lookup(codes, W))
    phase("rvq_lookup")
    print("\n  STAGE 1: RVQ lookup")
    r1 = check("rvq_lookup", dev_stage, host_stage, d0)

    dev_pm, d1 = run_twice(lambda: qd.post_module(dev_stage, W, FW))
    phase("post_module")
    print("\n  STAGE 2: post_module transformer")
    r2 = check("post_module", dev_pm, host_pm, d1)

    dev_up0, d2 = run_twice(lambda: qd.quantizer_upsample(dev_pm, W, FW, 0, "up0"))
    phase("upsample.0")
    print("\n  STAGE 3: upsample.0 (conv_transpose + ConvNeXt)")
    r3 = check("upsample.0", dev_up0, host_up0, d2)

    dev_up1, d3 = run_twice(lambda: qd.quantizer_upsample(dev_up0, W, FW, 1, "up1"))
    phase("upsample.1")
    print("\n  STAGE 4: upsample.1 (conv_transpose + ConvNeXt) -- THE GATE (== latent)")
    r4 = check("upsample.1 (latent)", dev_up1, host_up1, d3, gate=GATE)

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    for label, r in (("rvq_lookup", r1), ("post_module", r2), ("upsample.0", r3), ("upsample.1", r4)):
        print(f"  {label:14s} rel-L2 {r:.3e}  (gate {GATE:.1e})")
    print(f"  total dispatches: {qd.stats()['dispatches']}"
         f"{'  (x2 via run2run)' if not args.skip_run2run else '  (run2run skipped)'}")
    print("\nPASS")


if __name__ == "__main__":
    main()
