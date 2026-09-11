#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read every weight buffer BACK off the device and compare it to what the host wrote.

The 6-layer bisect leaves `xf` clean at 2.79e-02 -- indistinguishable from the 5-layer arm that
passes -- and then reports logits at 1.49e+01. A correct hidden state feeding a wrong lm-head means
the fault is not in the arithmetic the bisect walks; the candidate left is the ARENA, where a
buffer sized for one geometry can be overlapped by one sized for another.

This does not run the graph. It writes the weights, flushes, reads them straight back and diffs.
Anything that comes back different was overlapped by another buffer, and the name says by which
region.
"""
import argparse, os, sys
import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from gen_llm_decode import build_graph, isolate_build_dir, load_weight_buffer  # noqa: E402

BF16 = ml_dtypes.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    # Same isolation verify_llm_decode does. Without it the build lands in CWD and
    # params.txt is not found, so the ParameterScratchpad never binds and every
    # per-token write fails -- which shows up as `params` being None at depth.
    isolate_build_dir("probe")

    sp, fused, weights, md = build_graph(a.spec, a.weights, a.layers, 2048)
    c = fused.get_callable()
    for n, arr in weights.items():
        with c.get_buffer(n).overwrite() as buf:
            buf[:] = np.asarray(arr, BF16).reshape(-1)
    c.scratch_buffer.device = "cpu"
    c.scratch_buffer.to("npu")

    # ARENA OVERLAP CHECK, before anything is dispatched. The corrupted logits sit in the last
    # two of the lm-head's eight column blocks, and a correct xf feeding a wrong logits vector is
    # what a buffer landing on the output region looks like. The layout is queried off the compiled
    # object (the same call gen_llm_decode.py::main makes), never a hand-written list.
    names = ["logits", *md["inputs"], *weights.keys(), *md.get("cache_names", [])]
    for l in range(md["NL"]):
        names += [f"L{l}_{t}" for t in
                  ("q", "k", "v", "kr", "vr", "vt", "sc", "sw", "cx", "a",
                   "hn", "hf", "g", "u", "gh", "d", "ls", "qkv")]
        names += [f"x{l}", f"x{l+1}"]
    names += ["xf"]
    lay = {}
    for n in dict.fromkeys(names):
        try:
            lay[n] = fused.get_layout_for_buffer(n)
        except Exception:
            pass

    def ext(e):
        # get_layout_for_buffer returns a plain (buf_type, offset_bytes, length_bytes) tuple
        # (iron/common/sequence.py), with sliced buffers already resolved to the parent's
        # absolute offset.
        t, off, ln = e
        return t, int(off), int(ln)

    if "logits" in lay:
        la, lo, ll = ext(lay["logits"])
        print(f"\n[layout] logits: arena {la} off {lo} len {ll} ({ll//2} bf16 elements)")
        hits = []
        for n, e in lay.items():
            if n == "logits":
                continue
            oa, oo, ol = ext(e)
            if oa == la and oo < lo + ll and lo < oo + ol:
                hits.append((n, oo, ol, max(lo, oo) - lo))
        for n, oo, ol, rel in sorted(hits, key=lambda h: h[3]):
            print(f"[layout] OVERLAP {n:28} off {oo:10} len {ol:9} "
                  f"-> from byte {rel} of logits (element {rel//2}, column block {rel//2//32768})")
        if not hits:
            print(f"[layout] nothing overlaps logits in arena {la}")
    print(f"[layout] resolved {len(lay)} buffer locations\n")

    # Run the graph once at pos 0, then look at the logits ELEMENT-WISE. The bisect says xf is
    # clean and logits is not, and the weights survive a write/flush/read, so the remaining
    # candidate is a buffer written DURING the dispatch landing on the output region. Where the
    # vector stops agreeing names that region's boundary.
    D = sp.d_model
    # One ROW is all this needs; mmap so the 3.75 GiB table is never resident (see the same
    # idiom and its OOM history in gen_llm_decode.py::npy).
    embed = np.load(os.path.join(a.weights, f"{sp.weight_prefix}embed_tokens.weight.npy"),
                    mmap_mode="r")
    scale = np.sqrt(D) if sp.embed_scale == "sqrt_d_model" else 1.0
    with c.get_buffer("x").overwrite() as buf:
        buf[:] = np.asarray(embed[785].astype(np.float32) * scale, BF16).reshape(-1)
    for ang in ("rope_global", "rope_local"):
        if ang in md["inputs"]:
            with c.get_buffer(ang).overwrite() as buf:
                row = np.zeros(buf.size, np.float32); row[0::2] = 1.0
                buf[:] = np.asarray(row, BF16)
    for slot_name, _ in md["kv_slots"]:
        c.params.write(slot_name, 0)
    c.params.write("sm_mask", 1)
    c.params.sync()
    c()
    lg = np.asarray(c.get_buffer("logits").data, BF16).astype(np.float32).copy()
    # RE-SYNC AND RE-READ, the documented absorb for the host-only BO coherency race
    # (a host-only BO coherency race): sync(FROM_DEVICE) can return before the last DMA line is
    # visible, and the host then reads the PRE-DMA value -- which for a zeroed output buffer is
    # exactly zero. That is the shape of the 2169 unwritten elements. A genuine miscompute is
    # deterministic and survives every attempt, so a bounded re-read absorbs only the transient
    # window and cannot hide a real fault.
    for attempt in range(1, 6):
        arena = c.get_buffer("logits")
        try:
            arena.device = "npu"; arena.to("cpu")
        except Exception:
            pass
        lg_r = np.asarray(arena.data, BF16).astype(np.float32).copy()
        nz_before, nz_after = int((lg == 0).sum()), int((lg_r == 0).sum())
        if not np.array_equal(lg, lg_r):
            print(f"[coherency] re-sync attempt {attempt} CHANGED the buffer: "
                  f"zeros {nz_before} -> {nz_after}, "
                  f"{int((lg != lg_r).sum())} elements differ")
            lg = lg_r
        else:
            print(f"[coherency] re-sync attempt {attempt}: identical (zeros {nz_after})")
            break
    # Second dispatch on the SAME build with the SAME inputs. A race in BD reuse would land
    # differently run to run; a structural fault reproduces exactly. This is the fork that decides
    # whether the defect is a timing hazard or a wrong descriptor program.
    c()
    lg2 = np.asarray(c.get_buffer("logits").data, BF16).astype(np.float32).copy()
    same = np.array_equal(lg, lg2)
    print(f"\n[redispatch] second dispatch identical: {same}"
          + ("" if same else f"  ({int((lg != lg2).sum())} of {lg.size} elements differ)"))
    print(f"\nlogits buffer: {lg.size} elements, "
          f"finite {int(np.isfinite(lg).sum())}, zeros {int((lg == 0).sum())}, "
          f"|max| {float(np.abs(lg[np.isfinite(lg)]).max()):.6g}")
    V = int(sp.vocab)
    for lo, hi in [(0, 1024), (V//4, V//4+1024), (V//2, V//2+1024), (3*V//4, 3*V//4+1024), (V-1024, V)]:
        seg = lg[lo:hi]
        print(f"  [{lo:7}:{hi:7}] |max| {float(np.abs(seg).max()):12.6g}  "
              f"mean|.| {float(np.abs(seg).mean()):12.6g}  zeros {int((seg==0).sum()):5}")
    z = np.flatnonzero(lg == 0)
    if z.size:
        # Contiguity tells apart a missing TRANSFER (one run) from a missing element per
        # descriptor (a stride). Runs are maximal spans of consecutive zero indices.
        brk = np.flatnonzero(np.diff(z) != 1)
        starts = np.concatenate(([z[0]], z[brk + 1]))
        ends = np.concatenate((z[brk], [z[-1]]))
        runs = [(int(a), int(b - a + 1)) for a, b in zip(starts, ends)]
        runs.sort(key=lambda r: -r[1])
        print(f"  zeros: {z.size} in {len(runs)} runs; longest: {runs[:6]}")
        print(f"  zero run lengths seen: {sorted(set(l for _, l in runs))[:12]}")
        print(f"  zeros per column block: "
              f"{ {int(k): int(v) for k, v in zip(*np.unique(z // 32768, return_counts=True))} }")
    big = np.flatnonzero(np.abs(lg) > 1e3)
    print(f"  elements with |logit| > 1e3: {big.size}"
          + (f", first at {int(big[0])}, last at {int(big[-1])}" if big.size else ""))
    print(f"  device argmax {int(np.argmax(lg))}")
    if big.size:
        print(f"  outlier indices: {big.tolist()}")
        d = np.diff(big)
        print(f"  gaps between them: {d.tolist()}")
        print(f"  idx % 8   : {(big % 8).tolist()}")
        print(f"  idx % 64  : {sorted(set((big % 64).tolist()))}")
        print(f"  idx // 32768 (per-column block): {sorted(set((big // 32768).tolist()))}")

    bad = 0
    for n, arr in weights.items():
        want = np.asarray(arr, BF16).reshape(-1)
        got = np.asarray(c.get_buffer(n).data, BF16).reshape(-1)[:want.size]
        if not np.array_equal(np.asarray(want, np.float32), np.asarray(got, np.float32)):
            d = np.asarray(want, np.float32) - np.asarray(got, np.float32)
            nz = int(np.count_nonzero(d))
            print(f"CORRUPT {n:40} {nz}/{want.size} elements differ, "
                  f"first at {int(np.flatnonzero(d)[0])}, max|d| {float(np.abs(d).max()):.4g}")
            bad += 1
    print(f"\n{bad} of {len(weights)} weight buffers came back different after a write+flush+read")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
