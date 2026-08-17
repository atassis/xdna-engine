"""Can a core-staged column PAIR advance a transposed V cache one token at a time?

M0.5 (--coalesce-self-tr) needs V[h,d] written into column n of a transposed cache. A DMA cannot
do it: the innermost bf16 unit is a 4-byte granule spanning columns (n & ~1, n | 1), and an odd
element offset truncates down into it. What a DMA CAN do is write that whole pair at an even
offset. So this drives the surviving route end to end at small scale:

    core writes V(n) into slot n&1 of a staged pair, carrying the other slot  -> DMA writes the
    pair at column base n & ~1

The staged pair is one arena buffer read and rewritten in place, so the slot not written this
token still holds the previous token's V. The claim under test is the WHOLE chain over consecutive
tokens, not one write: after token n, columns 0..n of the cache must each hold their own token's V
for every head and dim, and both parities must survive the next token's write.

Usage: python probe_vpair_stage.py [--compile-only]     (device run is single-tenant)

The per-dispatch parity and column base arrive as scratchpad parameters, via
param_scratchpad_compat -- the pinned instance omits the pybind module they normally go through.
"""

import argparse
import glob
import os

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 — MUST precede iron imports
from iron.common import AIEContext
from elf_dispatch_compat import OperatorSequence
from iron.operators.strided_copy.op import StridedCopy
from vpair_stage_op import VPairStage

BF16 = ml_dtypes.bfloat16
H, HD, S = 2, 4, 8          # small enough to print, same shape family as the decode cache
N = H * HD
STEPS = S                   # fill the cache, so both parities are exercised to the last column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true",
                    help="build the ELF and the parameter table, skip the device run")
    a = ap.parse_args()
    ctx = AIEContext()
    stage = VPairStage(N=N, parity_parameter="par", context=ctx)
    # The pair write: [H,HD,2] contiguous pairs -> cache[h, d, base:base+2], base = (n & ~1).
    # Innermost run is 2 CONTIGUOUS bf16 = one granule, and the runtime column base is even, which
    # is the pattern measured to land both columns.
    pair_write = StridedCopy(
        input_sizes=(H, HD, 2), input_strides=(HD * 2, 2, 1), input_offset=0,
        output_sizes=(H, HD, 2), output_strides=(HD * S, S, 1), output_offset=0,
        input_buffer_size=N * 2, output_buffer_size=H * HD * S,
        num_aie_channels=1, output_offset_parameter="vc_off", context=ctx,
    )
    fused = OperatorSequence(
        "vpairprobe",
        [(stage, "v", "pair", "pair"), (pair_write, "pair", "vcache")],
        input_args=["v"], output_args=["vcache"],
        buffer_sizes={"v": N * 2, "pair": N * 2 * 2, "vcache": H * HD * S * 2},
        context=ctx,
    )
    print("compiling...", flush=True)
    fused.compile()
    params_path = sorted(glob.glob("**/params.txt", recursive=True), key=os.path.getmtime)[-1]
    params = open(params_path).read().strip()
    print("params.txt:", params.replace("\n", " | "))
    declared = {line.split()[0] for line in params.splitlines()[1:] if line.strip()}
    assert declared == {"par", "vc_off"}, f"expected both params, got {declared}"
    if a.compile_only:
        print("compile-only: kernel + design + in-place pair aliasing build, both params declared")
        return

    import pyxrt
    from param_scratchpad_compat import get_parameter_scratchpad

    ParameterScratchpad = get_parameter_scratchpad()
    callable_ = fused.get_callable()
    run = pyxrt.run(callable_.xrt_kernel)
    run.set_arg(0, callable_.input_buffer.buffer_object())
    run.set_arg(1, callable_.output_buffer.buffer_object())
    run.set_arg(2, callable_.scratch_buffer.buffer_object())
    sp = ParameterScratchpad(run, params_path)

    v_buf = callable_.get_buffer("v")
    vc_buf = callable_.get_buffer("vcache")
    with callable_.get_buffer("pair").overwrite() as pair:
        pair[:] = 0
    callable_.scratch_buffer.to("npu")

    # V(n)[h,d] carries its own token, head and dim, so a landed value names where it came from.
    # Kept inside 1..255, which bf16 holds exactly: the decimal-decade encoding this replaced ran
    # past the point where the bf16 step exceeds 1 (step 4 at 600), which both read correct columns
    # as wrong and collided two dims onto one value, so a misplaced column could hide.
    def vtok(n):
        return np.array([1 + n * (H * HD) + h * HD + d for h in range(H) for d in range(HD)],
                        dtype=np.float32)

    print(f"  n  parity  base  columns_correct  cache_ok")
    ok_all = True
    for n in range(STEPS):
        with v_buf.overwrite() as dst:
            np.copyto(dst, vtok(n).astype(BF16))
        callable_.input_buffer.to("npu")
        sp.write("par", n & 1)
        sp.write("vc_off", n & ~1)
        sp.sync()
        run.start()
        run.wait2()
        # This probe dispatches through pyxrt directly, so nothing tells the runtime the device
        # wrote the output. Without the mark, `to("cpu")` sees the range already host-resident
        # from the first readback and no-ops, and every later token reads a stale host copy.
        callable_.output_buffer.device = "npu"
        callable_.output_buffer.to("cpu")
        cache = np.asarray(vc_buf.data).astype(np.float32).reshape(H, HD, S)

        # Every token so far must still be readable in its own column.
        per_col = [bool(np.array_equal(cache[:, :, t].reshape(-1), vtok(t))) for t in range(n + 1)]
        cache_ok = all(per_col)
        ok_all &= cache_ok
        print(f"  {n}  {n & 1:>6}  {n & ~1:>4}  {sum(per_col)}/{n + 1:<14}  {cache_ok}")

    print("\n=== verdict ===")
    if ok_all:
        print("CONFIRMED: core-staged pair + even-base pair write advances the transposed cache "
              "one token per dispatch, both parities, previous tokens intact.")
    else:
        print("REFUTED: read the table -- a column that was correct earlier stopped being correct, "
              "or the parity slot did not take.")
        cache = np.asarray(vc_buf.data).astype(np.float32).reshape(H, HD, S)
        print("final cache[h=0]:\n", cache[0])


if __name__ == "__main__":
    main()
