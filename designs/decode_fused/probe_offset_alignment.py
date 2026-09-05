"""Does a MISALIGNED runtime element offset corrupt a bf16 StridedCopy write?

M0.5 (--coalesce-self-tr) writes the self-V cache transposed with a per-token column offset
`vcache_off = n_self` in ELEMENT units. At bf16 that is 2*n_self bytes, which lands on a 4-byte
boundary only for EVEN n_self -- while the contiguous kcache write uses kv_off = n_self*head_dim,
always a multiple of 128 bytes and so aligned for every token. The static path rejects a misaligned
offset outright (AIEXDialect.cpp:653, "Offset must be 4-byte-aligned"); the runtime scratchpad path
feeds the same BD address registers with nothing able to check it.

This isolates that one variable: one StridedCopy, sweeping the runtime output offset 0..5.

Why it is not covered already: iron/tests/infrastructure/scratchpad_parameters.py exercises the same
runtime `output_offset_parameter` on device, but every offset it passes is a multiple of HEAD_DIM=64
elements (128 bytes), so every arm it runs is aligned. Odd element offsets are the uncovered case.

bf16 only. The f32 control this probe was written with cannot run here: the fused sequence path is
bf16-only in four hardcodings, so no f32 operator can be fused at all (measured 2026-08-25, all four
byte/element spellings rejected at two gates). Run it
off fusion if the control is ever wanted.

Usage: python probe_offset_alignment.py
"""

import numpy as np
import ml_dtypes
import torch

import newstack_compat  # noqa: F401 — MUST precede iron imports
from iron.common import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.strided_copy.op import StridedCopy

N_IN = 16          # elements copied per dispatch
N_OUT = 64         # destination elements
OFFSETS = [0, 1, 2, 3, 4, 5]


def build():
    """One fused ELF whose only runtime parameter is the output offset, in elements."""
    ctx = AIEContext()
    op = StridedCopy(
        input_sizes=[N_IN], input_strides=[1], input_offset=0,
        output_sizes=[N_IN], output_strides=[1], output_offset=0,
        input_buffer_size=N_IN, output_buffer_size=N_OUT,
        num_aie_channels=1, output_offset_parameter="off", context=ctx,
    )
    seq = OperatorSequence(
        name="offprobe_bf16", runlist=[(op, "src", "dst")],
        input_args=["src"], output_args=["dst"], dispatch="fused", context=ctx,
    )
    print("[bf16] compiling...", flush=True)
    seq.compile()
    run = seq.get_callable()
    assert run.params is not None, "no runtime parameters reached params.txt"
    return run


def sweep(run):
    # Distinct nonzero ramp so a landing position is unambiguous.
    payload = np.arange(1, N_IN + 1, dtype=np.float32).astype(ml_dtypes.bfloat16)
    src = run.get_buffer("src")
    dst = run.get_buffer("dst")

    print("[bf16] elem=2B   off  bytes  aligned  landed  exact  clean@landed  nnz")
    results = []
    for off in OFFSETS:
        # No manual to("npu")/to("cpu"): the callable's _sync_inputs/_sync_outputs force both
        # directions, and _run checks the ERT return code.
        src.torch_view()[:N_IN] = torch.from_numpy(
            payload.astype(np.float32)).to(torch.bfloat16)
        dst.torch_view()[:N_OUT] = 0
        run.params.write("off", np.int32(off))
        run.params.sync()
        run()

        out = dst.torch_view()[:N_OUT].clone().float().numpy()
        nz = np.nonzero(out)[0]
        landed = int(nz.min()) if len(nz) else -1
        exact = (landed == off and
                 np.array_equal(out[off:off + N_IN], payload.astype(np.float32)))
        # Distinguishes "a correct copy at the WRONG position" from "a corrupted copy":
        # only the former is a silent wrong-slot write.
        clean = (landed >= 0 and
                 np.array_equal(out[landed:landed + N_IN], payload.astype(np.float32)))
        results.append((off, landed, exact, clean))
        aligned = (off * 2) % 4 == 0
        print(f"[bf16]           {off:>3}  {off*2:>5}  "
              f"{'yes' if aligned else 'NO':>7}  {landed:>6}  {str(exact):>5}  "
              f"{str(clean):>12}  {len(nz):>3}")
    return results


if __name__ == "__main__":
    bf = sweep(build())

    print("\n=== verdict ===")
    bad_bf = [(o, l) for o, l, e, c in bf if not e]
    odd_bad = [o for o, l, e, c in bf if o % 2 == 1 and not e]
    even_bad = [o for o, l, e, c in bf if o % 2 == 0 and not e]
    truncated = [(o, l) for o, l, e, c in bf if not e and c and l == o - 1]
    print(f"bf16 wrong: {bad_bf}   (odd wrong: {odd_bad}, even wrong: {even_bad})")
    if truncated:
        print(f"and every wrong arm is a CLEAN copy at off-1, not corruption: {truncated}")
    if odd_bad and not even_bad:
        print("CONFIRMED: bf16 fails on exactly the ODD (misaligned) offsets.")
    elif not bad_bf:
        print("REFUTED: every bf16 offset lands exactly, so misalignment is handled and "
              "M0.5's failure is something else.")
    else:
        print("NEITHER: the pattern does not match the alignment account; read the table.")
