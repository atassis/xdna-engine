"""Does a MISALIGNED runtime element offset corrupt a bf16 StridedCopy write?

M0.5 (--coalesce-self-tr) writes the self-V cache transposed with a per-token column offset
`vcache_off = n_self` in ELEMENT units. At bf16 that is 2*n_self bytes, which lands on a 4-byte
boundary only for EVEN n_self -- while the contiguous kcache write uses kv_off = n_self*head_dim,
always a multiple of 128 bytes and so aligned for every token. The static path rejects a misaligned
offset outright (AIEXDialect.cpp:649, "Offset must be 4-byte-aligned"); the runtime scratchpad path
feeds the same BD address registers with nothing able to check it.

This isolates that one variable: one StridedCopy, sweeping the runtime output offset 0..5.
bf16 is the arm under test; f32 is the control, where every element offset is inherently 4-byte
aligned and so every offset must land exactly.

Usage: python probe_offset_alignment.py
"""

import glob
import os

import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 — MUST precede iron imports
import pyxrt
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import ParameterScratchpad
from iron.common import AIEContext
from iron.common.fusion import FusedMLIROperator
from iron.operators.strided_copy.op import StridedCopy

N_IN = 16          # elements copied per dispatch
N_OUT = 64         # destination elements
OFFSETS = [0, 1, 2, 3, 4, 5]


def run_dtype(np_dt, elem_bytes, label):
    ctx = AIEContext()
    # A plain contiguous copy; the ONLY variable is the runtime output offset.
    op = StridedCopy(
        input_sizes=(1, N_IN), input_strides=(0, 1), input_offset=0,
        output_sizes=(1, N_IN), output_strides=(0, 1), output_offset=0,
        input_buffer_size=N_IN, output_buffer_size=N_OUT,
        num_aie_channels=1, output_offset_parameter="off", context=ctx,
    )
    name = f"offprobe{label}"
    fused = FusedMLIROperator(
        name, [(op, "src", "dst")],
        input_args=["src"], output_args=["dst"],
        buffer_sizes={"src": N_IN * elem_bytes, "dst": N_OUT * elem_bytes},
        context=ctx,
    )
    print(f"[{label}] compiling...", flush=True)
    fused.compile()
    callable_ = fused.get_callable()
    params_path = sorted(glob.glob(f"**/{name}*.mlir.prj/params.txt", recursive=True),
                         key=os.path.getmtime)[-1]
    print(f"[{label}] params.txt: {params_path} -> "
          f"{open(params_path).read().strip().replace(chr(10), ' | ')}")

    run = pyxrt.run(callable_.xrt_kernel)
    run.set_arg(0, callable_.input_buffer.buffer_object())
    run.set_arg(1, callable_.output_buffer.buffer_object())
    run.set_arg(2, callable_.scratch_buffer.buffer_object())
    sp = ParameterScratchpad(run, params_path)

    src = callable_.get_buffer("src")
    dst = callable_.get_buffer("dst")
    # Distinct nonzero ramp so a landing position is unambiguous.
    payload = np.arange(1, N_IN + 1).astype(np_dt)

    print(f"[{label}] elem={elem_bytes}B   off  bytes  aligned  landed  exact")
    results = []
    for off in OFFSETS:
        np.copyto(src.data, payload.reshape(-1))
        np.copyto(dst.data, np.zeros(N_OUT, dtype=np_dt))
        callable_.input_buffer.to("npu")
        callable_.output_buffer.to("npu")
        sp.write("off", off)
        sp.sync()
        run.start(); run.wait2()
        callable_.output_buffer.to("cpu")
        out = np.asarray(dst.data).astype(np.float32)

        nz = np.nonzero(out)[0]
        landed = int(nz.min()) if len(nz) else -1
        exact = (landed == off and
                 np.array_equal(out[off:off + N_IN], payload.astype(np.float32)))
        results.append((off, landed, exact))
        aligned = (off * elem_bytes) % 4 == 0
        print(f"[{label}]           {off:>3}  {off*elem_bytes:>5}  "
              f"{'yes' if aligned else 'NO':>7}  {landed:>6}  {exact}")
    return results


if __name__ == "__main__":
    bf = run_dtype(ml_dtypes.bfloat16, 2, "bf16")
    f32 = run_dtype(np.float32, 4, "f32")

    print("\n=== verdict ===")
    bad_bf = [(o, l) for o, l, e in bf if not e]
    bad_f32 = [(o, l) for o, l, e in f32 if not e]
    odd_bad = [o for o, l, e in bf if o % 2 == 1 and not e]
    even_bad = [o for o, l, e in bf if o % 2 == 0 and not e]
    print(f"bf16 wrong: {bad_bf}   (odd wrong: {odd_bad}, even wrong: {even_bad})")
    print(f"f32  wrong: {bad_f32}")
    if odd_bad and not even_bad and not bad_f32:
        print("CONFIRMED: bf16 fails on exactly the ODD (misaligned) offsets; f32 control clean.")
    elif not bad_bf and not bad_f32:
        print("REFUTED: every bf16 offset lands exactly, so misalignment is handled and "
              "M0.5's failure is something else.")
    else:
        print("NEITHER: the pattern does not match the alignment account; read the table.")
