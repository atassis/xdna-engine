#!/usr/bin/env python3
"""Device gate for the embedding-gather brick (aie_kernels/embedding-gather/).
Design: docs/s2-embedding-gather-design.md. Run under the device lock: ./run.sh
verify_embedding_gather.py.

Gate discipline (error-metrics-are-notes-not-gates): 1:1 run-to-run determinism is the
BLOCKING check (each test index dispatched twice, must land bit-identical); rel-L2 is printed
as a note. A pure gather is zero-compute, so this ALSO asserts bit-exact equality against the
numpy golden (golden.py) -- not just "under a gate", the way gather-rows' own verify script
does for the same reason (an exact result is the achievable, expected one, not a stretch goal).

Not built on bricklib.py: bricklib has no offset_parameter path, and the one proven use of it
in this repo (probe_bd_gather_offsets.py) drives dispatch via raw pyxrt, not bricklib's
iron.jit convenience wrapper. See gen_embedding_gather.py's module docstring for why this
generator is standalone.

Compile note discovered while authoring this (no prior art in this repo covered it): aiecc's
raw CLI does NOT compile an ExternalFunction's source file -- that staging normally happens
inside `iron.jit`'s build orchestration (mlir-aie python/iron/kernel.py:293-295, "the @jit
decorator discovers and compiles all source files before invoking the MLIR compilation
pipeline"). Bypassing iron.jit (required here, to keep dispatch on the raw pyxrt path) means
this file's `_compile` has to do that staging step itself: compile embedding_gather.cc with
Peano first, place the .o next to aie.mlir, THEN invoke aiecc. Skipping this reproduces
`ld.lld: error: cannot open ... embedding_gather_chunk_bf16.o`. Confirmed by a device-free
compile of the real D=2560 shape while authoring this file: 43/43 aiecc steps green, a real
aie.elf + params.txt (`row_off 0 i32 addr`) written, zero device access.

Also discovered while authoring this: probe_bd_gather_offsets.py's own `_compile()` docstring
claims its `--no-xchesscc --no-xbridge` flags mirror mlir-aie's scratchpad_addr_offset RUN
line verbatim -- against the toolchain.lock pin this repo has now (435f2cbb), neither flag
exists (`aiecc --help` has no such options; Peano is already aiecc's default backend, so there
is nothing to negate) and passing them is a hard error. The RUN line actually checked out at
this pin (mlir-aie test/python/npu-xrt/scratchpad_addr_offset/test.py:9) has never had them.
This file uses the flag set that is actually accepted by the current pin.
"""
import os
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import ParameterScratchpad

sys.path.insert(0, str((Path(__file__).parent.parent / "embedding-gather").resolve()))
import gen_embedding_gather as gen  # noqa: E402
import golden  # noqa: E402

HERE = Path(__file__).parent
GEN_DIR = HERE / "gen" / "embedding_gather"

AIECC = os.environ.get("AIECC_PATH")
if not AIECC:
    print("FATAL: AIECC_PATH not set -- run this under run.sh", file=sys.stderr)
    sys.exit(2)

N_ROWS, D, CHUNK_N = 64, 2560, 16   # D=2560 matches all three real AR tables; N_ROWS is a
                                    # small synthetic stand-in (design doc: the mechanism is
                                    # table-size-independent).
GATE_REL_L2 = 3e-2                  # note only, per this project's gate discipline
DEVICE_NAME = "embgather_verify"


def _peano_include_dir(instance_dir: Path) -> Path:
    for cand in (instance_dir / "include", instance_dir / "src" / "third_party" / "aie_api" / "include"):
        if (cand / "aie_api" / "aie.hpp").exists():
            return cand
    raise RuntimeError(f"could not find aie_api/aie.hpp under {instance_dir}")


def _compile(mlir_text: str, workdir: Path):
    """Stage the external kernel object, then run aiecc. See this file's module docstring for
    why both steps are needed when bypassing iron.jit."""
    workdir.mkdir(parents=True, exist_ok=True)

    # `clang++` on PATH is the SYSTEM compiler, not Peano's aie2p-capable one (confirmed while
    # authoring this: it resolves to /usr/bin/clang++ and fails with "unknown target triple
    # 'aie2p-none-unknown-elf'"). Peano lives under PEANO_INSTALL_DIR, which run.sh already
    # exports (compile_check.sh resolves the same variable the same way).
    peano_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_dir or not (Path(peano_dir) / "bin" / "clang++").exists():
        raise RuntimeError("PEANO_INSTALL_DIR not set to a valid Peano install -- run under run.sh")
    peano_cxx = str(Path(peano_dir) / "bin" / "clang++")
    inst_dir = Path(AIECC).resolve().parent.parent   # <instance>/bin/aiecc -> <instance>
    inc_dir = _peano_include_dir(inst_dir)
    kernel_obj = workdir / f"{gen.KERNEL_SYMBOL}.o"
    cmd = [peano_cxx, "--target=aie2p-none-unknown-elf", "-O2", "-std=c++20", "-DNDEBUG",
           "-D__AIE_API_AIE_ADF_HPP__", f"-I{inc_dir}", "-c", str(gen.BRICK_CC),
           "-o", str(kernel_obj)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"embedding_gather.cc compile failed:\n{r.stderr}")

    (workdir / "aie.mlir").write_text(mlir_text)
    cmd = [AIECC, "-v", "--get-full-elf", "--dynamic-objFifos",
           "--get-scratchpad-parameters", "aie.mlir"]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        print("---- aiecc stdout (tail) ----")
        print(r.stdout[-4000:])
        print("---- aiecc stderr (tail) ----")
        print(r.stderr[-4000:])
        raise RuntimeError(f"aiecc failed rc={r.returncode} in {workdir}")
    elf, params = workdir / "aie.elf", workdir / "params.txt"
    if not elf.exists():
        raise RuntimeError(f"aiecc reported success but {elf} is missing")
    if not params.exists():
        raise RuntimeError(f"aiecc reported success but {params} is missing")
    return elf, params


def main():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((N_ROWS, D)).astype(np.float32).astype(ml_dtypes.bfloat16)

    # Same edge-case discipline as gather-rows' own verify: first row, last row, a repeat, a
    # few unremarkable middle rows. No out-of-range/negative cases -- this kernel never sees
    # an index to clamp (see embedding_gather.cc's header); clamping is exercised in
    # gen_embedding_gather.row_offset_elements's own unit behavior, not on device here.
    test_idx = [0, N_ROWS - 1, 17, 17, 5, 40]

    mlir_text = gen.build_design(n_rows=N_ROWS, d=D, chunk_n=CHUNK_N, device_name=DEVICE_NAME)
    elf_path, params_path = _compile(mlir_text, GEN_DIR)

    device = pyxrt.device(0)
    elf = pyxrt.elf(str(elf_path))
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, f"{DEVICE_NAME}:sequence")

    # in_tensor/out_tensor are allocated ONCE and reused across every dispatch below (reset +
    # re-uploaded per rep), matching probe_bd_gather_offsets.py's run_part1 exactly -- not a
    # style choice: this project has an open stale-HOST_ONLY-BO read race (the one npu-s2's
    # NPU_S2_RESYNC counts), and reallocating a fresh BO per dispatch would test a different
    # code path than the one that bug has actually been hit on.
    in_tensor = iron.tensor(table.reshape(-1), dtype=ml_dtypes.bfloat16, device="cpu")
    in_tensor.to("npu")
    out_tensor = iron.zeros((D,), dtype=ml_dtypes.bfloat16, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_tensor.buffer_object())
    params = ParameterScratchpad(run, str(params_path))

    all_ok = True
    max_rel_l2 = 0.0
    for idx in test_idx:
        row_off_val = gen.row_offset_elements(idx, N_ROWS, D)
        exp = golden.embedding_gather_ref(table.astype(np.float32), [min(max(idx, 0), N_ROWS - 1)])[0]

        reads = []
        for _rep in range(2):   # run-to-run determinism, same idx, same dispatch
            out_tensor.data.fill(0)
            out_tensor.to("npu")

            params.write(gen.OFFSET_PARAM_NAME, np.int32(row_off_val))
            params.sync()

            run.start()
            run.wait2()
            out_tensor.to("cpu")
            reads.append(out_tensor.numpy().reshape(-1).astype(np.float32).copy())

        determ = float(np.max(np.abs(reads[1] - reads[0])))
        got = reads[0]
        rel_l2 = golden.rel_l2(got, exp)
        max_rel_l2 = max(max_rel_l2, rel_l2)
        exact = bool(np.array_equal(got.astype(ml_dtypes.bfloat16), exp.astype(ml_dtypes.bfloat16)))
        nz = float(np.abs(got).sum())
        ok = (nz > 0.0) and (determ == 0.0) and exact
        all_ok &= ok
        print(f"  idx={idx:4d} row_off={row_off_val:8d}  rel_l2={rel_l2:.3e} (note)  "
              f"run2run={determ:.3e}  exact_vs_golden={exact}  -> {'PASS' if ok else 'FAIL'}")

    print(f"max rel_l2 across all test rows: {max_rel_l2:.3e} (note, gate={GATE_REL_L2:.1e})")
    assert all_ok, "embedding_gather device gate FAILED -- see per-row lines above"
    print("PASS")


if __name__ == "__main__":
    main()
