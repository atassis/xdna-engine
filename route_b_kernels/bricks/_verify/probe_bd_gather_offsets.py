#!/usr/bin/env python3
"""Does AIE2P's BD/DMA engine support a RUNTIME (data-dependent) row offset -- the single
open question blocking the S2 AR embedding gather (155776x2560 / 40960x2560 bf16 tables,
~800 MB / ~200 MB, can never be L1- or MemTile-resident, so the gather must be driven by
the DMA/BD layer reading rows at runtime-computed L3 offsets)?

SOURCE READING (see docs/s2-bd-gather-feasibility.md for full citations) found:

  1. `aiex.npu.dma_memcpy_nd` and `aie.dma_bd` both have SSA-operand offset/size/stride
     slots in the dialect (not compile-time-attribute-only) -- the IR shape supports
     runtime values.
  2. BUT the toolchain's own correctness harness
     (mlir-aie src/test/Targets/NPU/static_vs_dynamic/README.md) states outright that
     "genuinely runtime-valued DMA sizes/strides arrive with the Phase-2 dynamic BD-word
     encoder" -- NOT YET SHIPPED in this pinned instance. A core cannot compute an index
     from streamed data and feed it into a BD field within a dispatch; no such example
     exists anywhere in the fork (programming_examples/, test/).
  3. The ONE runtime-offset mechanism that IS proven end-to-end, in-tree, and already
     shipping in THIS project: `aiex.scratchpad_parameter` + `offset_parameter=` on a
     dma_bd/dma_memcpy_nd (mlir-aie src/test/python/npu-xrt/scratchpad_addr_offset/,
     CI-gated) -- the exact mechanism this repo's own decode_fused/gen_decode.py already
     uses for the per-token `kv_off` KV-cache-append offset. The offset value is WRITTEN
     BY THE HOST into a small scratchpad (max 32 x i32 / 128 bytes total) BEFORE each
     dispatch; the command-processor firmware ADDS it to a BD address register. This is a
     per-dispatch (or per compile-time-unrolled-slot) host-supplied offset, not a value
     the chip computes for itself mid-dispatch from data it just read.

This probe tests exactly mechanism #3 -- the nearest LEGAL thing to a true data-dependent
gather -- since #1/#2 (a core driving its own BD from streamed data) has no toolchain
hook to even attempt. Two parts:

  PART 1 (primary, decisive, low-risk): one `offset_parameter`-patched dma_bd reading a
    single row from a table sized WELL PAST L1 (256 KB vs 64 KB/core), so the read MUST
    be a genuine multi-hop shim->core DMA, not a resident-buffer illusion. The SAME
    compiled ELF is re-dispatched 3x with 3 different HOST-WRITTEN offsets (start/mid/end
    of the table) and NO recompilation. Table is a ramp (row r's data == r, broadcast
    across the row) so a wrong offset is unmistakable. This is a close variant of mlir-aie's
    own CI-gated scratchpad_addr_offset test, scaled up and re-purposed as a row gather.

  PART 2 (secondary, exploratory, answers the T-rows-per-dispatch cost question): T=2
    INDEPENDENT offset_parameters, each driving its own dma_bd, both patched by the host
    and both fired in a SINGLE dispatch. This is a NEW combination not covered by any
    existing in-tree test (each ingredient -- offset_parameter singly; multi-tile IRON
    designs -- is proven separately, but not jointly). A pass here means T embedding rows
    can be gathered per xclbin call (bounded by the 32-slot state table); a build/run
    failure here does NOT overturn Part 1's answer to the primary question.

VERDICT semantics: any check that returns all-zero or the wrong row is FAIL, never
silently treated as pass. Part 1 failing means mechanism #3 itself is not usable as
documented -- a major, surprising finding since it contradicts a passing upstream CI
test. Part 2 failing only bounds T to 1-row-per-dispatch; it does not touch the primary
verdict.

Run: cd route_b_kernels/bricks/_verify && ./run.sh probe_bd_gather_offsets.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyxrt

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import ParameterScratchpad

from bricklib import GEN

AIECC = os.environ.get("AIECC_PATH")
if not AIECC:
    print("FATAL: AIECC_PATH not set -- this probe must run under run.sh", file=sys.stderr)
    sys.exit(2)


def _force_pdi_reload(mlir_text: str) -> str:
    """Insert an `aie.device(npu2) @empty {}` ahead of the real device.

    Per AIEX.td's NpuLoadPdiOp docs: "The firmware optimizes out repeated load_pdi
    operations if they refer to the same PDI. To force a reload ... intersperse a
    load_pdi to a different PDI." Every dispatch in this probe re-runs the whole
    runtime_sequence (including its npu_load_pdi calls), so without this the 2nd/3rd
    dispatch of the SAME PDI could be silently skipped by the dedup -- which would make
    a bug (offset not actually re-applied) look like a pass. Verbatim technique from
    mlir-aie's own scratchpad_addr_offset test (test/python/npu-xrt/scratchpad_addr_offset/aie_design.py).
    """
    empty = "  aie.device(npu2) @empty { }\n"
    return mlir_text.replace("module {\n", "module {\n" + empty, 1)


def _compile(mlir_text: str, workdir: Path):
    """aiecc invocation mirrors mlir-aie's own scratchpad_addr_offset run.lit RUN line
    verbatim (`-v --get-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos
    --get-scratchpad-parameters`), since that combination is the ONE proven-in-CI path
    for `offset_parameter`. Deviating from it (e.g. going through the higher-level
    `aie.utils.compile.utils.compile_mlir_module`, which does not wire the
    --get-scratchpad-parameters flag at all) is unproven and was avoided.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "aie.mlir").write_text(mlir_text)
    cmd = [AIECC, "-v", "--get-full-elf", "--no-xchesscc", "--no-xbridge",
           "--dynamic-objFifos", "--get-scratchpad-parameters", "aie.mlir"]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        print("---- aiecc stdout (tail) ----")
        print(r.stdout[-4000:])
        print("---- aiecc stderr (tail) ----")
        print(r.stderr[-4000:])
        raise RuntimeError(f"aiecc failed rc={r.returncode} in {workdir}")
    elf = workdir / "aie.elf"
    params = workdir / "params.txt"
    if not elf.exists():
        raise RuntimeError(f"aiecc reported success but {elf} is missing")
    if not params.exists():
        raise RuntimeError(f"aiecc reported success but {params} is missing "
                            "(--get-scratchpad-parameters did not emit params.txt)")
    return elf, params


# ============================================================================
# PART 1 -- single offset_parameter, table >> L1, 3 dispatches, no recompile.
# ============================================================================
TABLE_ROWS = 4096
ROW_LEN = 16  # i32 -> 64 B/row; TABLE_ROWS*ROW_LEN*4 = 262144 B = 256 KiB >> 64 KiB L1/core.
P1_ROWS = [0, 2047, 4095]  # start / middle / end -- rules out "only offset 0 works".
P1_DEVICE = "gatherp1"


def build_part1():
    in_ty = np.ndarray[(TABLE_ROWS * ROW_LEN,), np.dtype[np.int32]]
    out_ty = np.ndarray[(ROW_LEN,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(ROW_LEN,), np.dtype[np.int32]]

    row_off = ScratchpadParameter("row_off", np.int32)

    of_in = ObjectFifo(tile_ty, name="gin")
    of_out = ObjectFifo(tile_ty, name="gout")

    def core_fn(of_in, of_out):
        in_elem = of_in.acquire(1)
        out_elem = of_out.acquire(1)
        for i in range(ROW_LEN):
            out_elem[i] = in_elem[i]
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()], while_true=False)

    def sequence(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=P1_DEVICE)
        # Static tap offset is 0; the ACTUAL row is entirely driven by row_off (element
        # units, added by the firmware to whatever's in the BD address register -- see
        # AIEX.td NpuUpdateFromScratchpadOp: "always additive").
        tap = TensorAccessPattern(
            (TABLE_ROWS * ROW_LEN,), offset=0, sizes=[1, 1, 1, ROW_LEN], strides=[0, 0, 0, 1]
        )
        in_h.fill(in_tensor, tap=tap, offset_parameter=row_off)
        out_h.drain(out_tensor, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(device_name=P1_DEVICE)
    return _force_pdi_reload(str(module))


def run_part1():
    print("=" * 78)
    print("PART 1 -- offset_parameter row gather from a table >> L1 (256 KiB), "
          "3 dispatches, 1 compile")
    print("=" * 78)
    mlir_text = build_part1()
    elf_path, params_path = _compile(mlir_text, GEN / "bd_gather_part1")

    table = np.zeros((TABLE_ROWS, ROW_LEN), dtype=np.int32)
    table[:, :] = np.arange(TABLE_ROWS, dtype=np.int32)[:, None]  # row r's data == r.

    device = pyxrt.device(0)
    elf = pyxrt.elf(str(elf_path))
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, f"{P1_DEVICE}:sequence")

    in_tensor = iron.tensor(table.reshape(-1), dtype=np.int32, device="cpu")
    out_tensor = iron.zeros((ROW_LEN,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_tensor.buffer_object())
    params = ParameterScratchpad(run, str(params_path))

    in_tensor.to("npu")

    all_ok = True
    for row_idx in P1_ROWS:
        out_tensor.data.fill(0)
        out_tensor.to("npu")

        params.write("row_off", np.int32(row_idx * ROW_LEN))  # element units.
        params.sync()

        run.start()
        run.wait2()
        out_tensor.to("cpu")

        got = out_tensor.numpy().reshape(-1).astype(np.int64)
        expected = np.full(ROW_LEN, row_idx, dtype=np.int64)
        nz = int(np.abs(got).sum())
        ok = (nz > 0) and np.array_equal(got, expected)
        all_ok &= ok
        status = "PASS" if ok else ("FAIL-ZERO" if nz == 0 else "FAIL")
        print(f"  row_off={row_idx:5d}  expected={expected[:4].tolist()}...  "
              f"got={got[:4].tolist()}...  -> {status}")
        if not ok:
            print(f"    FULL got row: {got.tolist()}")

    print(f"PART 1 VERDICT: {'PASS' if all_ok else 'FAIL'} -- "
          + ("a host-written scratchpad parameter DOES retarget a dma_bd's row offset "
             "at runtime, against a table far larger than L1, with no recompilation "
             "between dispatches."
             if all_ok else
             "the offset_parameter mechanism did NOT behave as documented -- treat this "
             "as a MAJOR finding, it contradicts a passing upstream CI test "
             "(test/python/npu-xrt/scratchpad_addr_offset)."))
    return all_ok


# ============================================================================
# PART 2 -- two INDEPENDENT offset_parameters, both patched, ONE dispatch.
# Answers: can T>1 gather rows be pulled per xclbin call, or does each row need its
# own dispatch? NEW combination, not covered by any existing in-tree test.
# ============================================================================
P2_DEVICE = "gatherp2"
P2_ROW_A = 777
P2_ROW_B = 3141


def build_part2():
    in_ty = np.ndarray[(TABLE_ROWS * ROW_LEN,), np.dtype[np.int32]]
    out_ty = np.ndarray[(ROW_LEN,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(ROW_LEN,), np.dtype[np.int32]]

    off_a = ScratchpadParameter("row_off_a", np.int32)
    off_b = ScratchpadParameter("row_off_b", np.int32)

    of_in_a = ObjectFifo(tile_ty, name="gin_a")
    of_out_a = ObjectFifo(tile_ty, name="gout_a")
    of_in_b = ObjectFifo(tile_ty, name="gin_b")
    of_out_b = ObjectFifo(tile_ty, name="gout_b")

    def core_fn(of_in, of_out):
        in_elem = of_in.acquire(1)
        out_elem = of_out.acquire(1)
        for i in range(ROW_LEN):
            out_elem[i] = in_elem[i]
        of_in.release(1)
        of_out.release(1)

    # Two independent tile pairs so neither BD/channel/lock is shared -- the lowest-risk
    # way to prove two SIMULTANEOUS runtime offsets, at the cost of not being the literal
    # production topology (a real T-row gather brick would want one BD chain or repeated
    # re-use of one channel, not T dedicated compute tiles; see docs/s2-bd-gather-feasibility.md).
    worker_a = Worker(core_fn, [of_in_a.cons(), of_out_a.prod()], while_true=False)
    worker_b = Worker(core_fn, [of_in_b.cons(), of_out_b.prod()], while_true=False)

    def sequence(in_tensor, out_a, out_b, in_h_a, in_h_b, out_h_a, out_h_b):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=P2_DEVICE)
        tap = TensorAccessPattern(
            (TABLE_ROWS * ROW_LEN,), offset=0, sizes=[1, 1, 1, ROW_LEN], strides=[0, 0, 0, 1]
        )
        # Both fills read the SAME resident table tensor -- exactly the shape of a real
        # gather (one codebook, many independently-indexed row reads) -- each patched by
        # its OWN scratchpad parameter, both issued before a single wait.
        in_h_a.fill(in_tensor, tap=tap, offset_parameter=off_a)
        in_h_b.fill(in_tensor, tap=tap, offset_parameter=off_b)
        out_h_a.drain(out_a, wait=True)
        out_h_b.drain(out_b, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, out_ty,
                            of_in_a.prod(), of_in_b.prod(), of_out_a.cons(), of_out_b.cons()])
    module = Program(NPU2(), rt, workers=[worker_a, worker_b]).resolve_program(device_name=P2_DEVICE)
    return _force_pdi_reload(str(module))


def run_part2():
    print("=" * 78)
    print("PART 2 (exploratory) -- 2 independent offset_parameters, 1 dispatch")
    print("=" * 78)
    mlir_text = build_part2()
    elf_path, params_path = _compile(mlir_text, GEN / "bd_gather_part2")

    table = np.zeros((TABLE_ROWS, ROW_LEN), dtype=np.int32)
    table[:, :] = np.arange(TABLE_ROWS, dtype=np.int32)[:, None]

    device = pyxrt.device(0)
    elf = pyxrt.elf(str(elf_path))
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, f"{P2_DEVICE}:sequence")

    in_tensor = iron.tensor(table.reshape(-1), dtype=np.int32, device="cpu")
    out_a = iron.zeros((ROW_LEN,), dtype=np.int32, device="cpu")
    out_b = iron.zeros((ROW_LEN,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_a.buffer_object())
    run.set_arg(2, out_b.buffer_object())
    params = ParameterScratchpad(run, str(params_path))

    in_tensor.to("npu")
    out_a.data.fill(0); out_a.to("npu")
    out_b.data.fill(0); out_b.to("npu")

    params.write("row_off_a", np.int32(P2_ROW_A * ROW_LEN))
    params.write("row_off_b", np.int32(P2_ROW_B * ROW_LEN))
    params.sync()

    run.start()
    run.wait2()
    out_a.to("cpu"); out_b.to("cpu")

    got_a = out_a.numpy().reshape(-1).astype(np.int64)
    got_b = out_b.numpy().reshape(-1).astype(np.int64)
    exp_a = np.full(ROW_LEN, P2_ROW_A, dtype=np.int64)
    exp_b = np.full(ROW_LEN, P2_ROW_B, dtype=np.int64)

    ok_a = int(np.abs(got_a).sum()) > 0 and np.array_equal(got_a, exp_a)
    ok_b = int(np.abs(got_b).sum()) > 0 and np.array_equal(got_b, exp_b)
    ok_distinct = not np.array_equal(got_a, got_b)  # catches "both slots aliased" bugs.
    all_ok = ok_a and ok_b and ok_distinct

    print(f"  row_off_a={P2_ROW_A}  expected={exp_a[:4].tolist()}...  got={got_a[:4].tolist()}...  "
          f"-> {'PASS' if ok_a else 'FAIL'}")
    print(f"  row_off_b={P2_ROW_B}  expected={exp_b[:4].tolist()}...  got={got_b[:4].tolist()}...  "
          f"-> {'PASS' if ok_b else 'FAIL'}")
    print(f"  A/B distinct (not aliased): {'PASS' if ok_distinct else 'FAIL'}")
    print(f"PART 2 VERDICT: {'PASS' if all_ok else 'FAIL'} -- "
          + ("T independent runtime offsets CAN be patched and fired in a single "
             "dispatch; a real gather brick can batch T<=32 rows per xclbin call."
             if all_ok else
             "T>1 offsets in one dispatch did NOT both come back correct -- treat the "
             "gather as bounded to 1 row per dispatch until this is root-caused (it may "
             "be a probe-construction issue, not a hardware limit -- this combination "
             "is not covered by any existing in-tree test)."))
    return all_ok


def main():
    p1_ok = None
    p2_ok = None
    p1_err = None
    p2_err = None

    try:
        p1_ok = run_part1()
    except Exception as e:  # noqa: BLE001 -- report, don't crash before Part 2 / final verdict.
        p1_err = e
        print(f"PART 1 ERROR (build/run failed before a correctness check could run): {e}")

    try:
        p2_ok = run_part2()
    except Exception as e:  # noqa: BLE001
        p2_err = e
        print(f"PART 2 ERROR (build/run failed before a correctness check could run): {e}")

    print()
    print("=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    if p1_ok:
        print("Host-driven, per-dispatch scratchpad-parameter BD offset patching WORKS on "
              "AIE2P, including against a table far larger than L1 (SUPPORTED-WITH-CAVEATS: "
              "the offset must be known to the HOST before the dispatch launches; a core "
              "cannot compute a row index from streamed data and feed it into a BD field "
              "within a dispatch -- no such mechanism exists in this toolchain instance, "
              "see docs/s2-bd-gather-feasibility.md).")
    elif p1_err is not None:
        print("Part 1 could not even build/run -- INCONCLUSIVE on the primary question. "
              "Check the aiecc output above before concluding anything about hardware "
              "capability; this may be an environment issue (stale instance, missing "
              "aiebu-asm, etc.), not a toolchain limit.")
    else:
        print("Part 1 built and ran but did NOT reproduce the documented behavior -- "
              "NOT SUPPORTED as read from source. This contradicts mlir-aie's own passing "
              "CI test and should be escalated, not quietly worked around.")

    if p2_ok is True:
        print("Bonus: T=2 independent offsets fired correctly in ONE dispatch -- batched "
              "gather (up to the 32-slot state-table cap) looks viable.")
    elif p2_err is not None:
        print("Part 2 (batched, exploratory) could not build/run -- inconclusive on "
              "T-rows-per-dispatch; does not change the Part 1 verdict.")
    else:
        print("Part 2 (batched, exploratory) did not reproduce correctly -- treat T>1 "
              "rows-per-dispatch as unproven; each row may need its own dispatch. Does "
              "not change the Part 1 verdict.")

    print()
    print("WHAT THIS PROBE CANNOT DETERMINE: whether a value COMPUTED ON THE CHIP (e.g. "
          "a token id produced by a previous on-NPU op) can drive a BD offset WITHOUT a "
          "host write in between. No such mechanism exists anywhere in this fork to even "
          "attempt (source-read finding, see docs/s2-bd-gather-feasibility.md) -- that "
          "remains open pending the toolchain's own 'Phase-2 dynamic BD-word encoder'.")

    sys.exit(0 if p1_ok else 1)


if __name__ == "__main__":
    main()
