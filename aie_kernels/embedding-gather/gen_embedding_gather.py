#!/usr/bin/env python3
"""IRON design generator for the embedding-gather brick (contract (b): L3-resident table,
host-written per-dispatch row offset). Design + byte arithmetic: docs/s2-embedding-gather-design.md.
Mechanism proof: aie_kernels/_test/probe_bd_gather_offsets.py,
docs/s2-bd-gather-feasibility.md.

Standalone rather than a bricklib._build_* addition: bricklib.py has no offset_parameter path
(gather_rows.cc:57-60's own header names this as the extension point), and every proven use of
offset_parameter in this repo (the probe above; decode_fused/gen_decode.py's `kv_off`) drives the
kernel via the RAW pyxrt dispatch path, not bricklib's iron.jit convenience wrapper -- so this
generator is built and compiled the same way the probe is, not folded into the shared harness.

build_design() mirrors two independently-proven pieces glued together for the first time here:
bricklib._build_streamed's no-resident, multi-tile `TensorTiler2D.group_tiler` grouped `fill()`
(every green brick in this catalog) and the probe's single-tile `offset_parameter=`. ObjectFifoHandle
.fill()/._emit_transfer (mlir-aie python/iron/dataflow/objectfifo.py:804-839,692-700) take `tap` and
`offset_parameter` as independent keyword args feeding one DMATask, so the offset patches the BD's
base-address register once while the tap's own strides walk the n_tiles chunks from that base --
this composition is expected to work by construction, but is untested before this file; that is what
verify_embedding_gather.py's device gate actually checks.
"""
from pathlib import Path

import ml_dtypes
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorTiler2D

BRICK_CC = Path(__file__).parent / "embedding_gather.cc"
KERNEL_SYMBOL = "embedding_gather_chunk_bf16"
OFFSET_PARAM_NAME = "row_off"


def _force_pdi_reload(mlir_text: str) -> str:
    """One design, re-dispatched per row with a fresh `row_off`: without an interleaved
    load_pdi to a different PDI, the firmware's repeated-load_pdi dedup can skip re-applying
    the offset on the 2nd+ dispatch of the same PDI (mlir-aie's own
    test/python/npu-xrt/scratchpad_addr_offset technique; load-bearing here, not a probe-only
    artifact, since a real AR driver dispatches this design once per gathered row)."""
    empty = "  aie.device(npu2) @empty { }\n"
    return mlir_text.replace("module {\n", "module {\n" + empty, 1)


def build_design(n_rows: int, d: int, chunk_n: int = 16, device_name: str = "embgather",
                 compile_flags=None, stack_size=None):
    """n_rows/d: logical table shape [n_rows, d], bf16. chunk_n: this kernel's own copy width
    (embedding_gather.cc's GATHER_CHUNK_N) -- must divide d. The D//chunk_n repeat lives in the
    WORKER loop below, never inside the kernel (see embedding_gather.cc's header).

    Returns MLIR text for `aiecc -v --get-full-elf --dynamic-objFifos
    --get-scratchpad-parameters` (verify_embedding_gather.py's `_compile`; see its module
    docstring for why this flag set, not probe_bd_gather_offsets.py's, is what actually builds
    against the current toolchain.lock pin).

    stack_size: None takes the AIE dialect's 0x400 default. The kernel body is one load_v/
    store_v with no local array, so this is expected to be well under that -- but the frame
    was never measured on this pin (no device access for this task), and an undersized
    reservation overwrites the objectFIFO buffers SILENTLY (bricklib.py's own
    `_build_streamed` docstring; the private "hanging numbers are bugs" doctrine's own worked
    example is exactly this class of bug). Pass an explicit value if a build/device run shows
    otherwise.
    """
    if d % chunk_n != 0:
        raise ValueError(f"D={d} must be a multiple of chunk_n={chunk_n}")
    elem_bytes = 2  # bf16
    if (d * elem_bytes) % 4 != 0:
        # A misaligned runtime offset silently lands one slot short, with no error: the
        # firmware patches a BD base-address register that addresses 32-bit words. Measured
        # 2026-08-25; see the design doc's alignment-hazard section.
        raise ValueError(
            f"D*elem_bytes={d * elem_bytes} is not a multiple of 4 -- row_off would not land "
            "on a 4-byte boundary for every idx, and the runtime offset_parameter path "
            "truncates a misaligned offset silently rather than erroring. Widen D or the dtype.")
    n_tiles = d // chunk_n

    compile_flags = list(compile_flags or [])
    row_off = ScratchpadParameter(OFFSET_PARAM_NAME, np.int32)

    tile_ty = np.ndarray[(chunk_n,), np.dtype[ml_dtypes.bfloat16]]
    table_ty = np.ndarray[(n_rows * d,), np.dtype[ml_dtypes.bfloat16]]
    row_ty = np.ndarray[(d,), np.dtype[ml_dtypes.bfloat16]]

    kern = ExternalFunction(KERNEL_SYMBOL, source_file=str(BRICK_CC),
                            arg_types=[tile_ty, tile_ty], compile_flags=compile_flags)
    of_in = ObjectFifo(tile_ty, name="egin")
    of_out = ObjectFifo(tile_ty, name="egout")

    def core(of_in, of_out, kern):
        for _ in range_(n_tiles):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            kern(ei, eo)
            of_out.release(1)
            of_in.release(1)

    worker = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern], stack_size=stack_size)

    in_tap = TensorTiler2D.group_tiler((n_tiles, chunk_n), (1, chunk_n), (n_tiles, 1))[0]
    out_tap = TensorTiler2D.group_tiler((n_tiles, chunk_n), (1, chunk_n), (n_tiles, 1))[0]

    def sequence(table, out_row, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)
        # ONE fill covers the whole row (n_tiles chunks): row_off is an ELEMENT-unit offset
        # added once to this transfer's base address; in_tap's own strides walk the n_tiles
        # chunks from that patched base (see this file's module docstring).
        in_h.fill(table, tap=in_tap, offset_parameter=row_off)
        out_h.drain(out_row, tap=out_tap, wait=True)

    rt = Runtime(sequence, [table_ty, row_ty, of_in.prod(), of_out.cons()])
    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(device_name=device_name)
    return _force_pdi_reload(str(module))


def row_offset_elements(idx: int, n_rows: int, d: int) -> int:
    """Host-side clamp + element-unit offset for `row_off`. This kernel never sees `idx` (see
    embedding_gather.cc's header), so the [0, n_rows) clamp -- matching gather-rows.cc's
    clamp_code semantics -- has to happen here, before the scratchpad write."""
    clamped = max(0, min(int(idx), n_rows - 1))
    return clamped * d
