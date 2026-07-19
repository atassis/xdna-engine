# SPDX-License-Identifier: Apache-2.0
"""Deep-C port shim: let amd/IRON's operator library run on the NEW vendored mlir-aie (1.3.2),
which has the scratchpad feature (offset_parameter + aie-lower-scratchpad-parameters) but moved a
few APIs vs the old 0.0.1 bundle amd/IRON was written against.

Import this module FIRST — before any `from iron.operators...` — to install the shim. It is a no-op
on the old stack (where `aie.iron.placers` still exists), so the same gen/verify scripts run on both.

Discovered port deltas (keep this list as the canonical record):
  1. `aie.iron.placers` was removed (placement is now automatic). amd/IRON calls
     `Program(dev, rt).resolve_program(SequentialPlacer())`; the new signature is
     `resolve_program(device_name="main")`. We provide a stand-in `SequentialPlacer` and make
     `resolve_program` drop a leading placer positional arg.
  2. The explicit-placement kwarg was renamed `placement=` -> `tile=` across the dataflow/runtime
     API (`ObjectFifoHandle.{split,join,forward}`, `Worker.__init__`, `Runtime.{fill,drain}`). The
     GEMM operator's design.py still calls `placement=Tile(...)` (deep-C only exercised GEMV/LN/etc.,
     whose design.py the deep-C patch already ported, so this delta surfaced only when GEMM was first
     built on the new stack for the lever-3 batching probe). We rename the kwarg at call time so the
     unported GEMM design.py runs unchanged. No-op on the old stack (this branch never runs there).
  3. The device geometry helpers `get_shim_tiles` / `get_num_connections` / `get_mem_tiles` were all
     removed (hunhoffe #3213 moved placement into the compiler placer), breaking amd/IRON's
     `get_shim_dma_limit`. We patch `iron.common.utils.get_shim_dma_limit` to derive the ShimDMA
     budget from the surviving `dev._tm` (AIETargetModel): (#shim tiles) * 2 output channels = 16 on
     NPU2. See the block below for the validation. No-op on the old stack.
"""
import functools
import sys
import types

try:
    import aie.iron.placers  # noqa: F401  — present on the OLD stack: nothing to do.
except ImportError:
    # NEW vendored mlir-aie (1.3.2): synthesize the removed module + adapt resolve_program.
    _placers = types.ModuleType("aie.iron.placers")

    class SequentialPlacer:  # no-op stand-in; the new mlir-aie auto-places.
        def __init__(self, *a, **k):
            pass

    _placers.SequentialPlacer = SequentialPlacer
    _placers.Placer = type("Placer", (), {})
    sys.modules["aie.iron.placers"] = _placers

    from aie.iron.program import Program as _Program

    _orig_resolve = _Program.resolve_program

    def _resolve_program(self, *args, **kwargs):
        args = tuple(a for a in args if not isinstance(a, SequentialPlacer))
        return _orig_resolve(self, *args, **kwargs)

    _Program.resolve_program = _resolve_program

    # Delta 2: rename `placement=` -> `tile=` at call time on the methods amd/IRON's (unported)
    # GEMM design.py still calls with the old kwarg name.
    def _rename_placement(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if "placement" in kwargs:
                kwargs.setdefault("tile", kwargs.pop("placement"))
            return fn(*args, **kwargs)
        return _wrapped

    from aie.iron.dataflow import ObjectFifoHandle
    from aie.iron.worker import Worker
    from aie.iron import Runtime

    for _cls, _meth in (
        (ObjectFifoHandle, "split"),
        (ObjectFifoHandle, "join"),
        (ObjectFifoHandle, "forward"),
        (Worker, "__init__"),
        (Runtime, "fill"),
        (Runtime, "drain"),
    ):
        setattr(_cls, _meth, _rename_placement(getattr(_cls, _meth)))

    # Delta 3: amd/IRON's per-tile device-geometry helpers were deleted upstream -- hunhoffe's #3213
    # (+ 789e8bc1dc3) moved placement into the compiler placer, so the NPU2 device object lost BOTH
    # `get_shim_tiles` AND `get_num_connections`, and the older-wheel shim that once restored the latter
    # leaned on `get_mem_tiles`, itself now gone. amd/IRON's `get_shim_dma_limit`
    # (iron/common/utils.py, added by #114) is `sum(get_num_connections(t, output=True) for t in
    # get_shim_tiles())` -- it crashes on all three at the current pins (fb1f7095 and fa85bb34).
    # Fix at the source (do NOT resurrect the deleted per-tile methods): derive the ShimDMA budget
    # purely from the device target model `dev._tm` (an `aie._mlir_libs._aie.AIETargetModel`, which
    # survived). Each shim tile exposes 2 ShimDMA output (MM2S) channels on AIE2P/NPU2 (validated vs
    # the e4f35d6 wheel: 2 DMA channels/direction); the budget is (#shim tiles) * 2. Verified on both
    # pins: NPU2 has 8 shim tiles -> 16, identical to the wheel's native sum. Note `_tm.columns()` /
    # `_tm.rows()` are methods, and `is_shim_noc_or_pl_tile(col,row)` is on `_tm`, not the device.
    # No-op on the old stack (this whole branch only runs where aie.iron.placers is absent).
    import iron.common.utils as _iron_utils

    def _get_shim_dma_limit(dev):
        tm = dev._tm
        cols, rows = tm.columns(), tm.rows()
        n_shim = sum(
            1
            for c in range(cols)
            for r in range(rows)
            if tm.is_shim_noc_or_pl_tile(c, r)
        )
        return n_shim * 2  # 2 ShimDMA output channels per shim tile (AIE2P/NPU2)

    _iron_utils.get_shim_dma_limit = _get_shim_dma_limit

    # Belt-and-suspenders: if any other op path still calls the removed `get_num_connections`,
    # provide it deriving mem-tile rows from `_tm` (the old shim's `get_mem_tiles()` is also gone).
    from aie.iron.device import NPU2 as _NPU2

    def _get_num_connections(self, tile, output=True):
        return 6 if self._tm.is_mem_tile(tile.col, tile.row) else 2

    if not hasattr(_NPU2(), "get_num_connections"):
        _NPU2.get_num_connections = _get_num_connections
