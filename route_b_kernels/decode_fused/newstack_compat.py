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
