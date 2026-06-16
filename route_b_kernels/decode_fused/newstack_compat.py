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
"""
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
