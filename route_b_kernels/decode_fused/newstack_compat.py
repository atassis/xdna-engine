# SPDX-License-Identifier: Apache-2.0
"""Deep-C port shim: let amd/IRON's operator library run on the pinned mlir-aie instance, which has
the scratchpad feature (offset_parameter + aie-lower-scratchpad-parameters) but moves several APIs
amd/IRON is written against.

Import this module FIRST — before any `from iron.operators...` — to install the shim. It is a no-op
on the old stack (where `aie.iron.placers` still exists), so the same gen/verify scripts run on both.

Port deltas (keep this list as the canonical record):
  1. `aie.iron.placers` was removed (placement is automatic). amd/IRON calls
     `Program(dev, rt).resolve_program(SequentialPlacer())`; the current signature is
     `resolve_program(device_name="main")`. We provide a stand-in `SequentialPlacer` and make
     `resolve_program` drop a leading placer positional arg.
  2. The explicit-placement kwarg is `tile=`, spelled `placement=` in amd/IRON's gemm and mha
     design.py. Renamed at call time on the dataflow methods that still accept a tile.
  3. `Device.{get_shim_tiles,get_mem_tiles,get_compute_tiles,get_num_connections}` were removed in
     favour of `cols`/`rows`/`get_tile_type`. amd/IRON's `get_shim_dma_limit` (iron/common/utils.py)
     uses the enumeration to size the ShimDMA budget. Reconstructed from `get_tile_type`.
  4. #3387 reworked Runtime into an eager callback body: the no-arg `Runtime()` + `with
     rt.sequence(...)` spelling every amd/IRON design.py uses no longer exists, and its verbs moved
     onto other objects (fill/drain to the ObjectFifo handle, task groups to `TaskGroup`, barrier
     sets to the barrier, workers to `Program(workers=)`). The two spellings cannot coexist in one
     Runtime, so the compat class RECORDS an old-spelling body and replays it inside the callback
     the current Runtime requires. Recording is per-instance and keyed on the constructor: a
     `Runtime(seq_fn, fn_args)` call is the current spelling and passes straight through, so our own
     generators and amd/IRON's designs both run in one process.
  5. #3364 made every aiecc output an explicit request (`--get-<name>`), so the `--aie-generate-*`
     spellings amd/IRON's compilation rules emit are gone, and the Chess/host negations with them
     (Peano is the default backend, the host program is one more requestable output). Translated on
     the argv, which covers every rule that shells out to aiecc rather than each rule's flag list.
"""
import contextlib
import functools
import sys
import types

try:
    import aie.iron.placers  # noqa: F401  — present on the OLD stack: nothing to do.
except ImportError:
    _placers = types.ModuleType("aie.iron.placers")

    class SequentialPlacer:  # no-op stand-in; the compiler auto-places.
        def __init__(self, *a, **k):
            pass

    _placers.SequentialPlacer = SequentialPlacer
    _placers.Placer = type("Placer", (), {})
    sys.modules["aie.iron.placers"] = _placers

    import aie.iron as _iron
    from aie.iron.dataflow import ObjectFifoHandle
    from aie.iron.program import Program as _Program
    from aie.iron.runtime import Runtime as _Runtime
    from aie.iron.runtime import TaskGroup as _TaskGroup
    from aie.iron.runtime import sync_parameters as _sync_parameters
    from aie.iron.worker import Worker

    _orig_resolve = _Program.resolve_program

    def _resolve_program(self, *args, **kwargs):
        args = tuple(a for a in args if not isinstance(a, SequentialPlacer))
        return _orig_resolve(self, *args, **kwargs)

    _Program.resolve_program = _resolve_program

    # Delta 2.
    def _rename_placement(fn):
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if "placement" in kwargs:
                kwargs.setdefault("tile", kwargs.pop("placement"))
            return fn(*args, **kwargs)

        return _wrapped

    for _cls, _meth in (
        (ObjectFifoHandle, "split"),
        (ObjectFifoHandle, "join"),
        (ObjectFifoHandle, "forward"),
        (Worker, "__init__"),
    ):
        setattr(_cls, _meth, _rename_placement(getattr(_cls, _meth)))

    # Delta 3. Row 0 is shim, row 1 memtile, rows >= 2 compute on this device family, but read that
    # off get_tile_type rather than assuming it: the mapping is a target-model property.
    from aie.dialects.aie import AIETileType as _AIETileType
    from aie.iron.device import Device as _Device
    from aie.iron.device.tile import Tile as _Tile

    def _tiles_of_type(self, *kinds):
        return [
            _Tile(col, row)
            for col in range(self.cols)
            for row in range(self.rows)
            if self.get_tile_type(col, row) in kinds
        ]

    def _get_shim_tiles(self):
        return _tiles_of_type(
            self, _AIETileType.ShimNOCTile, _AIETileType.ShimPLTile
        )

    def _get_mem_tiles(self):
        return _tiles_of_type(self, _AIETileType.MemTile)

    def _get_compute_tiles(self):
        return _tiles_of_type(self, _AIETileType.CoreTile)

    def _get_num_connections(self, tile, output=True):
        """DMA source/dest connections on `tile`, as the removed method reported them."""
        kind = self.get_tile_type(tile.col, tile.row)
        return 6 if kind == _AIETileType.MemTile else 2

    for _name, _fn in (
        ("get_shim_tiles", _get_shim_tiles),
        ("get_mem_tiles", _get_mem_tiles),
        ("get_compute_tiles", _get_compute_tiles),
        ("get_num_connections", _get_num_connections),
    ):
        if not hasattr(_Device, _name):
            setattr(_Device, _name, _fn)

    # Delta 4.
    class _SeqBuf:
        """Stands in for one runtime_sequence buffer while a body is being recorded."""

        __slots__ = ("index",)

        def __init__(self, index):
            self.index = index

    class _TaskGroupToken:
        """Stands in for a TaskGroup, which can only be constructed inside the callback."""

        __slots__ = ()

    class _CompatRuntime(_Runtime):
        def __init__(self, seq_fn=None, fn_args=None, **kwargs):
            self._compat = seq_fn is None
            if not self._compat:
                super().__init__(seq_fn, fn_args, **kwargs)
                return
            self._rec = []
            self._rec_types = ()
            self._rec_workers = []
            self._rec_trace = None
            self._rec_kwargs = kwargs
            self._fifos = set()  # mem_copy's design.py adds handles here directly

        @contextlib.contextmanager
        def sequence(self, *types_):
            self._rec_types = types_
            bufs = tuple(_SeqBuf(i) for i in range(len(types_)))
            yield bufs[0] if len(bufs) == 1 else bufs

        def start(self, *workers):
            self._rec_workers.extend(workers)

        def task_group(self):
            token = _TaskGroupToken()
            self._rec.append(("task_group", token))
            return token

        def finish_task_group(self, token):
            self._rec.append(("finish_task_group", token))

        def fill(self, handle, source, tap=None, **kwargs):
            self._rec.append(("fill", handle, source, tap, kwargs))

        def drain(self, handle, dest, tap=None, **kwargs):
            self._rec.append(("drain", handle, dest, tap, kwargs))

        def inline_ops(self, fn, args):
            self._rec.append(("inline_ops", fn, args))

        def set_barrier(self, barrier, value):
            self._rec.append(("set_barrier", barrier, value))

        def sync_parameters(self):
            self._rec.append(("sync_parameters",))

        def enable_trace(self, *args, **kwargs):
            self._rec_trace = (args, kwargs)

        def _compat_replay(self, *body_args):
            bufs = body_args[: len(self._rec_types)]
            groups = {}
            for entry in self._rec:
                verb = entry[0]
                if verb == "task_group":
                    groups[entry[1]] = _TaskGroup()
                elif verb == "finish_task_group":
                    groups[entry[1]].finish()
                elif verb in ("fill", "drain"):
                    _, handle, buf, tap, kwargs = entry
                    kwargs = dict(kwargs)
                    if "placement" in kwargs:
                        raise NotImplementedError(
                            f"{verb}(placement=...) selects the shim tile, which is now fixed when "
                            f"the handle is taken: port the design to .prod(tile=)/.cons(tile=)"
                        )
                    token = kwargs.pop("task_group", None)
                    if token is not None:
                        kwargs["group"] = groups[token]
                    getattr(handle, verb)(bufs[buf.index], tap, **kwargs)
                elif verb == "inline_ops":
                    entry[1](*entry[2])
                elif verb == "set_barrier":
                    entry[1].set(entry[2])
                elif verb == "sync_parameters":
                    _sync_parameters()

        def _compat_bind(self):
            """Turn the recorded body into a real Runtime. Returns the workers to start."""
            # Every handle a recorded verb touches goes in fn_args, so its shim endpoint binds
            # before the Program collects fifos to resolve — the body itself emits last.
            handles = [e[1] for e in self._rec if e[0] in ("fill", "drain")]
            handles += list(self._fifos)
            kwargs = dict(self._rec_kwargs)
            # A design that groups only some of its transfers relies on the default group taking
            # the rest, which strict mode rejects.
            kwargs.setdefault("strict_task_groups", False)
            _Runtime.__init__(
                self, self._compat_replay, [*self._rec_types, handles], **kwargs
            )
            return self._rec_workers

    class _CompatProgram(_Program):
        def __init__(self, device, rt, workers=None):
            trace = None
            if getattr(rt, "_compat", False):
                workers = list(workers or []) + rt._compat_bind()
                trace = rt._rec_trace
            super().__init__(device, rt, workers)
            if trace is not None:
                self.enable_trace(*trace[0], **trace[1])

    _iron.Runtime = _CompatRuntime
    _iron.Program = _CompatProgram

    # Delta 5. Requesting an output no longer needs a matching negation: asking only for
    # --get-npu-insts already builds nothing else, so --no-compile / --no-compile-host drop out.
    _AIECC_FLAGS = {
        "--aie-generate-xclbin": "--get-xclbin",
        "--aie-generate-npu-insts": "--get-npu-insts",
        "--emit-scratchpad-parameters": "--get-scratchpad-parameters",
        "--generate-full-elf": "--get-full-elf",
        "--no-compile": None,
        "--no-compile-host": None,
        "--no-xbridge": None,
        "--no-xchesscc": None,
    }

    import os.path as _osp

    from iron.common.compilation.base import ShellCompilationCommand as _Shell

    _orig_shell_init = _Shell.__init__

    def _shell_init(self, command, *args, **kwargs):
        if command and _osp.basename(str(command[0])).startswith("aiecc"):
            command = [
                _AIECC_FLAGS.get(a, a)
                for a in command
                if _AIECC_FLAGS.get(a, a) is not None
            ]
        return _orig_shell_init(self, command, *args, **kwargs)

    _Shell.__init__ = _shell_init
