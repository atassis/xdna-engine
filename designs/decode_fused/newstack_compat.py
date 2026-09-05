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
  2. The explicit-placement kwarg was renamed `placement=` -> `tile=` across the dataflow/runtime
     API (`ObjectFifoHandle.{split,join,forward}`, `Worker.__init__`, `Runtime.{fill,drain}`).
     amd/IRON's gemm and mha design.py still call `placement=Tile(...)`, so we rename the kwarg at
     call time on the dataflow methods that still accept a tile, and those design.py run unchanged.
  3. `Device.{get_shim_tiles,get_mem_tiles,get_compute_tiles,get_num_connections}` were removed in
     favour of `cols`/`rows`/`get_tile_type`. amd/IRON's `get_shim_dma_limit` (iron/common/utils.py)
     uses the enumeration to size the ShimDMA budget. Reconstructed from `get_tile_type`.
  4. Xilinx/mlir-aie#3387 reworked Runtime into an eager callback body: the no-arg `Runtime()` +
     `with rt.sequence(...)` spelling every amd/IRON design.py uses no longer exists, and its verbs
     moved onto other objects (fill/drain to the ObjectFifo handle, task groups to `TaskGroup`,
     barrier sets to the barrier, workers to `Program(workers=)`). The two spellings cannot coexist
     in one Runtime, so the compat class RECORDS an old-spelling body and replays it inside the
     callback the current Runtime requires. Recording is per-instance and keyed on the constructor:
     a `Runtime(seq_fn, fn_args)` call is the current spelling and passes straight through, so our
     own generators and amd/IRON's designs both run in one process. The patch list itself is applied
     by ATTRIBUTE-PRESENCE rather than pruned, because this shim spans both sides of #3387 --
     patching an absent `Runtime.fill` raised AttributeError at IMPORT and took down every
     scratchpad probe that imports this module. Per-transfer placement moved to `.prod()/.cons()`,
     and the handle's fill/drain take no `tile=`, so delta 2 has nothing to rename there.
  5. #3364 made every aiecc output an explicit request (`--get-<name>`), so the `--aie-generate-*`
     spellings amd/IRON's compilation rules emit are gone, and the host-compile negations with them
     (the host program is one more requestable output). Translated on the argv, which covers every
     rule that shells out to aiecc rather than each rule's flag list. The BACKEND negations are not
     in that family: at this pin aiecc still defaults to Chess and needs them -- see Delta 5 below,
     and note the pin guard, since upstream #3501 deletes them 38 commits later.
"""
import contextlib
import functools
import sys
import types


def _assert_pinned_aie():
    """Fail loud when the resolved `aie` package is not this lock's toolchain instance.

    Every delta in this shim is written against ONE mlir-aie commit, so which instance `import aie`
    resolves to is load-bearing -- and nothing enforced it. `iron_env.sh` exports the right
    PYTHONPATH, but a generator run directly gets whatever `.venv-iron/aie.pth` happens to say, and
    that file is only rewritten by `toolchain_wire.sh on`. Measured 2026-08-17: a `.pth` from 08-11
    resolved builds to instance 185212afd5ca = mlir-aie d91f899ea9d, 38 commits past the pinned
    62be3ea3133, where upstream #3501 has already deleted --no-xchesscc/--no-xbridge -- so the
    correct-for-the-pin argv below read as a hard aiecc error and cost a device attempt.

    Set XDNA_ALLOW_PIN_DRIFT=1 to downgrade to a warning (deliberate off-pin A/B).
    """
    import hashlib
    import importlib.util
    import os

    root = os.path.dirname(os.path.abspath(__file__))
    while root != "/" and not os.path.exists(os.path.join(root, "toolchain.lock")):
        root = os.path.dirname(root)
    lock = os.path.join(root, "toolchain.lock")
    if not os.path.exists(lock):
        return  # not in a pinned checkout; nothing to judge against
    # SEMANTIC lock hash -- comments and blank lines stripped -- because that is how
    # toolchain_up.sh names the instance dir (`_lock_semantic | sha256sum | cut -c1-12`), and the
    # dir name is what this guard matches against. It used to hash the WHOLE file with the comment
    # "keyed exactly as toolchain_up.sh does": true when written 2026-08-17, false once
    # toolchain_up.sh moved off whole-file hashing so a prose edit would stop orphaning instances.
    # The comment asserting the equality was the only thing tying the two derivations together, and
    # it kept asserting it after it stopped holding -- so the guard refused EVERY correctly-wired
    # run (wanted 18c493a81285, instance is 9da6356ac521). Do not re-inline a second derivation
    # here: toolchain_up.sh is the authority for the name, this only has to agree with it.
    with open(lock, "r", encoding="utf-8") as f:
        text = f.read()
    _semantic = "".join(
        line.split("#", 1)[0].rstrip() + "\n"
        for line in text.splitlines()
        if line.split("#", 1)[0].strip()
    )
    want = hashlib.sha256(_semantic.encode()).hexdigest()[:12]
    # toolchain_up.sh ADOPTS an instance built under the old whole-file key by symlinking it to the
    # semantic name, so a checkout wired before that change can legitimately resolve through the
    # legacy directory. Accept both names rather than failing a correct setup.
    legacy = hashlib.sha256(text.encode()).hexdigest()[:12]
    spec = importlib.util.find_spec("aie")
    got = (spec.origin or (spec.submodule_search_locations or [""])[0]) if spec else None
    if got and any(os.path.join("instances", h, "python") in got for h in (want, legacy)):
        return
    msg = (f"[newstack_compat] resolved `aie` is not the pinned instance {want}: {got or 'unresolvable'}\n"
           f"  fix: scripts/toolchain_wire.sh on   (or: source scripts/iron_env.sh)")
    if os.environ.get("XDNA_ALLOW_PIN_DRIFT") == "1":
        print(f"WARNING {msg}", file=sys.stderr)
        return
    raise RuntimeError(msg)


_assert_pinned_aie()

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

    # Present-only: post-#3387 Runtime has neither fill nor drain (delta 4). An absent name here
    # means the API moved, which is this shim's subject -- not a silent failure to paper over.
    for _cls, _meth in (
        (ObjectFifoHandle, "split"),
        (ObjectFifoHandle, "join"),
        (ObjectFifoHandle, "forward"),
        (Worker, "__init__"),
    ):
        _fn = getattr(_cls, _meth, None)
        if _fn is not None:
            setattr(_cls, _meth, _rename_placement(_fn))

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
    # --no-xchesscc / --no-xbridge are NOT in that family and must survive: they select the
    # front-end, not an output, and aiecc still defaults both to Chess (cl::init(true) on
    # `xchesscc`/`xbridge`, CommandLineOptions.h). Dropping them sent every core down the chess
    # path, which dies at chess-llvm-link on a box with no aietools.
    _AIECC_FLAGS = {
        "--aie-generate-xclbin": "--get-xclbin",
        "--aie-generate-npu-insts": "--get-npu-insts",
        "--emit-scratchpad-parameters": "--get-scratchpad-parameters",
        "--generate-full-elf": "--get-full-elf",
        "--no-compile": None,
        "--no-compile-host": None,
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
