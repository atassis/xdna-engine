# SPDX-License-Identifier: Apache-2.0
"""Pure-Python `ParameterScratchpad`, for instances built without the pybind module.

`aie.utils.hostruntime.xrtruntime.parameter_scratchpad` imports
`aie._mlir_libs._parameter_scratchpad`, which the pinned instance does not build
(`toolchain_up.sh` passes `AIE_ENABLE_XRT_PYTHON_BINDINGS=OFF` beside
`CMAKE_DISABLE_FIND_PACKAGE_XRT=ON`). The XRT half is NOT missing: `.venv-iron`'s pyxrt is our
own build and does expose `run.get_ctrl_scratchpad_bo` (scripts/build_pyxrt_py314.sh). All the
C++ module contributes is a params.txt parser plus a word write, so it is reimplemented here
rather than paid for with a toolchain rebuild.

Semantics tracked verbatim from `runtime_lib/test_lib/parameter_scratchpad.h`:
core-kind values are left-shifted by 2 (the firmware UPDATE_REG Incr mode masks the low bits and
the core shifts back), addr-kind values are written raw, and the scratchpad is zeroed to
num_params words at construction.

    sp = ParameterScratchpad(run, params_path)
    sp.write("sm_mask", 42); sp.sync(); run.start()
"""

import struct
from pathlib import Path

import pyxrt

__all__ = ["ParameterScratchpad", "get_parameter_scratchpad"]


def _to_bytes(value) -> bytes:
    if isinstance(value, bool):
        raise TypeError("bool is not a scratchpad parameter type")
    if isinstance(value, int):
        if not -0x80000000 <= value <= 0xFFFFFFFF:
            raise ValueError(f"parameter value {value} does not fit in 32 bits")
        return value.to_bytes(4, "little", signed=value < 0)
    if isinstance(value, float):
        return struct.pack("<f", value)
    if hasattr(value, "tobytes"):
        return value.tobytes()
    raise TypeError(f"unsupported parameter type: {type(value)}")


def _parse_params(path):
    """params.txt: a count line, then `<name> <state_table_idx> <type> <kind>` per parameter."""
    fields = Path(path).read_text().split()
    num_params, rest = int(fields[0]), fields[1:]
    if len(rest) < 4 * num_params:
        raise ValueError(f"{path}: declares {num_params} parameters, has {len(rest) // 4}")
    index, core = {}, set()
    for name, idx, _type, kind in zip(*[iter(rest[: 4 * num_params])] * 4):
        idx = int(idx)
        if idx > 255:
            raise ValueError(f"{path}: state_table_idx {idx} for '{name}' exceeds uint8_t range")
        if name in index:
            raise ValueError(f"{path}: duplicate parameter name '{name}'")
        if kind not in ("core", "addr"):
            raise ValueError(f"{path}: invalid kind '{kind}' for parameter '{name}'")
        index[name] = idx
        if kind == "core":
            core.add(name)
    return index, core


class ParameterScratchpad:
    """Write named runtime parameters into the control scratchpad of an `xrt::run`."""

    def __init__(self, run, params_path):
        self._index, self._core = _parse_params(params_path)
        self._size = 4 * len(self._index)
        self._bo = run.get_ctrl_scratchpad_bo()
        if self._bo.size() < self._size:
            raise RuntimeError(
                f"ctrl scratchpad BO is {self._bo.size()} bytes, parameters need {self._size}"
            )
        self._mv = self._bo.map().cast("B")
        self._mv[: self._size] = b"\x00" * self._size

    def write(self, name, value) -> None:
        try:
            idx = self._index[name]
        except KeyError:
            raise KeyError(f"unknown parameter '{name}'; declared: {sorted(self._index)}") from None
        bits = int.from_bytes(_to_bytes(value)[:4].ljust(4, b"\x00"), "little")
        if name in self._core:
            bits = (bits << 2) & 0xFFFFFFFF
        self._mv[4 * idx : 4 * idx + 4] = bits.to_bytes(4, "little")

    def read(self, name) -> int:
        """The value as it sits in the scratchpad -- core-kind reads back shifted."""
        idx = self._index[name]
        return int.from_bytes(self._mv[4 * idx : 4 * idx + 4], "little")

    def sync(self) -> None:
        self._bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)


def get_parameter_scratchpad():
    """Prefer the upstream class; fall back here when the instance lacks the pybind module."""
    try:
        from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
            ParameterScratchpad as Upstream,
        )
    except ImportError:
        return ParameterScratchpad
    return Upstream
