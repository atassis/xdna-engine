"""The compiler's bounded, per-command scratchpad parameter file."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScratchpadParams:
    slot_count: int
    parameters: dict[str, tuple[int | None, str, str, int | None, int | None]]
    joint_bounds: list[tuple[str, str, int, int]]
    commands: dict[str, tuple[str, ...]]


def read_params(path: Path) -> ScratchpadParams:
    rows = iter(Path(path).read_text().splitlines())

    def take(width=None):
        try:
            fields = next(rows).split()
        except StopIteration:
            raise ValueError("truncated scratchpad parameters") from None
        if not fields or (width is not None and len(fields) != width):
            raise ValueError("malformed scratchpad parameter row")
        return fields

    def count():
        value = int(take(1)[0])
        if value < 0:
            raise ValueError("negative scratchpad count")
        return value

    number, slots = map(int, take(2))
    if number < 0 or slots < 0:
        raise ValueError("negative parameter or slot count")
    parameters = {}
    for _ in range(number):
        name, slot, dtype, kind, low, high = take(6)
        slot = None if slot == "-" else int(slot)
        if name in parameters or (slot is not None and not 0 <= slot < slots):
            raise ValueError("duplicate parameter or invalid slot")
        if kind not in {"addr", "core", "repeat"} or (low == "-") != (high == "-"):
            raise ValueError("invalid parameter kind or range")
        low, high = (None, None) if low == "-" else (int(low), int(high))
        if low is not None and low > high:
            raise ValueError("empty parameter range")
        parameters[name] = (slot, dtype, kind, low, high)
    joint_bounds = []
    for _ in range(count()):
        length, offset, step, maximum = take(4)
        step, maximum = int(step), int(maximum)
        if step <= 0 or any(name not in parameters or parameters[name][3] is None
                            for name in (length, offset)):
            raise ValueError("invalid joint parameter bound")
        joint_bounds.append((length, offset, step, maximum))
    commands = {}
    for _ in range(count()):
        name, reads, *names = take()
        if (int(reads) != len(names) or len(set(names)) != len(names) or name in commands
                or any(n not in parameters or parameters[n][0] is None for n in names)):
            raise ValueError("invalid command parameter bindings")
        commands[name] = tuple(names)
    if any(row.strip() for row in rows):
        raise ValueError("trailing scratchpad parameter data")
    return ScratchpadParams(slots, parameters, joint_bounds, commands)


def merge_params(parts: list[ScratchpadParams]) -> ScratchpadParams:
    if not parts:
        raise ValueError("no scratchpad parameter parts")
    first = parts[0]
    merged = ScratchpadParams(first.slot_count, dict(first.parameters), [], {})
    for part in parts:
        if part.slot_count != first.slot_count or part.parameters.keys() != first.parameters.keys():
            raise ValueError("incompatible parameter names or slot count")
        for name, value in part.parameters.items():
            slot, dtype, kind, low, high = value
            before = merged.parameters[name]
            if before[:3] != value[:3]:
                raise ValueError(f"incompatible parameter identity: {name}")
            # A packed scratchpad must satisfy every part's transfer bounds.
            lows = [v for v in (before[3], low) if v is not None]
            highs = [v for v in (before[4], high) if v is not None]
            low, high = (max(lows), min(highs)) if lows else (None, None)
            if low is not None and low > high:
                raise ValueError(f"disjoint parameter ranges: {name}")
            merged.parameters[name] = (slot, dtype, kind, low, high)
        for bound in part.joint_bounds:
            if bound not in merged.joint_bounds:
                merged.joint_bounds.append(bound)
        for name, reads in part.commands.items():
            if name in merged.commands and merged.commands[name] != reads:
                raise ValueError(f"incompatible command parameter bindings: {name}")
            merged.commands[name] = reads
    return merged


def write_params(path: Path, params: ScratchpadParams) -> None:
    def field(value):
        return "-" if value is None else str(value)

    rows = [f"{len(params.parameters)} {params.slot_count}"]
    rows.extend(" ".join([name, *(field(v) for v in values)])
                for name, values in params.parameters.items())
    rows.append(str(len(params.joint_bounds)))
    rows.extend(" ".join(map(str, bound)) for bound in params.joint_bounds)
    rows.append(str(len(params.commands)))
    rows.extend(" ".join([name, str(len(reads)), *reads]) for name, reads in params.commands.items())
    Path(path).write_text("\n".join(rows) + "\n")
