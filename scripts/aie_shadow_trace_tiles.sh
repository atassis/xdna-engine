#!/usr/bin/env bash
# Materialise a shadow `aie` package that can trace a NON-worker tile (a MemTile), and print the
# directory to put on PYTHONPATH.
#
# WHY A SHADOW: IRON's enable_trace() builds its tile list from Workers only
# (program.py: `for w in trace_workers: tiles_to_trace.append(w.tile.op)`), so a MemTile -- which has
# no Worker -- is unreachable by the public API even though aie.utils.trace.setup already knows how
# to configure one (it dispatches on tile_op.is_mem_tile() and ships a default memtile event set).
# The fix is five lines, but it lands in the BUILT INSTANCE, and rebuilding the pinned toolchain to
# test a measurement flag is the wrong trade. cp -as gives a symlink tree of the instance; only the
# one edited file is a real copy. See memory `mlir-aie-python-tests-shadow-package`.
#
# The same change is carried in the fork checkout (mlir-aie/python/iron/), which is its durable home
# and the upstreamable form; this script exists so the measurement reproduces against the CURRENT
# toolchain.lock pin without a rebuild. Delete it once a pin ships the change.
#
# Usage:  export PYTHONPATH="$(bash scripts/aie_shadow_trace_tiles.sh):$PYTHONPATH"
set -euo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADOW="${AIE_SHADOW_DIR:-/tmp/aie-shadow-trace-tiles}"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { echo "iron_env did not set MLIR_AIE_INSTANCE" >&2; exit 1; }

SRC="$MLIR_AIE_INSTANCE/python/aie"
rm -rf "$SHADOW"; mkdir -p "$SHADOW"
cp -as "$SRC" "$SHADOW/aie"
rm "$SHADOW/aie/iron/program.py"
cp "$SRC/iron/program.py" "$SHADOW/aie/iron/program.py"
chmod u+w "$SHADOW/aie/iron/program.py"

python3 - "$SHADOW/aie/iron/program.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
edits = [
    ("        core_trace_mode=TraceMode.EventTime,\n    ):",
     "        core_trace_mode=TraceMode.EventTime,\n        trace_tiles: list | None = None,\n    ):"),
    ("        self._trace_workers = workers\n",
     "        self._trace_workers = workers\n        self._trace_tiles = trace_tiles\n"),
    ("        self._trace_workers = None\n",
     "        self._trace_workers = None\n        self._trace_tiles = None\n"),
    ("                if self._trace_size is not None and self._trace_size > 0:",
     "                if self._trace_tiles:\n"
     "                    for t in self._trace_tiles:\n"
     "                        tiles_to_trace.append(t.op)\n"
     "                if self._trace_size is not None and self._trace_size > 0:"),
]
for old, new in edits:
    # the two _trace_workers lines are distinct strings (assignment vs init); each must hit once
    if s.count(old) != 1:
        sys.exit(f"shadow patch: expected exactly 1 match for {old!r}, got {s.count(old)}")
    s = s.replace(old, new)
open(p, "w").write(s)
PY

echo "$SHADOW"
