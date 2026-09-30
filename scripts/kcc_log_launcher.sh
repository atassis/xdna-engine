#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AIE_KERNEL_COMPILER_LAUNCHER shim for fleet enumeration: append one JSON line per kernel
# compile to $KCC_LOG (with wall start/end), then run the compile through $KCC_NEXT (e.g. ccache)
# or directly. argv: <compiler> <source> <args...>, the order cxx_core_compile_command() produces.
# Not exec'd: the compile must be waited on so its wall time can be logged after it returns.
set -uo pipefail
log="${KCC_LOG:?KCC_LOG must name the log file}"
start=$(date +%s.%N)
if [ -n "${KCC_NEXT:-}" ]; then "$KCC_NEXT" "$@"; else "$@"; fi
rc=$?
end=$(date +%s.%N)
python3 - "$rc" "$start" "$end" "$log" "$@" <<'PY'
import hashlib, json, sys
rc, start, end, log, cc, *rest = sys.argv[1:]
src = next(a for a in rest if a.endswith((".cc", ".cpp", ".c")))
keep, skip = [], False
for a in rest:
    if skip:
        skip = False
        continue
    if a in ("-o", "-MF", "-include-pch"):
        skip = True
        continue
    if a != src:
        keep.append(a)
with open(src, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()
with open(log, "a") as f:
    f.write(json.dumps({"compiler": cc, "source": src, "source_sha256": h, "args": keep,
                         "rc": int(rc), "start": float(start), "end": float(end)}) + "\n")
PY
exit "$rc"
