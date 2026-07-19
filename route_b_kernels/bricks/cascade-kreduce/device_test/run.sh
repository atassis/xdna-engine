#!/usr/bin/env bash
# Run the cascade-accessor device test on the NPU. Needs the device FREE
# (single-tenant): stop the decode daemon first if it is holding /dev/accel:
#   systemctl --user stop npu-serve.service      # (then restart it after)
# Announce + check nothing else holds the device:  fuser -v /dev/accel/accel0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[ -f cascade.xclbin ] && [ -f insts.bin ] && [ -f test.exe ] || { echo "build first: ./build.sh"; exit 1; }
./test.exe -x cascade.xclbin -k MLIR_AIE -i insts.bin
# PASS prints "PASS!" (host checks buf[5]==214, i.e. 14 -> +100 -> +100 across
# two cascade hops through aie::cascade_out / aie::cascade_in_i32).
