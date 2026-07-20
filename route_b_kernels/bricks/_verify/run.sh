#!/usr/bin/env bash
# Run a brick device-verify harness under the NPU lock with the wired IRON toolchain.
# Usage: ./run.sh verify_rmsnorm.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
LOCK="$WS/xdna-engine-private/journal/scripts/npu_lock.sh"
PROG="${1:?usage: run.sh <verify_xxx.py>}"
export NPU_WAIT_S="${NPU_WAIT_S:-600}"
exec "$LOCK" queue -- bash -c "
  set -euo pipefail
  source '$WS/wt-conveyor-dev/scripts/iron_env.sh' >/dev/null 2>&1
  cd '$HERE'
  exec python -u '$PROG'
"
