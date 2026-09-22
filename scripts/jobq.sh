#!/usr/bin/env bash
# Run one memory-hungry job under a HARD memory cap and a shared slot queue.
#
#   scripts/jobq.sh [--mem 6G] [--slots 2] [--class build] [--wait 7200] -- <cmd> [args...]
#
# Two independent problems, because they have different fixes:
#
#   CAP -- a systemd user scope with MemoryMax and no swap. The kernel's global OOM killer picks
#   its victim by badness score, not by who caused the pressure: a job that over-allocates takes
#   down whatever else is resident, and on 2026-09-11 that was a device bench holding three model
#   arenas plus five unrelated sessions. A cgroup cap kills only the offending job, and the exit
#   status says so instead of leaving a silent gap.
#
#   QUEUE -- N flock slots shared across every checkout, so concurrent agents do not all compile
#   or all load weights at the same moment. The lock lives under the build cache, which every
#   worktree resolves to the same directory (scripts/cache_env.sh), NOT under the repo.
#
# Sizing: an aiecc run on a large fused module has been measured at 11.5 GB, and a Gemma-4 decode
# arena is 15.3 GB at full depth (3.35 GB at 6 of 48 layers, because the bf16 lm-head is 1.88 GB
# of it). A weight tensor read with mmap costs page cache, which is reclaimable; read with
# np.load() it costs anonymous memory, which is not. Prefer mmap and size the cap for the build.
set -euo pipefail
MEM=6G
SLOTS=2
CLASS=build
WAIT_S=7200
while [ $# -gt 0 ]; do
  case "$1" in
    --mem)   MEM="$2"; shift 2 ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --class) CLASS="$2"; shift 2 ;;
    --wait)  WAIT_S="$2"; shift 2 ;;
    --)      shift; break ;;
    *) echo "jobq.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ $# -ge 1 ] || { echo "usage: jobq.sh [--mem 6G] [--slots N] [--class C] -- <cmd...>" >&2; exit 2; }

. "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"
QDIR="$XDNA_CACHE/jobq"; mkdir -p "$QDIR"

# Take the first free slot; wait on slot 1 only if every slot is busy, so the wait is FIFO-ish
# rather than a thundering herd on one file.
slot=""
for i in $(seq 1 "$SLOTS"); do
  exec {fd}>"$QDIR/$CLASS.$i"
  if flock -n "$fd"; then slot="$i"; break; fi
  eval "exec $fd>&-"
done
if [ -z "$slot" ]; then
  echo "[jobq] $CLASS: all $SLOTS slots busy, waiting up to ${WAIT_S}s" >&2
  exec {fd}>"$QDIR/$CLASS.1"
  flock -w "$WAIT_S" "$fd" || { echo "[jobq] timed out waiting for a $CLASS slot" >&2; exit 75; }
  slot=1
fi
echo "[jobq] $CLASS slot $slot/$SLOTS, MemoryMax=$MEM: $*" >&2

rc=0
systemd-run --user --scope --quiet --collect \
  -p MemoryMax="$MEM" -p MemorySwapMax=0 -- "$@" || rc=$?
# 137 is SIGKILL, which under this scope means the cap -- the one failure an agent would otherwise
# read as a crash in its own code.
[ "$rc" = 137 ] && echo "[jobq] KILLED at MemoryMax=$MEM -- raise --mem or reduce the job, this is not a bug in the command" >&2
exit "$rc"
