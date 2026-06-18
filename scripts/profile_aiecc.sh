#!/usr/bin/env bash
# =============================================================================================
# Profile the running aiecc final-ELF phase to find why it's slow. NEEDS sudo (ptrace_scope=1 means
# only root can attach gdb to a non-child process). Non-destructive: attaches, samples stacks, detaches
# (each sample briefly pauses the build for ~1-2s, 8 samples total — negligible vs a 20-min phase).
#
#   sudo bash $REPO/scripts/profile_aiecc.sh
#
# Auto-targets the aiecc process with the most accumulated CPU time (the one deepest in the serial
# full-ELF assembly). aiecc is a C++ binary (NOT the Python aiecc.py), and the real work runs in the
# llvm-worker-* threads under the async pass adaptor — the main thread just parks in the thread pool.
# So we dump NATIVE backtraces for ALL threads (the on-CPU worker frame is the hot function). Samples
# 8x over ~25s so the hot frame shows up consistently. Writes artifacts/aiecc_profile_<ts>.log.
# =============================================================================================
set -u
WT="$REPO"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/aiecc_profile_${TS}.log"
: > "$LOG"; log(){ echo -e "$*" | tee -a "$LOG"; }

command -v gdb >/dev/null || { echo "FATAL: gdb not installed"; exit 1; }

# pick the aiecc PID with the largest cumulative CPU time (= deepest in the serial phase).
# Match the C++ aiecc binary by exact process name (covers both the wheel `mlir_aie/bin/aiecc`
# and our isolated `mlir-aie/build-on2/bin/aiecc`); the Python wrapper is `aiecc.py`, not `aiecc`.
PID="$(for p in $(pgrep -x aiecc 2>/dev/null); do
         ct=$(ps -o cputimes= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$ct" ] && echo "$ct $p"
       done | sort -rn | head -1 | awk '{print $2}')"
[ -z "${PID:-}" ] && { log "FATAL: no running 'mlir_aie/bin/aiecc' process found"; exit 1; }

WD="$(ps -o args= -p "$PID" 2>/dev/null | grep -oE '/tmp/tmp[A-Za-z0-9.]*' | head -1)"
log "================ aiecc profile  $TS ================"
log "target PID=$PID  workdir=$WD"
log "cmd: $(ps -o args= -p "$PID" 2>/dev/null | tr -s ' ' | cut -c1-200)"
log "cputime=$(ps -o cputime= -p "$PID" 2>/dev/null|tr -d ' ')  %cpu=$(ps -o %cpu= -p "$PID" 2>/dev/null|tr -d ' ')  threads=$(ls /proc/$PID/task 2>/dev/null|wc -l)"
log "ptrace_scope=$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)  (running as uid $(id -u))"

for i in $(seq 1 8); do
  # which thread(s) are on-CPU (state R) at this instant
  RT="$(for t in /proc/$PID/task/*; do s=$(awk '{print $3}' "$t/stat" 2>/dev/null); [ "$s" = "R" ] && basename "$t"; done | tr '\n' ' ')"
  log "\n================= SAMPLE $i  $(date +%H:%M:%S)  running-TIDs=[$RT] ================="
  timeout 90 gdb -p "$PID" -batch \
    -ex "set pagination off" \
    -ex "set print frame-arguments none" \
    -ex "echo \n--- NATIVE STACKS (all threads) ---\n" \
    -ex "thread apply all bt 40" \
    >> "$LOG" 2>&1
  echo "[sample $i done]"
  sleep 2
done

log "\n[done] full log: $LOG"
echo "DONE — give the agent this path: $LOG"
