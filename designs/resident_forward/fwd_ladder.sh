#!/usr/bin/env bash
# fwd_ladder.sh <name> "<rlayer_design flags>" <emit list>...: build one image in parts, each part
# `emit=` one list (a comma list of sequence names), one at a time under the build scope, then pack
# them into <name> (fwd_pack.py). A forward ladder lowered as one design needs more than the 6G
# build cap (223 MB of MLIR swapped past 12 GB); a part of one or two forward sequences stays near 5 GB.
#
# Parts are admitted by MEMORY, not by whether some other session's aiecc/cargo happens to be
# running (that pgrep loop blocked on unrelated work on a shared box). RF_LADDER_JOBS caps
# concurrency (default 3; =1 reproduces the old fully-serial behaviour). RF_PEAK_KB below is an
# initial table of measured per-part peak RSS keyed by the part's own emit list
# (g4-resident-rf48c-repro-build-wall, 2026-09-30); an emit list not in the table assumes 5.5 GB.
# A part is admitted only once MemAvailable minus a 4 GB desktop reserve covers its expected peak;
# each part still carries its own systemd-run MemoryMax=6G cap as the hard backstop.
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
export RF_BUILD=${RF_BUILD:-${XDNA_DATA:-$REPO}/build/resident_forward}
name=$1; flags=$2; shift 2

RF_LADDER_JOBS=${RF_LADDER_JOBS:-3}
export RF_KOBJ_RUN="run-$(date +%s%N)-$$"
trap 'rm -rf "$RF_BUILD/kobj/$RF_KOBJ_RUN"' EXIT
RESERVE_KB=$((4 * 1024 * 1024))
DEFAULT_PEAK_KB=$((5632 * 1024))  # 5.5 GB, unknown emit list

declare -A RF_PEAK_KB=(
  ["p1,p2,g1w64,g1w256,g1w1024,g1w4096,g2w64,g2w256,g2w1024,g2w4096,g1s128,g1s256,g1s512,g1s1024,g1s2048,g1s4096,h1,f1"]=$((3774874))  # 3.6 GB, p0
  ["f2,f1s256"]=$((3879731))                                                                                                            # 3.7 GB, p1
  ["f1s512,f1s1024"]=$((4718592))                                                                                                        # 4.5 GB, p2
  ["f1s2048,f1s4096"]=$((5347738))                                                                                                        # 5.1 GB, p3
  ["f2w256"]=$((2516582))                                                                                                                 # 2.4 GB, p4
  ["f2w1024"]=$((3145728))                                                                                                                 # 3.0 GB, p5
)

emits=("$@")
parts=()
i=0
for emit in "${emits[@]}"; do
  parts+=("${name}_p$i")
  i=$((i + 1))
done
total=${#parts[@]}

mem_available_kb() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }

declare -A PID_PART PID_EMIT
running=0
next=0
fail_rc=0
fail_msg=""

launch() {
  local part=$1 emit=$2
  mkdir -p $RF_BUILD/$part
  ( cd $RF_BUILD/$part &&
    RF_BUILD=$RF_BUILD RF_ATTN_H=1 RF_FAST=1 RF_TEXT_MARGIN=128 AIECC_JOBS=8 systemd-run --user --scope -q -p CPUWeight=20 \
      -p MemoryMax=6G -p MemorySwapMax=1G nice -n 10 /usr/bin/time -v $PY $HERE/rf_build.py $part rlayer_design \
      emit=$emit $flags > build.log 2>&1 ) &
  PID_PART[$!]=$part
  PID_EMIT[$!]=$emit
  running=$((running + 1))
}

# Non-blocking: reports every tracked pid that has already exited (wait on a dead pid returns
# immediately) and updates $fail_rc/$fail_msg on the first failure seen.
reap() {
  local pid dead=()
  for pid in "${!PID_PART[@]}"; do
    kill -0 "$pid" 2>/dev/null || dead+=("$pid")
  done
  for pid in "${dead[@]}"; do
    wait "$pid"; rc=$?
    local part=${PID_PART[$pid]} emit=${PID_EMIT[$pid]}
    local log=$RF_BUILD/$part/build.log
    local peak=$(grep "Maximum resident" "$log" | awk '{print $NF}')
    local wall=$(grep "Elapsed" "$log" | awk '{print $NF}')
    echo "$part emit=$emit rc=$rc peak_kB=$peak wall=$wall"
    unset "PID_PART[$pid]" "PID_EMIT[$pid]"
    running=$((running - 1))
    if [ $rc -ne 0 ] && [ $fail_rc -eq 0 ]; then
      fail_rc=$rc; fail_msg="$part failed (rc=$rc), log: $log"
    fi
  done
}

while [ $next -lt $total ] || [ $running -gt 0 ]; do
  reap
  if [ $fail_rc -ne 0 ]; then
    while [ $running -gt 0 ]; do sleep 5; reap; done
    echo "$fail_msg" >&2
    exit $fail_rc
  fi
  if [ $next -lt $total ] && [ $running -lt $RF_LADDER_JOBS ]; then
    emit=${emits[$next]}
    expected=${RF_PEAK_KB[$emit]:-$DEFAULT_PEAK_KB}
    avail=$(mem_available_kb)
    if [ $((avail - RESERVE_KB)) -ge "$expected" ]; then
      launch "${parts[$next]}" "$emit"
      next=$((next + 1))
      continue
    fi
  fi
  sleep 5
done

$PY $HERE/fwd_pack.py $name "${parts[@]}"
