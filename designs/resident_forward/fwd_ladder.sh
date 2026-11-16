#!/usr/bin/env bash
# fwd_ladder.sh <name> "<rlayer_design flags>" <emit list>...: build one image in parts, each part
# `emit=` one list (a comma list of sequence names), one at a time under the build scope, then pack
# them into <name> (fwd_pack.py). A forward ladder lowered as one design needs more than the 6G
# build cap (223 MB of MLIR swapped past 12 GB); a part of one or two forward sequences stays near 5 GB.
#
# Parts run serially inside the caller's cgroup so its aggregate memory/swap/CPU limits
# also cover compilation and packing. The caller must provide the approved resource scope;
# this script does not create one. RF_PEAK_KB records per-part RSS from
# g4-resident-rf48c-repro-build-wall (2026-09-30), not aggregate cgroup memory.
# Admission also reserves 4 GB for the desktop; unknown emit lists assume 5.5 GB.
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(cd "$HERE/../.." && pwd)
set -euo pipefail
max_jobs=$(nproc)
max_jobs=$((max_jobs / 2))
(( max_jobs >= 1 )) || max_jobs=1
(( max_jobs <= 2 )) || max_jobs=2
if [[ ${RF_LADDER_JOBS:-1} != 1 ]] ||
   [[ ! ${AIECC_JOBS:-$max_jobs} =~ ^[12]$ ]] ||
   (( ${AIECC_JOBS:-$max_jobs} > max_jobs )); then
  echo "RF resource policy requires RF_LADDER_JOBS=1 and AIECC_JOBS=1..$max_jobs" >&2
  exit 2
fi
export AIECC_JOBS=${AIECC_JOBS:-$max_jobs}
export OMP_NUM_THREADS=$AIECC_JOBS OPENBLAS_NUM_THREADS=$AIECC_JOBS MKL_NUM_THREADS=$AIECC_JOBS
source "$HERE/env.sh"
export RF_BUILD=${RF_BUILD:-${XDNA_DATA:-$REPO}/build/resident_forward}
name=$1; flags=$2; shift 2
export RF_KOBJ_RUN="run-$(date +%s%N)-$$"
if [[ -n ${KEEP_WORK:-} ]]; then
  echo "[fwd] retaining fresh cache $RF_BUILD/kobj/$RF_KOBJ_RUN"
else
  trap 'rm -rf "$RF_BUILD/kobj/$RF_KOBJ_RUN"' EXIT
fi
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

for ((i=0; i<total; i++)); do
  part=${parts[$i]}
  emit=${emits[$i]}
  expected=${RF_PEAK_KB[$emit]:-$DEFAULT_PEAK_KB}
  while [ $(( $(mem_available_kb) - RESERVE_KB )) -lt "$expected" ]; do
    sleep 5
  done
  mkdir -p "$RF_BUILD/$part"
  if ( cd "$RF_BUILD/$part" &&
    RF_ATTN_H=1 RF_FAST=1 RF_TEXT_MARGIN=128 nice -n 10 /usr/bin/time -v \
      "$PY" "$HERE/rf_build.py" "$part" rlayer_design "emit=$emit" $flags > build.log 2>&1 ); then
    echo "$part emit=$emit rc=0 log=$RF_BUILD/$part/build.log"
  else
    rc=$?
    echo "$part failed (rc=$rc), log: $RF_BUILD/$part/build.log" >&2
    exit "$rc"
  fi
done

"$PY" "$HERE/fwd_pack.py" "$name" "${parts[@]}"
