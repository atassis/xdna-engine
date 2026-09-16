#!/usr/bin/env bash
# Bounded-concurrency builder for a whole_array kernel FAMILY. Makefile.modal /
# Makefile.modal.int8 / Makefile.silu key their kernel object by TILE DIMS only
# (build/mm_${m}x${k}x${n}.o, build/mm_silu_epilogue_${m}x${k}x${n}.o -- see
# design_override.mk / Makefile.modal), so every N in one FFN-width loop shares
# one object. Building them concurrently against the shared build/ dir races N
# `make`s on that one object path and on the FORCE-driven *_defines.stamp files.
#
# Fix: give each design an ISOLATED COPY of the whole_array dir (so nothing is
# shared to race on), build the copies concurrently, then copy each copy's
# final_*.xclbin / insts_* back into the caller's build/ -- the same place a
# serial build leaves them, so publish_kernels.sh and
# check_kernel_artifact_freshness.sh see no difference.
#
# Measured 2026-09-16 (task perf/kernel-build-dirs), cold, whole_array Makefile.silu
# FUSION family (4 designs) and Makefile.modal K=800 fast-tile family (7 designs):
# 1.3-2.5x over today's serial build depending on box load; gated byte-identical
# (insts_*.txt + all per-core ELFs) against a serial build, xclbin diffs (65-75
# bytes) within the same band a serial-vs-serial same-arm control shows (bootgen
# UUID/header nondeterminism, not a regression).
set -uo pipefail

# run_concurrent_capped <max_concurrent> <cmd...> -- each <cmd> a full shell string.
run_concurrent_capped() {
  local max="$1"; shift
  local -a cmds=("$@")
  local n=${#cmds[@]} i=0 failed=0
  while [ "$i" -lt "$n" ]; do
    local -a pids=()
    local j=0
    while [ "$j" -lt "$max" ] && [ "$i" -lt "$n" ]; do
      bash -c "${cmds[$i]}" & pids+=("$!")
      i=$((i+1)); j=$((j+1))
    done
    for p in "${pids[@]}"; do wait "$p" || failed=1; done
  done
  return "$failed"
}

# build_family_concurrent <mmw_dir> <max_concurrent> <spec>...
#   spec = "MAKEFILE|VARS|TARGET", e.g.
#   "Makefile.silu|M=512 K=800 N=3072 n_aie_cols=8|build/final_512x800x3072_32x32x32_8c_silu.xclbin"
# All specs in one call MUST share the same kernel object identity (tile dims +
# every define that feeds MM_DEFINES/EPI_DEFINES) -- that is what makes building
# them in parallel safe. Different calls (different families) still run in the
# script's normal sequential order.
#
# Each design's OWN aiecc runs single-threaded (AIECC_JOBS=1): with <max_concurrent>
# designs already sharing the core budget, giving each one aiecc's own internal -j
# on top oversubscribes the box and measured net SLOWER than plain serial (aiecc's
# own single-threaded floor dominates a 32-core design regardless -- see the perf
# task). KERNEL_BUILD_AIECC_JOBS overrides, for a box with cores to spare.
build_family_concurrent() {
  local mmw="$1" max="$2"; shift 2
  local -a specs=("$@")
  local n=${#specs[@]}
  [ "$n" -gt 0 ] || return 0
  mkdir -p "$mmw/build"
  local -a copies=() cmds=()
  local i=0
  for spec in "${specs[@]}"; do
    i=$((i+1))
    local makefile vars target copy
    IFS='|' read -r makefile vars target <<<"$spec"
    copy="${mmw}.par${i}"
    rm -rf "$copy"
    # Makefiles + generators only -- never build* (that is the whole point of the copy).
    rsync -a --exclude='build*' --exclude='__pycache__' --exclude='_build' "$mmw/" "$copy/"
    copies+=("$copy")
    cmds+=("AIECC_JOBS=${KERNEL_BUILD_AIECC_JOBS:-1} make -C '$copy' -f '$makefile' NPU2=1 $vars '$target'")
  done
  local rc=0
  run_concurrent_capped "$max" "${cmds[@]}" || rc=1
  for copy in "${copies[@]}"; do
    cp "$copy"/build/final_*.xclbin "$copy"/build/insts_*.* "$mmw/build/" 2>/dev/null || true
    rm -rf "$copy"
  done
  return "$rc"
}
