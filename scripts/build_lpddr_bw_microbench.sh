#!/usr/bin/env bash
# Build the pure-DMA LPDDR bandwidth microbench xclbins for a transfer-size sweep.
# CPU-only (no NPU) BUT it invokes aiecc against the SHARED mlir-aie/amd-IRON toolchain
# -> serialize / run in a coordinated window (see [[shared-iron-checkout-hazard]]).
#
# Outputs (consumed by scripts/lpddr_bw_microbench_harness.py):
#   <OUTDIR>/lpddr_{mode}_c{cols}_{bytes}.xclbin
#   <OUTDIR>/lpddr_{mode}_c{cols}_{bytes}.insts.bin
# OUTDIR defaults to artifacts/parakeet/lpddr_bw (gitignored, in the MAIN checkout).
#
# TOOLCHAIN: built ONLY against the blessed FORK instance (atassis/mlir-aie via toolchain.lock
# + toolchain_up.sh) -- the place-tiles model the generator targets. NEVER the stale wheel
# python (old Python-placer model; not supported). aiecc + the aie python both come from
# the same fork instance ($INST/bin/aiecc, $INST/python), mirroring scripts/toolchain_smoke.sh.
# If toolchain_up.sh cannot produce a complete instance (import aie.iron must succeed), this
# script FAILS LOUD -- it does not silently fall back to the wheel. Run scripts/toolchain_smoke.sh
# first to confirm the toolchain is green before spending an NPU window.
#
# Usage:
#   scripts/build_lpddr_bw_microbench.sh                       # default rdwr+read+write, cols=1
#   MODES="rdwr" COLS="1 8" scripts/build_lpddr_bw_microbench.sh
#   SWEEP_BYTES="65536 1048576 67108864" scripts/build_lpddr_bw_microbench.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"                 # worktree (holds the generator + this script)
REPO="$(cd "$(git -C "$SRC" rev-parse --git-common-dir)/.." && pwd)"   # MAIN toolchain root
cd "$REPO"
set -a; . "$REPO/toolchain.lock"; set +a
INST="$("$REPO/scripts/toolchain_up.sh")"               # build/locate the fork instance (current lock)
export PEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
export PATH="$REPO/.venv-iron/bin:$PATH"                # venv python (ml_dtypes etc.)
export PYTHONPATH="$INST/python:${PYTHONPATH:-}"        # `aie` resolves to the FORK instance
AIECC="$INST/bin/aiecc"

# Fail loud if the instance is incomplete (the toolchain-version saga symptom) -- never the wheel.
"$REPO/.venv-iron/bin/python" -c "import aie.iron" 2>/dev/null || {
  echo "FATAL: 'import aie.iron' fails against the fork instance $INST." >&2
  echo "  The blessed toolchain is not green. Run scripts/toolchain_smoke.sh / coordinate the" >&2
  echo "  toolchain.lock instance rebuild. Refusing to fall back to the stale wheel." >&2
  exit 1
}

GEN="$SRC/route_b_kernels/lpddr_bw/lpddr_bw_microbench.py"
OUTDIR="${OUTDIR:-$REPO/artifacts/parakeet/lpddr_bw}"; mkdir -p "$OUTDIR"
WORK="$OUTDIR/build"; mkdir -p "$WORK"

MODES="${MODES:-rdwr read write}"
COLS="${COLS:-1}"
# Sweep BD/transfer granularity + objectFIFO depth too (NOT just transfer size): a single
# (4KB-line, depth-2) point can under-drive the shim DMA and under-report achievable BW.
# The harness takes max-BW over (line,depth); the artifact name carries both so it can find them.
SWEEP_LINE="${SWEEP_LINE:-1024 4096 16384}"
SWEEP_DEPTH="${SWEEP_DEPTH:-2 4}"
# 64KB .. 64MB, x4 steps. Small points pin the fixed floor; large points the BW slope.
SWEEP_BYTES="${SWEEP_BYTES:-65536 262144 1048576 4194304 16777216 67108864}"

build_one() {  # $1=mode $2=cols $3=line $4=depth $5=bytes
  local mode=$1 cols=$2 line=$3 depth=$4 bytes=$5
  if (( bytes % (cols * line) != 0 )); then
    echo "  skip ${mode} c${cols} l${line} d${depth} ${bytes}: not divisible by cols*line"; return 0
  fi
  local tag="lpddr_${mode}_c${cols}_l${line}_d${depth}_${bytes}"
  local mlir="$WORK/${tag}.mlir"
  echo "== gen+build ${tag} =="
  .venv-iron/bin/python "$GEN" \
      --mode "$mode" --bytes "$bytes" --line "$line" --depth "$depth" --cols "$cols" --dev npu2 > "$mlir"
  ( cd "$WORK" && "$AIECC" --aie-generate-xclbin --xclbin-name="${tag}.xclbin" \
        --no-xchesscc --no-xbridge \
        --aie-generate-npu-insts --npu-insts-name="${tag}.insts.bin" "$mlir" )
  mv -f "$WORK/${tag}.xclbin" "$OUTDIR/${tag}.xclbin"
  mv -f "$WORK/${tag}.insts.bin" "$OUTDIR/${tag}.insts.bin"
  echo "   -> $OUTDIR/${tag}.{xclbin,insts.bin}"
}

for mode in $MODES; do
  for cols in $COLS; do
    for line in $SWEEP_LINE; do
      for depth in $SWEEP_DEPTH; do
        for bytes in $SWEEP_BYTES; do
          build_one "$mode" "$cols" "$line" "$depth" "$bytes"
        done
      done
    done
  done
done
echo "Done. Run (in an NPU window): scripts/run_lpddr_bw_microbench.sh"
