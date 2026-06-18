#!/usr/bin/env bash
# Establish the PROPER aiecc byte-gate: generate the B=128 NL=1 fused-decode MLIR
# ONCE into a PERSISTENT dir (frozen), and build the ELF. The frozen decode_b.mlir
# can then be fed DIRECTLY to aiecc (scripts/gate_aiecc_only.sh) to byte-gate
# compiler changes without generator variation.
#
# Why: the canonical reference 1e6098a3 was measured on a pre-generated MLIR
# (kernel-cache result), but the full generator path can vary run-to-run; gating
# a compiler change requires a FROZEN input.
#
#   bash scripts/gate_freeze_and_build.sh
#
# Leaves: $FROZEN_DIR/decode_b.mlir (frozen input), $FROZEN_DIR/work/ (aiecc .prj),
#         and prints the ELF sha256.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="${B:-128}"; NL="${NL:-1}"
FROZEN_DIR="${FROZEN_DIR:-$REPO/artifacts/gate_frozen_B${B}_L${NL}}"
VENV_IRON="$REPO/.venv-iron"
IRON="~/repositories/ns/amd/IRON"
AIEBU_DIR="~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm"
WEIGHTS="$REPO/artifacts/whisper-small/whisper_decoder"
GENDIR="$REPO/route_b_kernels/decode_fused"

export AIECC_PATH="${AIECC_PATH:-$REPO/mlir-aie/build-on2/bin/aiecc}"
export AIECC_PHASE_TIMERS="${AIECC_PHASE_TIMERS:-1}"
export AIECC_PHASE_TIMERS_FILE="${AIECC_PHASE_TIMERS_FILE:-$FROZEN_DIR/phase_timers.log}"
export AIECC_JOBS="${AIECC_JOBS:-16}"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
export PYTHONPATH="$IRON:$GENDIR${PYTHONPATH:+:$PYTHONPATH}"

# apply IRON patches idempotently (same set as build_batched_decode.sh)
apply_patch(){ local p="$1"; [ -f "$p" ] || return 0
  if git -C "$IRON" apply --reverse --check "$p" >/dev/null 2>&1; then echo "[freeze] $(basename "$p") already applied"
  else echo "[freeze] applying $(basename "$p")"; git -C "$IRON" apply "$p"; fi; }
apply_patch "$REPO/patches/amd-IRON-deepc.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-transpose-num-batches.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-gemm-fusion-prefix.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-aiecc-build-perf.patch"

rm -rf "$FROZEN_DIR"; mkdir -p "$FROZEN_DIR/work" "$FROZEN_DIR/out"
: > "$AIECC_PHASE_TIMERS_FILE"
echo "[freeze] aiecc=$AIECC_PATH B=$B NL=$NL  frozen=$FROZEN_DIR"
t0=$(date +%s)
( cd "$FROZEN_DIR/work" && SP=1 ENG=1 SKIP_EXPAND_PDIS=1 DISABLE_REPEATER=1 \
    "$VENV_IRON/bin/python" "$GENDIR/gen_decode_batched.py" \
      --weights "$WEIGHTS" --B "$B" --layers "$NL" --S "${S:-64}" --T "${T:-128}" \
      --scratchpad --engine-only --out "$FROZEN_DIR/out" )
rc=$?
t1=$(date +%s)
echo "[freeze] gen+build wall: $((t1-t0)) s (rc=$rc)"
[ $rc -eq 0 ] || { echo "FREEZE FAIL rc=$rc"; exit $rc; }

# freeze the generated fused MLIR (largest decode_b*.mlir under work/, not a .prj intermediate)
MLIR="$(find "$FROZEN_DIR/work" -maxdepth 2 -name 'decode_b*.mlir' ! -path '*.prj*' -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')"
if [ -n "$MLIR" ]; then cp "$MLIR" "$FROZEN_DIR/decode_b_frozen.mlir"; echo "[freeze] froze MLIR: $MLIR -> decode_b_frozen.mlir ($(du -h "$FROZEN_DIR/decode_b_frozen.mlir"|cut -f1))"; else echo "[freeze] WARN: no decode_b*.mlir found under work/"; find "$FROZEN_DIR/work" -name '*.mlir' | head; fi

ELF="$FROZEN_DIR/out/decode_b.elf"
[ -f "$ELF" ] || { echo "FREEZE FAIL: no ELF"; exit 3; }
echo "[freeze] ELF sha256=$(sha256sum "$ELF" | awk '{print $1}')"
echo "[freeze] phase timers -> $AIECC_PHASE_TIMERS_FILE"
echo "FREEZE DONE"
