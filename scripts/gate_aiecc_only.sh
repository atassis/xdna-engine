#!/usr/bin/env bash
# FAST, ISOLATED aiecc byte-gate: run build-on2 aiecc DIRECTLY on the frozen
# decode MLIR (scripts/gate_freeze_and_build.sh must have produced it). This
# isolates COMPILER (aiecc) changes from generator variation — the proper gate
# for Track A. Replicates IRON's AieccFullElfCompilationRule command for our
# flags (SKIP_EXPAND_PDIS=1 DISABLE_REPEATER=1).
#
#   bash scripts/gate_aiecc_only.sh            # builds + prints sha + phase timers
#   REF=<sha> bash scripts/gate_aiecc_only.sh  # also PASS/FAIL vs REF
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="${B:-128}"; NL="${NL:-1}"
FROZEN_DIR="${FROZEN_DIR:-$REPO/artifacts/gate_frozen_B${B}_L${NL}}"
MLIR="${MLIR:-$FROZEN_DIR/decode_b_frozen.mlir}"
VENV_IRON="$REPO/.venv-iron"
AIEBU_DIR="~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm"
PEANO="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"

export AIECC_PATH="${AIECC_PATH:-$REPO/mlir-aie/build-on2/bin/aiecc}"
export AIECC_JOBS="${AIECC_JOBS:-16}"
WORK="${WORK:-$(mktemp -d)}"
export AIECC_PHASE_TIMERS=1
export AIECC_PHASE_TIMERS_FILE="${AIECC_PHASE_TIMERS_FILE:-$WORK/phase_timers.log}"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$PEANO"

[ -x "$AIECC_PATH" ] || { echo "GATE FAIL: aiecc not at $AIECC_PATH"; exit 2; }
[ -f "$MLIR" ] || { echo "GATE FAIL: frozen MLIR not at $MLIR (run gate_freeze_and_build.sh)"; exit 2; }
# aiecc resolves `link_with` kernel objects relative to the INPUT MLIR's dir, so
# the companion .o files must sit beside the frozen MLIR. gate_freeze_and_build.sh
# leaves them under work/build/; copy them next to the MLIR if missing.
MLIR_DIR="$(dirname "$MLIR")"
if ! ls "$MLIR_DIR"/*.o >/dev/null 2>&1 && ls "$MLIR_DIR"/work/build/*.o >/dev/null 2>&1; then
  cp "$MLIR_DIR"/work/build/*.o "$MLIR_DIR"/ && echo "[gate-aiecc] co-located $(ls "$MLIR_DIR"/*.o|wc -l) kernel .o beside MLIR"
fi
: > "$AIECC_PHASE_TIMERS_FILE"
OUT="$WORK/decode_b.elf"
echo "[gate-aiecc] aiecc=$AIECC_PATH  mlir=$MLIR  work=$WORK  JOBS=$AIECC_JOBS"

t0=$(date +%s)
( cd "$WORK" && "$AIECC_PATH" -v -j"$AIECC_JOBS" --no-compile-host --no-xchesscc \
    --no-xbridge --peano "$PEANO" --disable-repeater-scripts --generate-full-elf \
    --full-elf-name "$OUT" "$MLIR" ) > "$WORK/aiecc.stdout.log" 2>&1
rc=$?
t1=$(date +%s)
echo "[gate-aiecc] aiecc wall: $((t1-t0)) s (rc=$rc)"
echo "[gate-aiecc] --- phase timers ---"; cat "$AIECC_PHASE_TIMERS_FILE" 2>/dev/null | grep -E "SERIAL|:|timed-sum"
[ $rc -eq 0 ] || { echo "GATE FAIL: aiecc rc=$rc (see $WORK/aiecc.stdout.log)"; tail -20 "$WORK/aiecc.stdout.log"; exit $rc; }
[ -f "$OUT" ] || { echo "GATE FAIL: no ELF at $OUT"; exit 3; }
GOT="$(sha256sum "$OUT" | awk '{print $1}')"
echo "[gate-aiecc] ELF=$OUT"
echo "[gate-aiecc] sha256=$GOT"
echo "[gate-aiecc] work kept at: $WORK"
if [ -n "${REF:-}" ]; then
  if [ "$GOT" = "$REF" ]; then echo "GATE PASS (sha == REF $REF)"; exit 0
  else echo "GATE FAIL (sha $GOT != REF $REF)"; exit 4; fi
fi
echo "GATE INFO (no REF given; sha recorded above)"
