#!/usr/bin/env bash
# Byte-gate harness for the build-chain attack (a batch run).
# Builds the B=128 NL=1 fused-decode ELF via build-on2 aiecc (SP=1 ENG=1
# SKIP_EXPAND_PDIS=1 DISABLE_REPEATER=1) and prints the ELF sha256, comparing it
# to the canonical reference 1e6098a3 (the kernel-cache byte-gate).
#
# Phase timers are enabled (AIECC_PHASE_TIMERS=1) — byte-neutral, stdout only.
#
#   bash scripts/gate_build_decode_l1.sh [NL]      # NL defaults to 1
#
# Exit 0 + "GATE PASS" if the sha matches; non-zero + "GATE FAIL" otherwise.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NL="${1:-1}"
REF_SHA="1e6098a303587b801b5f7e13ddafa5b7ffaa96fcc4f46290b554a555fba3d00b"

export AIECC_PATH="${AIECC_PATH:-$REPO/mlir-aie/build-on2/bin/aiecc}"
export AIECC_PHASE_TIMERS="${AIECC_PHASE_TIMERS:-1}"
export AIECC_JOBS="${AIECC_JOBS:-16}"

[ -x "$AIECC_PATH" ] || { echo "GATE FAIL: aiecc not at $AIECC_PATH"; exit 2; }
echo "[gate] aiecc=$AIECC_PATH  NL=$NL  JOBS=$AIECC_JOBS"

t0=$(date +%s)
SP=1 ENG=1 SKIP_EXPAND_PDIS=1 DISABLE_REPEATER=1 B=128 NL="$NL" \
  bash "$REPO/scripts/build_batched_decode.sh" decode
rc=$?
t1=$(date +%s)
echo "[gate] build wall: $((t1 - t0)) s  (rc=$rc)"
[ $rc -eq 0 ] || { echo "GATE FAIL: build rc=$rc"; exit $rc; }

sp_tag="_sp"; nopdi_tag="_nopdi"
OUT="$REPO/artifacts/decode_batched_B128_L${NL}${sp_tag}${nopdi_tag}"
ELF="$(ls "$OUT"/*.elf 2>/dev/null | head -1)"
[ -n "$ELF" ] || { echo "GATE FAIL: no ELF in $OUT"; exit 3; }
GOT="$(sha256sum "$ELF" | awk '{print $1}')"
echo "[gate] ELF=$ELF"
echo "[gate] sha256=$GOT"
if [ "$NL" = "1" ] && [ "$GOT" = "$REF_SHA" ]; then
  echo "GATE PASS (sha == 1e6098a3)"
  exit 0
elif [ "$NL" = "1" ]; then
  echo "GATE FAIL (sha != 1e6098a3 reference)"
  exit 4
else
  echo "GATE INFO (NL=$NL has no canonical ref; sha recorded above)"
  exit 0
fi
