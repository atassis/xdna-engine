#!/usr/bin/env bash
# Canonical fused-decode build entrypoint. Encodes the EXACT dims each consumer
# expects, so callers pick a PROFILE by name instead of hand-assembling S/T/P/NL/B
# env (the source of the T=128-vs-1500 / S=64-vs-448 build mistakes — those dims
# only lived in baseline meta.json + docs before this).
#
#   bash scripts/build_decode_profile.sh wer      [extra build_batched_decode args]
#   bash scripts/build_decode_profile.sh l1gate
#   COALESCE_GEMV=1 bash scripts/build_decode_profile.sh wer   # opt-in BD-iteration
#
# Profiles (single source of truth for decode build configs):
#   wer    : the config scripts/wer_batched_decode.sh + rust verify_batched_decode
#            REQUIRE — B=16 NL=12 S=448 T_enc=1500 P=5, scratchpad + engine-only.
#            (verify asserts T_enc==1500; S=448 is the production self-context.)
#   l1gate : the byte-gate proxy — B=128 NL=1 S=448 T=1500, scratchpad engine-only,
#            SKIP_EXPAND_PDIS + DISABLE_REPEATER (frozen-MLIR aiecc-only gate = 370686d).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"; shift || true
export AIECC_PATH="${AIECC_PATH:-$("$REPO/scripts/toolchain_up.sh")/bin/aiecc}"
export AIECC_JOBS="${AIECC_JOBS:-16}"
case "$PROFILE" in
  wer)
    echo "[profile wer] B=16 NL=12 S=448 T=1500 P=5 SP ENG  (COALESCE_GEMV=${COALESCE_GEMV:-0})"
    SP=1 ENG=1 B=16 NL=12 S=448 T=1500 PVAL=5 \
      bash "$REPO/scripts/build_batched_decode.sh" decode "$@" ;;
  l1gate)
    echo "[profile l1gate] B=128 NL=1 S=448 T=1500 SP ENG SKIP_EXPAND_PDIS DISABLE_REPEATER  (COALESCE_GEMV=${COALESCE_GEMV:-0})"
    SP=1 ENG=1 SKIP_EXPAND_PDIS=1 DISABLE_REPEATER=1 B=128 NL=1 S=448 T=1500 \
      bash "$REPO/scripts/build_batched_decode.sh" decode "$@" ;;
  *)
    echo "usage: $0 {wer|l1gate} [args]   (set COALESCE_GEMV=1 to opt into BD-iteration)"; exit 2 ;;
esac
