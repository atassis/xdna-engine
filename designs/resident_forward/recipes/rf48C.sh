#!/usr/bin/env bash
# One-command recipe for the gemma4-12b resident-forward rf48C design: the full 48-layer forward
# (attention + MLP + LM head) as one image, built as a ladder of parts (fwd_ladder.sh) and packed
# into one ELF (fwd_pack.py). Confirmed reproduction: toolchain.lock's MLIR_AIE_FORK_COMMIT
# 8e3958b596a + IRON at IRON_PIN below -> design.elf sha256 matching the served artifact.
#
# Required env: IRON_DIR (an IRON checkout; HEAD is checked against IRON_PIN below, warns on
# mismatch/dirty rather than failing, since a close-but-not-exact IRON often still builds -- it
# just may not reproduce the exact bytes), RF_BUILD (an isolated, disk-backed build dir; never
# share one build dir between recipe runs, the kernel-object cache under it is per-run but the
# per-part directories are not).
set -euo pipefail
IRON_PIN=22d10a51b083e854a8153767fc10b427a62eef58
export RF_ATTN_H=1 RF_FAST=1 RF_TEXT_MARGIN=128
: "${IRON_DIR:?set IRON_DIR to an IRON checkout on (or near) $IRON_PIN}"
: "${RF_BUILD:?set RF_BUILD to an isolated, disk-backed build dir}"

HERE=$(dirname "$(readlink -f "$0")")
RF=$(dirname "$HERE")
REPO=$(cd "$RF/../.." && pwd)
if ! git -C "$REPO" diff --quiet HEAD -- "$RF" 2>/dev/null; then
  echo "warning: $RF has uncommitted changes -- output may not match" >&2
fi

iron_got=$(git -C "$IRON_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$iron_got" != "$IRON_PIN" ]; then
  echo "warning: IRON_DIR is at $iron_got, not the confirmed pin $IRON_PIN -- output may not match" >&2
fi
if ! git -C "$IRON_DIR" diff --quiet 2>/dev/null; then
  echo "warning: IRON_DIR has a dirty working tree -- output may not match the pin" >&2
fi

exec "$RF/fwd_ladder.sh" rf48C \
  "m g h arena seg=64,256,1024,4096 fseg=64,256,1024 split=128,256,512,1024,2048,4096 gcap=262144 sring=1280 s64 fwd=0-47+h nbw=20 1 2" \
  "p1,p2,g1w64,g1w256,g1w1024,g1w4096,g2w64,g2w256,g2w1024,g2w4096,g1s128,g1s256,g1s512,g1s1024,g1s2048,g1s4096,h1,f1" \
  "f2,f1s256" \
  "f1s512,f1s1024" \
  "f1s2048,f1s4096" \
  "f2w256" \
  "f2w1024"
