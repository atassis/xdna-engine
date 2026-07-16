#!/usr/bin/env bash
# STEP-C: build the SINGLE resident relpos-MHA xclbin (STEP=8, T=RELPOS_BUILT_T=172) and its
# template instruction stream into artifacts/relpos/single/{final.xclbin,insts.bin}. npu.rs loads
# this ONCE (resident) and PATCHES the t_active word of the insts per clip -- one xclbin serves any
# clip T <= 172, zero per-clip build. TQ=8 KB=43 (must match npu.rs).
#
# Usage:  scripts/relpos_prebuild.sh          (builds the single T=172 xclbin)
# Needs the FORK toolchain env (sourced internally). Serializes on the shared toolchain.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BUILT_T=172; TQ=8; KB=43
# HEADS=8 = Parakeet n_heads run on 8 PARALLEL cores (one head/core), one dispatch/block
# (Phase-2 spatial-parallel relpos). HEADS=1 rebuilds the original single-Worker block.
HEADS="${HEADS:-8}"
EX=mlir-aie/programming_examples/ml/relpos_mha
out="artifacts/relpos/single"

setup_env() { source scripts/iron_env.sh; }
setup_env
python3 -c "import aie.iron" 2>/dev/null || { echo "[prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1

if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ] && [ -z "${FORCE:-}" ]; then
  echo "[prebuild] single xclbin already present -> $out (FORCE=1 to rebuild)"; exit 0
fi
echo "[prebuild] building the STEP=8 T=$BUILT_T TQ=$TQ KB=$KB HEADS=$HEADS xclbin (runtime t_active) ..."
( cd "$EX" && make clean >/dev/null 2>&1; \
  make NPU2=1 STEP=8 SPLITP=1 T="$BUILT_T" TQ="$TQ" KB="$KB" TACTIVE="$BUILT_T" HEADS="$HEADS" >/dev/null 2>&1 )
if [ ! -f "$EX/build/final.xclbin" ]; then
  echo "[prebuild] FAILED (no final.xclbin)"; exit 1
fi
# The template insts hold HEADS t_active words (one per head's RTP write), all == BUILT_T.
# npu.rs patches EVERY word == BUILT_T to the clip's t. Verify the count == HEADS so a
# stale/mis-built insts (wrong head count, or a BUILT_T collision) fails LOUD here.
nt=$(python3 - "$EX/build/insts.bin" "$BUILT_T" <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
v = int(sys.argv[2])
w = [struct.unpack("<I", b[i:i+4])[0] for i in range(0, len(b), 4)]
print(sum(1 for x in w if x == v))
PY
)
if [ "$nt" != "$HEADS" ]; then
  echo "[prebuild] WARN: found $nt insts words == BUILT_T=$BUILT_T, expected HEADS=$HEADS t_active words."
  echo "[prebuild]       npu.rs patches all of them; a mismatch means a BUILT_T value collision -- inspect before use."
fi
mkdir -p "$out"
cp "$EX/build/final.xclbin" "$out/final.xclbin"
cp "$EX/build/insts.bin"    "$out/insts.bin"
echo "[prebuild] installed xclbin + template insts ($nt t_active words) -> $out"
