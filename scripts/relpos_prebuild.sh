#!/usr/bin/env bash
# STEP-C: build the resident relpos-MHA xclbins (STEP=8) + template instruction streams, one per
# T-BUCKET, into artifacts/relpos/<subdir>/{final.xclbin,insts.bin}. npu.rs loads each ONCE
# (co-resident) and PATCHES the t_active word of the insts per clip -- a bucket serves any clip
# T <= its BUILT_T, zero per-clip build, and the dispatch picks the SMALLEST bucket that fits (so
# short clips run a smaller padded dataflow -> less wasted relpos padding compute).
#
# The BUCKETS list below MUST match RELPOS_BUCKETS in rust/npu-parakeet/src/npu.rs
# (BUILT_T:KB:subdir). KB (key-block rows) is baked into the xclbin buffer sizes, so it must match
# the Rust bucket entry. TQ=8 HEADS=8 (Parakeet n_heads on 8 parallel cores).
#
# Usage:  scripts/relpos_prebuild.sh          (builds every missing bucket; FORCE=1 rebuilds all)
# Needs the FORK toolchain env (sourced internally). Serializes on the shared toolchain.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
TQ=8
# HEADS=8 = Parakeet n_heads run on 8 PARALLEL cores (one head/core), one dispatch/block
# (Phase-2 spatial-parallel relpos). HEADS=1 rebuilds the original single-Worker block.
HEADS="${HEADS:-8}"
# BUILT_T:KB:subdir -- MUST match whichever RELPOS_BUCKETS table npu.rs selects, or the engine
# panics on a bucket this never built. The default here has to be RELPOS_BUCKETS_SHIPPED, since
# that is what npu.rs picks with no env override.
#   NOPEEL (PARAKEET_RELPOS_NOPEEL=1):  112:28:bucket_112 152:38:bucket_152 176:44:single
#   ROWTILE (PARAKEET_RELPOS_ROWTILE=1, ROWS=4): 160:40:bucket_152 192:48:single
# A BUILT_T that is not a multiple of TQ peels a ragged query tile and re-emits every block call
# (20% of the core's 16 KB program memory) -- which is what the NOPEEL set exists to avoid -- and
# must never be 104 (the insts header word[2] constant; the per-clip patch is by value).
BUCKETS="${BUCKETS:-100:25:bucket_100 152:38:bucket_152 172:43:single}"
# TSKIP=1 builds the runtime t_active block-skip arm (drop the blocks the softmax never reads).
# OUT_ROOT redirects the artifacts so both arms can sit side by side for a swap A/B.
TSKIP="${TSKIP:-0}"
# ROWS>1 row-tiles each head across ROWS cores (k/p/V multicast + MemTile split/join). Each CORE
# carries its own t_active RTP, so the insts hold HEADS*ROWS t_active words, not HEADS.
ROWS="${ROWS:-1}"
OUT_ROOT="${OUT_ROOT:-artifacts/relpos}"
EX=mlir-aie/programming_examples/ml/relpos_mha

setup_env() { source scripts/iron_env.sh; }
setup_env
python3 -c "import aie.iron" 2>/dev/null || { echo "[prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1

build_bucket() {
  local BUILT_T="$1" KB="$2" out="$OUT_ROOT/$3"
  if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ] && [ -z "${FORCE:-}" ]; then
    echo "[prebuild] bucket $BUILT_T present -> $out (FORCE=1 to rebuild)"; return 0
  fi
  echo "[prebuild] building STEP=8 T=$BUILT_T TQ=$TQ KB=$KB HEADS=$HEADS -> $out ..."
  ( cd "$EX" && make clean >/dev/null 2>&1; \
    make NPU2=1 STEP=8 SPLITP=1 TSKIP="$TSKIP" ROWS="$ROWS" T="$BUILT_T" TQ="$TQ" KB="$KB" TACTIVE="$BUILT_T" HEADS="$HEADS" >/dev/null 2>&1 )
  if [ ! -f "$EX/build/final.xclbin" ]; then
    echo "[prebuild] FAILED bucket $BUILT_T (no final.xclbin)"; return 1
  fi
  # The template insts hold HEADS t_active words (one per head's RTP write), all == BUILT_T.
  # npu.rs patches EVERY word == BUILT_T to the clip's t. Verify the count == HEADS so a
  # stale/mis-built insts (wrong head count, or a BUILT_T collision) fails LOUD here.
  local nt
  nt=$(python3 - "$EX/build/insts.bin" "$BUILT_T" <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
v = int(sys.argv[2])
w = [struct.unpack("<I", b[i:i+4])[0] for i in range(0, len(b), 4)]
print(sum(1 for x in w if x == v))
PY
)
  local want=$(( HEADS * ROWS ))
  if [ "$nt" != "$want" ]; then
    echo "[prebuild] WARN: bucket $BUILT_T found $nt insts words == BUILT_T, expected HEADS*ROWS=$want t_active words."
    echo "[prebuild]       npu.rs patches all of them; a mismatch means a BUILT_T value collision -- inspect before use."
  fi
  mkdir -p "$out"
  cp "$EX/build/final.xclbin" "$out/final.xclbin"
  cp "$EX/build/insts.bin"    "$out/insts.bin"
  echo "[prebuild] installed bucket $BUILT_T ($nt t_active words) -> $out"
}

rc=0
for spec in $BUCKETS; do
  IFS=: read -r BT KB SUB <<<"$spec"
  build_bucket "$BT" "$KB" "$SUB" || rc=1
done
exit $rc
