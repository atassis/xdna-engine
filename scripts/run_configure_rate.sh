#!/usr/bin/env bash
# The configure-rate control on the Gemma-4 decode graph: several arms that differ ONLY in how
# many aiex.configure blocks the token costs, resident and dispatched round-robin.
#
#   bash scripts/run_configure_rate.sh --warm              # build every arm + census, no device
#   bash scripts/run_configure_rate.sh [SESSIONS] [LAYERS] # time them under the device lock
#
# The arm axis is `g<G>` (SPLIT_QKNORM): the per-head qk-norm runs are dealt to two IDENTICAL
# RMSNorm designs in contiguous groups of G heads, so a group boundary is a design switch. Runs,
# bytes, designs (2 at every G>0) and output are all unchanged; the configure count is the only
# quantity that moves. That is the order-only shape D009's rate needs.
#
# Depth is REDUCED and that is deliberate: an arm's arena is ~15.3 GB at the full 48 layers and
# this box has 30 GB, so the arms cannot be resident together at full depth -- and resident
# round-robin is what makes the deltas survive DVFS drift. L must stay a multiple of 6 to keep the
# 5-sliding-1-global layer pattern the configure count depends on.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
SPEC=gemma4-12b
WEIGHTS="${WEIGHTS:-/mnt/data/xdna/artifacts/$SPEC/weights_int8g64}"
OUT="${CONFIGURE_RATE_OUT:-/mnt/data/xdna/scratch/gemma4/configure-rate}"
WORK="${CONFIGURE_RATE_WORK:-/mnt/data/xdna/build/gemma4-configure-rate}"
ARMS=(${CONFIGURE_RATE_ARMS:-g0 g4 g2 g1})
# --warm takes LAYERS as its only positional. Sharing one positional list with the timing form
# silently built L=6 for `--warm 12`, and a census of the wrong depth looks exactly like a census
# of the right one.
WARM_ONLY=0
if [ "${1:-}" = "--warm" ]; then
  WARM_ONLY=1; shift
  SESSIONS=0; LAYERS="${1:-6}"
else
  SESSIONS="${1:-3}"; LAYERS="${2:-6}"
fi
case "$LAYERS" in (*[!0-9]*|"") echo "LAYERS must be a positive integer, got '$LAYERS'" >&2; exit 2 ;; esac
[ $((LAYERS % 6)) -eq 0 ] || echo "[warn] LAYERS=$LAYERS is not a multiple of 6; the 5-sliding-1-global pattern is broken and the per-layer configure count will not match the shipped graph's" >&2
MAXSEQ="${MAXSEQ:-512}"
POS="${POS:-7}"
REPS="${REPS:-25}"

# Device serialisation, named by env rather than by path: the serialiser lives outside this repo,
# and more than one session runs against this box. Unset is a hard error, not a silent unlocked run
# -- two arms sharing the device produce plausible, wrong timings rather than a failure.
LOCK="${NPU_LOCK:-}"
if [ "$WARM_ONLY" != 1 ]; then
  [ -n "$LOCK" ] || { echo "set NPU_LOCK to a serialiser accepting \$NPU_LOCK queue -- <cmd...>" >&2; exit 2; }
  [ -x "$LOCK" ] || { echo "NPU_LOCK=$LOCK is not executable" >&2; exit 2; }
fi

mkdir -p "$OUT" "$WORK"
cd "$REPO"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
INST="$("$REPO/scripts/toolchain_up.sh")" || exit 1
export PYTHONPATH="$INST/python:$IRON"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE=npu2
export CUDA_VISIBLE_DEVICES=""
# The shipped Gemma-4 graph. GQA_GROUPED_V and FUSE_QKV_DP are NOT at their module defaults, so
# they are named here rather than inherited -- FUSE_QKV_DP=1 refuses this spec outright and
# GQA_GROUPED_V=0 would be a different graph with a different byte count. The weight FORMAT is not
# set at all: weights_int8g64 carries a quant.json and a packed dump is the authority.
export GQA_GROUPED_K=1 GQA_GROUPED_V=1 FUSE_QKV_DP=0 SPLIT_LM_HEAD=0
# ONE build dir for every arm, on the same condition build_llm_decode.sh relies on: the arm is in
# the artifact NAME (`splitqk<G>`), so two arms cannot share a filename. Verified in --warm, which
# prints each arm's resolved census before any device time is spent on it.
ARGS=(--spec "$SPEC" --weights "$WEIGHTS" --max-seq "$MAXSEQ" --pos "$POS"
      --arms "${ARMS[@]/#/$LAYERS}")

if [ "$WARM_ONLY" = 1 ]; then
  ( cd "$WORK" && "$VENV_IRON/bin/python" "$REPO/designs/decode_fused/bench_layer_arms.py" \
      "${ARGS[@]}" --build-only --out-json "$OUT/census-l$LAYERS.json" )
  exit $?
fi

# The serving units hold /dev/accel, and npu_lock.sh is deliberately non-destructive -- it DEFERS
# rather than stopping them, so without this every session here exits 75. Stop them around the whole
# session set and restart on any exit, including a trap: leaving the box without dictation is worse
# than a lost measurement.
. "$REPO/scripts/_npu_services.sh"
npu_svc_assert_units || exit 2
trap 'npu_svc_start' EXIT INT TERM
npu_svc_stop || exit 2
npu_svc_require_device_free || { echo "device still held after stopping the units"; exit 75; }

# Power mode is per-boot and often unpinned; two traces can legitimately disagree by 18% (D027).
xrt-smi examine -r platform 2>/dev/null | grep -i mode | tee "$OUT/power-mode.txt" || true

for s in $(seq 1 "$SESSIONS"); do
  echo "############ session $s  $(date +%H:%M:%S)"
  ( cd "$WORK" && "$LOCK" queue -- "$VENV_IRON/bin/python" \
      "$REPO/designs/decode_fused/bench_layer_arms.py" "${ARGS[@]}" \
      --reps "$REPS" --warmup 5 --out-json "$OUT/session-l$LAYERS-$s.json" )
  rc=$?
  [ "$rc" = 75 ] && { echo "DEVICE BUSY -- deferred at session $s"; exit 75; }
  [ "$rc" != 0 ] && { echo "session $s failed rc=$rc"; exit "$rc"; }
done
echo "############ done $(date +%H:%M:%S)"
