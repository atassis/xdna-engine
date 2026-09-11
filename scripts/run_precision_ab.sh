#!/usr/bin/env bash
# The precision A/B on device: several weight-format arms, resident and interleaved.
#
#   bash scripts/run_precision_ab.sh [SESSIONS] [POS]
#     SESSIONS  independent sessions, each a full round-robin (default 3)
#     POS       n_past every arm is dispatched at (default 1024)
#
# Sets up the same toolchain environment build_llm_decode.sh does, then runs
# designs/decode_fused/bench_precision_arms.py under the device lock. The interleaving that makes
# the deltas trustworthy is INSIDE the harness (all arms resident, round-robin per rep); the
# repeated sessions here are the coarser replication on top.
#
# NON-DESTRUCTIVE: npu_lock.sh `queue` waits its turn and defers with exit 75 if production holds
# the device. It never stops npu-serve.
#
# Pre-warm the per-arm build caches first or the first session compiles 28 layers per arm while
# holding the lock:  bash scripts/run_precision_ab.sh --warm
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
# Device serialisation helper. The NPU is single-tenant, so arms must not overlap. Point
# NPU_LOCK at a serialiser exposing `<lock> queue -- <cmd...>`; it lives outside this repo,
# so it is named by env rather than by path. Unset is a hard error, not a silent unlocked
# run: two arms sharing the device produce plausible, wrong timings rather than a failure.
LOCK="${NPU_LOCK:-}"
if [ -z "$LOCK" ] || [ ! -x "$LOCK" ]; then
  echo "run_precision_ab.sh: set NPU_LOCK to an executable device serialiser" >&2
  echo "  (it must accept: \$NPU_LOCK queue -- <command...>)" >&2
  exit 2
fi
OUT="${PRECISION_AB_OUT:-/mnt/data/xdna/scratch/precision/ab}"
ARMS=(bf16 '{"head":"int8a/g128"}' mlp-int8 mlp-head-int8)
WARM_ONLY=0
[ "${1:-}" = "--warm" ] && { WARM_ONLY=1; shift; }
SESSIONS="${1:-3}"
POS="${2:-1024}"
mkdir -p "$OUT"
cd "$REPO"

VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE=npu2
export DYNAMIC_WINDOW=1 WINDOW_RUNGS=256,512,1024,2048
export CUDA_VISIBLE_DEVICES=""
# ONE build dir, shared by every arm and kept across sessions, because the precision plan is in
# the artifact NAME at both levels: the design (GEMV..._wdtint8ag128,
# DecodeLayerDataParallel..._int8ag128) and the object (gemv_1024k_64vs_int8ag128.o,
# mlp_swiglu_mlp_dp_core_int8ag128.a). Two arms therefore cannot share a filename, which is the
# same condition build_llm_decode.sh relies on for its own shared cache. Verified per arm before
# trusting it -- a name that did NOT distinguish would have one arm silently assembling an ELF
# from another's binaries, which is a deterministic wrong answer, not a crash.
WORK=/mnt/data/xdna/build/precision-ab
mkdir -p "$WORK"

if [ "$WARM_ONLY" = 1 ]; then
  for arm in "${ARMS[@]}"; do
    tag="$(echo "$arm" | tr -c 'A-Za-z0-9' '-')"
    echo "[warm] $arm"
    PRECISION="$arm" KEEP_WORK="$WORK" GEN_EXTRA="--max-seq 4096" \
      bash scripts/build_llm_decode.sh qwen3-0.6b 28 \
      "/mnt/data/xdna/artifacts/precision-ab/$tag" || exit 1
  done
  exit 0
fi

# Power mode is per-boot and often unpinned; two traces can legitimately disagree by 18%, so it
# is recorded rather than assumed, once per session set.
xrt-smi examine -r platform 2>/dev/null | grep -i mode | tee "$OUT/power-mode.txt" || true

for s in $(seq 1 "$SESSIONS"); do
  echo "############ session $s  $(date +%H:%M:%S)"
  # Run FROM the warm dir: the harness writes IRON's build/ intermediates under its cwd, and
  # this is where --warm left them.
  ( cd "$WORK" && "$LOCK" queue -- "$VENV_IRON/bin/python" \
      "$REPO/designs/decode_fused/bench_precision_arms.py" \
      --spec qwen3-0.6b --weights "$REPO/artifacts/qwen3-0.6b/weights" \
      --layers 28 --max-seq 4096 --pos "$POS" --reps 30 --warmup 5 \
      --arms "${ARMS[@]}" --out-json "$OUT/session-$s.json" )
  rc=$?
  [ "$rc" = 75 ] && { echo "DEVICE BUSY -- deferred at session $s"; exit 75; }
  [ "$rc" != 0 ] && { echo "session $s failed rc=$rc"; exit "$rc"; }
done
echo "############ done $(date +%H:%M:%S)"
