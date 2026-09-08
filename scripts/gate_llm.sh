#!/usr/bin/env bash
# The rail's correctness gate for the LLM path, in two tiers. One command per tier.
#
# The gate this replaces was "produce byte-identical tokens to the existing path". That was free
# when prefill WAS the M=1 decode path; it stopped being satisfiable when the rail gained a second
# implementation of an op (batched prefill projects through GEMM/mm.cc, decode through GEMV/mv.cc
# -- different kernels, different operand formats, different rounding). Measured, the best
# achievable agreement between them is 1.18 bf16 ULP. Identity was never the standard being failed.
#
#   TIER 1  per kernel / per block: element-wise |out-ref| <= atol + rtol*|ref| over the FULL
#           output, against a float32 reference computed from the same bf16 inputs the device saw.
#           rtol = 1.6e-2 (PyTorch/vLLM's canonical bf16 tolerance); atol is per artifact and rides
#           in the artifact's own meta.json. -> scripts/gate_numeric.py
#   TIER 2  end to end: greedy-decode N tokens and require, at the first divergence from the
#           reference model, that the reference's token is inside the device's top-K.
#           N=32, K=5.                                   -> scripts/gate_token_set.py
#
#   bash scripts/gate_llm.sh --tier1              # DEVICE: run the probes, then judge
#   bash scripts/gate_llm.sh --tier1 --judge-only # no device: judge dumps that already exist
#   bash scripts/gate_llm.sh --tier2              # DEVICE: greedy decode, then judge
#   bash scripts/gate_llm.sh --tier2 --judge-only # no device
#   bash scripts/gate_llm.sh --all                # both, device
#   bash scripts/gate_llm.sh --refresh-goldens D  # no device: (re)write D's float32 reference
#   bash scripts/gate_llm.sh --make-ref           # no device: (re)write the Tier 2 reference
#
# The DEVICE steps are announced before they run and are the only ones that open /dev/accel. The
# NPU is single-tenant: run them under xdna-engine-private/journal/scripts/npu_lock.sh.
#
# Env: GATE_ARTIFACTS (space-separated Tier 1 artifact dirs), GATE_DUMP_ROOT, GATE_REF, GATE_NPU,
#      WEIGHTS, GATE_TOKENS, GATE_K, VENV_IRON, IRON.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
PY="$VENV_IRON/bin/python"
[ -x "$PY" ] || { echo "ERROR: no iron venv python at $PY"; exit 2; }

SCRATCH="${GATE_DUMP_ROOT:-/mnt/data/xdna-scratch/prefill/gate}"
WEIGHTS="${WEIGHTS:-$WS/artifacts-qwen3-0.6b/weights}"
REF="${GATE_REF:-$REPO/tests/refs/qwen3-0.6b/gate_ref_n32.json}"
NPU_JSON="${GATE_NPU:-$SCRATCH/npu_tokens.json}"
TOKENS="${GATE_TOKENS:-32}"
K="${GATE_K:-5}"
SPEC="${GATE_SPEC:-qwen3-0.6b}"
DECODE_ART="${DECODE_ART:-$WS/artifacts-qwen3-0.6b/decode}"
# Default Tier 1 subjects: the two BLOCK artifacts. Deliberately not the 28-layer stack -- bf16
# rounding compounds down a deep stack, so its `xout` is Tier 2's question, not Tier 1's. Its
# layer-0 KV slabs ARE Tier 1 subjects; add the artifact to GATE_ARTIFACTS and pass --tensors.
DEFAULT_ARTIFACTS="/mnt/data/xdna-scratch/prefill/mlp_m256 /mnt/data/xdna-scratch/prefill/attn_m256_s2048"
read -r -a ARTIFACTS <<<"${GATE_ARTIFACTS:-$DEFAULT_ARTIFACTS}"

MODE=""; JUDGE_ONLY=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tier1|--tier2|--all|--refresh-goldens|--make-ref) MODE="$1" ;;
    --judge-only) JUDGE_ONLY=1 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done
[ -n "$MODE" ] || { echo "usage: gate_llm.sh --tier1|--tier2|--all|--refresh-goldens <dir>|--make-ref"; exit 2; }

device_step() {
  echo
  echo "=== DEVICE STEP: $1 ==="
  echo "=== the NPU is single-tenant; this must run under npu_lock.sh with npu-asr/voxd stopped ==="
}

refresh_goldens() {
  "$PY" "$REPO/scripts/refresh_prefill_goldens.py" --weights "$WEIGHTS" "$@"
}

make_ref() {
  echo "[gate] building the Tier 2 reference ($TOKENS tokens, top-$K) -- no device"
  "$PY" "$REPO/scripts/gate_llm_reference.py" --spec "$SPEC" --weights "$WEIGHTS" \
      --tokens "$TOKENS" --k "$K" --out "$REF"
}

tier1() {
  mkdir -p "$SCRATCH"
  local rc=0 dumps=() art name out probe
  for art in "${ARTIFACTS[@]}"; do
    [ -f "$art/meta.json" ] || { echo "ERROR: no meta.json in $art"; return 2; }
    name="$(basename "$art")"
    dumps+=("$SCRATCH/dump_$name")
    if [ "$JUDGE_ONLY" = "0" ]; then
      # Which probe reads this artifact is a property of the artifact, so read it off meta.json
      # rather than keeping a list here that drifts from the generators.
      out="$(jq -r .output "$art/meta.json")"
      case "$out" in
        out|cx)  probe=fused_elf_probe ;;
        xout)    probe=prefill_golden_probe ;;
        *) echo "ERROR: $art/meta.json output=$out -- no probe known for it"; return 2 ;;
      esac
      ( cd "$REPO/rust" && cargo build --release -p npu-probes --bin "$probe" ) || return 2
      device_step "$probe on $name"
      GATE_DUMP_DIR="$SCRATCH/dump_$name" "$REPO/rust/target/release/$probe" "$art" || rc=1
    fi
  done
  echo
  "$PY" "$REPO/scripts/gate_numeric.py" "${ARTIFACTS[@]}" \
      --dump "$(IFS=,; echo "${dumps[*]}")" --json "$SCRATCH/tier1.json" || rc=1
  return $rc
}

tier2() {
  mkdir -p "$SCRATCH"
  [ -f "$REF" ] || make_ref || return 2
  if [ "$JUDGE_ONLY" = "0" ]; then
    [ -d "$DECODE_ART" ] || echo "[gate] NOTE no decode artifact at $DECODE_ART; verify_llm_decode "\
"builds its own graph, so this only matters if you meant to gate a built ELF"
    . "$REPO/scripts/amd_paths.sh"
    local IRONDIR="${IRON:-$IRON_DIR}" INST
    INST="$("$REPO/scripts/toolchain_up.sh")" || return 2
    export PYTHONPATH="$INST/python:$IRONDIR${PYTHONPATH:+:$PYTHONPATH}"
    export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
    export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
    export MLIR_AIE_INSTANCE="$INST"
    export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
    device_step "greedy decode, $TOKENS tokens, capturing top-$K per step"
    "$PY" "$REPO/designs/decode_fused/verify_llm_decode.py" --spec "$SPEC" --weights "$WEIGHTS" \
        --ref "$REF" --steps "$TOKENS" --topk "$K" --emit-topk "$NPU_JSON" "${EXTRA[@]}" || return 2
  fi
  echo
  "$PY" "$REPO/scripts/gate_token_set.py" --ref "$REF" --npu "$NPU_JSON" --k "$K"
}

rc=0
case "$MODE" in
  --refresh-goldens) refresh_goldens "${EXTRA[@]}"; rc=$? ;;
  --make-ref)        make_ref; rc=$? ;;
  --tier1)           tier1; rc=$? ;;
  --tier2)           tier2; rc=$? ;;
  --all)             tier1; rc=$?; tier2 || rc=1 ;;
esac
echo
echo "[gate] $MODE -> $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
exit $rc
