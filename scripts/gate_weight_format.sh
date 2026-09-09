#!/usr/bin/env bash
# The three device gates a weight-format change owes, in one run and one device hold.
#
#   bash scripts/gate_weight_format.sh <spec> <corpus> <out-dir> [ARM ...]
#   ARM = a comma-joined env list, e.g. QUANT_MLP_DTYPE=int4a,QUANT_MLP_GROUP=128
#         The first ARM should be the bf16 control.
#
# Why one run: this box drifts materially within a session, so the quantity is a DELTA against a
# CONTEMPORANEOUS control, never two absolutes taken hours apart. Holding the device across the
# whole set is what makes the control contemporaneous -- re-acquiring per arm invites another
# tenant in between two things you are comparing. Wrap the whole invocation in your single-tenant
# device lock, not each arm.
#
# What it gates, in the order that matters:
#   1. DETERMINISM at temperature 0. The project's stated blocking gate, and the one a format
#      change is supposed to PASS rather than expected to. A weight format is perfectly
#      deterministic and can be perfectly wrong-er, so this is necessary and nowhere near
#      sufficient -- which is exactly why 2 and 3 exist.
#   2. LATENCY, alternated, so the byte cut's CONVERSION is measured rather than assumed. A byte
#      arithmetic prediction is not a result: this project records narrow formats converting at
#      43-55%, so quoting the full byte arithmetic overstates the win roughly twofold.
#   3. QUALITY, paired, on the same corpus and the same positions for every arm. Compare two arms
#      with designs/decode_fused/hostlab/pairwise.py on the emitted .nll.npy files -- NEVER by
#      differencing their two control-relative percentages, which throws away the pairing and has
#      already produced a wrong sign on this exact comparison.
#
# NOT gated here, and it is the one that decides whether a 4-bit format is shippable at all:
# greedy generation. Every 4-bit scheme measured forks from the bf16 trajectory within one or two
# tokens while perplexity separates them by 3-12 points. Run hostlab/divergence.py as well.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: gate_weight_format.sh <spec> <corpus> <out-dir> [ARM ...]}"
CORPUS="${2:?corpus}"; OUT="${3:?out-dir}"; shift 3
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=("QUANT_MLP_DTYPE=bf16")
VENV="${VENV_IRON:-$REPO/.venv-iron}"
# bench_llm_decode.py drives the same graph the generator builds, so it needs the SAME toolchain
# environment build_llm_decode.sh assembles -- without it the harness dies at "[newstack_compat]
# resolved aie is not the pinned instance" before it reaches the device. run_llm_perplexity.sh
# sets this up for itself; the bench is invoked directly, so it is set up here once for both.
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV/bin:$VENV/cc-shim:$AIEBU_ASM_DIR:$PATH"
export CUDA_VISIBLE_DEVICES=""
POS="${GATE_POSITIONS:-1 64 256}"; REPS="${GATE_REPS:-30}"
TOK="${GATE_TOKENS:-2000}"; PASSES="${GATE_PASSES:-2}"
mkdir -p "$OUT"
echo "[gate] $SPEC  arms=${#ARMS[@]}  passes=$PASSES  positions=$POS  ppl-tokens=$TOK"
echo "[gate] power mode: $(xrt-smi examine -r platform 2>/dev/null | grep -i 'power mode' | head -1)"

tag_of() { echo "$1" | tr ',=' '__' | sed 's/QUANT_//g;s/_DTYPE//g;s/_GROUP//g'; }

# 1+2: determinism and latency, ABBA over the arm list so drift lands on every arm equally.
for pass in $(seq 1 "$PASSES"); do
  for arm in "${ARMS[@]}"; do
    t="$(tag_of "$arm")"
    echo "=== bench $t pass$pass ==="
    ( IFS=',' read -ra KVS <<< "$arm"; set -a; for kv in "${KVS[@]}"; do export "${kv?}"; done; set +a
      # shellcheck disable=SC2086  -- POS is a deliberate word list
      "$VENV/bin/python" "$REPO/designs/decode_fused/bench_llm_decode.py" --spec "$SPEC" \
        --weights "$REPO/artifacts/$SPEC/weights" --positions $POS --reps "$REPS" \
        --det-runs 5 --out-json "$OUT/bench-$t-p$pass.json" ) || echo "[gate] bench $t FAILED"
  done
done

# 3: quality, one pass per arm, same corpus and positions for all of them.
for arm in "${ARMS[@]}"; do
  t="$(tag_of "$arm")"
  echo "=== ppl $t ==="
  ( IFS=',' read -ra KVS <<< "$arm"; set -a; for kv in "${KVS[@]}"; do export "${kv?}"; done; set +a
    bash "$REPO/scripts/run_llm_perplexity.sh" "$SPEC" "$CORPUS" "$OUT/ppl-$t" "$TOK" ) \
    || echo "[gate] ppl $t FAILED"
done
echo "[gate] done -- results in $OUT"
echo "[gate] pair the arms:  python designs/decode_fused/hostlab/pairwise.py <tag> A:B"
