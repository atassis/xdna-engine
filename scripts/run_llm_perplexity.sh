#!/usr/bin/env bash
# Run the teacher-forced perplexity gate on device, with the SAME toolchain environment
# build_llm_decode.sh sets up.
#
#   bash scripts/run_llm_perplexity.sh <spec> <corpus> <out-prefix> [MAX_TOKENS]
#
# <corpus> is UTF-8 text, or a tokenize_corpus.py `.ids.json` -- taken as --ids when the name ends
# .json, because QwenBPE cannot read Gemma's tokenizer. LAYERS, MAX_SEQ, REF_NLL and REF_META pass
# through to the harness; REF_NLL makes the run print the paired verdict instead of a bare number.
#
# The arm is selected by gen_llm_decode.py's env flags (QUANT_MLP_DTYPE, QUANT_MLP_GROUP, ...), or
# by WEIGHTS alone when the dump is pre-packed -- its quant.json is then the authority and the plan
# cannot re-choose. A caller runs this once per arm with the same corpus and --max-tokens; the arms
# share positions and the comparison is PAIRED. designs/decode_fused/hostlab/pairwise.py does the
# test on the emitted .nll.npy files -- do NOT compare two arms by differencing their
# control-relative percentages.
#
# This exists because the harness needs the pinned instance on PYTHONPATH and dies at
# `[newstack_compat] resolved aie is not the pinned instance` without it, and every recorded run of
# it so far was assembled by hand in a throwaway worktree.
#
# The NPU is single-tenant. Wrap the WHOLE sequence of arms in your device lock, not each arm --
# holding it across the set is what makes the control contemporaneous, and re-acquiring per arm
# invites another tenant in between two things you are comparing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
SPEC="${1:?usage: run_llm_perplexity.sh <spec> <corpus> <out-prefix> [MAX_TOKENS]}"
CORPUS="${2:?corpus}"
OUTP="${3:?out-prefix}"
MAXTOK="${4:-2000}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/$SPEC/weights}"
[ -d "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS"; exit 1; }
[ -r "$CORPUS" ] || { echo "ERROR: cannot read corpus $CORPUS"; exit 1; }
case "$CORPUS" in *.json) CORPUS_FLAG=--ids ;; *) CORPUS_FLAG=--text ;; esac
EXTRA=()
[ -n "${LAYERS:-}" ]   && EXTRA+=(--layers "$LAYERS")
[ -n "${MAX_SEQ:-}" ]  && EXTRA+=(--max-seq "$MAX_SEQ")
[ -n "${REF_NLL:-}" ]  && EXTRA+=(--ref-nll "$REF_NLL")
[ -n "${REF_META:-}" ] && EXTRA+=(--ref-meta "$REF_META")

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$(dirname "$OUTP")"
echo "[ppl] spec=$SPEC inst=$(basename "$INST") iron=$(basename "$IRON") tokens=$MAXTOK"
echo "[ppl] arm: QUANT_MLP_DTYPE=${QUANT_MLP_DTYPE:-bf16} g=${QUANT_MLP_GROUP:-128}" \
     "QUANT_HEAD_DTYPE=${QUANT_HEAD_DTYPE:-bf16} QUANT_ATTN_DTYPE=${QUANT_ATTN_DTYPE:-bf16}"
exec "$VENV_IRON/bin/python" "$REPO/designs/decode_fused/eval_llm_perplexity.py" \
    --spec "$SPEC" --weights "$WEIGHTS" "$CORPUS_FLAG" "$CORPUS" --max-tokens "$MAXTOK" \
    "${EXTRA[@]}" --dump-nll "$OUTP.nll.npy" --out-json "$OUTP.json"
