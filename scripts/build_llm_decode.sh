#!/usr/bin/env bash
# Build a decoder-LLM fused decode ELF from an LlmSpec (designs/decode_fused/gen_llm_decode.py).
# Compile-only, no NPU needed.
#
#   bash scripts/build_llm_decode.sh <spec> [LAYERS] [OUT_DIR]
#     spec    qwen3-0.6b | gemma3-270m
#     LAYERS  default: the spec's full depth
#
# Env overrides: VENV_IRON, IRON (an IRON checkout carrying iron/common/fusion.py), WEIGHTS.
# NOTE the shared workspace IRON checkout is NOT usable by default: its local integration-stack has
# dropped the carried iron/common/fusion.py (upstream deleted it; we carry it). Point IRON at a
# worktree of origin/integration-stack instead of editing the shared checkout.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
SPEC="${1:?usage: build_llm_decode.sh <spec> [LAYERS] [OUT]}"
LAYERS="${2:-}"
OUT="${3:-$REPO/artifacts/$SPEC/decode${LAYERS:+_l$LAYERS}}"
. "$REPO/scripts/require_disk_backed.sh"
require_disk_backed "$OUT" "OUT (the built artifact)" || exit 1
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"        # -> IRON_DIR, AIEBU_ASM_DIR (relocatable; env-overridable)
IRON="${IRON:-$IRON_DIR}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/$SPEC/weights}"
GEN="$REPO/designs/decode_fused/gen_llm_decode.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: no iron venv at $VENV_IRON"; exit 1; }
# Gate on the API SURFACE gen_llm_decode.py imports, not a branch name or a worktree path. This
# used to default IRON to a wt-iron-qwen3 worktree and require the DELETED iron/common/fusion.py,
# which tethered the whole LLM decode build to one side branch; the generator is now on
# OperatorSequence like the other 15.
iron_require_pin || exit 1
# Gate on the modules gen_llm_decode.py ACTUALLY IMPORTS at module scope. The first two are the
# API surface the generator was ported to; the last two are operators that exist only on the
# integration stack, and WITHOUT THEM LISTED this gate passed against a checkout that then died at
# `ModuleNotFoundError: No module named 'iron.common.quant'`. A gate whose purpose is to
# fail early, failing late, on a message naming a Python module rather than a mis-pointed IRON_DIR.
iron_at="$(iron_require_api "gen_llm_decode.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/operators/strided_copy/op.py:output_offset_parameter" \
  "iron/operators/tmatvec/op.py:class TMatVec" \
  "iron/common/quant.py:def quantize_weight" \
  "iron/operators/qkv_head_dp/op.py:class QKVHeadDataParallel")" || exit 1
# WINDOW_RUNGS needs an API surface that the DEFAULT IRON_DIR does not have, so it is gated
# separately rather than added to the list above -- requiring it unconditionally would break every
# decode build against an IRON without it, including the rung-free default this arm leaves inert.
# Without this the failure is a late TypeError naming a kwarg, which is the same shape that broke
# every decode build on 2026-09-10 when window_parameter was handed through unconditionally.
if [ -n "${WINDOW_RUNGS:-}" ]; then
  iron_require_api "gen_llm_decode.py (WINDOW_RUNGS=$WINDOW_RUNGS)" \
    "iron/common/sequence.py:extra_runlists" >/dev/null || exit 1
fi
echo "[build] IRON on $iron_at (API surface verified)"
[ -d "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS (run scripts/dump_llm_weights.py)"; exit 1; }

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc missing at $AIECC_PATH"; exit 1; }

# IRON writes build/ intermediates under CWD, and IRON's own cache is mtime-vs-dependencies
# (iron/common/compilation/base.py::is_available_in_filesystem). A fresh mktemp per build threw
# that cache away every time: MEASURED, a rebuild of an already-built arm takes 41 s against
# roughly 4 min cold, ~6x, producing a byte-size identical ELF.
#
# The isolation the temp dir provided is OBSOLETE, and sequence_name()'s own docstring says why --
# "Isolating the build dir hides the collision; naming the arm removes it". Every graph-changing
# knob is now in the artifact name, so two arms cannot share a filename and a shared dir is safe.
#
# KEEP_WORK still pins a specific dir: the fused .mlir is the only input decode_ddr_bytes.py takes,
# and the shipped artifact does not carry the shim BDs, so a byte census needs a known location.
if [ -n "${KEEP_WORK:-}" ]; then
    WORK="$KEEP_WORK"; mkdir -p "$WORK"; echo "[build] keeping intermediates in $WORK"
else
    WORK="${BUILD_CACHE:-${XDNA_CACHE:-/mnt/data/xdna/cache}/llm-build/$SPEC}"
    mkdir -p "$WORK"
fi
# Serialise builds sharing one dir. The cache is shared BY DESIGN (kernel objects do not depend on
# the arm), and this box runs more than one session against this checkout -- two concurrent builds
# in one dir race on the same .o files rather than merely duplicating work. flock makes the second
# wait instead of corrupting the first; the temp-dir version never needed this because it never
# shared anything.
exec 9>"$WORK/.build.lock"
if ! flock -n 9; then echo "[build] another build holds $WORK -- waiting"; flock 9; fi
mkdir -p "$OUT"
echo "[build] spec=$SPEC layers=${LAYERS:-full} inst=$(basename "$INST") iron=$(basename "$IRON")"
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --spec "$SPEC" --weights "$WEIGHTS" \
    --out "$OUT" ${LAYERS:+--layers $LAYERS} ${GEN_EXTRA:-} )
