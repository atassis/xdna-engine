#!/usr/bin/env bash
# Build batched fused-decode block ELFs (vector-b, plan 2026-06-16-batched-decode-elf.md). Compile-only.
# Mirrors build_gemm_probe.sh env + applies the IRON patches (deep-C + transpose + gemm-fusion-prefix)
# idempotently. Each block generator emits a fused_elf_probe-compatible meta.
#
#   B=128 bash scripts/build_batched_decode.sh ffn         # Task 1: FFN block  -> artifacts/ffn_batched_B<B>
# (later: ln_qkv, decode --layers N — added as those generators land)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="${1:-ffn}"
B="${B:-128}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
IRON="${IRON:-~/repositories/ns/amd/IRON}"
AIEBU_DIR="${AIEBU_DIR:-~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/whisper-small/whisper_decoder}"
GENDIR="$REPO/route_b_kernels/decode_fused"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing"; exit 1; }
[ -d "$IRON/iron" ] || { echo "ERROR: amd/IRON not at $IRON"; exit 1; }
[ -x "$AIEBU_DIR/aiebu-asm" ] || { echo "ERROR: aiebu-asm not at $AIEBU_DIR"; exit 1; }

# apply IRON patches idempotently (deep-C scratchpad + transpose num_batches + GEMM fusion-prefix)
apply_patch(){ local p="$1"; [ -f "$p" ] || return 0
  if git -C "$IRON" apply --reverse --check "$p" >/dev/null 2>&1; then echo "[build] $(basename "$p") already applied"
  else echo "[build] applying $(basename "$p")"; git -C "$IRON" apply "$p"; fi; }
apply_patch "$REPO/patches/amd-IRON-deepc.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-transpose-num-batches.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-gemm-fusion-prefix.patch"
apply_patch "$REPO/route_b_kernels/patches/iron-aiecc-build-perf.patch"  # AIECC_JOBS (-j) + SKIP_EXPAND_PDIS env-gates

export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
export PYTHONPATH="$IRON:$GENDIR${PYTHONPATH:+:$PYTHONPATH}"
# aiecc per-core compile parallelism (the iron-aiecc-jobs patch reads this; default 1 keeps other
# sessions unchanged). 16 cuts the per-core .o phase ~16x on this 20-core box; the final-ELF assembly
# stays single-threaded. Override with AIECC_JOBS=N.
export AIECC_JOBS="${AIECC_JOBS:-16}"
echo "[build] AIECC_JOBS=$AIECC_JOBS (aiecc per-core parallelism)"

case "$WHAT" in
  ffn)
    OUT="$REPO/artifacts/ffn_batched_B${B}"; GEN="$GENDIR/gen_ffn_batched.py"; ARGS="--B $B" ;;
  ln_qkv)
    OUT="$REPO/artifacts/ln_qkv_batched_B${B}"; GEN="$GENDIR/gen_ln_qkv_batched.py"; ARGS="--B $B" ;;
  self_attn)
    OUT="$REPO/artifacts/self_attn_batched_B${B}"; GEN="$GENDIR/gen_self_attn_batched.py"; ARGS="--B $B --S ${S:-64}" ;;
  cross_attn)
    OUT="$REPO/artifacts/cross_attn_batched_B${B}"; GEN="$GENDIR/gen_cross_attn_batched.py"; ARGS="--B $B --T ${T:-128}" ;;
  decode)
    NL="${NL:-2}"; sp_tag=""; [ -n "${SP:-}" ] && sp_tag="_sp"; occ_tag=""; [ -n "${OCC:-}" ] && occ_tag="_occ"
    nopdi_tag=""; [ -n "${SKIP_EXPAND_PDIS:-}" ] && nopdi_tag="_nopdi"
    OUT="$REPO/artifacts/decode_batched_B${B}_L${NL}${sp_tag}${occ_tag}${nopdi_tag}"; GEN="$GENDIR/gen_decode_batched.py"
    ARGS="--B $B --layers $NL --S ${S:-64} --T ${T:-128}"
    [ -n "${SP:-}" ] && ARGS="$ARGS --scratchpad"
    [ -n "${ENG:-}" ] && ARGS="$ARGS --engine-only"
    [ -n "${OCC:-}" ] && ARGS="$ARGS --occ"
    [ -n "${PVAL:-}" ] && ARGS="$ARGS --P $PVAL" ;;
  *) echo "ERROR: unknown block '$WHAT' (have: ffn ln_qkv self_attn cross_attn decode)"; exit 1 ;;
esac

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT; mkdir -p "$OUT"
echo "=== building batched $WHAT B=$B -> $OUT (work=$WORK) ==="
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --weights "$WEIGHTS" $ARGS --out "$OUT" )
echo "[build] done: $OUT  (elf=$(du -h "$OUT"/*.elf | cut -f1))"
