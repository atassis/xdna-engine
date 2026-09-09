#!/usr/bin/env bash
# End-to-end reproducibility test for the mlir-aie pinned-submodule + tethered-patch vendoring
# (Task 0; design internal notes-..., record internal notes).
#
# Proves the contract: fresh clone -> `git submodule update --init` resolves the pinned SHA ->
# the tethered patch applies cleanly -> our kernels sync forward -> build_kernels.sh produces the
# encoder xclbins. It runs the REAL setup_kernel_env.sh + build_kernels.sh against a throwaway clone.
#
# Toolchain is REUSED (the existing .venv-iron is symlinked in), so we do NOT re-download the
# ~1.8 GB wheels (per the agreed test scope). By default the mlir-aie submodule is mirrored from
# the local clone for speed; pass --github to instead fetch it from GitHub (exercises that the
# pinned SHA is reachable on the real remote).
set -euo pipefail
ORIG="$(cd "$(dirname "$0")/.." && pwd)"
# The expected gitlink is DERIVED from toolchain.lock, never written here. A literal was wrong
# from 2026-07-11, when 59756ef moved the gitlink in passing and nothing linked the two records --
# so this assertion compared against a June SHA for two months and could only pass by coincidence.
SHA="$(. "$ORIG/toolchain.lock"; echo "$MLIR_AIE_FORK_COMMIT")"
[ -n "$SHA" ] || { echo "FAIL: toolchain.lock has no MLIR_AIE_FORK_COMMIT" >&2; exit 1; }
USE_GITHUB=0; [ "${1:-}" = "--github" ] && USE_GITHUB=1

[ -d "$ORIG/.venv-iron" ] || { echo "FAIL: this test reuses the existing .venv-iron toolchain, which is absent. Run scripts/setup_kernel_env.sh first." >&2; exit 1; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/repro-vendor.XXXXXX")"
cleanup(){ chmod -R u+w "$TMP" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT
echo "== temp workspace: $TMP =="
fail(){ echo "REPRO TEST FAILED: $1" >&2; exit 1; }

echo "== [1/6] local clone of our repo (committed state) =="
git clone --quiet "$ORIG" "$TMP/repo"
cd "$TMP/repo"

echo "== [2/6] reuse toolchain: symlink .venv-iron -> original (skips venv+wheel install) =="
ln -s "$ORIG/.venv-iron" "$TMP/repo/.venv-iron"
# Same reuse, one layer down. XDNA_CACHE now defaults INSIDE the repo, so without this the temp
# clone resolves its own empty .cache and toolchain_up.sh builds a fresh instance from source --
# 1-2 h, and outside this test's agreed scope (it reuses the toolchain, it does not provision one).
export TOOLCHAIN_HOME="$ORIG/.cache/instances"

echo "== [3/6] submodule update --init -> resolve the pinned gitlink =="
if [ "$USE_GITHUB" = 1 ]; then
  echo "   (fetching submodule from GitHub)"
  git submodule update --init mlir-aie
else
  echo "   (mirroring submodule from local clone for speed; use --github for the real fetch)"
  # protocol.file.allow=always: git blocks local-path submodule transport by default
  # (CVE-2022-39253); safe here since the source is our own local clone. The real-world
  # path (--github, https) needs no such override.
  git -c protocol.file.allow=always -c submodule.mlir-aie.url="$ORIG/mlir-aie" \
    submodule update --init mlir-aie
fi
GOT="$(git -C mlir-aie rev-parse HEAD)"
[ "$GOT" = "$SHA" ] && echo "   OK: submodule at pinned SHA $GOT" || fail "submodule SHA $GOT != pinned $SHA"

echo "== [4/6] run the real setup_kernel_env.sh (skips venv/wheels/init via guards; applies patch + syncs) =="
bash scripts/setup_kernel_env.sh
# assert the patch landed on all 3 upstream files
for f in programming_examples/common.cmake \
         programming_examples/basic/matrix_multiplication/common.h \
         programming_examples/ml/layernorm/Makefile; do
  git -C mlir-aie diff --quiet -- "$f" && fail "patch did not modify $f"
done
grep -q 'LOCAL PATCH (CachyOS)' mlir-aie/programming_examples/common.cmake || fail "cmake patch marker missing"
echo "   OK: tethered patch applied to the 3 upstream files"
# assert our kernels synced forward
for k in aie_kernels/aie2p/dwconv1d.cc aie_kernels/aie2p/mm_silu_epilogue.cc \
         programming_examples/ml/dwconv1d/Makefile programming_examples/ml/softmax400/softmax400.py \
         programming_examples/basic/matrix_multiplication/whole_array/whole_array_silu_iron.py; do
  [ -f "mlir-aie/$k" ] || fail "kernel not synced: $k"
done
echo "   OK: custom kernels copied-forward"

echo "== [5/6] build_kernels.sh against the fresh tree (reusing toolchain) =="
# Do NOT let a partial build short-circuit step 6. build_kernels.sh now exits non-zero with a NAMED
# list when some shapes fail rather than dying at the first one, and step 6 is the only thing in
# this tree that asserts WHICH xclbins must exist -- so aborting here threw away the completeness
# check to report a failure step 6 would have described precisely. The build's own failure list is
# already on stderr; step 6 decides the verdict.
bash scripts/build_kernels.sh || echo "   (build reported failures -- step 6 says whether any REQUIRED xclbin is affected)"

echo "== [6/6] assert the encoder xclbins were produced =="
MM=programming_examples/basic/matrix_multiplication
# THE DECLARED SHAPE SET. This is the only place that says which xclbins must EXIST, and it is why
# it is worth keeping wider than feels necessary: neither guard downstream can express completeness.
# `.toolchain-stamp` is per-DIRECTORY and passes on one file being present; `kernel_manifest.json`
# is explicitly descriptive ("cannot assert anything the directory doesn't currently contain"), so
# regenerating it after a partial build simply adopts the smaller reality. On 2026-09-09 the
# installed dir lost seven shapes and every guard reported OK.
#
# Every entry below is CURRENTLY BUILDABLE and was verified present after a from-zero run. The
# K=768 fast tiles (64x32x96, 64x64x96) are deliberately NOT here: they overflow L1 on this
# toolchain AND on the previous one, so listing them would paint the test permanently red for a
# known, separate defect rather than for a regression.
MUST=(
  programming_examples/ml/dwconv1d/build/final.xclbin
  programming_examples/ml/layernorm/build/final.xclbin
  "$MM/whole_array/build/final_512x800x3072_32x32x32_8c_silu.xclbin"
  "$MM/whole_array/build/final_512x3104x768_32x32x32_8c_bias.xclbin"
  programming_examples/ml/softmax400/build/final.xclbin
  # K_aug=800 modal (Whisper-small / Parakeet, d_model 768 + the 32-row bias augment)
  "$MM/whole_array/build/final_512x800x768_64x32x96_8c_modalsilu.xclbin"
  "$MM/whole_array/build/final_512x800x1536_64x32x96_8c_modalsilu.xclbin"
  "$MM/whole_array/build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin"
  "$MM/whole_array/build/final_512x800x3072_64x32x96_8c_modalid.xclbin"
  "$MM/whole_array/build/final_512x800x3072_64x32x96_8c_modalgelu.xclbin"
  # K_aug=1312 (Whisper-turbo, d_model 1280 + 32). ctx2.rs:1583 asserts kaug()==1312; these were
  # absent from every build script until 2026-09-09 and vanished from the live path at a re-pin.
  "$MM/whole_array/build/final_512x1312x1280_32x32x32_8c_modalid.xclbin"
  "$MM/whole_array/build/final_512x1312x5120_32x32x32_8c_modalsilu.xclbin"
  "$MM/whole_array/build/final_512x1312x5120_32x32x32_8c_modalgelu.xclbin"
  # K=1024 modal resident (Parakeet zero-switch encoder) -- the delegated half of the build
  "$MM/whole_array/build/final_512x1024x4096_64x32x128_8c_modalsilu.xclbin"
  "$MM/whole_array/build/final_512x4096x1024_64x32x128_8c_modalid.xclbin"
)
ok=1
for x in "${MUST[@]}"; do
  if [ -f "mlir-aie/$x" ]; then echo "   OK   $x"; else echo "   MISS $x"; ok=0; fi
done
[ "$ok" = 1 ] || fail "one or more expected xclbins missing"
echo
echo "REPRO TEST PASSED — fresh clone reproduces the pinned build (SHA $SHA)."
