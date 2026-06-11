#!/usr/bin/env bash
# End-to-end reproducibility test for the mlir-aie pinned-submodule + tethered-patch vendoring
# (Task 0; design internal notes-..., record internal notes).
#
# Proves the contract: fresh clone -> `git submodule update --init` resolves the pinned SHA ->
# the tethered patch applies cleanly -> our kernels sync forward -> build_kernels.sh produces the
# encoder xclbins. It runs the REAL setup_route_b.sh + build_kernels.sh against a throwaway clone.
#
# Toolchain is REUSED (the existing .venv-iron is symlinked in), so we do NOT re-download the
# ~1.8 GB wheels (per the agreed test scope). By default the mlir-aie submodule is mirrored from
# the local clone for speed; pass --github to instead fetch it from GitHub (exercises that the
# pinned SHA is reachable on the real remote).
set -euo pipefail
ORIG="$(cd "$(dirname "$0")/.." && pwd)"
SHA=8373e49165649644f1ec414c2e406c0abbbf51cf
USE_GITHUB=0; [ "${1:-}" = "--github" ] && USE_GITHUB=1

[ -d "$ORIG/.venv-iron" ] || { echo "FAIL: this test reuses the existing .venv-iron toolchain, which is absent. Run scripts/setup_route_b.sh first." >&2; exit 1; }

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

echo "== [4/6] run the real setup_route_b.sh (skips venv/wheels/init via guards; applies patch + syncs) =="
bash scripts/setup_route_b.sh
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
bash scripts/build_kernels.sh

echo "== [6/6] assert the encoder xclbins were produced =="
MM=programming_examples/basic/matrix_multiplication
MUST=(
  programming_examples/ml/dwconv1d/build/final.xclbin
  programming_examples/ml/layernorm/build/final.xclbin
  "$MM/whole_array/build/final_512x800x3072_32x32x32_8c_silu.xclbin"
  "$MM/whole_array/build/final_512x3104x768_32x32x32_8c_bias.xclbin"
  programming_examples/ml/softmax400/build/final.xclbin
)
ok=1
for x in "${MUST[@]}"; do
  if [ -f "mlir-aie/$x" ]; then echo "   OK   $x"; else echo "   MISS $x"; ok=0; fi
done
[ "$ok" = 1 ] || fail "one or more expected xclbins missing"
echo
echo "REPRO TEST PASSED — fresh clone reproduces the pinned build (SHA $SHA)."
