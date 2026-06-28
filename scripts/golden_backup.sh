#!/usr/bin/env bash
# Back up known-good xclbins OUTSIDE the gitignored build dir + record their sha in the manifest, so a
# working artifact can never be lost to an in-place overwrite again.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${GOLDEN_HOME:-$HOME/.cache/xdna2-build/goldens}"; mkdir -p "$DST"
WB="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
MAN="$REPO/artifacts/goldens/manifest.tsv"
for x in final_512x800x3072_64x32x96_8c_modalsilu.xclbin \
         final_512x800x3072_64x32x96_8c_modalgelu.xclbin; do
  [ -f "$WB/$x" ] || { echo "missing $x"; exit 1; }
  cp -f "$WB/$x" "$DST/$x"
  sha="$(sha256sum "$DST/$x" | cut -c1-16)"
  sed -i "s|^$x\t[^\t]*|$x\t$sha|" "$MAN"
  echo "backed up $x ($sha)"
done
