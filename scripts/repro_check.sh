#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build one aie.mlir twice, in different directories with different --tmpdir names, and compare
# every top-level output byte for byte. Exit 0 = all identical, 1 = a difference, 2 = usage.
# usage: repro_check.sh <aie.mlir> <out-dir> -- <aiecc flags...>
set -uo pipefail
src="${1:?usage: repro_check.sh <aie.mlir> <out-dir> -- <aiecc flags...>}"; out="${2:?}"
shift 2; [ "${1:-}" = "--" ] && shift
[ -n "${PEANO_INSTALL_DIR:-}" ] || { echo "repro_check: set PEANO_INSTALL_DIR (aiecc otherwise takes Peano tools from \$PATH)" >&2; exit 2; }
aiecc="${AIECC_BIN:-${AIECC_PATH:?set AIECC_BIN or AIECC_PATH}}"
rm -rf "$out"; mkdir -p "$out/a" "$out/b"
for r in a b; do
  cp "$src" "$out/$r/aie.mlir"
  ( cd "$out/$r" && "$aiecc" aie.mlir --tmpdir="tmp_$r" "$@" > build.log 2>&1 ) \
    || { echo "repro_check: build $r failed, see $out/$r/build.log" >&2; exit 2; }
done
status=0
for f in $(cd "$out/a" && find . -maxdepth 1 -type f ! -name build.log ! -name aie.mlir -printf '%f\n' | sort); do
  if [ ! -f "$out/b/$f" ]; then echo "MISSING $f"; status=1; continue; fi
  n=$(cmp -l "$out/a/$f" "$out/b/$f" 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ] && cmp -s "$out/a/$f" "$out/b/$f"; then
    echo "IDENTICAL $f"
  else
    echo "DIFF $f $n bytes of $(stat -c %s "$out/a/$f"), first at: $(cmp -l "$out/a/$f" "$out/b/$f" 2>/dev/null | awk 'NR<=8{printf "0x%x ", $1-1}')"
    status=1
  fi
done
exit "$status"
