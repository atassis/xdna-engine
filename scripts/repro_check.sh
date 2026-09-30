#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build one aie.mlir twice, in different directories with different --tmpdir names, and compare
# every output file (recursively, excluding the tmpdirs) byte for byte. Exit 0 = all identical,
# 1 = a difference, 2 = usage error or a build that failed.
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
while IFS= read -r -d '' f; do
  if [ ! -f "$out/b/$f" ]; then echo "MISSING $f"; status=1; continue; fi
  if cmp -s "$out/a/$f" "$out/b/$f"; then
    echo "IDENTICAL $f"
  elif [ "$(stat -c %s "$out/a/$f")" != "$(stat -c %s "$out/b/$f")" ]; then
    echo "DIFF $f sizes differ ($(stat -c %s "$out/a/$f") vs $(stat -c %s "$out/b/$f"))"
    status=1
  else
    echo "DIFF $f $(cmp -l "$out/a/$f" "$out/b/$f" | wc -l) bytes of $(stat -c %s "$out/a/$f"), first at: $(cmp -l "$out/a/$f" "$out/b/$f" | awk 'NR<=8{printf "0x%x ", $1-1}')"
    status=1
  fi
done < <(cd "$out/a" && find . -path ./tmp_a -prune -o -type f ! -name build.log ! -name aie.mlir -printf '%P\0' | sort -z)
exit "$status"
