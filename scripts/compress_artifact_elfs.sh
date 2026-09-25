#!/usr/bin/env bash
# Add `<name>.elf.zst` beside every `*.elf` under the given artifact dirs, without a rebuild.
# The plain `.elf` is left in place -- this only ADDS the compressed sibling.
#
#   scripts/compress_artifact_elfs.sh <artifact_dir> [<artifact_dir>...]
#
# Each `.elf.zst` is decompressed right back and compared byte-for-byte against the source before
# being counted as a success; a mismatch removes the `.zst` and fails the run rather than leaving
# a bad compressed copy next to a good plain one.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  sed -n '2,9p' "$0"
  exit 2
fi
command -v zstd >/dev/null || { echo "ERROR: zstd not on PATH" >&2; exit 1; }

ok=0
skip=0
fail=0

for root in "$@"; do
  [ -d "$root" ] || { echo "ERROR: not a directory: $root" >&2; fail=$((fail + 1)); continue; }
  while IFS= read -r -d '' elf; do
    zst="$elf.zst"
    if [ -e "$zst" ]; then
      skip=$((skip + 1))
      continue
    fi
    tmp="$zst.tmp.$$"
    if ! zstd -q -3 --long=27 -o "$tmp" "$elf"; then
      echo "FAIL compress: $elf" >&2
      rm -f "$tmp"
      fail=$((fail + 1))
      continue
    fi
    # Byte-identity, not just "zstd exited 0": decompress the just-written .zst and diff it
    # against the source before this counts as done.
    if ! cmp -s <(zstd -dc "$tmp") "$elf"; then
      echo "FAIL roundtrip mismatch: $elf" >&2
      rm -f "$tmp"
      fail=$((fail + 1))
      continue
    fi
    mv "$tmp" "$zst"
    before=$(stat -c%s "$elf")
    after=$(stat -c%s "$zst")
    LC_NUMERIC=C printf 'ok %s (%d -> %d, %.1fx)\n' "$elf" "$before" "$after" \
      "$(LC_NUMERIC=C awk "BEGIN{print $before/$after}")"
    ok=$((ok + 1))
  done < <(find "$root" -type f -name '*.elf' -print0)
done

echo "compress_artifact_elfs: $ok compressed, $skip already had .elf.zst, $fail failed"
[ "$fail" -eq 0 ]
