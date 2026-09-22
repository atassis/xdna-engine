#!/usr/bin/env bash
# Collapse byte-identical artifact buffers onto one inode.
#
#   scripts/dedup_artifacts.sh                    # dry run over the artifact store
#   scripts/dedup_artifacts.sh --apply            # link them
#   scripts/dedup_artifacts.sh --apply <root>...  # other roots
#
# For adopting a tree whose blobs were packed per arm; buffer_blob pools them as it writes,
# so this is a one-off per tree, not a job that keeps up with builds. Measured 2026-09-22:
# 84.5 GB across /mnt/data/xdna/artifacts.

# SCOPE IS `buffers/` ONLY, and that is load-bearing. A link is safe exactly when every writer
# replaces the file rather than truncating it -- buffer_blob.write_blob does, and
# test_buffer_blob.py fails on a writer that does not. `meta.json` and `*.elf` are still
# written in place, so a link there would write through. A filesystem with reflink clones
# instead and needs no such invariant.

set -euo pipefail

STORE="${XDNA_ARTIFACT_STORE:-/mnt/data/xdna/artifacts}"
MIN_SIZE="${DEDUP_MIN_SIZE:-1M}"

apply=0
roots=()
for arg in "$@"; do
    case "$arg" in
        --apply) apply=1 ;;
        -h | --help) sed -n '2,19p' "$0"; exit 0 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) roots+=("$arg") ;;
    esac
done
[ ${#roots[@]} -gt 0 ] || roots=("$STORE")

for r in "${roots[@]}"; do
    [ -d "$r" ] || { echo "ERROR: not a directory: $r" >&2; exit 1; }
done

mapfile -t -d '' dirs < <(find "${roots[@]}" -type d -name buffers -print0)
if [ ${#dirs[@]} -eq 0 ]; then
    echo "no buffers/ directories under: ${roots[*]}"
    exit 0
fi
echo "${#dirs[@]} buffers/ directories under: ${roots[*]}"

# --mount keeps each pass inside one filesystem, since a link cannot cross one and the roots
# may span /mnt and /home. -c compares content alone: an arm's build time is not its identity.
opts=(--content --mount --minimum-size "$MIN_SIZE" --reflink=auto --verbose)
[ "$apply" -eq 1 ] || opts+=(--dry-run)

hardlink "${opts[@]}" "${dirs[@]}"

[ "$apply" -eq 1 ] || echo $'\nDry run. Re-run with --apply to link.'
