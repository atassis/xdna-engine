#!/usr/bin/env bash
# Refuse to write build artifacts onto a RAM filesystem. Sourced by the build scripts.
#
# WHY THIS IS A HARD ERROR AND NOT A STYLE NOTE. On this class of box /tmp is a tmpfs -- 16 GB of
# 30 GB total on the dev machine -- so anything written there is held in MEMORY, including whatever
# TMPDIR-less `mktemp -d` picks. A single fused decode artifact is ~1.4 GB and a weight dump is
# ~1.2 GB, so three or four builds silently consume a fifth of RAM. That is the same RAM the model
# needs: Gemma-4-12B decode holds roughly 6.7-8.1 GB of weight BOs resident, and its build was
# already OOM-killed once. A build that competes with the workload it exists to produce should fail
# loudly rather than succeed and slow everything down invisibly.
#
# ALLOW_TMPFS_BUILD=1 overrides, for genuinely small probe builds.
# XDNA_SCRATCH names the disk-backed scratch root (default /mnt/data/xdna/scratch).

_fs_type() { df -PT "$1" 2>/dev/null | awk 'NR==2 {print $2}'; }

require_disk_backed() {
    local path="$1" label="${2:-output directory}" fs
    mkdir -p "$path" 2>/dev/null || true
    fs="$(_fs_type "$path")"
    case "$fs" in
        tmpfs | ramfs)
            {
                echo "ERROR: $label is on $fs, which is RAM: $path"
                echo "  A decode artifact is ~1.4 GB. Building into RAM competes with the weight"
                echo "  buffers the model itself must hold resident (Gemma-4-12B: ~8 GB)."
                echo "  Use a disk-backed path, e.g. \${XDNA_SCRATCH:-/mnt/data/xdna/scratch}/<name>."
                echo "  Set ALLOW_TMPFS_BUILD=1 to override (small probe builds only)."
            } >&2
            [ "${ALLOW_TMPFS_BUILD:-0}" = "1" ] || return 1
            echo "  ALLOW_TMPFS_BUILD=1 set -- proceeding on $fs anyway." >&2
            ;;
    esac
    return 0
}

# A mktemp -d that is not on RAM. Falls back to plain mktemp only when the disk root is unusable,
# and says so -- silently landing back on tmpfs is the failure this file exists to stop.
disk_backed_mktemp() {
    local root="${XDNA_SCRATCH:-/mnt/data/xdna/scratch}"
    if mkdir -p "$root" 2>/dev/null && [ -w "$root" ]; then
        case "$(_fs_type "$root")" in
            tmpfs | ramfs) ;;
            *) mktemp -d "$root/build.XXXXXX"; return 0 ;;
        esac
    fi
    echo "WARN: no disk-backed scratch at $root; intermediates go to $(dirname "$(mktemp -u)")" >&2
    mktemp -d
}
