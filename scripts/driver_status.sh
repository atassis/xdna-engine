#!/usr/bin/env bash
# driver_status.sh -- is the running amdxdna what driver.lock says, and is a bump worth your attention?
#
# Prints ONE verdict and, when there is something to do, the exact command to run. It never installs
# anything: the install needs root, so this hands the command over rather than escalating.
#
#   NO-CHANGE  pin == upstream tip. Nothing to do, nothing to read.
#   MINOR      new commits, but none touch the hardware path we run. Bump without thinking about it.
#   REVIEW     commits touch our aie2/npu4 path or the ioctl ABI. Read the summary before bumping.
#
# WHY the split: the tree took 106 commits in 90 days, most of them for hardware that is not ours.
# We are PCI 0x17f0:0x10 -> dev_npu4_info, so aie4_*, npu1/3/5/6_regs.c and ve2 changes cannot reach
# us. Treating every commit as significant would make the report noise, and noise gets ignored.
# The ABI header moved 7 times in the same window, which is why it is called out separately: that is
# the one change class that can break userspace (XRT) rather than just the module.
#
# Usage:
#   scripts/driver_status.sh            # assess; print verdict + the command to run
#   scripts/driver_status.sh --verbose  # also list the commits in each bucket
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
DRV="${XDNA_DRIVER_SRC:-$WS/xdna-driver}"
VERBOSE=0; [ "${1:-}" = "--verbose" ] && VERBOSE=1

log() { echo "[driver_status] $*"; }

# The handover. Everything up to pacman is unprivileged and I can run it; the install and the module
# reload need root, so they are printed for you rather than escalated. Uses AMD's own Arch packaging
# (build/arch/PKGBUILD-amdxdna-driver), which is a DKMS package -- so it rebuilds itself on every
# future kernel update and pacman can uninstall it cleanly.
emit_command() {
  local drv="$DRV" jobs; jobs=$(( $(nproc) / 2 )); [ "$jobs" -lt 1 ] && jobs=1
  cat <<EOF

  --- build + package (no root; I can run these for you) ---
  cd $drv && ./build/build.sh -release -j $jobs
  cd $drv/build/arch && XDNA_BUILD_DIR=../Release makepkg -f

  --- install + load (needs root; run these yourself) ---
  sudo pacman -U $drv/build/arch/amdxdna-driver-*.pkg.tar.zst
  sudo modprobe -r amdxdna && sudo modprobe amdxdna

  --- verify (no root) ---
  $REPO/scripts/driver_status.sh          # expect: loaded module -> dkms (ours)
  xrt-smi examine                         # expect: the NPU still enumerates

  --- rollback, if the NPU misbehaves ---
  sudo pacman -R amdxdna-driver && sudo modprobe -r amdxdna && sudo modprobe amdxdna
  # DKMS restores the in-tree module by removing the updates/ copy depmod was preferring.
EOF
}
lock_field() { sed -n "s/^$1=\([^ #]*\).*/\1/p" "$REPO/driver.lock" | head -1; }

[ -f "$REPO/driver.lock" ] || { echo "no driver.lock at $REPO" >&2; exit 2; }
[ -d "$DRV/.git" ]         || { echo "no xdna-driver checkout at $DRV (set XDNA_DRIVER_SRC)" >&2; exit 2; }

PIN="$(lock_field XDNA_DRIVER_COMMIT)"
CARRIES="$(lock_field XDNA_DRIVER_CARRIES)"
PIN_KERNEL="$(lock_field XDNA_DRIVER_KERNEL)"
RUNNING_KERNEL="$(uname -r)"

# --- what is actually loaded ------------------------------------------------------------------
KO="$(modinfo -n amdxdna 2>/dev/null || true)"
case "$KO" in
  */updates/*) SOURCE="dkms (ours)" ;;
  "")          SOURCE="not installed" ;;
  *)           SOURCE="in-tree (distro)" ;;
esac
log "running kernel : $RUNNING_KERNEL"
log "loaded module  : ${KO:-<none>}  -> $SOURCE"
log "pinned commit  : ${PIN:0:12}${CARRIES:+  + carries: $CARRIES}"

# --- how far is upstream ahead of the pin ------------------------------------------------------
git -C "$DRV" fetch origin --quiet 2>/dev/null || log "WARN fetch failed; comparing against stale refs"
TIP="$(git -C "$DRV" rev-parse origin/main)"
if ! git -C "$DRV" cat-file -e "$PIN" 2>/dev/null; then
  log "VERDICT REVIEW -- pinned commit $PIN is not in this checkout (force-push upstream, or a bad pin)"
  exit 1
fi
AHEAD="$(git -C "$DRV" rev-list --count "$PIN..$TIP")"

if [ "$AHEAD" = "0" ]; then
  if [ "$SOURCE" = "dkms (ours)" ]; then
    log "VERDICT NO-CHANGE -- pin is at upstream tip and our module is installed. Nothing to do."
    exit 0
  fi
  log "VERDICT INSTALL -- pin is current, but the running module is the $SOURCE one."
  emit_command; exit 0
fi

# --- classify the delta -------------------------------------------------------------------------
# OURS: the shared core plus the aie2/npu4 path we actually execute.
# OTHER-HW: generations we do not have. A change there cannot reach our device.
ours=0; otherhw=0; abi=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    */uapi/*amdxdna_accel.h)                     abi=$((abi+1));     ours=$((ours+1)) ;;
    */aie4_*|*/npu1_regs.c|*/npu3_regs.c|*/npu5_regs.c|*/npu6_regs.c|*ve2*|*aie2ps*)
                                                 otherhw=$((otherhw+1)) ;;
    drivers/accel/amdxdna/*)                     ours=$((ours+1)) ;;
    *)                                           : ;;   # userspace shim, tests, docs: not the module
  esac
done < <(git -C "$DRV" diff --name-only "$PIN..$TIP")

echo
log "upstream is $AHEAD commits ahead of the pin:"
log "  touches our aie2/npu4 path : $ours file(s)"
log "  other hardware only        : $otherhw file(s)"
log "  ioctl ABI (amdxdna_accel.h): $abi file(s)"

if [ "$VERBOSE" = "1" ]; then
  echo; git -C "$DRV" log --format='    %h %cs %s' "$PIN..$TIP" | head -40
fi

echo
if [ "$abi" -gt 0 ]; then
  log "VERDICT REVIEW -- the ioctl ABI changed. That is the one class that can break XRT userspace,"
  log "                 not just the module. Read the ABI diff before bumping:"
  log "                 git -C $DRV diff $PIN..$TIP -- '*uapi*amdxdna_accel.h'"
elif [ "$ours" -gt 0 ]; then
  log "VERDICT REVIEW -- $ours file(s) on the path we execute. Skim them, then bump:"
  log "                 git -C $DRV log --oneline $PIN..$TIP -- drivers/accel/amdxdna/"
else
  log "VERDICT MINOR -- nothing on our hardware path. Bump without reading."
fi
emit_command
