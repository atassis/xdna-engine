#!/usr/bin/env bash
# reset_npu.sh -- recover a wedged AMD XDNA NPU without a reboot.
#
# A hung / timed-out NPU kernel can wedge the hw_context: every later submission
# fails with `DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22)`, and a plain
# `modprobe -r amdxdna && modprobe amdxdna` leaves the PCI device present-but-
# UNBOUND with /dev/accel/accel0 gone. This does the full PCI remove + rescan +
# driver reload in one atomic sequence, which forces a clean firmware re-init.
#
#   sudo bash scripts/reset_npu.sh
#
# If it does NOT restore /dev/accel/accel0, a reboot is the guaranteed fix.
# Root cause + the proper driver-side fix are tracked in the KB task
# `xdna-driver-hung-context-recovery`.
set -uo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0" >&2; exit 1; }

# Auto-detect the NPU PCI address (AMD XDNA NPU = PCI device 1022:17f0).
PCI=$(lspci -Dn 2>/dev/null | awk '$3=="1022:17f0"{print $1}' | head -1)
[ -n "$PCI" ] || PCI=$(lspci -D 2>/dev/null | awk '/Neural Processing Unit/{print $1}' | head -1)
[ -n "$PCI" ] || { echo "could not locate the NPU PCI device" >&2; exit 1; }
echo "NPU PCI device: $PCI"

echo "[1/5] unload amdxdna (unbinds the device)"
modprobe -r amdxdna 2>/dev/null || true
echo "[2/5] remove the PCI device"
[ -e "/sys/bus/pci/devices/$PCI/remove" ] && echo 1 > "/sys/bus/pci/devices/$PCI/remove" || true
sleep 1
echo "[3/5] rescan PCI (re-enumerate the device fresh)"
echo 1 > /sys/bus/pci/rescan
sleep 2
echo "[4/5] reload amdxdna (probe + bind the fresh device)"
modprobe amdxdna 2>/dev/null || true
sleep 2
echo "[5/5] verify"
if [ -e /dev/accel/accel0 ]; then
  echo "RECOVERED: /dev/accel/accel0 is back"
  exit 0
else
  echo "NOT recovered -- the firmware wedge needs a reboot (see xdna-driver-hung-context-recovery)" >&2
  exit 1
fi
