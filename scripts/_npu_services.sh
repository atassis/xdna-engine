#!/usr/bin/env bash
# Sourced by the device-timing scripts to quiesce the NPU's single-tenant contenders.
#
# The NPU is single-tenant, so a timed run must be the only thing holding /dev/accel. That guard
# used to be a bare `systemctl --user stop <units> >/dev/null 2>&1` in each script, which failed
# open twice over:
#
#   1. A RENAME voids it at a distance. The scripts named voxd.service; the unit is npu-vox.service.
#      systemd DOES object -- rc=5, "Failed to stop voxd.service: Unit voxd.service not loaded." on
#      stderr -- but `>/dev/null 2>&1` plus a discarded rc threw away both, so the voice daemon ran
#      through timed runs while the script printed "device clear". Measured 2026-08-17: npu-vox was
#      active since 15:13 and still active after a 17:48 run that announced the device clear.
#   2. `fuser /dev/accel/accel0` does not cover it. A voice daemon idling on a hotkey holds no FD,
#      so it passes the device check and can still take the device mid-run. fuser proves the device
#      is free NOW; only stopping the daemon says anything about the next 60 seconds.
#
# So: the unit names live here once, every one is asserted to EXIST before the timed section, and
# the stop is verified rather than assumed. A future rename fails the run instead of silently
# downgrading it.
#
# Usage:
#   . "$(dirname "${BASH_SOURCE[0]}")/_npu_services.sh"
#   trap 'npu_svc_start' EXIT
#   npu_svc_stop
#   npu_svc_require_device_free
#
# Callers may define log(); it is used when present. Override the unit list with NPU_UNITS.

NPU_UNITS=${NPU_UNITS:-"xdna-engine.service npu-vox.service"}
NPU_ENGINE_URL=${NPU_ENGINE_URL:-http://127.0.0.1:11434/}

_npu_log(){ if declare -F log >/dev/null 2>&1; then log "$*"; else echo -e "$*"; fi; }

# Every unit must resolve. This is the check the old one-liner lacked: `is-active` prints
# "inactive" for a nonexistent unit exactly as it does for a stopped one, so only `cat` (or
# list-unit-files) distinguishes "stopped" from "never existed".
npu_svc_assert_units(){
  local u missing=""
  for u in $NPU_UNITS; do
    systemctl --user cat "$u" >/dev/null 2>&1 || missing="$missing $u"
  done
  [ -z "$missing" ] && return 0
  _npu_log "[ERR] unknown systemd unit(s):$missing"
  _npu_log "[ERR] the quiesce guard cannot hold -- a renamed unit is not stopped, and the run"
  _npu_log "[ERR] would time against an uncontrolled device. Fix NPU_UNITS in scripts/_npu_services.sh."
  _npu_log "[ERR] units present: $(systemctl --user list-unit-files --no-legend 'npu*' 'xdna*' 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
  return 1
}

# Stop each unit on its own so one bad name cannot mask another's failure, then confirm.
npu_svc_stop(){
  npu_svc_assert_units || return 1
  local u rc still=""
  for u in $NPU_UNITS; do
    systemctl --user stop "$u"; rc=$?
    [ $rc -eq 0 ] || { _npu_log "[ERR] stop $u failed (rc=$rc)"; return 1; }
  done
  sleep 1
  for u in $NPU_UNITS; do
    [ "$(systemctl --user is-active "$u")" = active ] && still="$still $u"
  done
  [ -n "$still" ] && { _npu_log "[ERR] still active after stop:$still"; return 1; }
  _npu_log "[svc] stopped:$(printf ' %s' $NPU_UNITS) (verified inactive)"
}

npu_svc_start(){
  local u
  for u in $NPU_UNITS; do systemctl --user start "$u" >/dev/null 2>&1 || true; done
  # The engine needs ~4 s here, so poll rather than sleep a guessed constant. Any HTTP code means
  # something answered; curl -w always prints one, so test curl's own rc, not the code string --
  # `code=$(curl ... || echo 000)` concatenates onto the printed 000 and reports a dead engine up.
  local i
  for i in $(seq 1 20); do
    curl -s -o /dev/null --max-time 2 "$NPU_ENGINE_URL" && { _npu_log "[svc] restarted; $NPU_ENGINE_URL answering"; return 0; }
    sleep 1
  done
  _npu_log "[warn] restarted, but $NPU_ENGINE_URL did not answer within 20s"
}

# Necessary but not sufficient on its own -- see the header. Run it AFTER npu_svc_stop.
npu_svc_require_device_free(){
  local dev=${1:-/dev/accel/accel0}
  if fuser "$dev" >/dev/null 2>&1; then
    _npu_log "[ERR] $dev busy after quiesce:"
    _npu_log "$(fuser -v "$dev" 2>&1)"
    return 1
  fi
  _npu_log "[svc] device clear ($dev)"
}
