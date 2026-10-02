#!/usr/bin/env bash
# CPU-only regression coverage for the fork-only `aie` import binding.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
FAIL=0
TEMP_DIRS=()

fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }
cleanup() { rm -rf "${TEMP_DIRS[@]}"; }
trap cleanup EXIT

site_packages() {
  "$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}

import_origin() {
  env -u PYTHONPATH "$1" -c 'import pathlib, aie.iron; print(pathlib.Path(aie.iron.__file__).resolve())'
}

fixture() {
  local t="$1" site
  "$PYTHON" -m venv "$t/venv"
  site="$(site_packages "$t/venv/bin/python")"
  mkdir -p "$t/instance/python/aie/iron" "$t/wheel/aie/iron" "$site"
  printf 'SOURCE = "fork"\n' > "$t/instance/python/aie/iron/__init__.py"
  printf 'SOURCE = "wheel"\n' > "$t/wheel/aie/iron/__init__.py"
  printf '%s\n' "$t/wheel" > "$site/aie.pth"
}

test_binds_the_selected_fork_over_the_conflicting_wheel() {
  local t before after site first second
  t="$(mktemp -d)"
  TEMP_DIRS+=("$t")
  fixture "$t"
  before="$(import_origin "$t/venv/bin/python")"
  if [ "$before" = "$t/wheel/aie/iron/__init__.py" ]; then
    pass "fixture naked import selects conflicting wheel before binding"
  else
    fail "fixture did not reproduce wheel import: $before"
    return
  fi

  # shellcheck source=/dev/null
  . "$ROOT/scripts/lib/fork_python_binding.sh"
  bind_fork_python "$t/venv/bin/python" "$t/instance"
  site="$(site_packages "$t/venv/bin/python")"
  after="$(import_origin "$t/venv/bin/python")"
  if [ "$after" = "$t/instance/python/aie/iron/__init__.py" ]; then
    pass "naked venv import selects the selected fork instance"
  else
    fail "binding did not select fork import: $after"
  fi
  if [ "$(cat "$site/aie.pth")" = "$t/instance/python" ]; then
    pass "binding writes the canonical fork path in the venv site-packages"
  else
    fail "binding did not replace aie.pth with the fork path"
  fi

  first="$(cat "$site/aie.pth")"
  bind_fork_python "$t/venv/bin/python" "$t/instance/../instance"
  second="$(cat "$site/aie.pth")"
  if [ "$first" = "$second" ]; then
    pass "canonical source paths make the binding idempotent"
  else
    fail "second binding changed the site binding"
  fi
}

test_rejects_missing_unsafe_and_nonvenv_targets() {
  local t result base_python
  t="$(mktemp -d)"
  TEMP_DIRS+=("$t")
  fixture "$t"
  # shellcheck source=/dev/null
  . "$ROOT/scripts/lib/fork_python_binding.sh"

  if bind_fork_python "$t/venv/bin/python" "$t/missing" >"$t/out" 2>&1; then
    fail "missing instance was accepted"
  elif grep -Fq "missing fork Python package" "$t/out"; then
    pass "missing instance fails loud"
  else
    fail "missing instance failure lacked cause: $(cat "$t/out")"
  fi

  if bind_fork_python "$t/venv/bin/python" "$t/instance"$'\nimport sys; sys.exit(0)' >"$t/out" 2>&1; then
    fail "unsafe instance path was accepted"
  elif grep -Fq "unsafe instance path" "$t/out"; then
    pass "unsafe instance path fails before writing a .pth"
  else
    fail "unsafe path failure lacked cause: $(cat "$t/out")"
  fi

  base_python="$("$PYTHON" -c 'import sys; print(sys._base_executable)')"
  if bind_fork_python "$base_python" "$t/instance" >"$t/out" 2>&1; then
    fail "non-venv interpreter was accepted"
  elif grep -Fq "virtual environment" "$t/out"; then
    pass "binding refuses a site-packages outside its virtual environment"
  else
    fail "non-venv failure lacked cause: $(cat "$t/out")"
  fi
}

test_toolchain_uses_its_captured_instance_and_export_venv_stays_separate() {
  local source bootstrap export_env
  source="$(cat "$ROOT/scripts/toolchain_up.sh")"
  bootstrap="$(cat "$ROOT/scripts/bootstrap_public_env.sh")"
  export_env="$(cat "$ROOT/scripts/setup_export_venv.sh")"
  if [[ "$source" == *'bind_fork_python "$REPO/.venv-iron/bin/python" "$INST"'* ]]; then
    pass "toolchain binding consumes the captured instance value"
  else
    fail "toolchain binding does not consume its captured instance value"
  fi
  if [[ "$bootstrap" == *'scripts/toolchain_up.sh'* && "$bootstrap" == *'scripts/setup_export_venv.sh'* ]]; then
    pass "public bootstrap resolves the fork toolchain before creating export consumers"
  else
    fail "bootstrap environment order changed"
  fi
  if [[ "$export_env" != *'PYTHONPATH='* && "$export_env" != *'import aie'* ]]; then
    pass "export venv remains separate from AIE Python resolution"
  else
    fail "export venv unexpectedly owns AIE Python resolution"
  fi
}

test_binds_the_selected_fork_over_the_conflicting_wheel
test_rejects_missing_unsafe_and_nonvenv_targets
test_toolchain_uses_its_captured_instance_and_export_venv_stays_separate
exit "$FAIL"
