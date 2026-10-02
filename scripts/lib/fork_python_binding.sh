#!/usr/bin/env bash

_fork_python_binding_error() {
  echo "[fork_python_binding] ERROR: $*" >&2
  return 1
}

bind_fork_python() {
  if [ "$#" -ne 2 ]; then
    _fork_python_binding_error "usage: bind_fork_python <venv-python> <instance>"
    return 1
  fi

  local python="$1" instance="$2" prefix base_prefix site instance_root fork_python resolved tmp
  local -a _fork_python_binding_env
  case "$instance" in
    /*) ;;
    *) _fork_python_binding_error "instance path must be absolute: $instance"; return 1 ;;
  esac
  case "$instance" in
    *$'\n'*|*$'\r'*) _fork_python_binding_error "unsafe instance path"; return 1 ;;
  esac
  [ -x "$python" ] || { _fork_python_binding_error "Python interpreter is not executable: $python"; return 1; }

  readarray -t _fork_python_binding_env < <(
    "$python" -c 'import sys, sysconfig; print(sys.prefix); print(sys.base_prefix); print(sysconfig.get_paths()["purelib"])'
  ) || { _fork_python_binding_error "cannot query Python environment: $python"; return 1; }
  prefix="${_fork_python_binding_env[0]:-}"
  base_prefix="${_fork_python_binding_env[1]:-}"
  site="${_fork_python_binding_env[2]:-}"
  [ -n "$prefix" ] && [ "$prefix" != "$base_prefix" ] \
    || { _fork_python_binding_error "Python interpreter is not a virtual environment: $python"; return 1; }
  [ -d "$site" ] || { _fork_python_binding_error "venv site-packages is missing: $site"; return 1; }
  case "$site" in
    "$prefix"/lib/python*/site-packages|"$prefix"/Lib/site-packages) ;;
    *) _fork_python_binding_error "venv site-packages escapes its environment: $site"; return 1 ;;
  esac

  instance_root="$(realpath -e -- "$instance")" \
    || { _fork_python_binding_error "missing fork Python package: $instance/python/aie/iron"; return 1; }
  fork_python="$instance_root/python"
  [ -f "$fork_python/aie/iron/__init__.py" ] \
    || { _fork_python_binding_error "missing fork Python package: $fork_python/aie/iron"; return 1; }

  tmp="$(mktemp "$site/.aie.pth.XXXXXX")" || { _fork_python_binding_error "cannot create $site/aie.pth"; return 1; }
  printf '%s\n' "$fork_python" > "$tmp"
  if ! cmp -s "$tmp" "$site/aie.pth" 2>/dev/null; then
    mv -f "$tmp" "$site/aie.pth"
  else
    rm -f "$tmp"
  fi

  resolved="$(env -u PYTHONPATH "$python" -c 'import pathlib, aie.iron; print(pathlib.Path(aie.iron.__file__).resolve())' 2>&1)" \
    || { _fork_python_binding_error "fork import verification failed: $resolved"; return 1; }
  case "$resolved" in
    "$fork_python"/aie/iron/*) ;;
    *) _fork_python_binding_error "fork import resolved outside selected instance: $resolved"; return 1 ;;
  esac
}
