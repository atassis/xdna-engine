#!/usr/bin/env bash

init_source_aiebu() {
  local source="$1" entry
  entry="$(git -C "$source" ls-tree HEAD -- third_party/aiebu)" || return 1
  [ -n "$entry" ] || return 0
  case "$entry" in
    "160000 commit "*) ;;
    *) echo "ERROR: third_party/aiebu is not a pinned submodule" >&2; return 1 ;;
  esac
  if [ -L "$source/third_party/aiebu" ]; then
    echo "ERROR: refusing to initialize a shared symlinked aiebu checkout" >&2
    return 1
  fi
  git -C "$source" submodule update --init --recursive -- third_party/aiebu
}
