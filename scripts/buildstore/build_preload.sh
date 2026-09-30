#!/usr/bin/env bash
# scripts/buildstore/build_preload.sh
# SPDX-License-Identifier: Apache-2.0
# Compile rec_preload.c once per source content; print the .so path.
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)/rec_preload.c"
. "$(dirname "$0")/../cache_env.sh"
h=$(sha256sum "$src" | cut -c1-16)
out="$XDNA_CACHE/buildstore/rec_preload-$h.so"
if [ ! -f "$out" ]; then
  mkdir -p "$(dirname "$out")"
  cc -O2 -shared -fPIC -Wall -Werror -o "$out.tmp$$" "$src" -ldl
  mv "$out.tmp$$" "$out"
fi
echo "$out"
