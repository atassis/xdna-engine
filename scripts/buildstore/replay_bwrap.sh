#!/usr/bin/env bash
# scripts/buildstore/replay_bwrap.sh
# SPDX-License-Identifier: Apache-2.0
# Rebuild from a manifest with ONLY its recorded reads visible (read-only) and its writable roots
# fresh and empty, then compare the output root with the recorded run's.
# Exit 0 = same bytes; anything else = the manifest is not sufficient.
# usage: replay_bwrap.sh <manifest.json> <recorded-out-dir> <scratch>
set -euo pipefail
m="$1"; ref="$2"; scratch="$3"
rm -rf "$scratch"; mkdir -p "$scratch"
# Merged-/usr layout (CachyOS): the ELF interpreter is reached through /lib64, which ld.so opens
# itself and the shim cannot see.
args=(--unshare-all --die-with-parent --dev /dev --proc /proc --tmpfs /tmp --clearenv
      --symlink usr/lib /lib64 --symlink usr/lib /lib --symlink usr/bin /bin --symlink usr/bin /sbin)
# System files glibc opens through internal calls the shim cannot hook (ld.so cache, NSS, locale,
# zone). Declared here, not key inputs: they are furniture, not build inputs.
while IFS= read -r p; do [ -e "$p" ] && args+=(--ro-bind "$p" "$p"); done \
  < <(grep -v '^#' "$(dirname "$0")/replay_base.txt")
while IFS=$'\t' read -r kind p q; do
  case "$kind" in
    ro) args+=(--ro-bind "$p" "$p") ;;
    rw) mkdir -p "$scratch/rw$p"; args+=(--bind "$scratch/rw$p" "$p") ;;
    cwd) args+=(--dir "$p" --chdir "$p") ;;
    env) args+=(--setenv "${p%%=*}" "${p#*=}") ;;
    # A given-name lookup (e.g. a DT_NEEDED "libfoo.so.8") resolves through this symlink to the
    # real file already ro-bound above; ld.so opens the symlink name, not the resolved path.
    # ldalias entries (below) are the same shape, derived from ld.so.cache rather than the shim.
    link | ldalias) args+=(--symlink "$q" "$p") ;;
    arg) argv+=("$p") ;;
  esac
done < <(python3 - "$m" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
for p in sorted(set(m["reads"]) | set(m["dirs"]), key=len):
    print(f"ro\t{p}")
for p in m["writable"]:
    print(f"rw\t{p}")
print(f"cwd\t{m['cwd']}")
for k, v in m["env"].items():
    print(f"env\t{k}={v}")
for g, t in m["links"].items():
    print(f"link\t{g}\t{t}")
# ld.so resolves a DT_NEEDED soname through ld.so.cache to a *symlink* path (e.g.
# libreadline.so.8 -> libreadline.so.8.3), a lookup its own internal opens make outside any
# hooked call, so the shim never sees it (rec_preload.c's documented gap). ld.so.cache is
# already trusted furniture (replay_base.txt); derive the alias from it for any target this
# manifest did record, rather than trying to make the shim see ld.so's own opens.
import subprocess
reads = set(m["reads"])
for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout.splitlines()[1:]:
    if "=>" not in line:
        continue
    alias, path = line.split("=>")
    alias, path = alias.split("(")[0].strip(), path.strip()
    real = os.path.realpath(path)
    if path != real and real in reads:
        print(f"ldalias\t{path}\t{real}")
for a in m["argv"]:
    print(f"arg\t{a}")
PY
)
bwrap "${args[@]}" "${argv[@]}"
diff -r "$ref" "$scratch/rw$ref" > /dev/null
