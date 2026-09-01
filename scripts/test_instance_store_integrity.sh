#!/usr/bin/env bash
# The toolchain instance store is content-addressed: .cache/instances/<key>/src IS the commit its
# key names. It stopped being that because route_b_override.mk resolved OUR kernels through
# kernels_dir (the store), so sync_kernels.sh re-exec'd into it on every build -- 43 dirty entries,
# and verify_kernel_source.sh reads VENDOR ground truth from the same tree, so the gate's reference
# was the mutated store. Our kernels now build from rb_kernels_dir (the tracked tree).
#
# Run standalone. Exit 0 = the store is what its key says and nothing can write to it.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
ok(){ echo "  PASS $1"; }
no(){ echo "  FAIL $1"; fail=1; }

# Ask the resolver, never re-derive the key: the key moved from the whole lock file to its fields on
# 2026-09-01, and every hand-rolled copy of that hashing rule went stale silently.
INST="$(scripts/toolchain_up.sh 2>/dev/null)"
[ -n "$INST" ] && [ -d "$INST/src" ] || { echo "  SKIP no built instance"; exit 0; }
echo "== instance $INST =="

dirty="$(git -C "$INST/src" status --porcelain -- aie_kernels/ programming_examples/ | wc -l)"
if [ "$dirty" = "0" ]; then
  ok "instance src has no modified or untracked kernel/example files"
else
  no "instance src is mutated ($dirty entries) -- it is not the commit its key names"
  git -C "$INST/src" status --short -- aie_kernels/ programming_examples/ | head -10
fi

echo "== sync_kernels.sh must refuse the store =="
out="$(bash scripts/sync_kernels.sh "$INST/src" 2>&1)"; rc=$?
[ "$rc" -ne 0 ] && ok "sync_kernels.sh refuses an instance-store target (exit $rc)" \
                || no "sync_kernels.sh wrote into the store (rc=0): $out"

echo "== our kernels must not resolve through kernels_dir =="
bad="$(grep -rn '${kernels_dir}/[A-Za-z0-9_]*\.cc' route_b_kernels/*/Makefile* route_b_kernels/*/*.mk 2>/dev/null \
       | while IFS= read -r l; do
           f="$(sed 's/.*${kernels_dir}\///; s/ .*//' <<<"$l")"
           [ -f "route_b_kernels/aie_kernels/$f" ] && echo "$l"
         done)"
[ -z "$bad" ] && ok "no route_b kernel is declared through \${kernels_dir}" \
              || { no "route_b kernels declared through \${kernels_dir} (they only resolve if copied into the store):"; echo "$bad"; }

exit $fail
