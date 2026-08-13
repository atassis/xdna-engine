#!/usr/bin/env bash
# Regenerate kernel_manifest.json for one or more artifact dirs, with the fail-closed policy.
#
# Every script that stages final_*.xclbin into a dir npu-parakeet's resolve_verified reads must end
# by calling this, or NPU_KERNEL_MANIFEST_VERIFY=1 rejects a CORRECT artifact: a dir nobody generated
# for fails MissingManifest, and a dir rebuilt without regenerating fails a hash mismatch. Four
# scripts stage into whole_array/build and none of them regenerated, which is how the gate came to
# abort a required resident matmul that loads fine with the flag unset.
#
# Fail-closed on failure: delete the old manifest rather than leave a stale one. resolve_checked
# would PASS on stale hashes -- silently wrong is the bug the manifest exists to catch, so "absent"
# is the safer failure.
#
# Usage: bash scripts/refresh_kernel_manifest.sh <artifact-dir> [<artifact-dir> ...]
#        Paths may be relative to the caller's cwd or absolute; both are resolved here, so a caller
#        that has cd'd into the build dir does not have to think about it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

[ "$#" -ge 1 ] || { echo "usage: refresh_kernel_manifest.sh <artifact-dir> [...]" >&2; exit 2; }

for dir in "$@"; do
  abs="$(cd "$dir" 2>/dev/null && pwd)" || { echo "WARNING: $dir does not exist -- no manifest written." >&2; continue; }
  echo "== artifact manifest: $abs =="
  if ( cd "$REPO/rust" && cargo run -q --release -p npu-asr --bin gen_kernel_manifest -- "$abs" ); then
    :
  else
    rm -f "$abs/kernel_manifest.json"
    echo "WARNING: gen_kernel_manifest failed -- removed $abs/kernel_manifest.json rather than leave a stale one." >&2
    echo "         NPU_KERNEL_MANIFEST_VERIFY=1 will now fail closed here until this is re-run." >&2
  fi
done
