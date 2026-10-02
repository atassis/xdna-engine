#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/scripts/iron_env.sh"
source "$REPO/scripts/kernel_sandbox.sh"

cmd="${1:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
stem="${2:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
mlir_aie_root="${3:?usage: whole_array.sh build <stem> <mlir-aie-root>}"
MMW="$mlir_aie_root/programming_examples/basic/matrix_multiplication/whole_array"

case "$cmd" in
  build) ;;
  *) echo "[whole_array] unsupported command '$cmd'" >&2; exit 2 ;;
esac

recipe_output="$(python3 "$REPO/scripts/kernel_families/whole_array_recipe.py" "$stem")" || exit $?
mapfile -t recipe <<< "$recipe_output"
[ "${#recipe[@]}" -eq 3 ] || { echo "[whole_array] invalid recipe output for '$stem'" >&2; exit 1; }
makefile="${recipe[0]}"
read -r -a variables <<< "${recipe[1]}"
target="${recipe[2]}"

ensure_fresh_sandbox "$MMW/build"
if [ -n "$makefile" ]; then
  scripts/verify_kernel_source.sh "$makefile"
  make -C "$MMW" -f "$makefile" NPU2=1 "${variables[@]}" "$target"
else
  make -C "$MMW" NPU2=1 "${variables[@]}" "$target"
fi

built="$MMW/$target"
[ -f "$built" ] || { echo "[whole_array] build did not produce $built" >&2; exit 1; }
echo "[whole_array] built $stem in $MMW/build"