#!/usr/bin/env bash
# Reproduce the Route B (open mlir-aie/Peano) build environment on this CachyOS box.
# Idempotent: safe to re-run. mlir-aie is a PINNED git submodule (see docs/11) checked out on our fork
# integration branch atassis/mlir-aie:xdna2-asr (the CachyOS fixes + toolchain patches are COMMITS on it,
# no apply-patch step); .venv-iron is .gitignored. Durable record of the env (fork branch + gcc-13 shims +
# pinned toolchain wheels) needed to build/run on Arch/CachyOS.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# toolchain.lock is the single source of truth for the exact toolchain (Peano pin, fork commit,
# nanobind). Source it ONCE up front so the wheel-install block below reads PEANO_DIST instead of
# hardcoding a nightly that rotates out of the release window.
set -a; . "$REPO/toolchain.lock"; set +a   # -> PEANO_DIST, MLIR_AIE_FORK_COMMIT, NANOBIND

# The exact upstream mlir-aie commit our build is pinned to. This MUST match both the
# submodule gitlink (recorded in our history) and the toolchain wheel below (the wheel
# version string embeds +g<short-sha>). Bumping = change all three together + re-fit the
# patch + re-run scripts/test_repro_vendoring.sh. See internal notes.
MLIR_AIE_SHA=8373e49165649644f1ec414c2e406c0abbbf51cf

# 1. Python 3.14 venv with system pyxrt visible (matches the system pyxrt 3.14 .so)
[ -d .venv-iron ] || uv venv --python 3.14 --system-site-packages .venv-iron

# 2. Toolchain wheels (~1.8 GB: llvm-aie/Peano + mlir_aie). These provide only the version-agnostic
#    vendored binaries (bootgen, aie-translate) + `import aie` for the venv; the BLESSED toolchain is
#    the fork INSTANCE built by toolchain_up.sh, never the wheel python. PINNED to the EXACT versions
#    prefetched into the uv cache (overnight/PREFETCH-STATE.md):
#      - Peano: derived from toolchain.lock PEANO_DIST (currently 21.0.0.2026062301+cb664e8c),
#        NOT the old literal 2026052701 which has rotated out of the nightly window.
#      - mlir_aie: 0.0.1.2026033104+e4f35d6 (resolves the earlier provenance gap; NOT 1.3.2.dev126).
#      - nanobind: pinned via toolchain.lock NANOBIND (2.13.0 silently breaks the bindings).
#    PEANO_DIST looks like 'llvm_aie-21.0.0.2026062301+cb664e8c' -> pip spec 'llvm-aie==21.0.0...'.
PEANO_PIN="llvm-aie==${PEANO_DIST#llvm_aie-}"
MLIR_AIE_PIN='mlir_aie==0.0.1.2026033104+e4f35d6'
NANOBIND_PIN="nanobind==${NANOBIND:-2.12.0}"
SITE=".venv-iron/lib/python3.14/site-packages"
have_aie()   { .venv-iron/bin/python -c 'import aie' 2>/dev/null; }
have_peano() { ls "$SITE"/llvm-aie/bin/clang >/dev/null 2>&1; }
if ! { have_aie && have_peano; }; then
  # Prefer an OFFLINE install from the uv cache (these exact versions are prefetched). On a cache
  # miss, use the LOCAL wheelhouse (vendor/wheelhouse) BEFORE the network -- vendor/ is gitignored, so
  # a fresh checkout rebuilds the wheel on demand from the uv archive cache via build_wheelhouse.sh
  # (works off the owner box too, not just where a wheel was hand-repacked).
  # IMPORTANT: llvm-aie (Peano) ships only a cp310 wheel that is UNRESOLVABLE on this py3.14 venv, so
  # it is EXCLUDED from every pip tier here -- bundling "$PEANO_PIN" would fail the whole resolve
  # (installing nothing) and, as the last statement in this block, abort the script under `set -e`
  # before Peano can be provided. Peano is handled EXCLUSIVELY by the tree-copy block below.
  WHEELHOUSE="$REPO/vendor/wheelhouse"
  if ! uv pip install --python .venv-iron --offline "$MLIR_AIE_PIN" "$NANOBIND_PIN"; then
    ls "$WHEELHOUSE"/mlir_aie-*.whl >/dev/null 2>&1 || bash "$REPO/scripts/build_wheelhouse.sh"
    uv pip install --python .venv-iron --find-links "$WHEELHOUSE" --offline \
        "$MLIR_AIE_PIN" "$NANOBIND_PIN" \
      || uv pip install --python .venv-iron \
        --find-links "$WHEELHOUSE" \
        --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 \
        "$MLIR_AIE_PIN" "$NANOBIND_PIN"
  fi

  # llvm-aie (Peano) ships a cp310-tagged wheel; pip will not install it into this py3.14 venv. If
  # it is still missing after the steps above, reconstruct it by copying the unpacked archive tree
  # from the uv cache straight into site-packages (the wheel only vendors binaries; no python import
  # is needed -- the blessed Peano is the fork instance, this just provides bootgen/clang/etc.).
  if ! have_peano; then
    LAR=$(find "$HOME/.cache/uv/archive-v0" -maxdepth 2 \
            -name 'llvm_aie-'"${PEANO_PIN#llvm-aie==}"'.dist-info' 2>/dev/null | head -1 | xargs -r dirname)
    if [ -n "${LAR:-}" ] && [ -d "$LAR/llvm-aie" ]; then
      cp -a "$LAR"/llvm-aie "$LAR"/llvm_aie-*.dist-info "$SITE"/
      echo "  llvm-aie (Peano) copied from uv archive cache -> $SITE/llvm-aie"
    else
      echo "  WARNING: llvm-aie tree not found in uv archive cache; Peano binaries unavailable" >&2
    fi
  fi
fi

# 3. gcc-13/g++-13 shims -> real gcc (makefile-common hardcodes CC?=gcc-13; we have gcc16)
mkdir -p .venv-iron/cc-shim
ln -sf "$(command -v gcc)" .venv-iron/cc-shim/gcc-13
ln -sf "$(command -v g++)" .venv-iron/cc-shim/g++-13

# 4. Ensure a local mlir-aie checkout exists. If it is ALREADY present (the prefetch clones it, or a
#    prior run initialized the submodule), this is a NO-OP -- we do NOT re-clone and we do NOT run the
#    broad `git submodule update --init mlir-aie` fallback, which can fetch the wrong (default) branch.
#    We rely instead on toolchain_up.sh's fetch-BY-SHA (and the fork-branch checkout just below) to
#    land the exact pinned commit. Only when there is no checkout at all do we bootstrap one.
if [ -e mlir-aie/.git ]; then
  echo "  mlir-aie already present -> skip clone (fork commit ensured by fetch-by-SHA below)"
else
  git submodule update --init --depth 1 mlir-aie 2>/dev/null \
    || git submodule update --init mlir-aie
fi
# Check out our FORK INTEGRATION BRANCH: atassis/mlir-aie:xdna2-asr = the upstream base + our toolchain
# patch stack carried as COMMITS (the CachyOS build fixes + the bf16 mm.cc microkernel + aiecc-jobs are
# all on the branch). There is no apply-patch step. toolchain.lock pins the exact commit; toolchain_up.sh
# builds the toolchain INSTANCE from a clean git-worktree of it; route_b kernels are overlaid by
# sync_kernels. toolchain.lock is already sourced at the top -> MLIR_AIE_FORK_COMMIT is in scope.
git -C mlir-aie remote get-url fork >/dev/null 2>&1 \
  || git -C mlir-aie remote add fork "${MLIR_AIE_FORK_URL:-https://github.com/atassis/mlir-aie}"
git -C mlir-aie cat-file -e "${MLIR_AIE_FORK_COMMIT}^{commit}" 2>/dev/null \
  || git -C mlir-aie fetch fork xdna2-asr
git -C mlir-aie checkout -B xdna2-asr "$MLIR_AIE_FORK_COMMIT" \
  && echo "  mlir-aie on xdna2-asr @ ${MLIR_AIE_FORK_COMMIT:0:12}"

# INSTALL D: our custom kernels/designs. route_b_kernels/ (tracked) is the single source of
# truth; copy them FORWARD into the gitignored mlir-aie build sandbox (one-directional => no
# drift; real files so relative-path Makefiles/includes work). See docs/08-10 + sync_kernels.sh.
bash "$REPO/scripts/sync_kernels.sh"

echo "Route B env ready. Use:  source scripts/iron_env.sh  then  make NPU2=1 run  in an example dir."
echo "Build dwconv1d:  make -C mlir-aie/programming_examples/ml/dwconv1d NPU2=1"
