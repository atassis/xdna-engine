#!/usr/bin/env bash
# Reproduce the open mlir-aie/Peano kernel build environment on this CachyOS box.
# Idempotent: safe to re-run. mlir-aie is a PINNED git submodule (see docs/11) checked out on our fork
# branch xdna2-asr, which this script RECREATES at MLIR_AIE_FORK_COMMIT on every run -- so it is a
# label for the pin, not an integration line, and nothing may be carried as a commit on it. (It was
# described here as carrying "the CachyOS fixes + toolchain patches" as commits; that stopped being
# true when the pin went zero-carry, and the stale sentence is why 34 orphaned commits looked
# maintained. Our source lives in designs/ and aie_kernels/ and is synced forward by
# sync_kernels.sh.) .venv-iron is .gitignored. Durable record of the env (fork branch + gcc-13 shims +
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
MLIR_AIE_PIN='mlir_aie==0.0.1.2026033104+e4f35d6'
MLIR_AIE_INDEX='https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3'
NANOBIND_PIN="nanobind==${NANOBIND:-2.12.0}"
SITE=".venv-iron/lib/python3.14/site-packages"
KERNEL_ENV_REQUIREMENTS="$REPO/scripts/requirements-kernel-env.txt"
have_aie()   { .venv-iron/bin/python -c 'import aie' 2>/dev/null; }
have_peano() { ls "$SITE"/llvm-aie/bin/clang >/dev/null 2>&1; }
have_pinned_peano() {
  have_peano && "$SITE"/llvm-aie/bin/clang++ --version 2>/dev/null | grep -Fq "$PEANO_FORK_COMMIT"
}
if ! { have_aie && have_pinned_peano; }; then
  # Prefer an OFFLINE install from the uv cache (these exact versions are prefetched). On a cache
  # miss, use the LOCAL wheelhouse (vendor/wheelhouse) BEFORE the network -- vendor/ is gitignored, so
  # a fresh checkout rebuilds the wheel on demand from the uv archive cache via build_wheelhouse.sh
  # (works off the owner box too, not just where a wheel was hand-repacked).
  WHEELHOUSE="$REPO/vendor/wheelhouse"
  PIP_REQUIREMENTS=("$MLIR_AIE_PIN" "$NANOBIND_PIN" -r "$KERNEL_ENV_REQUIREMENTS")
  if ! uv pip install --python .venv-iron --offline "${PIP_REQUIREMENTS[@]}"; then
    # Rebuild the wheelhouse if absent. A failure here (e.g. empty uv cache off the owner box) must
    # NOT abort under set -e: fall through so the network tier below can still fetch mlir_aie.
    ls "$WHEELHOUSE"/mlir_aie-*.whl >/dev/null 2>&1 || bash "$REPO/scripts/build_wheelhouse.sh" || true
    WHEELHOUSE_ARGS=()
    if ls "$WHEELHOUSE"/mlir_aie-*.whl >/dev/null 2>&1; then
      WHEELHOUSE_ARGS=(--find-links "$WHEELHOUSE")
      uv pip install --python .venv-iron --offline "${WHEELHOUSE_ARGS[@]}" \
          "${PIP_REQUIREMENTS[@]}" || true
    fi
    uv pip install --python .venv-iron "${WHEELHOUSE_ARGS[@]}" \
        --find-links "$MLIR_AIE_INDEX" \
        "${PIP_REQUIREMENTS[@]}"
  fi
fi

if ! have_pinned_peano; then
  bash "$REPO/scripts/provision_peano.sh"
fi

# Terminal guard (GAP #3): the tiers above WARN-and-continue on a cache/wheel miss, which under set -e
# would otherwise surface as a confusing failure deep in the toolchain build (Step 3/4). Fail LOUD and
# EARLY here instead. Idempotent: on the warm-cache path both checks pass and this is a silent no-op.
if ! have_pinned_peano; then
  echo "ERROR: Peano (llvm-aie) is missing or does not report PEANO_FORK_COMMIT=$PEANO_FORK_COMMIT." >&2
  echo "       scripts/provision_peano.sh must fetch the checked PEANO_DIST seed and activate codegen" >&2
  echo "       built from the pinned fork; see its preceding error." >&2
  exit 1
fi
if ! have_aie; then
  echo "ERROR: the mlir_aie wheel ($MLIR_AIE_PIN) is not installed -- '.venv-iron import aie' fails." >&2
  echo "       Its resolution needs a warm uv archive cache, a vendor/wheelhouse/ wheel, or the network" >&2
  echo "       find-links index (which rotates dated assets out). Pre-warm the uv cache or vendor the" >&2
  echo "       ~290 MB wheel into vendor/wheelhouse/ (scripts/build_wheelhouse.sh repacks it from the cache)." >&2
  exit 1
fi

# 3. gcc-13/g++-13 shims -> real gcc (makefile-common hardcodes CC?=gcc-13; we have gcc16)
mkdir -p .venv-iron/cc-shim
ln -sf "$(command -v gcc)" .venv-iron/cc-shim/gcc-13
ln -sf "$(command -v g++)" .venv-iron/cc-shim/g++-13

# 4. Ensure a local mlir-aie checkout exists. If it is ALREADY present (the prefetch clones it, or a
#    prior run initialized it), this is a NO-OP -- we do NOT re-clone. When there is no checkout at all
#    (a bare clone) we bootstrap an empty repo and fetch the pinned fork commit BY SHA; we deliberately
#    avoid the broad `git submodule update --init mlir-aie` (it errors on the untracked path, and would
#    fetch the wrong default branch). The fork-branch checkout just below lands the exact pinned commit.
if [ -e mlir-aie/.git ]; then
  echo "  mlir-aie already present -> skip clone (fork commit ensured by fetch-by-SHA below)"
else
  # Bare clone: mlir-aie is UNTRACKED in this repo (a .gitmodules entry with no committed gitlink),
  # so `git submodule update --init mlir-aie` errors ("pathspec did not match"). Bootstrap an empty
  # repo and fetch the pinned fork commit BY SHA; the fork-branch checkout block just below lands it.
  echo "  mlir-aie absent -> bootstrapping fork checkout @ ${MLIR_AIE_FORK_COMMIT:0:12}"
  mkdir -p mlir-aie
  git -C mlir-aie init -q
  git -C mlir-aie remote add fork "${MLIR_AIE_FORK_URL:-https://github.com/atassis/mlir-aie}" 2>/dev/null || true
  git -C mlir-aie fetch --depth 1 fork "$MLIR_AIE_FORK_COMMIT" 2>/dev/null \
    || git -C mlir-aie fetch fork xdna2-asr
fi
# Check out our FORK INTEGRATION BRANCH: atassis/mlir-aie:xdna2-asr = the upstream base + our toolchain
# patch stack carried as COMMITS (the CachyOS build fixes + the bf16 mm.cc microkernel + aiecc-jobs are
# all on the branch). There is no apply-patch step. toolchain.lock pins the exact commit; toolchain_up.sh
# builds the toolchain INSTANCE from a clean git-worktree of it; our kernels are overlaid by
# sync_kernels. toolchain.lock is already sourced at the top -> MLIR_AIE_FORK_COMMIT is in scope.
git -C mlir-aie remote get-url fork >/dev/null 2>&1 \
  || git -C mlir-aie remote add fork "${MLIR_AIE_FORK_URL:-https://github.com/atassis/mlir-aie}"
git -C mlir-aie cat-file -e "${MLIR_AIE_FORK_COMMIT}^{commit}" 2>/dev/null \
  || git -C mlir-aie fetch fork xdna2-asr
# FAIL LOUD. This used to be `checkout -B ... && echo`, so a checkout blocked by an untracked file
# short-circuited the && and the script carried on to sync_kernels.sh and printed "Route B env ready"
# with exit 0. The sandbox then stayed on whatever commit it happened to hold. Measured 2026-09-09:
# it had been failing that way since the pin went zero-carry, leaving the tree 252 commits behind and
# containing NONE of the last five pins -- while build_kernels.sh's newer ancestor check reported the
# drift correctly and was read as a new problem. A checkout that cannot land is a hard error.
if ! git -C mlir-aie checkout -B xdna2-asr "$MLIR_AIE_FORK_COMMIT"; then
  echo "[setup_kernel_env] FAIL: cannot check out MLIR_AIE_FORK_COMMIT (${MLIR_AIE_FORK_COMMIT:0:12}) in mlir-aie." >&2
  echo "  The sandbox is a DERIVED tree: designs/ + aie_kernels/ are the tracked source and are re-synced" >&2
  echo "  below, so local files there are expendable -- but resolve it deliberately, do not delete blindly:" >&2
  echo "    git -C mlir-aie status --short" >&2
  echo "  Anything of value must be moved into designs/ or aie_kernels/ FIRST; see sync_kernels.sh." >&2
  exit 1
fi
echo "  mlir-aie on xdna2-asr @ ${MLIR_AIE_FORK_COMMIT:0:12}"

# INSTALL D: our custom kernels/designs. designs/ (tracked) is the single source of
# truth; copy them FORWARD into the gitignored mlir-aie build sandbox (one-directional => no
# drift; real files so relative-path Makefiles/includes work). See docs/08-10 + sync_kernels.sh.
bash "$REPO/scripts/sync_kernels.sh"

echo "Route B env ready. Use:  source scripts/iron_env.sh  then  make NPU2=1 run  in an example dir."
echo "Build dwconv1d:  make -C mlir-aie/programming_examples/ml/dwconv1d NPU2=1"
