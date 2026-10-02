#!/usr/bin/env bash
# Regression coverage for cold bootstrap dependency dispatch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0

fail() { echo "FAIL: $*" >&2; FAIL=1; }
pass() { echo "PASS: $*"; }
contains() { grep -Fq -- "$2" "$1"; }

setup_fixture() {
  local t="$1"
  mkdir -p "$t/repo/scripts/lib" "$t/bin" "$t/cache" "$t/home"
  cp "$ROOT/scripts/setup_kernel_env.sh" "$t/repo/scripts/"
  cp "$ROOT/scripts/provision_peano.sh" "$t/repo/scripts/"
  cp "$ROOT/scripts/lib/data_root.sh" "$t/repo/scripts/lib/"
  cat > "$t/repo/toolchain.lock" <<'EOF'
MLIR_AIE_FORK_COMMIT=498f452a811a065a22c503ed8da13dafed5f1f91
PEANO_DIST=llvm_aie-22.0.0.2026091901+0006955e
PEANO_DIST_SHA256=c0fb42c9808b7aef059cd6df98794d458a62a8fa732e5bbb3780380864e2e514
PEANO_FORK_COMMIT=aa4acd25694884aadad7bdb1479b72ec0399744c
NANOBIND=2.12.0
EOF
  cat > "$t/repo/scripts/requirements-kernel-env.txt" <<'EOF'
markdown
pyyaml
jinja2
EOF
  cat > "$t/repo/scripts/build_wheelhouse.sh" <<'EOF'
#!/bin/sh
exit 1
EOF
  cat > "$t/repo/scripts/sync_kernels.sh" <<'EOF'
#!/bin/sh
exit 0
EOF
  cat > "$t/repo/scripts/build_peano_fast.sh" <<'EOF'
#!/bin/sh
printf 'build LLVM_AIE_SRC=%s BUILD_DIR=%s\n' "$LLVM_AIE_SRC" "$BUILD_DIR" >> "$MOCK_LOG"
mkdir -p "$BUILD_DIR/lib" "$BUILD_DIR/bin"
touch "$BUILD_DIR/lib/libLLVM.so.22git" "$BUILD_DIR/lib/libclang-cpp.so.22git"
printf '#!/bin/sh\necho "clang version 22.0.0git (git@github.com:atassis/llvm-aie.git aa4acd25694884aadad7bdb1479b72ec0399744c)"\n' > "$BUILD_DIR/bin/clang++"
chmod +x "$BUILD_DIR/bin/clang++"
EOF
  cat > "$t/repo/scripts/install_peano_local.sh" <<'EOF'
#!/bin/sh
printf 'install %s\n' "$*" >> "$MOCK_LOG"
case "$1" in
  --resolve) printf '%s/pinned\n' "$PEANO_LOCAL_HOME" ;;
  --activate)
    mkdir -p "$PWD/.venv-iron/lib/python3.14/site-packages"
    ln -sfn "$2" "$PWD/.venv-iron/lib/python3.14/site-packages/llvm-aie"
    ;;
  *)
    mkdir -p "$PEANO_LOCAL_HOME/pinned/bin"
    printf '#!/bin/sh\necho "clang version 22.0.0git (git@github.com:atassis/llvm-aie.git aa4acd25694884aadad7bdb1479b72ec0399744c)"\n' > "$PEANO_LOCAL_HOME/pinned/bin/clang++"
    chmod +x "$PEANO_LOCAL_HOME/pinned/bin/clang++"
    cp "$PEANO_LOCAL_HOME/pinned/bin/clang++" "$PEANO_LOCAL_HOME/pinned/bin/clang"
    ;;
esac
EOF
  chmod +x "$t/repo/scripts/"*.sh

  cat > "$t/bin/uv" <<'EOF'
#!/bin/sh
printf 'uv %s\n' "$*" >> "$MOCK_LOG"
if [ "$1" = venv ]; then
  mkdir -p "$PWD/.venv-iron/bin"
  cat > "$PWD/.venv-iron/bin/python" <<'PY'
#!/bin/sh
test -e "$PWD/.venv-iron/.aie-ready"
PY
  chmod +x "$PWD/.venv-iron/bin/python"
  exit 0
fi
case " $* " in
  *' --offline '*) exit 1 ;;
  *) touch "$PWD/.venv-iron/.aie-ready"; exit 0 ;;
esac
EOF
  cat > "$t/bin/git" <<'EOF'
#!/bin/sh
printf 'git %s\n' "$*" >> "$MOCK_LOG"
if [ "$1" = -C ]; then
  dir="$2"; shift 2
else
  dir=.
fi
case "${1:-}" in
  init) mkdir -p "$dir/.git" ;;
  checkout) mkdir -p "$dir/llvm"; : > "$dir/llvm/CMakeLists.txt" ;;
  rev-parse) printf '%s\n' aa4acd25694884aadad7bdb1479b72ec0399744c ;;
esac
exit 0
EOF
  cat > "$t/bin/gh" <<'EOF'
#!/bin/sh
printf 'gh %s\n' "$*" >> "$MOCK_LOG"
out=
while [ $# -gt 0 ]; do
  [ "$1" = --output ] && { out="$2"; shift 2; continue; }
  shift
done
[ -z "$out" ] || : > "$out"
EOF
  cat > "$t/bin/sha256sum" <<'EOF'
#!/bin/sh
printf 'sha256sum %s\n' "$*" >> "$MOCK_LOG"
cat >/dev/null
EOF
  cat > "$t/bin/python3" <<'EOF'
#!/usr/bin/env bash
printf 'python3 %s\n' "$*" >> "$MOCK_LOG"
dest="${@: -1}"
mkdir -p "$dest/llvm-aie/bin"
: > "$dest/llvm-aie/bin/clang"
chmod +x "$dest/llvm-aie/bin/clang"
EOF
  chmod +x "$t/bin/"*
}

test_setup_uses_optional_wheelhouse_and_provisions_peano() {
  local t; t="$(mktemp -d)"
  trap 'rm -rf "$t"' RETURN
  setup_fixture "$t"
  export MOCK_LOG="$t/log" PATH="$t/bin:$PATH" HOME="$t/home" UV_CACHE_DIR="$t/cache" PEANO_LOCAL_HOME="$t/peano"
  if ! (cd "$t/repo" && bash scripts/setup_kernel_env.sh); then
    cat "$MOCK_LOG" >&2
    fail "setup completes after its mocked network and Peano provision paths"
    return
  fi
  if contains "$MOCK_LOG" "$t/repo/vendor/wheelhouse"; then
    fail "setup passes an absent wheelhouse to uv"
  else
    pass "setup omits an absent wheelhouse"
  fi
  if contains "$MOCK_LOG" 'latest-wheels-3'; then
    pass "setup reaches the declared mlir_aie network index"
  else
    fail "setup does not use the declared mlir_aie network index"
  fi
  if contains "$MOCK_LOG" 'requirements-kernel-env.txt'; then
    pass "setup installs the declared aiebu Python requirements"
  else
    fail "setup omits the declared aiebu Python requirements"
  fi
  if contains "$MOCK_LOG" "gh release download nightly --repo Xilinx/llvm-aie"; then
    pass "setup dispatches Peano provisioning after an empty UV cache"
  else
    fail "setup does not dispatch Peano provisioning"
  fi
}

test_provision_uses_declared_seed_and_fork() {
  local t; t="$(mktemp -d)"
  trap 'rm -rf "$t"' RETURN
  setup_fixture "$t"
  export MOCK_LOG="$t/log" PATH="$t/bin:$PATH" HOME="$t/home" UV_CACHE_DIR="$t/cache" XDNA_CACHE="$t/xdna-cache" PEANO_LOCAL_HOME="$t/peano"
  mkdir -p "$t/repo/.venv-iron/lib/python3.14/site-packages"
  if ! (cd "$t/repo" && bash scripts/provision_peano.sh); then
    cat "$MOCK_LOG" >&2
    fail "Peano provision completes with mocked release, source, build, and install commands"
    return
  fi
  local asset='llvm_aie-22.0.0.2026091901+0006955e-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
  if contains "$MOCK_LOG" "--pattern $asset"; then
    pass "provision requests the exact official Linux seed asset"
  else
    fail "provision does not request the exact official Linux seed asset"
  fi
  if contains "$MOCK_LOG" 'sha256sum -c -'; then
    pass "provision verifies the declared seed checksum"
  else
    fail "provision does not verify the seed checksum"
  fi
  if contains "$MOCK_LOG" "build LLVM_AIE_SRC=$t/xdna-cache/peano-source/aa4acd25694884aadad7bdb1479b72ec0399744c BUILD_DIR=$t/xdna-cache/peano-build/aa4acd25694884aadad7bdb1479b72ec0399744c"; then
    pass "provision builds the declared fork commit"
  else
    fail "provision does not build the declared fork commit"
  fi
  if contains "$MOCK_LOG" "--from $t/xdna-cache/peano-seed/llvm_aie-22.0.0.2026091901+0006955e --build $t/xdna-cache/peano-build/aa4acd25694884aadad7bdb1479b72ec0399744c"; then
    pass "provision installs fork codegen over the declared seed"
  else
    fail "provision does not install fork codegen over the declared seed"
  fi
}

test_provision_prefers_uv_cache_dir() {
  local t archive dist
  t="$(mktemp -d)"
  trap 'rm -rf "$t"' RETURN
  setup_fixture "$t"
  archive="$t/cache/archive-v0/entry"
  dist='llvm_aie-22.0.0.2026091901+0006955e.dist-info'
  mkdir -p "$archive/$dist" "$archive/llvm-aie/bin" \
    "$t/repo/.venv-iron/lib/python3.14/site-packages"
  : > "$archive/llvm-aie/bin/clang"
  chmod +x "$archive/llvm-aie/bin/clang"
  export MOCK_LOG="$t/log" PATH="$t/bin:$PATH" HOME="$t/home" UV_CACHE_DIR="$t/cache" XDNA_CACHE="$t/xdna-cache" PEANO_LOCAL_HOME="$t/peano"
  if ! (cd "$t/repo" && bash scripts/provision_peano.sh > "$t/out" 2>&1); then
    cat "$t/out" >&2
    fail "Peano provision completes from UV_CACHE_DIR"
    return
  fi
  if contains "$t/out" "seed restored from $t/cache/archive-v0" && ! contains "$MOCK_LOG" 'gh release download'; then
    pass "provision uses UV_CACHE_DIR as a seed fast path"
  else
    fail "provision does not prefer UV_CACHE_DIR over a release download"
  fi
}

test_setup_uses_optional_wheelhouse_and_provisions_peano
test_provision_uses_declared_seed_and_fork
test_provision_prefers_uv_cache_dir
exit "$FAIL"
