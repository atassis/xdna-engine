#!/usr/bin/env bash
# The Rust gate. Run before pushing anything that touches rust/.
#
# This gate exists because all three of the following were broken at once, silently, with no CI:
#   1. `cargo test --workspace` did not compile (a bin was never updated for a 6->7 arg signature change)
#   2. `cargo check -p npu-parakeet` did not compile without features, breaking the crate's own
#      "builds alone, without XRT" contract
#   3. `cargo clippy --workspace` hit deny-by-default lints in three separate crates
# Each check below pins one of those.
#
# Needs XRT (/usr/include/xrt + libxrt_coreutil) and onnxruntime, so it runs on the dev box, not on a
# hosted runner. .github/workflows/rust-ci.yml covers the XRT-free subset upstream.
#
# Usage:  scripts/ci_gate.sh            # full gate
#         SKIP_RUST_GATE=1 ...          # opt out (pre-push honors this)
set -euo pipefail

if [ "${SKIP_RUST_GATE:-0}" = "1" ]; then
  echo "[ci_gate] SKIP_RUST_GATE=1 -- skipping"; exit 0
fi

cd "$(dirname "$0")/../rust"

# Half the cores: heavy local builds should leave the box usable.
J=$(( $(nproc) / 2 )); [ "$J" -lt 1 ] && J=1

fail=0
step() {
  echo
  echo "[ci_gate] === $1 ==="
  shift
  if "$@"; then echo "[ci_gate] OK"; else echo "[ci_gate] FAILED"; fail=1; fi
}

# 1. Lint. Plain clippy (not -D warnings) is deliberate for now: every failure this gate was built to
#    catch is an error or a deny-by-default lint, and those already fail here. Ratchet to
#    `-- -D warnings` once the remaining ~17 warning sites are cleared.
step "clippy --workspace --all-targets" \
  cargo clippy --workspace --all-targets -j "$J"

# 2. Tests. `--nocapture`, because the fixture-guard pattern (`eprintln!("SKIP: ...")` + early
#    `return`) prints nothing without it -- a test whose fixture is missing still reports `ok`, so
#    a passing-by-skipping test read as green here too. Tee to a scratch file so the tally below
#    survives the scrollback; `pipefail` so `tee` succeeding doesn't hide a `cargo test` failure.
echo
echo "[ci_gate] === test --workspace ==="
test_log="$(mktemp)"
if (set -o pipefail; cargo test --workspace --no-fail-fast -j "$J" -- --nocapture 2>&1 | tee "$test_log"); then
  echo "[ci_gate] OK"
else
  echo "[ci_gate] FAILED"; fail=1
fi
skip_count=$(grep -c '^SKIP' "$test_log" || true)
if [ "$skip_count" -gt 0 ]; then
  echo "[ci_gate] $skip_count SKIP line(s) above -- a guarded test's fixture was missing and it returned early, counted as a pass:"
  grep '^SKIP' "$test_log" | sort -u | sed 's/^/[ci_gate]   /'
fi
rm -f "$test_log"

# 3. The decoupling contract: npu-parakeet must build standalone, no features, no XRT. This is the
#    check that would have caught the missing `required-features` on a cfg-gated device probe.
step "check -p npu-parakeet (no default features)" \
  cargo check -p npu-parakeet --no-default-features -j "$J"

echo
if [ "$fail" -ne 0 ]; then
  echo "[ci_gate] GATE RED"; exit 1
fi
echo "[ci_gate] GATE GREEN"
