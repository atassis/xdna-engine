#!/usr/bin/env bash
# DEPRECATED shim -> ../install.sh
#
# There used to be two installers for one service, and that is exactly how the deployed unit and
# the documented one drifted apart: this script wrote npu-serve.service and hand-installed
# ~/.local/bin/npu, while the repo-root install.sh wrote npu-asr.service and built binaries nobody
# ran. The result was a production binary that no script ever refreshed -- 15 days and 72 commits
# stale -- while every install reported success.
#
# The repo-root install.sh is now the single entrypoint. It builds the workspace, INSTALLS
# ~/.local/bin/npu, preflights the engine config against on-disk artifacts (refusing to install a
# service that cannot serve), writes xdna-engine.service, and retires the superseded units.
#
# This file only forwards, so existing muscle memory and older docs keep working.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '\033[1;33m[deprecated]\033[0m scripts/install.sh -> forwarding to %s/install.sh\n' "$REPO" >&2
exec "$REPO/install.sh" "$@"
