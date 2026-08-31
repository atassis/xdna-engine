#!/usr/bin/env bash
# install.sh -- install the `s2say` CLI for the current user.
#
#   cli/install.sh                     install to ~/.local/bin, write ~/.config/s2say/config
#   cli/install.sh --prefix ~/opt      install elsewhere
#   cli/install.sh --uninstall
#
# USER-LEVEL ONLY, no sudo anywhere. It installs a launcher into $PREFIX/bin and records the
# resolved model/binary paths in a config file, so the installed command keeps working if the
# checkout later moves -- and so the CLI itself contains no absolute paths.
#
# It does NOT build s2.cpp or download the model. Both are large and belong to their own workflows;
# if either is missing this reports where it looked and exits, rather than guessing.
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
UNINSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"
BIN_DIR="$PREFIX/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/s2say"
TARGET="$BIN_DIR/s2say"

if [[ "$UNINSTALL" == 1 ]]; then
  rm -f "$TARGET" && echo "removed $TARGET"
  echo "config left at $CONFIG_DIR/config (delete by hand if you want it gone)"
  exit 0
fi

S2_BIN="${S2_BIN:-$WS/s2.cpp/build-cpu/s2}"
S2_MODEL="${S2_MODEL:-$WS/s2.cpp/models/s2-pro-q6_k.gguf}"
S2_TOKENIZER="${S2_TOKENIZER:-$WS/s2.cpp/models/tokenizer.json}"

echo "workspace  $WS"
missing=0
for pair in "s2 binary:$S2_BIN" "model:$S2_MODEL" "tokenizer:$S2_TOKENIZER"; do
  label="${pair%%:*}"; path="${pair#*:}"
  if [[ -e "$path" ]]; then
    printf '  %-10s %s\n' "$label" "$path"
  else
    printf '  %-10s %s   [MISSING]\n' "$label" "$path"
    missing=1
  fi
done

if [[ "$missing" == 1 ]]; then
  cat >&2 <<'MSG'

Something above is missing. Nothing was installed.

  s2 binary : build it with   cmake --build s2.cpp/build-cpu --target s2
  model     : place the GGUF at s2.cpp/models/ (or point S2_MODEL at it)

Or override any of them and re-run, e.g.
  S2_MODEL=/path/to/model.gguf cli/install.sh
MSG
  exit 1
fi

command -v python3 >/dev/null || { echo "python3 is required (wav header parsing)" >&2; exit 1; }
PLAYER=""
for p in paplay pw-play aplay; do command -v "$p" >/dev/null && { PLAYER="$p"; break; }; done
[[ -n "$PLAYER" ]] || echo "note: no paplay/pw-play/aplay found -- s2say will write files but not play them" >&2

mkdir -p "$BIN_DIR" "$CONFIG_DIR"
# _CFG suffixes so an env var of the plain name still wins at run time (see s2say's resolution order).
cat > "$CONFIG_DIR/config" <<EOF
# Written by cli/install.sh on $(date -Iseconds). Edit freely.
# s2say resolves: environment  ->  this file  ->  repo-relative defaults.
S2_BIN_CFG="$S2_BIN"
S2_MODEL_CFG="$S2_MODEL"
S2_TOKENIZER_CFG="$S2_TOKENIZER"
S2SAY_OUT_DIR_CFG="\${S2SAY_OUT_DIR_CFG:-\$HOME/Music/s2}"
EOF

install -m 0755 "$HERE/s2say" "$TARGET"

echo
echo "installed  $TARGET"
echo "config     $CONFIG_DIR/config"
[[ -n "$PLAYER" ]] && echo "player     $PLAYER"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo
     echo "NOTE: $BIN_DIR is not on your PATH. Add this to your shell rc:"
     echo "  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

echo
echo "try:  s2say --where"
echo "      s2say \"hello from the s2 model\""
