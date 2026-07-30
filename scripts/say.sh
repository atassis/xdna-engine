#!/usr/bin/env bash
# say.sh -- synthesize speech from text, write a timestamped wav, play it.
#
#   scripts/say.sh "some text to speak"
#   scripts/say.sh --no-play "text"          write the file, do not play
#   scripts/say.sh --voice alice "text"      use a saved .s2voice profile
#   scripts/say.sh --out-dir ~/clips "text"
#
# WHICH HARDWARE THIS USES, stated plainly because it is the whole question right now:
# this runs the s2.cpp CPU reference end to end. It is NOT the NPU path.
#
# The NPU work so far covers the codec DECODER's operators and its window-stitching driver -- a
# complete decoder stage is device-green at rel-L2 8.869e-07 against the true stream. It is not yet
# a runnable pipeline, for two independent reasons:
#
#   1. Only stage 4 of 4 is wired end to end; stages 1-3 need their -D shapes and channel chunking
#      (the mechanism is in place and validated, the wiring is not).
#   2. Even wired, a full-length decode is impractical today: stage 4's dilation-9 convolution
#      advances 10 samples per dispatch over a 112640-sample stream, so a 2.5 s clip is ~11k
#      dispatches for that one op. Window length is the fix, and the lever is a bf16 resident.
#
# And the AR half (text -> codes) has no NPU kernels at all. So "TTS on the NPU" is not something
# this script can honestly claim yet; when it can, this script gets a --npu flag and this comment
# gets shorter.
set -euo pipefail

WS="$WORKSPACE"
S2="${S2_BIN:-$WS/s2.cpp/build-cpu/s2}"
MODEL="${S2_MODEL:-$WS/s2.cpp/models/s2-pro-q6_k.gguf}"
TOKENIZER="${S2_TOKENIZER:-$WS/s2.cpp/models/tokenizer.json}"
OUT_DIR="${SAY_OUT_DIR:-$HOME/Music/s2}"
PLAY=1
VOICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-play) PLAY=0; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --voice)   VOICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) break ;;
  esac
done

TEXT="${*:-}"
if [[ -z "$TEXT" ]]; then
  echo "usage: $(basename "$0") [--no-play] [--voice ID] [--out-dir DIR] \"text to speak\"" >&2
  exit 2
fi

for f in "$S2" "$MODEL" "$TOKENIZER"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
# Slug from the text so a directory of clips is browsable rather than a wall of timestamps.
SLUG="$(printf '%s' "$TEXT" | tr '[:upper:]' '[:lower:]' \
        | tr -cs 'a-z0-9' '-' | sed 's/^-*//; s/-*$//' | cut -c1-40)"
OUT="$OUT_DIR/${STAMP}${SLUG:+-$SLUG}.wav"

ARGS=(-m "$MODEL" -t "$TOKENIZER" -text "$TEXT" -o "$OUT" -v -1
      -threads "$(( $(nproc) / 2 ))")
[[ -n "$VOICE" ]] && ARGS+=(--voice "$VOICE")

echo "synthesizing (CPU reference path) -> $OUT"
START=$(date +%s.%N)
"$S2" "${ARGS[@]}" >/dev/null 2>&1 || { echo "s2 failed; re-run without >/dev/null to see why" >&2; exit 1; }
END=$(date +%s.%N)

[[ -s "$OUT" ]] || { echo "no audio produced" >&2; exit 1; }

# Report duration and realtime factor: for a TTS clip those are the two numbers worth seeing, and
# they make the CPU-vs-NPU comparison concrete once there is an NPU path to compare against.
# Parse RIFF by hand: s2.cpp writes IEEE-float wavs (format tag 3) and python's `wave` module
# rejects anything that is not PCM.
read -r DUR SIZE < <(python3 -c "
import os, struct, sys
p = sys.argv[1]
b = open(p, 'rb').read()
assert b[:4] == b'RIFF' and b[8:12] == b'WAVE', 'not a RIFF/WAVE file'
i, rate, ch, bits, ndata = 12, None, None, None, 0
while i + 8 <= len(b):
    cid, sz = b[i:i+4], struct.unpack('<I', b[i+4:i+8])[0]
    if cid == b'fmt ':
        _, ch, rate, _, _, bits = struct.unpack('<HHIIHH', b[i+8:i+24])
    elif cid == b'data':
        ndata = sz
    i += 8 + sz + (sz & 1)
print((ndata / (ch * bits // 8)) / rate, os.path.getsize(p))
" "$OUT")
# All formatting in python: the shell's printf honours LC_NUMERIC, so on a comma-decimal locale
# it rejects the floats python just produced.
python3 -c "
import sys
dur, size, start, end = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
wall = end - start
print(f'  {dur:.2f} s audio, {size/1000:.0f} kB, {wall:.1f} s wall, '
      f'RTF {wall/max(dur,1e-9):.2f} (1.0 = realtime, lower is faster)')
" "$DUR" "$SIZE" "$START" "$END"

if [[ "$PLAY" == 1 ]]; then
  for p in paplay pw-play aplay; do
    if command -v "$p" >/dev/null; then
      echo "playing via $p"
      "$p" "$OUT" && break
    fi
  done
fi
