#!/usr/bin/env bash
# Patch a local FFmpeg 8.0 checkout with vf_xdna_sr, link libxdna_sr.so, and build a minimal ffmpeg that
# can run the filter. LOCAL ONLY -- never pushed; upstreaming the filter is a separate owner-gated act.
#
# Assumes the checkout already exists at target/ffmpeg-xdna (n8.0) and libxdna_sr.so is built
# (cargo build -p npu-sr-capi). Idempotent: re-running re-copies + rebuilds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"        # xdna-engine repo root
FFDIR="$ROOT/target/ffmpeg-xdna"
LIBDIR="$ROOT/rust/target/debug"                # where libxdna_sr.so lives
HDR="$ROOT/rust/npu-sr-capi/include"

[ -d "$FFDIR" ] || { echo "clone first: git clone --depth 1 -b n8.0 https://git.ffmpeg.org/ffmpeg.git $FFDIR"; exit 1; }
[ -f "$LIBDIR/libxdna_sr.so" ] || { echo "build the lib first: (cd $ROOT/rust && cargo build -p npu-sr-capi)"; exit 1; }

cp "$ROOT/ffmpeg/vf_xdna_sr.c" "$FFDIR/libavfilter/"
cp "$HDR/xdna_sr.h" "$FFDIR/libavfilter/"

# Register the filter: extern decl in allfilters.c + the Makefile object (gated by CONFIG_XDNA_SR_FILTER).
grep -q "ff_vf_xdna_sr" "$FFDIR/libavfilter/allfilters.c" || \
  sed -i 's/^extern const FFFilter ff_vf_vflip;/extern const FFFilter ff_vf_xdna_sr;\nextern const FFFilter ff_vf_vflip;/' \
    "$FFDIR/libavfilter/allfilters.c"
grep -q "vf_xdna_sr.o" "$FFDIR/libavfilter/Makefile" || \
  echo 'OBJS-$(CONFIG_XDNA_SR_FILTER)             += vf_xdna_sr.o' >> "$FFDIR/libavfilter/Makefile"

cd "$FFDIR"
# Minimal build: just enough to decode/encode a test clip and run the filter. Link libxdna_sr.so.
./configure \
  --enable-filter=xdna_sr \
  --extra-cflags="-I$HDR" \
  --extra-ldflags="-L$LIBDIR -Wl,-rpath,$LIBDIR" \
  --extra-libs="-lxdna_sr" \
  --disable-doc --disable-htmlpages --disable-manpages --disable-txtpages \
  >/tmp/xdna-ffconf.log 2>&1 || { echo "configure FAILED (see /tmp/xdna-ffconf.log)"; tail -20 /tmp/xdna-ffconf.log; exit 1; }
make -j"$(nproc)" ffmpeg >/tmp/xdna-ffmake.log 2>&1 || { echo "make FAILED (see /tmp/xdna-ffmake.log)"; tail -30 /tmp/xdna-ffmake.log; exit 1; }

echo "built: $FFDIR/ffmpeg"
echo "run with: LD_LIBRARY_PATH=$LIBDIR $FFDIR/ffmpeg -i in.mp4 -vf xdna_sr=schedule=artifacts/espcn/espcn.json out.mp4"
