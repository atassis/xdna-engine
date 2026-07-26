# vf_xdna_sr -- NPU super-resolution FFmpeg filter

A standalone libavfilter video filter that upscales frames on the AMD XDNA2 NPU by calling the npu-sr
engine's C ABI (`libxdna_sr.so`). ffmpeg keeps decode/encode; the filter only upscales.

    ffmpeg -i in.mp4 -vf xdna_sr=schedule=artifacts/espcn/espcn.json:npu=1 out.mp4

Options:
- `schedule` -- path to the net schedule `<net>.json` (default `artifacts/espcn/espcn.json`).
- `npu` -- 1 uses the NPU frontier, 0 forces the CPU reference frontier (default 1).

## Build (local, out-of-tree)

FFmpeg has no stable filter-plugin ABI, so the filter is compiled into a local FFmpeg checkout. This is a
local dev build; **upstreaming the filter is a separate, owner-gated act.**

    # 1. build the engine C ABI (produces rust/target/debug/libxdna_sr.so + the generated header)
    (cd rust && cargo build -p npu-sr-capi)
    # 2. get a matching FFmpeg checkout (8.0 -- the API vf_xdna_sr.c targets)
    git clone --depth 1 -b n8.0 https://git.ffmpeg.org/ffmpeg.git target/ffmpeg-xdna
    # 3. patch + configure + build
    bash ffmpeg/apply.sh
    # 4. run (the lib must be on LD_LIBRARY_PATH; run from the repo root so the schedule's checkpoint resolves)
    LD_LIBRARY_PATH=rust/target/debug target/ffmpeg-xdna/ffmpeg \
      -i in.mp4 -vf xdna_sr=schedule=artifacts/espcn/espcn.json:npu=0 out.mp4

`apply.sh` copies `vf_xdna_sr.c` + `xdna_sr.h` into `libavfilter/`, registers the filter in
`allfilters.c` + the Makefile, and builds a minimal `ffmpeg` linking `libxdna_sr.so`.

## Notes

- Pixel format: the filter negotiates RGB24 (the engine does Y-only SR + bicubic chroma internally).
- Output dims = input dims x the schedule's scale factor.
- The filter is written against FFmpeg 8.0's `FFFilter` API; adjust for other major versions.
