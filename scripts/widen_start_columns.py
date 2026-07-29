#!/usr/bin/env python3
"""Widen an xclbin's AIE_PARTITION start_columns so the driver's column solver can place it.

WHY THIS EXISTS. A narrow design (column_width=N < device columns) is emitted by aiecc with a
SINGLE allowed start column, e.g. column_width=4, start_columns=["1"]. Two such designs then
cannot be placed side by side even though 4+4 fits in 8 columns -- they both demand column 1, so
every xclbin-to-xclbin transition is a full array reprogram. Widening start_columns to every legal
offset (0 .. device_columns - column_width) lets the driver place them DISJOINTLY, which is exactly
the co-residency arm `partition_ab_probe` measures (`ctxln(4colW) <-> cast(4colW)`).

There is no generator or aiecc knob for this: `from_name()` exposes only `n_cols`, and aiecc emits
one start column. The `_p4cW` xclbins were originally produced by hand-editing the section, which is
why nothing in-tree could rebuild them. This script is that edit, made reproducible.

The AIE program is untouched -- only the partition claim changes. The instruction stream is
byte-identical between the narrow and widened builds (verified for ctxln/cast 512x1024_p4c), so no
re-gate of the kernel itself is implied by running this.

Usage:
  widen_start_columns.py --in final_x_p4c.xclbin --out final_x_p4cW.xclbin [--device-columns 8]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _xclbinutil() -> str:
    exe = shutil.which("xclbinutil") or "/opt/xilinx/xrt/bin/xclbinutil"
    if not Path(exe).exists():
        sys.exit("ERROR: xclbinutil not found (need XRT on PATH)")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument(
        "--device-columns",
        type=int,
        default=8,
        help="columns on the target device (NPU2=8, NPU1=4). Default 8.",
    )
    a = ap.parse_args()
    xu = _xclbinutil()

    with tempfile.TemporaryDirectory() as td:
        js = Path(td) / "aie_partition.json"
        subprocess.run(
            [xu, "--dump-section", f"AIE_PARTITION:JSON:{js}", "--input", str(a.src), "--force"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if not js.exists() or js.stat().st_size == 0:
            sys.exit(
                f"ERROR: {xu} could not dump AIE_PARTITION as JSON from {a.src.name}.\n"
                "       Need an xclbinutil whose AIE_PARTITION section has a JSON payload parser."
            )
        doc = json.loads(js.read_text())
        part = doc["aie_partition"]["partition"]
        width = int(part["column_width"])
        if width > a.device_columns:
            sys.exit(f"ERROR: column_width {width} > device columns {a.device_columns}")
        before = list(part["start_columns"])
        # Every legal offset: a width-W partition can start at 0 .. device_columns - W.
        part["start_columns"] = [str(c) for c in range(a.device_columns - width + 1)]
        js.write_text(json.dumps(doc, indent=2))
        a.dst.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            [
                xu,
                "--input", str(a.src),
                "--replace-section", f"AIE_PARTITION:JSON:{js}",
                "--output", str(a.dst),
                "--force",
            ],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0:
            # KNOWN GAP (measured 2026-07-29, XRT 2.21.75): this xclbinutil can DUMP
            # AIE_PARTITION as JSON but not re-encode it -- "Section 'AIE_PARTITION' (32) missing
            # payload parser". Dumping it as RAW yields 0 bytes, and the metadata is not stored as
            # text in the container, so there is nothing to patch in place either. Changing the
            # start_columns array length also shifts every downstream section offset (measured: the
            # shipped p4c/p4cW pair differs from byte 305 to EOF), so a hand-rolled container patch
            # would have to rewrite the axlf section table -- a silent-corruption risk not worth
            # taking for a diagnostic artifact. Needs an xclbinutil with the JSON writer.
            sys.exit(
                "ERROR: this xclbinutil cannot re-encode AIE_PARTITION from JSON.\n"
                f"       {rc.stderr.strip().splitlines()[0] if rc.stderr.strip() else ''}\n"
                "       Widening start_columns needs an xclbinutil with an AIE_PARTITION JSON\n"
                "       payload parser; RAW dump is empty and the section is not text-patchable.\n"
                "       Until then the _p4cW xclbins cannot be rebuilt -- keep the shipped copies."
            )
    print(
        f"[widen_start_columns] {a.src.name}: column_width={width} "
        f"start_columns {before} -> {part['start_columns']}  =>  {a.dst.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
