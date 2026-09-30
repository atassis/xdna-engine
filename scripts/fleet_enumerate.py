#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fleet enumeration (F0.2): dedup kernel compiles and operator devices across served designs.

Kernel compiles: read each design's kcc.jsonl (scripts/kcc_log_launcher.sh) and count distinct
(source_sha256, normalized args) pairs. Normalization drops the two things that vary per tmpdir
for an otherwise-identical compile: the -include-pch operand and any -I<path> under a work dir.

Operator devices: split an aie.mlir file on its top-level `aie.device(` blocks, strip `loc(...)`
spans and the symbol name after `@`, and sha256 each block. This is WEAKER than aiecc's own
DeviceCache::deviceKey (which prints the device without locations and hashes linked kernel-object
content, not just the device text) -- internal buffer/lock names can still differ between two
otherwise-identical devices. So the device-unique count from this script is a LOWER BOUND on real
device dedup, not the number DeviceCache would produce.
"""
import hashlib
import json
import re
import sys
from pathlib import Path


def _normalize_args(args):
    out = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "-include-pch":
            skip = True
            continue
        if a.startswith("-I") and "/work/" in a:
            out.append("-I<work>")
            continue
        out.append(a)
    return out


def summarize(jsonl_paths):
    """summarize(jsonl_paths) -> {"compiles", "unique", "per_source": {path: {...}}}."""
    per_source = {}
    seen = set()
    total = 0
    for p in jsonl_paths:
        p = Path(p)
        compiles = 0
        unique_keys = set()
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["source_sha256"], tuple(_normalize_args(rec["args"])))
            compiles += 1
            unique_keys.add(key)
            seen.add(key)
        total += compiles
        per_source[str(p)] = {"compiles": compiles, "unique": len(unique_keys)}
    return {"compiles": total, "unique": len(seen), "per_source": per_source}


def compile_seconds(jsonl_paths):
    """Total and unique-key wall seconds from each record's start/end (older logs lack these
    fields; records missing either are skipped and do not count toward either total)."""
    total_s = 0.0
    unique_s = 0.0
    seen = set()
    for p in jsonl_paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "start" not in rec or "end" not in rec:
                continue
            dur = rec["end"] - rec["start"]
            total_s += dur
            key = (rec["source_sha256"], tuple(_normalize_args(rec["args"])))
            if key not in seen:
                seen.add(key)
                unique_s += dur
    return {"total_s": total_s, "unique_s": unique_s}


_DEVICE_START = re.compile(r"^\s*aie\.device\(")
_LOC = re.compile(r'\s*loc\("[^"]*":\d+:\d+\)')
_LOC_NAMED = re.compile(r'\s*loc\("[^"]*"\)')
_AT_NAME = re.compile(r"@[A-Za-z_][A-Za-z0-9_]*")


def _strip_for_key(text):
    text = _LOC.sub("", text)
    text = _LOC_NAMED.sub("", text)
    text = _AT_NAME.sub("@_", text)
    return text


def device_keys(aie_mlir_path):
    """device_keys(path) -> [(symbol_name, sha256_hex), ...] for each top-level aie.device block.

    LOWER BOUND on device dedup -- see module docstring.
    """
    text = Path(aie_mlir_path).read_text()
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)
    while i < n:
        if not _DEVICE_START.match(lines[i]):
            i += 1
            continue
        m = _AT_NAME.search(lines[i])
        name = m.group(0)[1:] if m else "<anon>"
        depth = lines[i].count("{") - lines[i].count("}")
        start = i
        i += 1
        while i < n and depth > 0:
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
        block = "".join(lines[start:i])
        key = hashlib.sha256(_strip_for_key(block).encode()).hexdigest()
        out.append((name, key))
    return out


def _find_kcc_logs(s0f0_dir):
    return sorted(Path(s0f0_dir).glob("*/kcc.jsonl"))


def _find_mlir_files(s0f0_dir):
    return sorted(Path(s0f0_dir).glob("*/mlir/**/aie.mlir"))


def main(argv):
    if len(argv) != 2:
        print("usage: fleet_enumerate.py <s0f0-dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])

    kcc_logs = _find_kcc_logs(root)
    kernels = summarize(kcc_logs) if kcc_logs else {"compiles": 0, "unique": 0, "per_source": {}}
    seconds = compile_seconds(kcc_logs) if kcc_logs else {"total_s": 0.0, "unique_s": 0.0}

    dev_total = 0
    dev_seen = set()
    per_design_devices = {}
    for mlir in _find_mlir_files(root):
        design = mlir.relative_to(root).parts[0]
        keys = device_keys(mlir)
        entry = per_design_devices.setdefault(design, {"devices": 0, "unique_keys": set()})
        for _name, k in keys:
            entry["devices"] += 1
            entry["unique_keys"].add(k)
            dev_seen.add(k)
            dev_total += 1

    print(f"kernel_compiles\t{kernels['compiles']}\t{kernels['unique']}")
    print(f"kernel_compile_seconds\t{seconds['total_s']:.2f}\t{seconds['unique_s']:.2f}")
    print(f"operator_devices\t{dev_total}\t{len(dev_seen)}")
    print("# per-design (kernels: compiles/unique from kcc.jsonl; devices: count/unique-lower-bound)")
    designs = sorted(set(list(per_design_devices) +
                          [Path(p).parent.name for p in kernels["per_source"]]))
    for design in designs:
        k = kernels["per_source"].get(str(root / design / "kcc.jsonl"), {"compiles": 0, "unique": 0})
        d = per_design_devices.get(design, {"devices": 0, "unique_keys": set()})
        print(f"{design}\t{k['compiles']}\t{k['unique']}\t{d['devices']}\t{len(d['unique_keys'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
