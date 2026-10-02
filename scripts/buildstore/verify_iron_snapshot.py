#!/usr/bin/env python3
"""Check that a replay IRON snapshot exactly covers its recorded source reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import NoReturn


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fail(message: str) -> NoReturn:
    print(f"IRON snapshot: {message}", file=sys.stderr)
    raise SystemExit(1)


def snapshot_fields(path: Path) -> tuple[str, str, str, str, dict[str, str]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    if len(lines) < 5 or lines[0] != "iron-source-snapshot-v1":
        fail("invalid snapshot header")
    identity_parts = lines[1].split("\t", 1)
    dir_parts = lines[2].split("\t", 1)
    manifest_parts = lines[3].split("\t", 2)
    if len(identity_parts) != 2 or len(dir_parts) != 2 or len(manifest_parts) != 3:
        fail("invalid snapshot metadata")
    identity_key, identity = identity_parts
    dir_key, source_dir = dir_parts
    manifest_key, manifest_digest, manifest_path = manifest_parts
    if identity_key != "identity" or dir_key != "dir" or manifest_key != "manifest" \
            or not manifest_digest or not manifest_path:
        fail("invalid snapshot metadata")
    files = {}
    for line in lines[4:]:
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0] != "file" or not parts[1] or not parts[2] or parts[2] in files:
            fail("invalid snapshot file list")
        files[parts[2]] = parts[1]
    return identity, source_dir, manifest_digest, manifest_path, files


def verify(snapshot_path: Path, manifest_path: Path, source_dir: str) -> str:
    identity, snapshot_dir, manifest_digest, declared_manifest, files = snapshot_fields(snapshot_path)
    if declared_manifest != str(manifest_path) or digest(manifest_path) != manifest_digest:
        fail("manifest does not match the snapshot")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read manifest: {error}")
    env, reads = manifest.get("env"), manifest.get("reads")
    if not isinstance(env, dict) or not isinstance(reads, dict):
        fail("manifest has no source inputs")
    if snapshot_dir != source_dir or env.get("IRON_DIR") != source_dir \
            or env.get("IRON_SOURCE_IDENTITY") != identity:
        fail("source identity does not match the manifest")
    prefix = source_dir.rstrip("/") + "/"
    expected = {path: value for path, value in reads.items() if path.startswith(prefix)}
    if not expected or set(files) != set(expected):
        fail("snapshot file set does not match recorded IRON reads")
    for name, expected_digest in expected.items():
        path = Path(name)
        if not isinstance(expected_digest, str) or files[name] != expected_digest \
                or not path.is_file() or digest(path) != expected_digest:
            fail(f"recorded IRON input changed: {name}")
    return identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    print(verify(args.snapshot, args.manifest, args.dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
