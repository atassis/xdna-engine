#!/usr/bin/env python3
"""Validate release artifact IRON provenance against the declared source commit."""
import argparse
import json
from pathlib import Path


def validate(meta_path: Path, expected: str) -> str | None:
    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"cannot read artifact metadata {meta_path}: {error}"
    iron = metadata.get("iron")
    if not isinstance(iron, dict):
        return f"artifact records no IRON provenance: {meta_path}"
    if iron.get("commit") != expected:
        return f"artifact IRON commit is {iron.get('commit')!r}, expected {expected}: {meta_path}"
    if iron.get("dirty") is not False:
        return f"artifact IRON provenance is not explicitly clean: {meta_path}"
    if iron.get("identity") != f"pinned:{expected}":
        return f"artifact IRON identity is not the pinned release source: {meta_path}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    parser.add_argument("meta", type=Path)
    args = parser.parse_args()
    error = validate(args.meta, args.expected)
    if error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
