#!/usr/bin/env python3
"""Build/verify `$ENGINE_ROOT/artifacts.manifest.json`: one identity per served
artifact, so a source artifact that changed underneath an install (rebuilt in
place, wrong pin, truncated copy) is DETECTED instead of silently served.

Identity per artifact (one of):
  - dir with meta.json:  meta.json's own sha256 + whatever of {toolchain.hash,
    iron.commit, artifact_hash, sequence_name} it carries. These are the
    fields the rest of this toolchain already uses to tell one build from
    another (see install.sh's own toolchain/IRON gates) -- reuse them rather
    than inventing a new identity scheme.
  - dir with no meta.json: sha256 over the sorted (relative path, size) of
    every file inside. Cheap, catches added/removed/resized files; does not
    read weight bytes.
  - file <= 64 MiB: sha256 of the file.
  - file > 64 MiB: size_bytes + mtime_ns_unverified (named so a reader knows
    this is NOT a content hash -- mtime survives `cp -a`/reflink but not
    every transfer, so a mismatch here is a strong signal, agreement is not
    a proof).

Usage:
  artifact_manifest.py build  <engine_root> <artifacts_root> <rel1> [rel2 ...] > manifest.json
  artifact_manifest.py verify <engine_root> <manifest.json>
"""
import hashlib
import json
import sys
from pathlib import Path

LARGE_FILE_THRESHOLD = 64 * 1024 * 1024
META_FIELDS_OF_INTEREST = ("artifact_hash", "sequence_name")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_listing_identity(d: Path) -> dict:
    entries = sorted(
        (str(p.relative_to(d)), p.stat().st_size)
        for p in d.rglob("*") if p.is_file()
    )
    h = hashlib.sha256(repr(entries).encode()).hexdigest()
    return {"kind": "dir_listing", "dir_listing_sha256": h, "file_count": len(entries)}


def meta_identity(meta_path: Path) -> dict:
    ident = {"kind": "dir_meta", "meta_sha256": sha256_file(meta_path)}
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ident
    for field in META_FIELDS_OF_INTEREST:
        if field in meta:
            ident[field] = meta[field]
    tc = meta.get("toolchain")
    if isinstance(tc, dict) and "hash" in tc:
        ident["toolchain_hash"] = tc["hash"]
    iron = meta.get("iron")
    if isinstance(iron, dict) and "commit" in iron:
        ident["iron_commit"] = iron["commit"]
    return ident


def file_identity(p: Path) -> dict:
    size = p.stat().st_size
    if size <= LARGE_FILE_THRESHOLD:
        return {"kind": "file_sha256", "sha256": sha256_file(p)}
    st = p.stat()
    return {"kind": "file_size_mtime_unverified", "size_bytes": st.st_size,
            "mtime_ns_unverified": st.st_mtime_ns}


def identity_of(target: Path) -> dict:
    if target.is_dir():
        meta = target / "meta.json"
        if meta.is_file():
            return meta_identity(meta)
        return dir_listing_identity(target)
    return file_identity(target)


def build(engine_root: Path, artifacts_root: Path, rels: list[str]) -> dict:
    out = []
    for rel in rels:
        # `rel` is "artifacts/<sub>" (as scenario fields spell it); `artifacts_root` is
        # $ENGINE_ARTIFACTS, which already IS that "artifacts" directory -- strip the prefix
        # the same way install.sh's staging loop does, or every path doubles it.
        sub = rel[len("artifacts/"):] if rel.startswith("artifacts/") else rel
        target = (artifacts_root / sub).resolve()
        if not target.exists():
            print(f"# WARN: not staged (missing under artifacts root): {rel}", file=sys.stderr)
            continue
        out.append({"path": rel, "target": str(target), "identity": identity_of(target)})
    return {"engine_root": str(engine_root), "artifacts_root": str(artifacts_root), "artifacts": out}


def verify(engine_root: Path, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    mismatches = 0
    for entry in manifest["artifacts"]:
        link = engine_root / entry["path"]  # entry["path"] already starts with "artifacts/"
        target = Path(entry["target"])
        if not target.exists():
            print(f"MISMATCH {entry['path']}: source gone ({target})")
            mismatches += 1
            continue
        if not link.exists():
            print(f"MISMATCH {entry['path']}: not staged at {link}")
            mismatches += 1
            continue
        current = identity_of(target)
        if current != entry["identity"]:
            print(f"MISMATCH {entry['path']}: identity changed\n"
                  f"  recorded: {entry['identity']}\n"
                  f"  current:  {current}")
            mismatches += 1
    if mismatches:
        print(f"{mismatches} artifact(s) changed since install.")
    else:
        print(f"OK: {len(manifest['artifacts'])} artifact(s) match the recorded identity.")
    return 1 if mismatches else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    mode = sys.argv[1]
    if mode == "build":
        if len(sys.argv) < 4:
            print(__doc__, file=sys.stderr)
            return 2
        engine_root, artifacts_root, rels = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
        print(json.dumps(build(engine_root, artifacts_root, rels), indent=1))
        return 0
    if mode == "verify":
        if len(sys.argv) != 4:
            print(__doc__, file=sys.stderr)
            return 2
        return verify(Path(sys.argv[2]), Path(sys.argv[3]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
