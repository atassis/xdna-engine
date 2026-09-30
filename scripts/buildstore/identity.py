# SPDX-License-Identifier: Apache-2.0
"""Artifact identity: what a device gate result is valid for. Provenance does not reach the device."""
import hashlib, json, pathlib

PROVENANCE_KEYS = ("toolchain", "iron", "generator")


def _file_sha(p):
    if p.name == "meta.json":
        m = json.loads(p.read_bytes())
        for k in PROVENANCE_KEYS:
            m.pop(k, None)
        return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def of(root):
    root = pathlib.Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(root)}\0{_file_sha(p)}\n".encode())
    return h.hexdigest()
