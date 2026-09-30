# SPDX-License-Identifier: Apache-2.0
"""Artifact identity: what a device gate result is valid for. Provenance does not reach the device."""
import hashlib, json, pathlib

PROVENANCE_KEYS = ("toolchain", "iron", "generator")


def _artifact_dir(p):
    """The artifact root `p` names: itself if it holds meta.json, else its nearest such ancestor."""
    if p.is_dir() and (p / "meta.json").is_file():
        return p
    for anc in p.parents:
        if anc.is_dir() and (anc / "meta.json").is_file():
            return anc
    return None


def _resolve_refs(value, visited):
    # meta.json fields that are absolute paths into another artifact (e.g. prefill's
    # decode_artifact.meta / weights_from) must not key identity on checkout location: substitute
    # the referenced artifact's own identity, recursively.
    if isinstance(value, str) and value.startswith("/"):
        p = pathlib.Path(value)
        if p.exists():
            art = _artifact_dir(p)
            if art is not None:
                art = art.resolve()
                if art in visited:
                    return value
                return "identity:" + of(art, visited | {art})
        return value
    if isinstance(value, dict):
        return {k: _resolve_refs(v, visited) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, visited) for v in value]
    return value


def _file_sha(p, visited):
    if p.name == "meta.json":
        m = json.loads(p.read_bytes())
        for k in PROVENANCE_KEYS:
            m.pop(k, None)
        m = _resolve_refs(m, visited)
        return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def of(root, _visited=frozenset()):
    root = pathlib.Path(root).resolve()
    visited = _visited | {root}
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(root)}\0{_file_sha(p, visited)}\n".encode())
    return h.hexdigest()
