#!/usr/bin/env python3
"""Refuse a served artifact that would reach production through someone else's payload.

install.sh stages only the artifacts the served scenarios reference, as symlinks from the
production root into $ENGINE_ARTIFACTS. That single hop is what keeps tens of GB of model
weights out of the install root, and it only holds while the hop names the canonical store.

On 2026-11-12 the live install resolved 16 of its 18 served identities through
`<store>/accepted-cls-baseline-20261021/artifacts/` (an accepted-run snapshot) and 2 through a
`build/` lane dir, so a storage census that walked the "cold" lane payload was walking
production's own inputs -- and `readlink -f`, which install.sh uses to stage, is precisely the
hop that hides it: a store member that is itself a symlink resolves clean out of the store.

Two rules, nothing else -- a different disk or partition stays legal:

  root    the artifacts root resolves to a real directory that is not the repository, not a
          lane payload root ($XDNA_BUILD / $XDNA_SCRATCH / a `wt-*` worktree), and carries no
          snapshot or rollback component.
  target  a staged source resolves INSIDE the resolved root and carries no such component.

A snapshot/rollback component is any `accepted-*`, `*.prev-*`, `*.prepin-*`, `*.staging-*` or
`*baseline-<digits>` name: those are dated copies of somebody's payload, so they are never the
canonical identity a served scenario should load.

Usage:
  check_served_artifact_source.py root      <artifacts_root>
  check_served_artifact_source.py target    <artifacts_root> <source_path>
  check_served_artifact_source.py installed <engine_root> [<artifacts_root>]

`installed` audits an existing install's artifacts.manifest.json and names every served identity
that resolves outside the canonical store -- run it before any census or cleanup of a directory
that looks cold.

Exits 0 when the path is legal, 1 with the reason on stderr. Both checks also import as
check_root() / check_target() (ValueError) for scripts/prepared_install.py.
"""
import os
from pathlib import Path
import re
import sys

# Date-stamped or generation-stamped copies of a payload. See the module docstring.
SNAPSHOT_COMPONENT = re.compile(
    r"(?:^accepted-)"
    r"|(?:\.prev-)"
    r"|(?:\.prepin-)"
    r"|(?:^prepin-)"
    r"|(?:\.staging-)"
    r"|(?:baseline[-_]\d{4,})"
)
# Lane payload roots that are always wrong for served bytes, beyond the $XDNA_* roots passed in.
# `build`/`builds`/`scratch` are the lane families this workspace writes under one data root
# (scripts/lib/data_root.sh), `wt-` a linked worktree: all three are someone's working payload,
# and their names are what survives when $XDNA_BUILD/$XDNA_SCRATCH are not in the environment.
LANE_COMPONENT = re.compile(r"^(?:wt-|build$|builds$|scratch$)")


def _offending_components(path: Path):
    """Snapshot- or lane-shaped components in `path`, in order."""
    return [c for c in path.parts if SNAPSHOT_COMPONENT.search(c) or LANE_COMPONENT.search(c)]


def _forbidden_roots():
    """The lane payload roots this machine declares, by their real names.

    Deliberately not $REPO: `XDNA_DATA` defaults to `$REPO/data`, so a repo-ancestry rule would
    refuse the plain-laptop layout's own canonical store.
    """
    roots = []
    for var in ("XDNA_BUILD", "XDNA_SCRATCH", "XDNA_CAS"):
        value = os.environ.get(var, "")
        if value:
            roots.append(Path(value).resolve())
    return roots


def _describe(path: Path) -> str:
    return str(path)


def _reject(what: str, path: Path, resolved: Path, why: str) -> None:
    note = "" if resolved == path else f" (resolves to {_describe(resolved)})"
    raise ValueError(f"{what} {_describe(path)}{note} refused: {why}")


def check_root(artifacts_root) -> Path:
    """Return the resolved artifacts root, or raise ValueError explaining why it is refused."""
    root = Path(artifacts_root)
    resolved = root.resolve()
    # Shape before existence: a snapshot- or lane-shaped root is refused for the structural
    # reason whether or not it happens to be present, and "is not a directory" is the weaker claim.
    offenders = _offending_components(resolved)
    if offenders:
        _reject("artifacts root", root, resolved,
                f"it names a snapshot or lane payload ({', '.join(offenders)}) -- served bytes must "
                f"live in the canonical store, and a dated copy is not a served identity")
    for forbidden in _forbidden_roots():
        if resolved == forbidden or resolved.is_relative_to(forbidden):
            _reject("artifacts root", root, resolved,
                    f"it sits inside a lane payload root {_describe(forbidden)} -- builds and "
                    f"scratch work are not where production reads its models")
    if not resolved.is_dir():
        _reject("artifacts root", root, resolved, "it is not a directory")
    return resolved


def check_target(artifacts_root, source) -> Path:
    """Return the resolved staged source, or raise ValueError explaining why it is refused."""
    root = check_root(artifacts_root)
    src = Path(source)
    resolved = src.resolve()
    if not resolved.exists():
        _reject("served artifact source", src, resolved, "it does not exist")
    if not (resolved == root or resolved.is_relative_to(root)):
        _reject("served artifact source", src, resolved,
                f"it resolves outside the artifacts root {_describe(root)} -- staging it would "
                f"make the install depend on a payload outside the canonical store")
    offenders = _offending_components(resolved)
    if offenders:
        _reject("served artifact source", src, resolved,
                f"it names a snapshot or lane payload ({', '.join(offenders)})")
    return resolved


def check_installed(engine_root, artifacts_root=None):
    """Audit what an EXISTING install actually resolves through.

    The cleanup lane needed this on 2026-11-12 and did not have it: an install can verify
    cleanly against its own manifest while every served identity resolves through a lane payload,
    because the manifest records where the link points, not where the bytes may not come from.
    Returns a list of (path, reason); empty means the install is free of such hops.
    """
    root = Path(artifacts_root) if artifacts_root else Path(os.environ.get("XDNA_ARTIFACTS", ""))
    if not str(root):
        raise ValueError("no artifacts root: pass one or set XDNA_ARTIFACTS")
    engine_root = Path(engine_root)
    manifest = engine_root / "artifacts.manifest.json"
    if not manifest.is_file():
        raise ValueError(f"no artifacts manifest at {manifest}")
    import json
    violations = []
    for entry in json.loads(manifest.read_text()).get("artifacts", []):
        staged = engine_root / entry["path"]
        if not staged.exists() and not staged.is_symlink():
            violations.append((entry["path"], "staged entry is missing"))
            continue
        try:
            check_target(root, staged)
        except ValueError as exc:
            violations.append((entry["path"], str(exc)))
    return violations


def main(argv) -> int:
    usage = (__doc__ or "").split("Usage:", 1)[-1].strip()
    try:
        if len(argv) in (3, 4) and argv[1] == "root":
            resolved = check_root(argv[2])
        elif len(argv) == 4 and argv[1] == "target":
            resolved = check_target(argv[2], argv[3])
        elif len(argv) in (3, 4) and argv[1] == "installed":
            violations = check_installed(argv[2], argv[3] if len(argv) == 4 else None)
            for path, reason in violations:
                print(f"[artifact-root-gate] VIOLATION {path}: {reason}", file=sys.stderr)
            if violations:
                print(f"[artifact-root-gate] REFUSED: {len(violations)} served artifact(s) in "
                      f"{argv[2]} resolve outside the canonical store", file=sys.stderr)
                return 1
            print(f"OK: every served artifact in {argv[2]} resolves inside the canonical store")
            return 0
        else:
            print(usage, file=sys.stderr)
            return 2
    except (ValueError, OSError, ValueError) as exc:
        print(f"[artifact-root-gate] REFUSED: {exc}", file=sys.stderr)
        return 1
    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
