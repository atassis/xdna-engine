"""Package a self-contained resident-forward artifact from existing build outputs."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from artifact_meta import ARTIFACT_KIND, PARAMETER_ABI, elf_sha256


BUILD_FILES = ("pack.json", "fwd_layout.json", "params.txt", "gen_args.txt")
# full_elf_config.json names build-only PDI/TXN inputs, not packaged companions.
OPTIONAL_BUILD_FILES = ("gen_env.txt",)
RESIDENT_KINDS = {
    "resident_forward_raw",
    "resident_forward_onecmd",
    "resident_forward_ladder",
    ARTIFACT_KIND,
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing required companion: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"missing required directory: {path}")


def copy_file(source: str, destination: str) -> str:
    try:
        os.link(Path(source).resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=copy_file, symlinks=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def derive_store(source: Path, out: Path) -> None:
    """Derive portable metadata, preserving verified blobs and original provenance by hash."""
    if out.exists() or out.is_symlink():
        raise ValueError(f'derived store output already exists: {out}')
    manifest_path = source / 'manifest.json'
    receipt_path = source / '.recipe-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    files = receipt['files']
    actual = {str(p.relative_to(source)) for p in source.rglob('*')
              if p.is_file() and p.name != '.recipe-manifest.json'}
    if set(files) != actual or 'manifest.json' not in files:
        raise ValueError('store receipt file inventory mismatch')
    for rel, digest in files.items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha256_file(source / rel) != digest:
            raise ValueError(f'store receipt digest mismatch: {rel}')
    if 'source_dir' not in manifest:
        # Already portable: still validate the entire receipt before copying it.
        copy_tree(source, out)
        return
    source_dir = manifest.pop('source_dir')
    if not isinstance(source_dir, str):
        raise ValueError('store source_dir must be a string')
    copy_tree(source, out)
    # copy_tree may hardlink: replace metadata inodes, never write through them.
    (out / 'manifest.json').unlink()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / '.recipe-manifest.json').unlink()
    derived = {
        'command': 'package_artifact.py --derive-store (portable-store-v1)',
        'derivation': {
            'version': 1,
            'source_manifest_sha256': sha256_file(manifest_path),
            'source_receipt_sha256': sha256_file(receipt_path),
            'removed_fields': ['source_dir'],
            'source_dir_sha256': hashlib.sha256(source_dir.encode()).hexdigest(),
        },
        'files': files | {'manifest.json': sha256_file(out / 'manifest.json')},
    }
    (out / '.recipe-manifest.json').write_text(json.dumps(derived, indent=2, sort_keys=True) + '\n')


def relative_build_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"meta elf must stay inside build: {value}")
    return path


def required_weight_files(meta: dict) -> tuple[str, ...]:
    nlayer = meta.get("nlayer")
    if isinstance(nlayer, bool) or not isinstance(nlayer, int) or nlayer < 0:
        raise ValueError("meta.json is missing a non-negative integer `nlayer`")
    return tuple(f"w{layer}.npy" for layer in range(nlayer))


def manifest_blob_paths(store: Path, kind: str) -> tuple[Path, ...]:
    manifest_path = store / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid store manifest: {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid store manifest: {manifest_path}")

    paths: list[Path] = []

    def blob_path(blob: object) -> Path:
        if not isinstance(blob, str) or not blob or Path(blob).is_absolute() or ".." in Path(blob).parts:
            raise ValueError(f"invalid store blob reference: {blob!r}")
        return store / "blobs" / f"{blob}.bin"

    def visit(value: object) -> None:
        if isinstance(value, dict):
            blob = value.get("blob")
            if blob is not None:
                paths.append(blob_path(blob))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    entries = ("embedding", "final_norm") if kind == "resident_forward_raw" else ("embedding",)
    for name in entries:
        entry = manifest.get(name)
        if not isinstance(entry, dict) or "blob" not in entry:
            raise ValueError(f"store manifest is missing `{name}` blob entry")
        paths.append(blob_path(entry["blob"]))
    visit(manifest)
    return tuple(paths)


def package(out: Path, build: Path, meta_path: Path, weights: Path, store: Path) -> None:
    out = Path(out)
    build = Path(build)
    meta_path = Path(meta_path)
    weights = Path(weights)
    store = Path(store)
    require_dir(build)
    require_file(meta_path)
    require_dir(weights)
    require_dir(store)
    require_file(store / "manifest.json")
    if out.exists() or out.is_symlink():
        raise ValueError(f"artifact output already exists: {out}")

    meta = json.loads(meta_path.read_text())
    kind = meta.get("kind")
    if kind not in RESIDENT_KINDS:
        raise ValueError(f"meta kind {kind!r} is not a resident-forward artifact")
    elf = meta.get("elf")
    if not isinstance(elf, str) or not elf:
        raise ValueError("meta.json is missing `elf`")
    elf_path = relative_build_path(elf)
    elf_source = build / elf_path
    if not elf_source.is_file():
        elf_source = build / f"{elf_path}.zst"
    require_file(elf_source)
    if kind == ARTIFACT_KIND:
        if meta.get("kv_ring_parameter_abi") != PARAMETER_ABI:
            raise ValueError("incompatible KV-ring parameter ABI")
        if meta.get("elf_sha256") != elf_sha256(build / elf_path):
            raise ValueError("ELF digest differs from total-count metadata")
    for name in required_weight_files(meta) + ("w_head.npy",):
        require_file(weights / name)
    for blob_path in manifest_blob_paths(store, kind):
        require_file(blob_path)
    for name in BUILD_FILES:
        require_file(build / name)

    pack = json.loads((build / "pack.json").read_text())
    provenance = pack.get("provenance") if isinstance(pack, dict) else None
    if isinstance(provenance, dict):
        iron = {
            "commit": provenance.get("iron_commit"),
            "dirty": provenance.get("iron_dirty"),
            "identity": provenance.get("iron_identity"),
        }
        if all(value is not None for value in iron.values()):
            meta["iron"] = iron

    staging = out.with_name(f"{out.name}.staging-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise ValueError(f"staging path already exists: {staging}")
    try:
        staging.mkdir(parents=True)
        meta["weight_dir"] = "weights"
        meta["embedding_store"] = "store"
        (staging / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        elf_destination = staging / elf_source.relative_to(build)
        elf_destination.parent.mkdir(parents=True, exist_ok=True)
        copy_file(str(elf_source), str(elf_destination))
        for name in BUILD_FILES + OPTIONAL_BUILD_FILES:
            source = build / name
            if source.is_file():
                if name == 'gen_env.txt':
                    # Keep RF_INST and runtime flags; these two describe build-only locations.
                    lines = source.read_text().splitlines(keepends=True)
                    removed = [line for line in lines if line.startswith(('RF_BUILD=', 'RF_STORE='))]
                    (staging / name).write_text(''.join(line for line in lines if line not in removed))
                    if removed:
                        (staging / 'packaging-derivation.json').write_text(json.dumps({
                            'command': 'package_artifact.py (portable-gen-env-v1)',
                            'source_gen_env_sha256': sha256_file(source),
                            'gen_env_sha256': sha256_file(staging / name),
                            'removed_fields': [line.split('=', 1)[0] for line in removed],
                            'removed_values_sha256': [hashlib.sha256(line.split('=', 1)[1].rstrip('\n').encode()).hexdigest()
                                                      for line in removed],
                        }, indent=2) + '\n')
                else:
                    copy_file(str(source), str(staging / name))
        copy_file(str(weights / "w_head.npy"), str(staging / "w_head.npy"))
        copy_tree(weights, staging / "weights")
        if (store / '.recipe-manifest.json').is_file() or 'source_dir' in json.loads((store / 'manifest.json').read_text()):
            derive_store(store, staging / 'store')
        else:
            copy_tree(store, staging / "store")
        os.replace(staging, out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--derive-store", action="store_true")
    parser.add_argument("--build", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    if args.derive_store:
        derive_store(args.store, args.out)
        return 0
    if any(value is None for value in (args.build, args.meta, args.weights)):
        parser.error('packaging needs --build --meta --weights')
    package(args.out, args.build, args.meta, args.weights, args.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
