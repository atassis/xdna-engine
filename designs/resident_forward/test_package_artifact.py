import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from package_artifact import package


def write_build(build: Path) -> None:
    for name in ("design.elf.zst", "pack.json", "fwd_layout.json", "params.txt", "gen_args.txt"):
        (build / name).write_text(name)


def write_meta(path: Path, *, elf: str = "design.elf", nlayer: int = 1) -> None:
    path.write_text(json.dumps({
        "kind": "resident_forward_ladder",
        "elf": elf,
        "nlayer": nlayer,
        "weight_dir": "/retired-scratch/streams",
        "embedding_store": "/retired-store",
    }))


def write_store(store: Path) -> None:
    (store / "blobs").mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({
        "embedding": {"blob": "embed"},
        "towers": {"vision": {"blob": "tower"}},
    }))
    (store / "blobs" / "embed.bin").write_bytes(b"embedding-store")
    (store / "blobs" / "tower.bin").write_bytes(b"tower-store")


class PackageArtifactTests(unittest.TestCase):
    def test_package_is_relocatable_after_sources_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            out = tmp_path / "artifact"
            for directory in (build, weights):
                directory.mkdir(parents=True)
            write_build(build)
            write_meta(source_meta)
            weight_source = tmp_path / "weight-source.npy"
            weight_source.write_bytes(b"weight-stream")
            (weights / "w0.npy").symlink_to(weight_source)
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)

            package(out, build, source_meta, weights, store)
            shutil.rmtree(build)
            shutil.rmtree(weights)
            shutil.rmtree(store)
            source_meta.unlink()
            weight_source.unlink()
            moved = tmp_path / "relocated-artifact"
            out.rename(moved)

            meta = json.loads((moved / "meta.json").read_text())
            self.assertEqual(meta["weight_dir"], "weights")
            self.assertEqual(meta["embedding_store"], "store")
            self.assertNotIn("/retired-", (moved / "meta.json").read_text())
            self.assertEqual((moved / meta["weight_dir"] / "w0.npy").read_bytes(), b"weight-stream")
            self.assertEqual((moved / "w_head.npy").read_bytes(), b"head-stream")
            self.assertEqual((moved / meta["embedding_store"] / "blobs" / "embed.bin").read_bytes(), b"embedding-store")
            self.assertEqual((moved / meta["embedding_store"] / "blobs" / "tower.bin").read_bytes(), b"tower-store")
            self.assertTrue(Path(f"{moved / meta['elf']}.zst").is_file())
            self.assertTrue(all(not path.is_symlink() for path in moved.rglob("*")))

    def test_package_refuses_missing_build_weight_and_store_companions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            write_meta(source_meta)
            (weights / "w0.npy").write_bytes(b"weight-stream")
            write_store(store)

            with self.assertRaisesRegex(ValueError, "w_head.npy"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

            (weights / "w_head.npy").write_bytes(b"head-stream")
            (build / "pack.json").unlink()
            with self.assertRaisesRegex(ValueError, "pack.json"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

            (build / "pack.json").write_text("pack.json")
            (store / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "manifest.json"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

    def test_package_refuses_missing_layer_streams_for_metadata_layer_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            write_meta(source_meta, nlayer=2)
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)

            with self.assertRaisesRegex(ValueError, "w0.npy"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

    def test_package_refuses_a_missing_manifest_blob_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            write_meta(source_meta)
            (weights / "w0.npy").write_bytes(b"weight-stream")
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)
            (store / "blobs" / "tower.bin").unlink()

            with self.assertRaisesRegex(ValueError, "tower.bin"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

    def test_package_requires_the_ladder_store_embedding_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            write_meta(source_meta)
            (weights / "w0.npy").write_bytes(b"weight-stream")
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)
            (store / "manifest.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "embedding"):
                package(tmp_path / "artifact", build, source_meta, weights, store)

    def test_package_preserves_a_nested_elf_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            out = tmp_path / "artifact"
            for directory in (build / "nested", weights):
                directory.mkdir(parents=True)
            for name in ("pack.json", "fwd_layout.json", "params.txt", "gen_args.txt"):
                (build / name).write_text(name)
            (build / "nested" / "design.elf").write_text("nested elf")
            write_meta(source_meta, elf="nested/design.elf")
            (weights / "w0.npy").write_bytes(b"weight-stream")
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)

            package(out, build, source_meta, weights, store)

            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["elf"], "nested/design.elf")
            self.assertEqual((out / meta["elf"]).read_text(), "nested elf")

    def test_package_refuses_elf_paths_outside_the_build_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            outside = tmp_path / "outside.elf"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            (weights / "w0.npy").write_bytes(b"weight-stream")
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)
            outside.write_text("outside elf")

            for elf in (str(outside), "../outside.elf"):
                source_meta = tmp_path / f"{len(elf)}.json"
                write_meta(source_meta, elf=elf)
                with self.assertRaisesRegex(ValueError, "inside build"):
                    package(tmp_path / f"artifact-{len(elf)}", build, source_meta, weights, store)

    def test_package_refuses_a_dangling_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build = tmp_path / "build"
            weights = tmp_path / "streams"
            store = tmp_path / "store"
            source_meta = tmp_path / "meta.json"
            out = tmp_path / "artifact"
            for directory in (build, weights):
                directory.mkdir()
            write_build(build)
            write_meta(source_meta)
            (weights / "w0.npy").write_bytes(b"weight-stream")
            (weights / "w_head.npy").write_bytes(b"head-stream")
            write_store(store)
            out.symlink_to(tmp_path / "missing-target")

            with self.assertRaisesRegex(ValueError, "artifact output already exists"):
                package(out, build, source_meta, weights, store)

    def test_served_collector_selects_the_resident_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scenarios").mkdir()
            (root / "scenarios" / "resident.toml").write_text(
                "[scenario]\nkind = \"generate\"\nname = \"resident\"\n"
                "[artifacts]\nresident = \"artifacts/gemma4-12b/resident\"\n"
            )
            config = root / "engine.toml"
            config.write_text('[[model]]\nname = "resident"\nscenario = "scenarios/resident.toml"\n')
            repo = Path(__file__).parents[2]
            result = subprocess.run(
                [sys.executable, repo / "scripts/lib/collect_served_artifacts.py", config, root],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout, "artifacts/gemma4-12b/resident\n")
