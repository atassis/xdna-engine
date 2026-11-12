"""Offline fixtures: the served-artifact gate must refuse the 2026-11-12 shape.

That day the live install resolved 16 of 18 served identities through an accepted-run snapshot
inside the artifact store and 2 through a build lane, while every ordinary proof (manifest
verify, exclusivity, coldness) passed. The known-bad fixtures below are that layout.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lib.check_served_artifact_source import check_installed, check_root, check_target

CHECKER = Path(__file__).resolve().parent / "lib" / "check_served_artifact_source.py"
SNAPSHOT = "accepted-cls-baseline-20261021"


def store_fixture(root: Path) -> Path:
    """A data root with a canonical store, a snapshot inside it, and sibling lane roots."""
    artifacts = root / "data" / "artifacts"
    (artifacts / "bge-base" / "encoder").mkdir(parents=True)
    (artifacts / "bge-base" / "encoder" / "L0.npy").write_bytes(b"weights")
    snapshot = artifacts / SNAPSHOT / "artifacts" / "bge-base" / "encoder"
    snapshot.mkdir(parents=True)
    (snapshot / "L0.npy").write_bytes(b"weights")
    lane = root / "data" / "build" / "asr-repair-20261025" / "artifacts" / "parakeet"
    lane.mkdir(parents=True)
    (lane / "vocab.txt").write_text("vocab")
    (root / "data" / "scratch" / "fsr1-main").mkdir(parents=True)
    return artifacts


def env_for(root: Path):
    return {"XDNA_BUILD": str(root / "data" / "build"),
            "XDNA_SCRATCH": str(root / "data" / "scratch"),
            "XDNA_ARTIFACTS": str(root / "data" / "artifacts")}


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="artifact-root-gate-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.artifacts = store_fixture(self.root)
        patcher = patch.dict(os.environ, env_for(self.root), clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- accepted ---------------------------------------------------------
    def test_canonical_store_and_member_are_accepted(self):
        self.assertEqual(check_root(self.artifacts), self.artifacts.resolve())
        member = self.artifacts / "bge-base" / "encoder"
        self.assertEqual(check_target(self.artifacts, member), member.resolve())

    def test_link_that_stays_inside_the_store_is_accepted(self):
        target = self.artifacts / "bge-base" / "encoder"
        link = self.artifacts / "bge-base" / "encoder_p8e3958"
        link.symlink_to(target, target_is_directory=True)
        self.assertEqual(check_target(self.artifacts, link), target.resolve())

    # --- refused ----------------------------------------------------------
    def test_snapshot_root_is_refused(self):
        # The exact 2026-11-12 install: an artifacts root aimed at an accepted-run snapshot that
        # sits INSIDE the store, so "it is under $XDNA_ARTIFACTS" looked clean.
        with self.assertRaisesRegex(ValueError, "snapshot or lane payload"):
            check_root(self.artifacts / SNAPSHOT / "artifacts")

    def test_build_lane_root_is_refused(self):
        lane = self.root / "data" / "build" / "asr-repair-20261025" / "artifacts"
        with self.assertRaisesRegex(ValueError, "snapshot or lane payload"):
            check_root(lane)
        # And by the declared root, even when the lane dir is spelled some other way.
        with patch.dict(os.environ, env_for(self.root) | {"XDNA_BUILD": str(self.artifacts)}):
            with self.assertRaisesRegex(ValueError, "lane payload root"):
                check_root(self.artifacts / "bge-base")

    def test_lane_named_paths_are_refused_without_env(self):
        # The name rule carries the layout even when a caller bypasses scripts/lib/data_root.sh,
        # and stays quiet about a legitimate store on another disk.
        with patch.dict(os.environ, {}, clear=True):
            elsewhere = self.root / "elsewhere" / "artifacts"
            elsewhere.mkdir(parents=True)
            self.assertEqual(check_root(elsewhere), elsewhere.resolve())
            for rel in ("build/lane", "scratch/lane", "builds/lane"):
                with self.assertRaisesRegex(ValueError, "snapshot or lane payload"):
                    check_root(self.root / "data" / rel)

    def test_member_symlinked_into_a_snapshot_inside_the_store_is_refused(self):
        # The 2026-11-12 shape exactly: the snapshot is INSIDE the store, so containment passes
        # and the snapshot component is what catches it.
        member = self.artifacts / "bge-base" / "encoder"
        snapshot = self.artifacts / SNAPSHOT / "artifacts" / "bge-base" / "encoder"
        shutil.rmtree(member)
        member.symlink_to(snapshot, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "snapshot or lane payload"):
            check_target(self.artifacts, member)

    def test_member_symlinked_out_of_the_store_is_refused(self):
        member = self.artifacts / "bge-base" / "encoder"
        shutil.rmtree(member)
        escape = self.root / "data" / "scratch" / "lane" / "encoder"
        escape.mkdir(parents=True)
        (escape / "L0.npy").write_bytes(b"weights")
        member.symlink_to(escape, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "resolves outside the artifacts root"):
            check_target(self.artifacts, member)

    def test_dangling_member_is_refused(self):
        member = self.artifacts / "bge-base" / "encoder"
        shutil.rmtree(member)
        member.symlink_to(self.artifacts / SNAPSHOT / "artifacts" / "absent", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            check_target(self.artifacts, member)

    def test_symlinked_store_root_is_judged_where_it_lands(self):
        # `readlink -f` in install.sh sees only the resolved path, so that is what gets judged --
        # and the message names both sides of the link.
        real = self.root / "scratch" / "instance-artifacts"
        (real / "bge-base").mkdir(parents=True)
        link = self.root / "store-link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "scratch"):
            check_root(link)

    def test_rollback_and_worktree_components_are_refused(self):
        for rel in ("artifacts.prev-20261026T214701/bge-base",
                    "artifacts.staging-1234/bge-base",
                    "bge-base.prepin-20261014",
                    "wt-insts-abi-investigation/bge-base"):
            path = self.artifacts / rel
            path.mkdir(parents=True, exist_ok=True)
            (path / "L0.npy").write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "snapshot or lane payload"):
                check_target(self.artifacts, path)

    def test_worktree_outside_the_store_is_refused(self):
        outside = self.root / "wt-qwen-embedding-engine" / "artifacts" / "bge-base"
        outside.mkdir(parents=True)
        (outside / "L0.npy").write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, "resolves outside the artifacts root"):
            check_target(self.artifacts, outside)

    def test_missing_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            check_target(self.artifacts, self.artifacts / "bge-base" / "absent")

    # --- CLI --------------------------------------------------------------
    def test_cli_exit_codes(self):
        def run(*args):
            return subprocess.run([sys.executable, str(CHECKER), *args], capture_output=True,
                                  text=True, env=os.environ | {"PYTHONPATH": str(Path(__file__).parent)})
        good = run("target", str(self.artifacts), str(self.artifacts / "bge-base" / "encoder"))
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("encoder", good.stdout)
        bad = run("root", str(self.artifacts / SNAPSHOT / "artifacts"))
        self.assertEqual(bad.returncode, 1)
        self.assertIn("REFUSED", bad.stderr)
        self.assertEqual(run("root").returncode, 2)


class InstalledAuditTests(unittest.TestCase):
    """The audit the cleanup lane needed on 2026-11-12: what does an install really resolve to?"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="artifact-install-audit-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.artifacts = store_fixture(self.root)
        self.engine = self.root / "engine"
        (self.engine / "artifacts").mkdir(parents=True)
        patcher = patch.dict(os.environ, env_for(self.root), clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def stage(self, links):
        """Lay staged links down and write the manifest an install would have written."""
        entries = []
        for rel, target in links.items():
            path = self.engine / "artifacts" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
            entries.append({"path": f"artifacts/{rel}", "target": str(target)})
        (self.engine / "artifacts.manifest.json").write_text(json.dumps(
            {"engine_root": str(self.engine),
             "artifacts_root": str(self.root / "data" / "build" / "asr-repair-20261025" / "artifacts"),
             "artifacts": entries}))
        return check_installed(self.engine, self.artifacts)

    def test_clean_install_audits_empty(self):
        self.assertEqual(self.stage({"bge-base": self.artifacts / "bge-base"}), [])

    def test_snapshot_and_lane_hops_are_named(self):
        violations = self.stage({
            "whisper-turbo": self.artifacts / SNAPSHOT / "artifacts" / "bge-base",
            "parakeet": self.root / "data" / "build" / "asr-repair-20261025" / "artifacts" / "parakeet"})
        self.assertEqual({v[0] for v in violations},
                         {"artifacts/whisper-turbo", "artifacts/parakeet"})
        # It is the resolved hop that gets reported, not the staged link's own name.
        self.assertIn(SNAPSHOT, " ".join(v[1] for v in violations))

    def test_missing_staged_entry_is_named(self):
        (self.engine / "artifacts.manifest.json").write_text(json.dumps(
            {"artifacts": [{"path": "artifacts/gemma3-270m/tokenizer_qat", "target": "x"}]}))
        violations = check_installed(self.engine, self.artifacts)
        self.assertEqual([v[0] for v in violations], ["artifacts/gemma3-270m/tokenizer_qat"])
        self.assertIn("missing", violations[0][1])

    def test_root_comes_from_the_environment_when_not_given(self):
        self.stage({"bge-base": self.artifacts / "bge-base"})
        self.assertEqual(check_installed(self.engine), [])


if __name__ == "__main__":
    unittest.main()
