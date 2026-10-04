#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build_llm_decode.sh"


class LlmBuildLocation(unittest.TestCase):
    def select_workspace(self, repo, build, **overrides):
        selection = re.search(
            r'^if \[ -n "\$\{KEEP_WORK:-\}" \]; then\n.*?^fi$',
            SCRIPT.read_text(), re.MULTILINE | re.DOTALL,
        )
        assert selection is not None
        env = dict(os.environ)
        for name in ("BUILD_CACHE", "KEEP_WORK"):
            env.pop(name, None)
        env.update(REPO=str(repo), XDNA_BUILD=str(build),
                   XDNA_CACHE=str(build.parent / "cache"), SPEC="model")
        env.update({key: str(value) for key, value in overrides.items()})
        result = subprocess.run(
            ["bash", "-eu", "-c", selection.group(0) + '\nprintf "%s\\n" "$WORK"'],
            env=env, capture_output=True, text=True, check=True,
        )
        return Path(result.stdout.splitlines()[-1])

    def test_default_workspace_is_not_a_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "engine-worktree"
            repo.mkdir()
            work = self.select_workspace(repo, root / "build")
            self.assertTrue(work.is_relative_to(root / "build"))
            self.assertFalse(work.is_relative_to(root / "cache"))
            self.assertEqual(work.parts[-2:], ("llm", "model"))

    def test_same_basename_checkouts_have_distinct_stable_workspaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repos = [root / parent / "engine" for parent in ("one", "two")]
            for repo in repos:
                repo.mkdir(parents=True)
            works = [self.select_workspace(repo, root / "build") for repo in repos]
            self.assertNotEqual(*works)
            for repo, work in zip(repos, works):
                self.assertEqual(work, self.select_workspace(repo, root / "build"))
                digest = hashlib.sha256(os.fsencode(repo.resolve())).hexdigest()
                self.assertIn(digest, work.relative_to(root / "build").parts[0])

    def test_resolved_checkout_alias_uses_same_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "engine"
            repo.mkdir()
            alias = root / "alias"
            alias.symlink_to(repo, target_is_directory=True)
            self.assertEqual(self.select_workspace(repo, root / "build"),
                             self.select_workspace(alias, root / "build"))

    def test_explicit_workspace_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "engine"
            repo.mkdir()
            cache = root / "explicit cache"
            keep = root / "kept work"
            self.assertEqual(self.select_workspace(repo, root / "build", BUILD_CACHE=cache), cache)
            self.assertEqual(self.select_workspace(repo, root / "build", KEEP_WORK=keep), keep)
            self.assertEqual(self.select_workspace(repo, root / "build",
                                                  BUILD_CACHE=cache, KEEP_WORK=keep), keep)


if __name__ == "__main__":
    unittest.main()
