#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build_peano_fast.sh"


class PeanoCacheEnv(unittest.TestCase):
    def test_worktrees_share_store_but_normalize_against_own_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            ninja = bin_dir / "ninja"
            ninja.write_text('#!/bin/sh\nprintf "TEST_BASEDIR=%s\\nTEST_CACHE=%s\\n" "$CCACHE_BASEDIR" "$CCACHE_DIR"\n')
            ninja.chmod(0o755)
            shared = root / "shared"
            for name in ("tree-a", "tree-b"):
                source = root / name
                (source / "llvm").mkdir(parents=True)
                (source / "llvm/CMakeLists.txt").touch()
                (source / "build-fast").mkdir()
                (source / "build-fast/CMakeCache.txt").touch()
                env = dict(os.environ)
                env.pop("CCACHE_BASEDIR", None)
                env.pop("BUILD_DIR", None)
                env.update(LLVM_AIE_SRC=str(source), CCACHE_DIR=str(shared),
                           XDNA_CACHE=str(root / "cache"), FB_NPROC="1",
                           PATH=str(bin_dir) + os.pathsep + env["PATH"])
                result = subprocess.run(["bash", str(SCRIPT), "llc"], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("TEST_BASEDIR=" + str(source), result.stdout)
                self.assertIn("TEST_CACHE=" + str(shared), result.stdout)


if __name__ == "__main__":
    unittest.main()
