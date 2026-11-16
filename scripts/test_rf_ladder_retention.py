#!/usr/bin/env python3
"""CPU-only lifecycle checks for fwd_ladder's fresh kernel-object cache."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
FIXTURE_PARENT = Path(tempfile.gettempdir())


class RfLadderRetentionTests(unittest.TestCase):
    def setUp(self):
        FIXTURE_PARENT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="rf-ladder-", dir=FIXTURE_PARENT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = self.root / "repo" / "designs" / "resident_forward"
        self.fixture.mkdir(parents=True)
        shutil.copy2(REPO / "designs/resident_forward/fwd_ladder.sh", self.fixture / "fwd_ladder.sh")
        (self.fixture / "env.sh").write_text("# fixture: do not load the real environment\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "awk").write_text("#!/bin/sh\necho 99999999\n")
        (self.bin / "nproc").write_text("#!/bin/sh\necho 4\n")
        for command in ("awk", "nproc"):
            (self.bin / command).chmod(0o755)
        self.python = self.root / "fake-python"
        self.python.write_text("""#!/bin/sh
case "$1" in
  */rf_build.py)
    mkdir -p "$RF_BUILD/kobj/$RF_KOBJ_RUN"
    : > "$RF_BUILD/kobj/$RF_KOBJ_RUN/fixture.o"
    printf '%s' "$RF_KOBJ_RUN" > "$RF_BUILD/observed-runname"
    exit "${RF_FIXTURE_RC:-0}"
    ;;
  */fwd_pack.py) : > "$RF_BUILD/packed" ;;
esac
""")
        self.python.chmod(0o755)

    def run_ladder(self, *, keep=None, failure=False):
        build = self.root / "build"
        other = build / "kobj" / "other-cache"
        other.mkdir(parents=True, exist_ok=True)
        sentinel = other / "preexisting"
        sentinel.write_text("keep")
        env = os.environ | {"PY": str(self.python), "RF_BUILD": str(build), "RF_LADDER_JOBS": "1",
                            "AIECC_JOBS": "2", "PATH": str(self.bin) + os.pathsep + os.environ["PATH"]}
        env.pop("KEEP_WORK", None)
        env.pop("RF_FIXTURE_RC", None)
        if keep is not None:
            env["KEEP_WORK"] = keep
        if failure:
            env["RF_FIXTURE_RC"] = "17"
        result = subprocess.run(["bash", str(self.fixture / "fwd_ladder.sh"), "fixture", "", "emit=fixture"],
                                env=env, text=True, capture_output=True)
        run = (build / "observed-runname").read_text()
        return result, build / "kobj" / run, sentinel

    def test_nonempty_keep_work_retains_fresh_cache_on_success_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                result, cache, sentinel = self.run_ladder(keep=str(self.root / "build"), failure=failure)
                self.assertEqual(result.returncode, 17 if failure else 0, result.stderr)
                self.assertTrue((cache / "fixture.o").is_file())
                self.assertTrue(sentinel.is_file())

    def test_unset_or_empty_keep_work_removes_only_fresh_cache(self):
        for keep in (None, ""):
            for failure in (False, True):
                with self.subTest(keep=keep, failure=failure):
                    result, cache, sentinel = self.run_ladder(keep=keep, failure=failure)
                    self.assertEqual(result.returncode, 17 if failure else 0, result.stderr)
                    self.assertFalse(cache.exists())
                    self.assertTrue(sentinel.is_file())


if __name__ == "__main__":
    unittest.main()
