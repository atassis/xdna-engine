"""Fixture-only execution of the ladder; no toolchain, model or device access."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LadderTests(unittest.TestCase):
    def run_ladder(self, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(Path(__file__).with_name("fwd_ladder.sh"), root / "fwd_ladder.sh")
            (root / "env.sh").write_text('set -euo pipefail\nPY=python3\n')
            (root / "rf_build.py").write_text(
                'import json, os, pathlib, sys\n'
                'with open(os.environ["EVENTS"], "a") as f:\n'
                ' f.write(json.dumps({"part":sys.argv[1], "jobs":os.environ.get("AIECC_JOBS"), '
                '"cgroup":pathlib.Path("/proc/self/cgroup").read_text()}) + "\\n")\n'
                'sys.exit(7 if os.environ.get("FAIL_PART") == sys.argv[1] else 0)\n'
            )
            (root / "fwd_pack.py").write_text(
                'import os, pathlib\npathlib.Path(os.environ["PACKED"]).touch()\n'
            )
            # Any attempt to escape into a sibling scope fails without contacting systemd.
            (root / "systemd-run").write_text('#!/bin/sh\nexit 91\n')
            (root / "systemd-run").chmod(0o755)
            (root / "nproc").write_text('#!/bin/sh\necho "${TEST_CPUS:-8}"\n')
            (root / "nproc").chmod(0o755)
            (root / "awk").write_text('#!/bin/sh\necho "${TEST_AVAILABLE_KB:-999999999}"\n')
            (root / "awk").chmod(0o755)
            (root / "sleep").write_text('#!/bin/sh\nexit 92\n')
            (root / "sleep").chmod(0o755)
            env = os.environ.copy()
            for key in ("RF_LADDER_JOBS", "AIECC_JOBS", "FAIL_PART"):
                env.pop(key, None)
            env.update(PATH=f'{root}:{env["PATH"]}', RF_BUILD=str(root / "build"),
                       EVENTS=str(root / "events"), PACKED=str(root / "packed"), **settings)
            result = subprocess.run(
                ["bash", str(root / "fwd_ladder.sh"), "image", "", "p1", "p2"],
                env=env, capture_output=True, text=True, timeout=15,
            )
            events = [json.loads(line) for line in (root / "events").read_text().splitlines()] if (root / "events").exists() else []
            return result, events, (root / "packed").exists()

    def test_parts_inherit_cgroup_and_bounded_jobs(self):
        result, events, packed = self.run_ladder()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([event["part"] for event in events], ["image_p0", "image_p1"])
        self.assertTrue(packed)
        for event in events:
            self.assertEqual(event["cgroup"], Path("/proc/self/cgroup").read_text())
            self.assertEqual(int(event["jobs"]), 2)

    def test_failure_stops_before_next_part_and_pack(self):
        result, events, packed = self.run_ladder(FAIL_PART="image_p0")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([event["part"] for event in events], ["image_p0"])
        self.assertFalse(packed)

    def test_invalid_resource_requests_fail_before_build(self):
        for settings in ({"RF_LADDER_JOBS": "2"}, {"AIECC_JOBS": "8"},
                         {"AIECC_JOBS": "0"}, {"AIECC_JOBS": "bad"}):
            with self.subTest(settings=settings):
                result, events, packed = self.run_ladder(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("resource", result.stderr)
                self.assertEqual(events, [])
                self.assertFalse(packed)

    def test_memory_reserve_blocks_admission(self):
        result, events, packed = self.run_ladder(TEST_AVAILABLE_KB="1")
        self.assertEqual(result.returncode, 92)
        self.assertEqual(events, [])
        self.assertFalse(packed)

    def test_one_cpu_host_uses_one_job(self):
        result, events, packed = self.run_ladder(TEST_CPUS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(packed)
        self.assertTrue(all(event["jobs"] == "1" for event in events))
        result, events, packed = self.run_ladder(TEST_CPUS="1", AIECC_JOBS="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(events, [])
        self.assertFalse(packed)

    def test_explicit_single_job(self):
        result, events, packed = self.run_ladder(AIECC_JOBS="1", RF_LADDER_JOBS="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(packed)
        self.assertTrue(all(event["jobs"] == "1" for event in events))


if __name__ == "__main__":
    unittest.main()
