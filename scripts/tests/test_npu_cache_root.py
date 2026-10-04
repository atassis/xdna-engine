#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


LIB = Path(__file__).resolve().parents[1] / "lib"


class NpuCacheRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        lib = self.repo / "scripts/lib"
        lib.mkdir(parents=True)
        for name in ("data_root.sh", "data_root.py"):
            shutil.copy2(LIB / name, lib / name)

    def probe(self, language, settings, variable="NPU_CACHE_HOME"):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("XDNA_") and key not in ("NPU_CACHE_HOME", "CCACHE_DIR", "PYTHONPATH")}
        env.update(settings)
        env.update(REPO=str(self.repo), HOME=str(self.root / "home"), PROBE_VARIABLE=variable)
        if language == "shell":
            command = ["bash", "-c", '. "$REPO/scripts/lib/data_root.sh"; '
                       'printf "%s\\n" "${!PROBE_VARIABLE-unset}"']
            return subprocess.check_output(command, env=env, text=True).strip()
        code = ('import os,runpy,json; '
                'm=runpy.run_path(os.environ["REPO"]+"/scripts/lib/data_root.py"); '
                'v=os.environ["PROBE_VARIABLE"]; print(json.dumps([str(m.get(v)),os.environ.get(v)]))')
        values = json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
        self.assertEqual(values[0], values[1])
        return values[0]

    def test_default_npu_cache_is_under_selected_data_root(self):
        data = self.root / "data"
        for language in ("shell", "python"):
            with self.subTest(language=language):
                self.assertEqual(self.probe(language, {"XDNA_DATA": str(data)}), str(data / "cache/npu"))

    def test_cache_root_override_selects_npu_subdirectory(self):
        cache = self.root / "shared-cache"
        for language in ("shell", "python"):
            with self.subTest(language=language):
                self.assertEqual(self.probe(language, {"XDNA_CACHE": str(cache)}), str(cache / "npu"))

    def test_explicit_npu_cache_override_is_preserved(self):
        isolated = self.root / "isolated-test-cache"
        for language in ("shell", "python"):
            with self.subTest(language=language):
                self.assertEqual(self.probe(language, {"NPU_CACHE_HOME": str(isolated)}), str(isolated))

    def test_resolving_paths_does_not_create_cache_directories(self):
        data = self.root / "data"
        for language in ("shell", "python"):
            self.probe(language, {"XDNA_DATA": str(data)})
        self.assertFalse(data.exists())

    def test_engine_compiler_cache_uses_shared_store(self):
        cache = self.root / "shared-cache"
        for language in ("shell", "python"):
            with self.subTest(language=language):
                self.assertEqual(self.probe(language, {"XDNA_CACHE": str(cache)}, "CCACHE_DIR"),
                                 str(cache / "ccache"))

    def test_explicit_compiler_cache_override_is_preserved(self):
        isolated = self.root / "isolated-compiler-cache"
        for language in ("shell", "python"):
            with self.subTest(language=language):
                self.assertEqual(self.probe(language, {"CCACHE_DIR": str(isolated)}, "CCACHE_DIR"),
                                 str(isolated))


if __name__ == "__main__":
    unittest.main()
