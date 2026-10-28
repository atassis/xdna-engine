#!/usr/bin/env python3
"""An instance's header links must resolve to its pinned source, not a shared checkout."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "toolchain_up.sh"


class IncludeOwnerTest(unittest.TestCase):
    def test_configure_uses_pinned_modules_and_keeps_host_binding_detection(self):
        source = SCRIPT.read_text()
        self.assertTrue('-DCMAKE_MODULE_PATH="$SRC/cmake/modulesXilinx"' in source)
        self.assertNotRegex(source, r"(?m)^\s*-DCMAKE_DISABLE_FIND_PACKAGE_XRT=ON(?:\s|$)")

    def test_fork_capability_is_layout_tolerant_and_rejects_missing_parameter(self):
        source = SCRIPT.read_text()
        marker = "_has_fork_program() {"
        self.assertTrue(marker in source, "Fork capability detection must tolerate multiline signatures")
        start = source.index(marker)
        end = source.index("\n}\n", start) + 3
        function = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv-iron/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to("/usr/bin/python3")
            program = root / "program.py"
            for signature, expected in [("self, device_name='main'", 0),
                                        ("self,\n        device_name='main', context=None", 0),
                                        ("self, placer=None", 1)]:
                program.write_text(f"class Program:\n    def resolve_program({signature}):\n        pass\n")
                result = subprocess.run(["bash", "-c", function + '\n_has_fork_program "$PROGRAM"'],
                                        env={"REPO": str(root), "PROGRAM": str(program), "PATH": "/usr/bin:/bin"})
                self.assertEqual(result.returncode, expected)

    def test_carried_compile_features_do_not_reapply_a_legacy_patch(self):
        source = SCRIPT.read_text()
        start = source.index("_apply_kernel_compile_speedups() {")
        function = source[start:source.index("\n}\n", start) + 3]
        current = "_COMPILER_LAUNCHER = ''\n_AIECC_CACHE_ENABLED = True\ndef _kernel_pch(cmd): pass\nwith concurrent.futures.ThreadPoolExecutor(max_workers=jobs): pass\n"
        for body, should_patch in [("_PCH_ENABLED = True\n", False), (current, False),
                                   (current.replace("ThreadPoolExecutor", "OtherExecutor"), True)]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                module = root / "instance/src/python/utils/compile/utils.py"
                module.parent.mkdir(parents=True)
                module.write_text(body)
                patch = root / "patches/mlir-aie-kernel-compile-speedups.patch"
                patch.parent.mkdir()
                patch.write_text("fixture\n")
                tools = root / "tools"
                tools.mkdir()
                git = tools / "git"
                git.write_text('#!/bin/sh\nprintf "attempt\\n" >> "$LOG"\nexit 1\n')
                git.chmod(0o755)
                log = root / "git.log"
                subprocess.run(["bash", "-euc", function + "\n_apply_kernel_compile_speedups"], check=True,
                               env={"REPO": str(root), "INST": str(root / "instance"), "LOG": str(log),
                                    "PATH": str(tools) + ":/usr/bin:/bin"})
                self.assertEqual(log.exists(), should_patch)

    def test_build_invocations_use_the_requested_job_budget(self):
        calls = [line.strip() for line in SCRIPT.read_text().splitlines()
                 if re.search(r"\bninja\s", line) and not line.lstrip().startswith("#")
                 and "-t targets" not in line]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn('-j "$FB_NPROC"', call)

    def test_instance_owns_kernel_headers(self):
        source = SCRIPT.read_text()
        start = source.index("_link_include_dirs() {")
        end = source.index("\n}\n", start) + 3
        function = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance = root / "instance"
            shared = root / "shared"
            (instance / "build/include").mkdir(parents=True)
            (instance / "src/third_party/aie_api/include/aie_api").mkdir(parents=True)
            (instance / "src/third_party/aie_api/include/aie_api/aie.hpp").write_text("pinned\n")
            (instance / "src/aie_kernels").mkdir(parents=True)
            (shared / "mlir-aie/aie_kernels/aie2p").mkdir(parents=True)
            (shared / "mlir-aie/aie_kernels/aie2p/mm.cc").write_text("stale\n")
            shell = function + '\n_shared_link_root() { printf "%s\\n" "$SHARED"; }\n_link_include_dirs\n'
            subprocess.run(["bash", "-euc", shell], check=True,
                           env={"INST": str(instance), "SHARED": str(shared), "PATH": "/usr/bin:/bin"})
            self.assertEqual((instance / "build/include/aie_api").resolve(),
                             instance / "src/third_party/aie_api/include/aie_api")
            self.assertEqual((instance / "build/include/aie_kernels").resolve(),
                             instance / "src/aie_kernels")


if __name__ == "__main__":
    unittest.main()
