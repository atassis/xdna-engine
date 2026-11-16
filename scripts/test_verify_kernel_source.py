"""Focused fixtures for verify_kernel_source.sh's vendor source layout checks."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parent.parent
VERIFIER = REPO / "scripts" / "verify_kernel_source.sh"
PIN = "d71ba4b99b4468f4c3df49e8f2bfe0d9a7289ae2"


class VerifyKernelSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="verify-kernel-source-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.instance = self.root / "instance"
        self.instance.mkdir()
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        (self.scripts / "verify_kernel_source.sh").write_bytes(VERIFIER.read_bytes())
        toolchain_up = self.scripts / "toolchain_up.sh"
        toolchain_up.write_text(f"#!/bin/sh\nprintf '%s\\n' '{self.instance}'\n")
        toolchain_up.chmod(0o755)
        (self.root / "toolchain.lock").write_text(f"PEANO_FORK_COMMIT={PIN}\n")
        clang = self.instance / "peano" / "bin" / "clang++"
        clang.parent.mkdir(parents=True)
        clang.write_text(
            "#!/bin/sh\n"
            f"echo 'clang version 21 (https://example.test/llvm-aie {PIN})'\n"
        )
        clang.chmod(0o755)
        self.compiler = clang.parent.parent
        self.mlir = self.root / "mlir-aie"
        self.mmw = self.mlir / "programming_examples/basic/matrix_multiplication/whole_array"
        self.mmw.mkdir(parents=True)
        self.canonical = self.root / "designs/whole_array_fused/Makefile.modal"
        self.canonical.parent.mkdir(parents=True)
        self.kernels = self.root / "aie_kernels"
        for rel in ("mm-zero/mm_zero.cc", "mm-mode-lnaffcast/mm_mode_lnaffcast.cc",
                    "mm-mode-resadd2a/mm_mode_resadd2a.cc", "mm-silu-epilogue/mm_silu_epilogue.cc"):
            path = self.kernels / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// owned project kernel\n")
        self.layout = "linalg"
        self.source_root = self.instance / "src/aie_kernels"
        self._write_vendor_tree(self.source_root, "current")
        self._write_makefile()

    def _write_vendor_tree(self, root: Path, layout: str):
        if layout == "current":
            base = root / "linalg"
            (base / "mm.cc").parent.mkdir(parents=True, exist_ok=True)
            (root / "aie_arch.h").write_text("// architecture selector\n")
            (root / "aie_kernel_utils.h").write_text("// vendor helper v1\n")
            (base / "mm_aie2.h").write_text('#include "../aie_kernel_utils.h"\n// aie2\n')
            (base / "mm_aie2p.h").write_text('#include "../aie_kernel_utils.h"\n// aie2p\n')
            (base / "mm.cc").write_text(
                '#include "../aie_arch.h"\n#include "mm_aie2.h"\n#include "mm_aie2p.h"\n'
                'void matmul_bf16_f32(short *, short *, float *) {}\n'
            )
        else:
            base = root / layout
            base.mkdir(parents=True, exist_ok=True)
            (base / "mm.cc").write_text("// legacy vendor mm\n")

    def _write_makefile(self, *, compiler_pin: str = PIN):
        del compiler_pin
        makefile = (
            "srcdir := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))\n"
            "devicename ?= npu2\n"
            "KERNEL_CC=${PEANO_INSTALL_DIR}/bin/clang++\n"
            "kernels_dir=$(KERNEL_SOURCE_ROOT)/src/aie_kernels/$(KERNEL_LAYOUT)\n"
            "lib_kernels_dir=$(abspath ${srcdir}/../../../../../aie_kernels)\n"
            "build/mm.o: ${kernels_dir}/mm.cc ${lib_kernels_dir}/mm-zero/mm_zero.cc\n"
            "\tmkdir -p ${@D}\n"
            "\t${KERNEL_CC} -c $< -o $@\n"
            "build/mm-mode-lnaffcast.o: ${lib_kernels_dir}/mm-mode-lnaffcast/mm_mode_lnaffcast.cc\n"
            "build/mm-mode-resadd2a.o: ${lib_kernels_dir}/mm-mode-resadd2a/mm_mode_resadd2a.cc\n"
            "build/mm-silu-epilogue.o: ${lib_kernels_dir}/mm-silu-epilogue/mm_silu_epilogue.cc\n"
        )
        self.canonical.write_text(makefile)
        (self.mmw / "Makefile.modal").write_text(makefile)

    def _run(self, *, source_root: Path | None = None, layout: str | None = None,
             compiler_pin: str = PIN):
        env = os.environ.copy()
        env.update({
            "PEANO_INSTALL_DIR": str(self.compiler),
            "KERNEL_SOURCE_ROOT": str(source_root or self.instance),
            "KERNEL_LAYOUT": layout or self.layout,
        })
        if compiler_pin != PIN:
            clang = self.compiler / "bin/clang++"
            clang.write_text(
                "#!/bin/sh\n"
                f"echo 'clang version 21 (https://example.test/llvm-aie {compiler_pin})'\n"
            )
            clang.chmod(0o755)
        return subprocess.run(
            ["bash", str(self.scripts / "verify_kernel_source.sh"), "Makefile.modal"],
            cwd=self.root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )

    def test_current_linalg_layout_and_transitive_root_headers_are_accepted(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("OK   Makefile.modal/linalg/mm.cc (vendor)", result.stdout)
        self.assertIn("aie_arch.h", (self.mmw / "build/.kernel_source_manifest").read_text())
        self.assertIn("aie_kernel_utils.h", (self.mmw / "build/.kernel_source_manifest").read_text())

    def test_current_linalg_wrong_vendor_bytes_are_rejected(self):
        wrong = self.root / "wrong-source/src/aie_kernels"
        self._write_vendor_tree(wrong, "current")
        (wrong / "linalg/mm.cc").write_text("// wrong resolved vendor source\n")
        result = self._run(source_root=self.root / "wrong-source")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does NOT match its ground truth", result.stdout)

    def test_current_linalg_wrong_transitive_root_header_is_rejected(self):
        wrong = self.root / "wrong-header/src/aie_kernels"
        self._write_vendor_tree(wrong, "current")
        (wrong / "aie_kernel_utils.h").write_text("// wrong transitive helper\n")
        result = self._run(source_root=self.root / "wrong-header")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("aie_kernel_utils.h", result.stdout)
        self.assertIn("does NOT match its ground truth", result.stdout)

    def test_current_linalg_project_kernel_in_vendor_namespace_is_rejected(self):
        wrong = self.source_root / "linalg/mm-zero/mm_zero.cc"
        wrong.parent.mkdir(parents=True)
        wrong.write_bytes((self.kernels / "mm-zero/mm_zero.cc").read_bytes())
        for path in (self.canonical, self.mmw / "Makefile.modal"):
            path.write_text(path.read_text().replace(
                "${lib_kernels_dir}/mm-zero/mm_zero.cc",
                "${kernels_dir}/mm-zero/mm_zero.cc",
            ))
        result = self._run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("is OURS", result.stdout)
        self.assertIn("declared through ${kernels_dir}", result.stdout)

    def test_wrong_compiler_banner_is_rejected(self):
        result = self._run(compiler_pin="0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does NOT match toolchain.lock's PEANO_FORK_COMMIT", result.stdout)

    def test_copied_makefile_drift_is_rejected(self):
        (self.mmw / "Makefile.modal").write_text((self.mmw / "Makefile.modal").read_text() + "# drift\n")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sandbox copy differs from its canonical source", result.stdout)

    def test_legacy_aie2p_and_aie2_layouts_remain_supported(self):
        for layout in ("aie2p", "aie2"):
            with self.subTest(layout=layout):
                for old_layout in ("aie2p", "aie2"):
                    old_file = self.source_root / old_layout / "mm.cc"
                    if old_file.exists():
                        old_file.unlink()
                self._write_vendor_tree(self.source_root, layout)
                self.layout = layout
                self._write_makefile()
                result = self._run(layout=layout)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("OK   Makefile.modal/mm.cc (vendor)", result.stdout)
                self.assertIn(f"/aie_kernels/{layout}/mm.cc", result.stdout)


if __name__ == "__main__":
    unittest.main()
