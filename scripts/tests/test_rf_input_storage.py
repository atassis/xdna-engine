import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO / "designs" / "resident_forward" / "rf_build.py"


class ResidentInputStorage(unittest.TestCase):
    def run_builder(self, work, text="module {}\n", fail=False):
        generator = types.ModuleType("storage_test_generator")
        setattr(generator, "build_text", lambda args: text)
        setattr(generator, "kernels", lambda: [])
        compiler = types.ModuleType("aie.utils.compile.utils")

        def compile_module(source, *, full_elf_path, work_dir, options):
            Path(work_dir, "aie.mlir").write_text(source)
            if fail:
                raise RuntimeError("compiler failed")
            Path(full_elf_path).write_bytes(b"compiler-output-fixture")

        setattr(compiler, "compile_mlir_module", compile_module)
        setattr(compiler, "compile_cxx_core_function", lambda *args, **kwargs: None)
        modules = {"storage_test_generator": generator,
                   "aie.utils.compile.utils": compiler}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, RF_BUILD=str(work.parent)), \
                patch.object(sys, "argv", [str(BUILD_SCRIPT), work.name, generator.__name__]), \
                patch.object(sys, "path", sys.path.copy()):
            runpy.run_path(str(BUILD_SCRIPT), run_name="__main__")

    def assert_single_input(self, work, text):
        self.assertTrue((work / "design.mlir").is_symlink())
        self.assertEqual(os.readlink(work / "design.mlir"), "aie.mlir")
        self.assertEqual((work / "design.mlir").read_text(), text)
        self.assertEqual((work / "aie.mlir").read_text(), text)

    def test_success_keeps_one_physical_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            self.run_builder(work)
            self.assert_single_input(work, "module {}\n")
            self.assertEqual((work / "design.elf").read_bytes(), b"compiler-output-fixture")

    def test_changed_input_replaces_both_pathnames(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            self.run_builder(work)
            self.run_builder(work, "module { changed }\n")
            self.assert_single_input(work, "module { changed }\n")

    def test_failed_compile_retains_one_inspectable_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                self.run_builder(work, fail=True)
            self.assert_single_input(work, "module {}\n")

    def test_matching_regular_inputs_can_be_consolidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            work.mkdir()
            for name in ("design.mlir", "aie.mlir"):
                (work / name).write_text("old input\n")
            self.run_builder(work)
            self.assert_single_input(work, "module {}\n")

    def test_divergent_regular_input_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            work.mkdir()
            (work / "design.mlir").write_text("unique input\n")
            (work / "aie.mlir").write_text("other input\n")
            with self.assertRaises(ValueError):
                self.run_builder(work)
            self.assertEqual((work / "design.mlir").read_text(), "unique input\n")
            self.assertEqual((work / "aie.mlir").read_text(), "other input\n")

    def test_unexpected_symlink_is_not_followed(self):
        for name in ("design.mlir", "aie.mlir"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "arm"
                work.mkdir()
                outside = Path(tmp) / "outside.mlir"
                outside.write_text("unique input\n")
                (work / name).symlink_to(outside)
                with self.assertRaises(ValueError):
                    self.run_builder(work)
                self.assertEqual(outside.read_text(), "unique input\n")

    def test_hardlinked_input_does_not_mutate_another_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "arm"
            work.mkdir()
            outside = Path(tmp) / "outside.mlir"
            outside.write_text("shared input\n")
            os.link(outside, work / "aie.mlir")
            self.run_builder(work)
            self.assertEqual(outside.read_text(), "shared input\n")
            self.assert_single_input(work, "module {}\n")


if __name__ == "__main__":
    unittest.main()
