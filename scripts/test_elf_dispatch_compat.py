import ast
import unittest
from pathlib import Path


class ElfDispatchImportTests(unittest.TestCase):
    def test_compile_helpers_do_not_import_xrt_runtime(self):
        source = Path(__file__).resolve().parents[1] / "designs/decode_fused/elf_dispatch_compat.py"
        tree = ast.parse(source.read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        modules = [name.name for node in imports if isinstance(node, ast.Import) for name in node.names]
        modules += [node.module for node in imports if isinstance(node, ast.ImportFrom)]
        self.assertNotIn("pyxrt", modules)
        self.assertNotIn("aie.utils.hostruntime.xrtruntime.tensor", modules)


if __name__ == "__main__":
    unittest.main()
