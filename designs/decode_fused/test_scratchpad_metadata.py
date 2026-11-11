import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


CURRENT = """3 2
kv_off 0 i32 addr 0 466816
sm_mask 1 i32 core - -
unused - i32 addr 0 7
0
1
main:sequence 2 kv_off sm_mask
"""


class ScratchpadMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mlir = self.root / "build/model.mlir"
        self.params = self.mlir.parent / (self.mlir.name + ".d") / "params.txt"
        self.params.parent.mkdir(parents=True)
        self.params.write_text(CURRENT)
        self.output = self.root / "output"
        self.output.mkdir()
        self.sequence = SimpleNamespace(artifacts=[SimpleNamespace(
            mlir_input=SimpleNamespace(filename=str(self.mlir)))])

    def read(self):
        from scratchpad_metadata import sequence_scratchpad_metadata
        return sequence_scratchpad_metadata(self.sequence, self.output)

    def test_current_bounded_contract_preserves_offsets_bounds_and_exact_sidecar(self):
        result = self.read()
        self.assertEqual(result, {
            "kv_off": {"byte_offset": 0, "dtype": "i32", "kind": "addr", "minimum": 0, "maximum": 466816},
            "sm_mask": {"byte_offset": 4, "dtype": "i32", "kind": "core", "minimum": None, "maximum": None},
        })
        self.assertEqual((self.output / "params.txt").read_bytes(), self.params.read_bytes())
        self.assertNotIn("unused", result)

    def test_newer_unrelated_sidecar_is_not_selected(self):
        other = self.root / "unrelated/params.txt"
        other.parent.mkdir()
        other.write_text("not this ELF's metadata\n")
        os.utime(other, (self.params.stat().st_mtime + 100, self.params.stat().st_mtime + 100))
        self.assertIn("kv_off", self.read())

    def test_missing_owner_sidecar_does_not_fall_back_to_an_unrelated_one(self):
        self.params.unlink()
        other = self.root / "unrelated/params.txt"
        other.parent.mkdir()
        other.write_text(CURRENT)
        with self.assertRaises(FileNotFoundError):
            self.read()

    def test_empty_current_contract_is_supported(self):
        self.params.write_text("0 0\n0\n1\nmain:sequence 0\n")
        self.assertEqual(self.read(), {})

    def test_malformed_unknown_or_truncated_contracts_are_rejected(self):
        for value in ("2\nkv_off 0 i32 addr\nsm_mask 1 i32 core\n", CURRENT + "trailing\n",
                      CURRENT.replace("0 466816", "466816 0"),
                      CURRENT.replace("main:sequence 2 kv_off sm_mask", "main:sequence 1 missing"),
                      CURRENT.replace("kv_off 0", "kv_off 2"),
                      CURRENT.replace("i32 core", "i32 repeat"),
                      CURRENT.replace("i32 core", "f32 core")):
            self.params.write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read()


if __name__ == "__main__":
    unittest.main()
