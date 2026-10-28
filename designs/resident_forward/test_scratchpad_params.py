"""Packing retains the current compiler's parameter safety contract."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scratchpad_params import merge_params, read_params, write_params


FIRST = """2 2
length 0 i32 addr 1 64
offset 1 i32 addr 0 1024
1
length offset 16 1024
2
main:boot 0
main:f1 2 length offset
"""
SECOND = FIRST.replace("1 64", "1 32").replace("16 1024", "16 512").replace("main:f1", "main:f2")


class ScratchpadParamsTests(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.txt"
            path.write_text(text)
            return read_params(path)

    def test_merge_intersects_ranges_retains_joint_checks_and_unions_commands(self):
        merged = merge_params([self.read(FIRST), self.read(SECOND)])
        self.assertEqual(merged.parameters["length"], (0, "i32", "addr", 1, 32))
        self.assertEqual(set(merged.joint_bounds), {("length", "offset", 16, 1024),
                                                   ("length", "offset", 16, 512)})
        self.assertEqual(merged.commands, {"main:boot": (), "main:f1": ("length", "offset"),
                                            "main:f2": ("length", "offset")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.txt"
            write_params(path, merged)
            self.assertEqual(read_params(path), merged)

    def test_merged_file_enforces_bounds_in_the_compiler_host_reader(self):
        from aie._mlir_libs._parameter_scratchpad import ParameterScratchpad
        merged = merge_params([self.read(FIRST), self.read(SECOND)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "params.txt"
            write_params(path, merged)
            for command in ("main:f1", "main:f2"):
                params = ParameterScratchpad(bytearray(8), str(path), command)
                self.assertEqual(set(params.names()), {"length", "offset"})
                params.write_bytes("length", (32).to_bytes(4, "little"))
                params.write_bytes("offset", (0).to_bytes(4, "little"))
                params.validate()
                with self.assertRaises(ValueError):
                    params.write_bytes("length", (33).to_bytes(4, "little"))
                params.write_bytes("length", (32).to_bytes(4, "little"))
                params.write_bytes("offset", (1).to_bytes(4, "little"))
                with self.assertRaises(ValueError):
                    params.validate()
            boot = ParameterScratchpad(bytearray(8), str(path), "main:boot")
            self.assertEqual(set(boot.names()), set())
            with self.assertRaises(RuntimeError):
                boot.write_bytes("length", (1).to_bytes(4, "little"))

    def test_merge_rejects_incompatible_parameter_identity(self):
        first = self.read(FIRST)
        for old, new in (("length 0", "length 1"), ("i32 addr", "f32 addr"),
                         ("i32 addr", "i32 core"), ("1 64", "65 128")):
            with self.subTest(new=new), self.assertRaises(ValueError):
                merge_params([first, self.read(FIRST.replace(old, new, 1))])
        for other in (replace(first, slot_count=3),
                      replace(first, parameters={"length": first.parameters["length"]})):
            with self.assertRaises(ValueError):
                merge_params([first, other])

    def test_merge_rejects_conflicting_readers_for_a_shared_command(self):
        with self.assertRaises(ValueError):
            merge_params([self.read(FIRST), self.read(FIRST.replace("main:f1 2 length offset", "main:f1 1 length"))])

    def test_parser_rejects_legacy_truncated_and_unknown_contracts(self):
        bad = ["2\nlength 0 i32 addr\noffset 1 i32 addr\n", FIRST + "trailing\n",
               FIRST.rsplit("\n", 3)[0], FIRST.replace("length offset 16", "unknown offset 16"),
               FIRST.replace("main:f1 2 length offset", "main:f1 1 unknown"),
               FIRST.replace("length 0", "length 2"), FIRST.replace("1 64", "64 1"),
               FIRST.replace("length 0 i32 addr 1 64", "offset 0 i32 addr 1 64"),
               FIRST.replace("i32 addr", "i32 mystery", 1), FIRST.replace("1 64", "- 64"),
               FIRST.replace("main:f1", "main:boot")]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.read(text)


if __name__ == "__main__":
    unittest.main()
