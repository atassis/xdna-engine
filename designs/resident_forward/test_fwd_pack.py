"""The packed ELF carries the merged scratchpad contract, without recompiling parts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fwd_pack
from scratchpad_params import read_params
from test_scratchpad_params import FIRST, SECOND


class FwdPackTests(unittest.TestCase):
    def fixture(self, root):
        for i, text in enumerate((FIRST, SECOND)):
            part = root / f"p{i}"
            part.mkdir()
            (part / "cdo_main").mkdir()
            (part / "cdo_main/image.bin").write_bytes(b"same core image")
            (part / "params.txt").write_text(text)
            (part / "gen_args.txt").write_text(f"rlayer_design emit=f{i + 1} same")
            config = {"xrt-kernels": [{"name": "main", "arguments": [], "PDIs": [],
                       "instance": [{"id": "boot", "TXN_ctrl_code_file": "boot.bin"},
                                    {"id": f"f{i + 1}", "TXN_ctrl_code_file": "run.bin"},
                                    {"id": "aiecc_empty", "TXN_ctrl_code_file": "empty.bin"}]}]}
            (part / "full_elf_config.json").write_text(json.dumps(config))
            (part / "boot.bin").write_bytes(b"same boot")
            (part / "empty.bin").write_bytes(b"same empty sequence")
            (part / "run.bin").write_bytes(f"run{i}".encode())

    def pack(self, root, assemble):
        with patch.object(fwd_pack, "BUILD", str(root)), \
             patch.object(fwd_pack.sys, "argv", ["fwd_pack.py", "packed", "p0", "p1"]), \
             patch.object(fwd_pack.subprocess, "run", assemble), \
             patch.object(fwd_pack, "write_elf", return_value={"sha256": "fixture"}), \
             patch.object(fwd_pack, "provenance", return_value={}):
            fwd_pack.main()

    def test_full_pack_writes_bounds_and_all_command_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            def assemble(argv, **kwargs):
                self.assertTrue(kwargs["check"])
                Path(argv[argv.index("-o") + 1]).write_bytes(b"ELF fixture")
            self.pack(root, assemble)
            params = read_params(root / "packed/params.txt")
            self.assertEqual(params.parameters["length"][3:], (1, 32))
            self.assertEqual(len(params.joint_bounds), 2)
            self.assertEqual(set(params.commands), {"main:boot", "main:f1", "main:f2"})
            config = json.loads((root / "packed/full_elf_config.json").read_text())
            self.assertEqual({i["id"] for i in config["xrt-kernels"][0]["instance"]},
                             {"boot", "f1", "f2", "aiecc_empty"})

    def test_conflicting_inputs_reject_before_assembly(self):
        changes = {"bounds": ("params.txt", SECOND.replace("1 32", "65 128")),
                   "command": ("params.txt", SECOND.replace("main:f2", "main:missing")),
                   "slots": ("params.txt", SECOND.replace("length 0", "length 1")),
                   "image": ("cdo_main/image.bin", "different core image")}
        for name, (path, text) in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                (root / "p1" / path).write_text(text)
                with patch.object(fwd_pack.subprocess, "run") as assembler:
                    with self.assertRaises((ValueError, AssertionError)):
                        self.pack(root, assembler)
                    assembler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
