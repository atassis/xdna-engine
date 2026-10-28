import hashlib
import importlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ArtifactMetaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for compressed ELF metadata")
    def test_elf_digest_uses_decompressed_bytes_and_rejects_corruption(self):
        from artifact_meta import elf_sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design.elf"
            data = b"compiled ELF fixture"
            path.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(elf_sha256(path), expected)
            subprocess.run(["zstd", "--quiet", str(path)], check=True)
            path.unlink()
            self.assertEqual(elf_sha256(path), expected)
            Path(f"{path}.zst").write_bytes(b"not zstd")
            with self.assertRaisesRegex(ValueError, "cannot decompress ELF"):
                elf_sha256(path)
            Path(f"{path}.zst").unlink()
            with self.assertRaises(FileNotFoundError):
                elf_sha256(path)

    def test_metadata_uses_compiled_layout_and_scratchpad_slots(self):
        from artifact_meta import metadata
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            store.mkdir()
            (store / "manifest.json").write_text(json.dumps({"full_attention_layers": [5]}))
            config = root / "config.json"
            config.write_text(json.dumps({"text_config": {
                "hidden_size": 3840, "num_hidden_layers": 6, "vocab_size": 262144,
                "sliding_window": 1024, "final_logit_softcapping": 30.0}}))
            args = "m g h arena seg=64,256,1024 fseg=64,256,1024 split=128,256,512 gcap=65536 sring=1280 s64 fwd=0-5+h nbw=20 1 2"
            with patch.dict(os.environ, {"RF_FAST": "1", "RF_ATTN_H": "1"}):
                layer = importlib.import_module("rlayer_design")
                forward = importlib.import_module("rforward")
                with patch.object(forward, "STORE", str(store / "manifest.json")):
                    layer.configure(args.split())
                    layout = forward.layout(layer)
            (root / "gen_args.txt").write_text("rlayer_design " + args)
            (root / "gen_env.txt").write_text("RF_FAST=1\nRF_ATTN_H=1\n")
            (root / "fwd_layout.json").write_text(json.dumps(layout))
            slots = list(reversed(range(len(layout["params"]))))
            rows = [f"{name} {slot} i32 addr - -" for name, slot in zip(layout["params"], slots)]
            (root / "params.txt").write_text(f"{len(rows)} {len(rows)}\n" + "\n".join(rows) + "\n0\n0\n")
            elf = b"compiled ELF fixture"
            (root / "design.elf").write_bytes(elf)
            result = metadata(root, store, config)
            self.assertEqual(result["kind"], "resident_forward_ladder_v2")
            self.assertEqual(result["kv_ring_parameter_abi"], "zero-base-total-v1")
            self.assertEqual(layout["kv_ring_parameter_abi"], result["kv_ring_parameter_abi"])
            self.assertEqual(result["elf_sha256"], hashlib.sha256(elf).hexdigest())
            fixture = Path(__file__).resolve().parents[2] / "rust/npu-models/tests/fixtures/resident_ladder_v2_meta.json"
            self.assertEqual(result, json.loads(fixture.read_text()))
            self.assertEqual(result["scratchpad_params"], dict(zip(layout["params"], slots)))
            self.assertEqual(result["scratch_bytes"], layout["SF"])
            self.assertEqual(result["cache_bytes"], layout["KF"])
            self.assertEqual(result["s_ring_layout"], layout["s_ring_layout"])
            self.assertEqual(result["rungs"], layout["rungs"])
            self.assertEqual(result["nlayer"], 6)
            self.assertEqual(result["row_block"], 16)
            self.assertEqual(result["pmax"], max(rung["nt"] for rung in layout["rungs"]) * result["row_block"])
            for offset in result["split_widths_x_off"]:
                self.assertLessEqual(result["rope_g_off"] + offset - result["xrows"] + result["widths_bytes"], result["xbuf"])
            self.assertEqual(result["weight_dir"], "weights")
            legacy = dict(layout)
            legacy.pop("kv_ring_parameter_abi", None)
            (root / "fwd_layout.json").write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError, "layout"):
                metadata(root, store, config)
            layout["SF"] += 8
            (root / "fwd_layout.json").write_text(json.dumps(layout))
            with self.assertRaisesRegex(ValueError, "layout"):
                metadata(root, store, config)


if __name__ == "__main__":
    unittest.main()
