import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ArtifactMetaTests(unittest.TestCase):
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
            rows = [f"{name} {slot} i32 addr" for name, slot in zip(layout["params"], slots)]
            (root / "params.txt").write_text(str(len(rows)) + "\n" + "\n".join(rows) + "\n")
            result = metadata(root, store, config)
            self.assertEqual(result["scratchpad_params"], dict(zip(layout["params"], slots)))
            self.assertEqual(result["scratch_bytes"], layout["SF"])
            self.assertEqual(result["cache_bytes"], layout["KF"])
            self.assertEqual(result["s_ring_layout"], layout["s_ring_layout"])
            self.assertEqual(result["rungs"], layout["rungs"])
            self.assertEqual(result["nlayer"], 6)
            self.assertEqual(result["row_block"], 16)
            self.assertEqual(result["weight_dir"], "weights")
            layout["SF"] += 8
            (root / "fwd_layout.json").write_text(json.dumps(layout))
            with self.assertRaisesRegex(ValueError, "layout"):
                metadata(root, store, config)


if __name__ == "__main__":
    unittest.main()
