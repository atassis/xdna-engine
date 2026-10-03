import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("weight_store", HERE / "weight_store.py")
assert spec is not None and spec.loader is not None
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)


class WeightStoreTests(unittest.TestCase):
    def test_explicit_prototype_verification_rejects_mismatched_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ws, "BLOBS", directory + "/blobs"), \
                patch.object(ws, "gate_layer0", return_value=[("fixture", False, 4, 4)]) as gate, \
                patch.object(ws, "main") as build:
            with self.assertRaises(SystemExit) as failed:
                ws.cli(["--verify-prototypes"])
            self.assertEqual(failed.exception.code, 1)
            gate.assert_called_once_with()
            build.assert_not_called()
            self.assertFalse(Path(directory, "blobs").exists())

    def test_pack_does_not_require_prototype_scratch_files(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            weights = root / "weights"
            weights.mkdir()
            np.save(weights / "model.language_model.norm.weight.npy", np.array([1.0], dtype=np.float32))
            store = root / "store"
            for name, value in {"WDIR": str(weights), "STORE": str(store), "BLOBS": str(store / "blobs"),
                                "NLAYERS": 1, "FULL_ATTN_LAYERS": set()}.items():
                patches.enter_context(patch.object(ws, name, value))
            fixture = lambda: ws.entry(b"weight-store-test", "fixture")
            patches.enter_context(patch.object(ws, "pack_mlp_layer", side_effect=lambda li: fixture()))
            patches.enter_context(patch.object(ws, "pack_attn_sliding", side_effect=lambda li: (fixture(), fixture())))
            patches.enter_context(patch.object(ws, "pack_embedding", side_effect=fixture))
            patches.enter_context(patch.object(ws, "pack_towers", return_value={}))
            gate = patches.enter_context(patch.object(ws, "gate_layer0", side_effect=AssertionError(
                "normal packing must not require historical prototype scratch files")))
            ws.cli([])
            gate.assert_not_called()
            manifest = json.loads((store / "manifest.json").read_text())
            self.assertEqual(manifest["num_layers"], 1)
            entries = list(manifest["layers"]["0"]["matrices"].values()) + [manifest["embedding"], manifest["final_norm"]]
            for entry in entries:
                blob = store / "blobs" / f"{entry['blob']}.bin"
                self.assertEqual(blob.stat().st_size, entry["length"])


if __name__ == "__main__":
    unittest.main()
