#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "exec_graph", Path(__file__).with_name("exec_graph.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("load scripts/exec_graph.py")
exec_graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exec_graph)


class EncoderDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.ops = {op[0]: op for op in exec_graph.ENCODER_OPS}

    def test_removed_encoder_gate_names_are_not_declared(self):
        gates = {op[4] for op in exec_graph.ENCODER_OPS}
        self.assertNotIn("NPU_ENC_CONV_NPU", gates)
        self.assertNotIn("NPU_ENC_MHA_NPU", gates)

    def test_mha_declares_artifact_and_max_layer_conditions(self):
        mha = self.ops["mha"]
        self.assertEqual(mha[4], "NPU_ENC_MHA_MAXLAYER")
        self.assertTrue(mha[5])
        self.assertIn("artifact", mha[6].lower())
        model = exec_graph.build_model(
            {
                "name": "whisper-small",
                "scenario": "whisper-small.toml",
                "d_model": 768,
                "ffn": 3072,
                "n_heads": 12,
                "head_dim": 64,
                "n_layers": 12,
                "max_seq": 1500,
                "K_candidates": [800, 768],
                "registered_in_engine_toml": True,
            },
            [],
            [],
        )
        declared = next(op for op in model["ops"] if op["op"] == "mha")
        self.assertFalse(declared["artifact_present"])
        self.assertTrue(
            any("no StaticMHA" in missing for missing in model["health"]["missing"])
        )

    def test_conv_stem_declaration_is_host_without_a_gate(self):
        conv = self.ops["conv_stem"]
        self.assertEqual(conv[1], "host")
        self.assertIsNone(conv[4])


if __name__ == "__main__":
    unittest.main()
