import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class ParakeetExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.graph = self.source / "encoder-model.onnx"
        arrays = [numpy_helper.from_array(np.arange(4, dtype=np.float32).reshape(2, 2), "weight"),
                  numpy_helper.from_array(np.ones(2, dtype=np.float32), "scale"),
                  numpy_helper.from_array(np.zeros(2, dtype=np.float32), "bias")]
        nodes = [helper.make_node("Slice", ["audio_signal"], ["pos"], name="/pos_enc/Slice")]
        for block in range(24):
            prefix = f"/layers.{block}/"
            nodes.extend([
                helper.make_node("LayerNormalization", ["audio_signal", "scale", "bias"],
                                 [f"norm{block}"], name=prefix + "norm_feed_forward1/LayerNormalization"),
                helper.make_node("MatMul", [f"norm{block}", "weight"], [f"mm{block}"],
                                 name=prefix + "self_attn/linear_q/MatMul"),
                helper.make_node("LayerNormalization", [f"mm{block}", "scale", "bias"],
                                 [f"out{block}"], name=prefix + "norm_out/LayerNormalization"),
            ])
            for name in ("pos_bias_u", "pos_bias_v"):
                arrays.append(numpy_helper.from_array(np.ones((2, 2), dtype=np.float32),
                                                     f"layers.{block}.self_attn.{name}"))
        graph = helper.make_graph(nodes, "fixture", [
            helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, [1, 128, 256]),
            helper.make_tensor_value_info("length", TensorProto.INT64, [1]),
        ], [helper.make_tensor_value_info("out23", TensorProto.FLOAT, [1, 2, 2])], arrays)
        onnx.save_model(helper.make_model(graph), str(self.graph), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
        spec = importlib.util.spec_from_file_location("parakeet_export_fixture", Path(__file__).with_name("extract_parakeet_encoder.py"))
        self.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.exporter)
        self.exporter.SOURCE = str(self.source)
        self.exporter.ONNX = str(self.graph)
        self.exporter.OUT = str(self.output / "encoder")
        self.calls = []
        self.before = {p.name: p.read_bytes() for p in self.source.iterdir()}

    def run_export(self, fail=None):
        owner = self

        class Session:
            def __init__(self, model, sess_options=None, providers=None):
                owner.calls.append((Path(model), sess_options, providers))
                loaded = onnx.load(model, load_external_data=True)
                value = numpy_helper.to_array(next(i for i in loaded.graph.initializer if i.name == "weight"))
                np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32).reshape(2, 2))
                if fail == "construct":
                    raise RuntimeError("fixture session construction failure")

            def run(self, outputs, feed):
                if fail == "run":
                    raise RuntimeError("fixture reference execution failure")
                return [np.ones((1, 2, 2), dtype=np.float32) for _ in outputs]

        with patch.object(self.exporter.ort, "InferenceSession", Session):
            self.exporter.main()

    def assert_clean(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, self.before)
        self.assertFalse(any(p.is_symlink() or p.name == "_ref_graph.onnx"
                             or p.name.startswith("parakeet-ref-") for p in self.output.rglob("*")))

    def test_read_only_input_exports_through_output_temporary_graph(self):
        self.source.chmod(0o555)
        self.addCleanup(self.source.chmod, 0o755)
        self.run_export()
        self.assertTrue(self.calls[0][0].is_relative_to(self.output))
        self.assertTrue((self.output / "encoder/manifest.json").is_file())
        self.assert_clean()

    def test_reference_session_has_explicit_thread_bounds(self):
        self.run_export()
        _, options, providers = self.calls[0]
        self.assertIsNotNone(options)
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assert_clean()

    def test_external_data_location_cannot_escape_temporary_directory(self):
        load = self.exporter.onnx.load

        def escaped_location(*args, **kwargs):
            model = load(*args, **kwargs)
            if kwargs.get("load_external_data") is False:
                for tensor in model.graph.initializer:
                    for entry in tensor.external_data:
                        if entry.key == "location":
                            entry.value = "../escape.data"
            return model

        with patch.object(self.exporter.onnx, "load", escaped_location):
            with self.assertRaisesRegex(ValueError, "external data location must stay relative"):
                self.run_export()
        self.assertEqual(self.calls, [])
        self.assert_clean()

    def test_constructor_failure_cleans_temporary_graph(self):
        with self.assertRaisesRegex(RuntimeError, "fixture session construction"):
            self.run_export(fail="construct")
        self.assert_clean()

    def test_execution_failure_cleans_temporary_graph(self):
        with self.assertRaisesRegex(RuntimeError, "fixture reference execution"):
            self.run_export(fail="run")
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
