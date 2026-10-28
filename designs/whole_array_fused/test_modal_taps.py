"""Ordered modal DMA descriptors captured from the shipped tiler contract."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


class ModalTapTests(unittest.TestCase):
    def test_current_tap_api_preserves_shipped_walks(self):
        spec = importlib.util.spec_from_file_location("modal_generator", HERE / "whole_array_modal_iron.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        baseline = json.loads((HERE / "modal_taps_baseline.json").read_text())
        for case in baseline["cases"]:
            with self.subTest(args=case["args"]):
                actual = generator.matrix_taps(*case["args"])
                self.assertEqual(len(actual), 3)
                for taps, expected in zip(actual, case["taps"]):
                    self.assertEqual(len(taps), len(expected))
                    for tap, old in zip(taps, expected):
                        self.assertEqual(list(tap.tensor_dims), old["shape"])
                        self.assertEqual(tap.offset, old["offset"])
                        self.assertEqual(list(tap.sizes), old["sizes"])
                        self.assertEqual(list(tap.strides), old["strides"])


if __name__ == "__main__":
    unittest.main()
