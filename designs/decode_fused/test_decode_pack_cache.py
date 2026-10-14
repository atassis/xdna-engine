#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import pack_cache


GENERATOR = Path(__file__).with_name("gen_llm_decode.py")


def quantize_function(quantizer):
    tree = ast.parse(GENERATOR.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_quantize_cached")
    namespace = {"os": os, "np": np, "pack_cache": pack_cache,
                 "hashlib": hashlib, "json": json, "sys": sys, "quantize_weight": quantizer}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATOR), "exec"), namespace)
    return namespace[function.name]


class DecodePackCache(unittest.TestCase):
    def fixture(self, root):
        calls = []

        def quantizer(weights, *args, **kwargs):
            calls.append((args, kwargs))
            return weights * 2

        quantizer.__module__ = "cache_fixture_quantizer"
        source = root / "packer.py"
        source.write_text("# fixture packer version one\n")
        module = types.ModuleType(quantizer.__module__)
        module.__file__ = str(source)
        return quantizer, module, source, calls

    def test_hit_keeps_lease_through_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(max_workers=1) as pool:
            root = Path(directory)
            quantizer, module, source, calls = self.fixture(root)
            weights = np.arange(4, dtype=np.float32)
            env = {"PACK_CACHE_DIR": str(root / "cache"), "DECODE_PACK_CACHE": "1"}
            original_load = np.load
            pending = []

            def pinned_load(cached):
                self.assertTrue(hasattr(cached, "read"), "cache hit escaped as an unleased path")
                started = threading.Event()

                def prune():
                    started.set()
                    pack_cache.ContentStore(str(root / "cache"), max_bytes=0).prune()

                future = pool.submit(prune)
                pending.append(future)
                self.assertTrue(started.wait(2))
                with self.assertRaises(TimeoutError):
                    future.result(timeout=0.05)
                return original_load(cached)

            with patch.dict(os.environ, env), patch.dict(sys.modules, {module.__name__: module}):
                quantize = quantize_function(quantizer)
                expected = quantize(weights, 4, "int4", {})
                with patch.object(np, "load", side_effect=pinned_load):
                    actual = quantize(weights, 4, "int4", {})
                for future in pending:
                    future.result(timeout=2)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(len(calls), 1)

    def test_content_axes_invalidate_and_oversized_results_are_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quantizer, module, source, calls = self.fixture(root)
            weights = np.arange(4, dtype=np.float32)
            env = {"PACK_CACHE_DIR": str(root / "cache"), "DECODE_PACK_CACHE": "1"}
            with patch.dict(os.environ, env), patch.dict(sys.modules, {module.__name__: module}):
                quantize = quantize_function(quantizer)
                variants = [(weights, 4, "int4", {}), (weights + 1, 4, "int4", {}),
                            (weights.reshape(2, 2), 4, "int4", {}),
                            (weights.astype(np.float64), 4, "int4", {}),
                            (weights, 8, "int4", {}), (weights, 4, "int8", {}),
                            (weights, 4, "int4", {"clip": True})]
                for count, args in enumerate(variants, 1):
                    expected = args[0] * 2
                    np.testing.assert_array_equal(quantize(*args), expected)
                    np.testing.assert_array_equal(quantize(*args), expected)
                    self.assertEqual(len(calls), count)
                source.write_text("# fixture packer version two\n")
                quantize(weights, 4, "int4", {})
                self.assertEqual(len(calls), len(variants) + 1)
                with patch.object(np, "__version__", np.__version__ + ".fixture"):
                    quantize(weights, 4, "int4", {})
                self.assertEqual(len(calls), len(variants) + 2)
                objects = list((root / "cache" / "objects").glob("*/*.bin"))
                with patch.dict(os.environ, {"PACK_CACHE_MAX_BYTES": "1"}):
                    quantize(weights + 10, 4, "int4", {})
                    quantize(weights + 10, 4, "int4", {})
                self.assertEqual(len(calls), len(variants) + 4)
                self.assertEqual(objects, list((root / "cache" / "objects").glob("*/*.bin")))
                self.assertFalse(list((root / "cache" / "objects").glob(".pack-*")))


if __name__ == "__main__":
    unittest.main()
