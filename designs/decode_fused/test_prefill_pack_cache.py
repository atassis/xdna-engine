#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pack_cache


GENERATOR = Path(__file__).with_name("gen_llm_prefill.py")


def pack_function():
    tree = ast.parse(GENERATOR.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "pack_quant_weights")
    namespace = {"os": os, "np": np, "pack_cache": pack_cache,
                 "tempfile": tempfile, "__file__": str(GENERATOR)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATOR), "exec"), namespace)
    return namespace[function.name]


class PrefillPackCache(unittest.TestCase):
    def test_changed_input_does_not_overwrite_other_materializations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            source = root / "source.npy"
            original = np.array([1, 2, 3, 4], dtype=np.int8)
            replacement = np.array([5, 6, 7, 8], dtype=np.int8)
            np.save(source, original)
            entry = {"buf": "W", "src_dir": str(root), "src_file": source.name,
                     "N": 1, "K": 4, "tile_k": 4, "tile_n": 1,
                     "group_size": 4, "weight_dtype": "int4", "cols": 1,
                     "scale_dtype": "f32", "mmul": [1, 1, 1]}
            quant = types.ModuleType("iron.common.quant")
            quant.__file__ = str(GENERATOR)
            setattr(quant, "repack_gemm_weight", lambda rows, *args, **kwargs: rows)
            common = types.ModuleType("iron.common")
            setattr(common, "quant", quant)
            env = {"PACK_CACHE_DIR": str(root / "cache"),
                   "PREFILL_PACK_CACHE_DIR": str(root / "cache"), "PREFILL_PACK_CACHE": "1"}
            with patch.dict(os.environ, env), patch.dict("sys.modules", {"iron.common": common}), \
                    redirect_stdout(io.StringIO()):
                pack = pack_function()
                pack([entry], str(first), {})
                pack([entry], str(first), {})
                pack([entry], str(second), {})
                self.assertEqual((first / "W.bin").stat().st_ino,
                                 (second / "W.bin").stat().st_ino)
                np.save(source, replacement)
                pack([entry], str(first), {})
            self.assertEqual((first / "W.bin").read_bytes(), replacement.tobytes())
            self.assertEqual((second / "W.bin").read_bytes(), original.tobytes())


if __name__ == "__main__":
    unittest.main()
