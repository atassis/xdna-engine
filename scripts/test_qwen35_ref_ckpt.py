import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_ref import Ckpt  # noqa: E402


def test_single_file_checkpoint_needs_no_index(tmp_path):
    save_file({"model.language_model.norm.weight": torch.full((4,), 2.0)}, str(tmp_path / "model.safetensors"))
    json.dump({"text_config": {"hidden_size": 4}}, open(tmp_path / "config.json", "w"))
    assert Ckpt(str(tmp_path)).w("norm.weight").tolist() == [2.0] * 4
