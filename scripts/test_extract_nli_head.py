import json
import os
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def ckpt(d, arch="Qwen3_5ForSequenceClassification", rows=3):
    w = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4).to(torch.bfloat16)
    save_file({"score.weight": w, "model.language_model.norm.weight": torch.ones(4, dtype=torch.bfloat16)},
              os.path.join(d, "model.safetensors"))
    json.dump({"architectures": [arch], "id2label": {str(i): l for i, l in
               enumerate(["contradiction", "entailment", "neutral"][:rows])},
               "nli_template": "Premise: {premise}\nHypothesis: {hypothesis}",
               "text_config": {"hidden_size": 4}}, open(os.path.join(d, "config.json"), "w"))
    return w


def run(src, out):
    return subprocess.run([sys.executable, os.path.join(HERE, "extract_nli_head.py"),
                           "--checkpoint-dir", str(src), "--out", str(out)], capture_output=True, text=True)


def test_writes_bf16_rows_and_head_json(tmp_path):
    w = ckpt(tmp_path)
    r = run(tmp_path, tmp_path / "head")
    assert r.returncode == 0, r.stderr
    raw = np.fromfile(tmp_path / "head" / "score.bin", dtype=np.uint16)
    assert raw.tolist() == w.view(torch.int16).numpy().astype(np.uint16).reshape(-1).tolist()
    h = json.load(open(tmp_path / "head" / "head.json"))
    assert h["labels"] == ["contradiction", "entailment", "neutral"]
    assert h["d_model"] == 4 and h["template"].startswith("Premise:")


def test_refuses_a_causal_lm(tmp_path):
    ckpt(tmp_path, arch="Qwen3_5ForConditionalGeneration")
    r = run(tmp_path, tmp_path / "head")
    assert r.returncode != 0 and "SequenceClassification" in r.stderr
