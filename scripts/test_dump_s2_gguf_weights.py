# SPDX-License-Identifier: Apache-2.0
"""Tests for dump_s2_gguf_weights.py -- GGUF q6_k -> .npy dumper for S2-Pro's Slow-AR, remapped to
the HF-style keys designs/decode_fused/gen_llm_prefill.py already reads.

Run:
  .venv-iron/bin/python -m pytest scripts/test_dump_s2_gguf_weights.py -v
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

gguf = pytest.importorskip("gguf")

from dump_s2_gguf_weights import count_layers, dequantize_tensor, dump, dump_slow_ar_layer, load_gguf_tensor_names  # noqa: E402

S2_GGUF_PATH = os.environ.get("S2_GGUF_PATH", os.path.join(_HERE, "..", "s2.cpp", "models", "s2-pro-q6_k.gguf"))

if not os.path.exists(S2_GGUF_PATH):
    pytest.skip(f"no S2-Pro GGUF at {S2_GGUF_PATH}; set S2_GGUF_PATH", allow_module_level=True)


def test_load_gguf_tensor_names_finds_slow_ar_layer0():
    names = load_gguf_tensor_names(S2_GGUF_PATH)
    assert "layers.0.attention.wqkv.weight" in names
    assert "model.layers.0.self_attn.q_proj.weight" not in names  # confirms the GGUF's native naming


def test_dequantize_tensor_matches_gguf_quants_reference():
    reader = gguf.GGUFReader(S2_GGUF_PATH)
    tensor = next(t for t in reader.tensors if t.name == "layers.0.attention.wqkv.weight")
    ours = dequantize_tensor(tensor)
    reference = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    assert ours.shape == reference.shape
    np.testing.assert_array_equal(ours, reference)


def test_dump_slow_ar_layer0_produces_hf_style_keys_with_correct_shapes():
    reader = gguf.GGUFReader(S2_GGUF_PATH)
    tensor_dict = {t.name: t for t in reader.tensors}
    tensors = dump_slow_ar_layer(tensor_dict, layer=0)
    expected_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (4096, 2560),
        "model.layers.0.self_attn.k_proj.weight": (1024, 2560),
        "model.layers.0.self_attn.v_proj.weight": (1024, 2560),
        "model.layers.0.self_attn.o_proj.weight": (2560, 4096),
        "model.layers.0.self_attn.q_norm.weight": (128,),
        "model.layers.0.self_attn.k_norm.weight": (128,),
        "model.layers.0.input_layernorm.weight": (2560,),
        "model.layers.0.post_attention_layernorm.weight": (2560,),
        "model.layers.0.mlp.gate_proj.weight": (9728, 2560),
        "model.layers.0.mlp.up_proj.weight": (9728, 2560),
        "model.layers.0.mlp.down_proj.weight": (2560, 9728),
    }
    assert set(tensors) == set(expected_shapes)
    for key, shape in expected_shapes.items():
        assert tensors[key].shape == shape, f"{key}: {tensors[key].shape} != {shape}"

    wqkv = dequantize_tensor(tensor_dict["layers.0.attention.wqkv.weight"])
    recombined = np.concatenate([
        tensors["model.layers.0.self_attn.q_proj.weight"],
        tensors["model.layers.0.self_attn.k_proj.weight"],
        tensors["model.layers.0.self_attn.v_proj.weight"],
    ], axis=0)
    np.testing.assert_array_equal(recombined, wqkv)


def test_count_layers_auto_detects_full_stack():
    """Test the n_layers=None auto-detection branch (no full dump, just count assertion)."""
    reader = gguf.GGUFReader(S2_GGUF_PATH)
    tensor_dict = {t.name: t for t in reader.tensors}
    n_layers = count_layers(tensor_dict)
    assert n_layers == 36, f"expected 36 layers, got {n_layers}"


def test_dump_writes_expected_files_for_one_layer(tmp_path):
    """Test the dump() function (file-writing path and main() coverage)."""
    dump(S2_GGUF_PATH, str(tmp_path), layers=1)
    expected_files = [
        "model.layers.0.self_attn.q_proj.weight.npy",
        "model.layers.0.self_attn.k_proj.weight.npy",
        "model.layers.0.self_attn.v_proj.weight.npy",
        "model.layers.0.self_attn.o_proj.weight.npy",
        "model.layers.0.self_attn.q_norm.weight.npy",
        "model.layers.0.self_attn.k_norm.weight.npy",
        "model.layers.0.input_layernorm.weight.npy",
        "model.layers.0.post_attention_layernorm.weight.npy",
        "model.layers.0.mlp.gate_proj.weight.npy",
        "model.layers.0.mlp.up_proj.weight.npy",
        "model.layers.0.mlp.down_proj.weight.npy",
        "model.norm.weight.npy",
        "model.embed_tokens.weight.npy",
    ]
    files = sorted(os.listdir(tmp_path))
    assert files == sorted(expected_files), f"got {len(files)} files, expected {len(expected_files)}"
    # Spot-check one file: load and verify shape
    q_proj = np.load(os.path.join(tmp_path, "model.layers.0.self_attn.q_proj.weight.npy"))
    assert q_proj.shape == (4096, 2560)
