# SPDX-License-Identifier: Apache-2.0
"""FUSE_ATTN_GLOBAL_FLASH: name, runlist and meta agree on which geometries the op claims.

  PYTHONPATH=designs/decode_fused:$IRON_DIR .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_gen_llm_decode_global_flash.py -v
"""
import importlib

import pytest


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.setenv("FUSE_ATTN_GLOBAL_FLASH", "1")
    monkeypatch.setenv("KV_ALLOC", "262144")
    import gen_llm_decode
    return importlib.reload(gen_llm_decode)


def test_global_geometry_is_claimed_and_named(gen):
    from llm_decode_spec import SPECS
    sp = SPECS["gemma4-12b"]
    why = gen._attn_global_flash_why(sp, (512, 1, False))
    assert why is None
    assert gen._attn_global_flash_why(sp, (256, 8, True)) is not None


def test_name_token_is_present_only_when_claimed(gen):
    assert "agf_512_hpc16" in gen.flash_name_token(((512, 1, False),), 16)
    assert gen.flash_name_token((), 16) == ""
