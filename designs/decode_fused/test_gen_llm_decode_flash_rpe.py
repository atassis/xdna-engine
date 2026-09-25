# SPDX-License-Identifier: Apache-2.0
"""GLOBAL_FLASH_RPE: default is a no-op on the name, non-default reaches AttnGlobalFlash and the name.

  PYTHONPATH=designs/decode_fused:$IRON_DIR .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_gen_llm_decode_flash_rpe.py -v
"""
import importlib

import pytest


@pytest.fixture
def gen(monkeypatch):
    monkeypatch.setenv("FUSE_ATTN_GLOBAL_FLASH", "1")
    monkeypatch.setenv("KV_ALLOC", "262144")
    import gen_llm_decode
    return importlib.reload(gen_llm_decode)


def test_rpe_default_is_one_and_name_unchanged(gen):
    assert gen.GLOBAL_FLASH_RPE == 1
    assert gen.flash_name_token(((512, 1, False),), 4) == "_agf_512_hpc4"


def test_rpe_env_reaches_the_module_global(monkeypatch):
    monkeypatch.setenv("FUSE_ATTN_GLOBAL_FLASH", "1")
    monkeypatch.setenv("KV_ALLOC", "262144")
    monkeypatch.setenv("GLOBAL_FLASH_RPE", "8")
    import gen_llm_decode
    gen8 = importlib.reload(gen_llm_decode)
    assert gen8.GLOBAL_FLASH_RPE == 8


def test_name_token_carries_rpe_only_when_non_default(gen):
    assert gen.flash_name_token(((512, 1, False),), 4, rpe=1) == "_agf_512_hpc4"
    assert gen.flash_name_token(((512, 1, False),), 4, rpe=8) == "_agf_512_hpc4_rpe8"
    assert gen.flash_name_token((), 4, rpe=8) == ""
