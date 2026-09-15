# SPDX-License-Identifier: Apache-2.0
"""Which specs the --text path may tokenize (kernel-contract K019).

`--text` reads its corpus with QwenBPE. The tokenizer it reaches for used to be a hardcoded Qwen
cache path regardless of spec, so a Gemma spec silently scored Qwen ids: mean NLL 19.836 against
ln(262144)=12.477, and top-1 EXACTLY 0.0 over 2560 positions -- a vocabulary mismatch, not a
degraded model, and indistinguishable from a bad weight format in the number it produces.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "qwen_bpe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_bpe.py"))
qwen_bpe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qwen_bpe)


def test_qwen_spec_resolves_to_a_tokenizer():
    assert qwen_bpe.text_tokenizer_hint("qwen3-0.6b") is not None


@pytest.mark.parametrize("spec", ["gemma4-12b", "gemma3-270m"])
def test_gemma_specs_have_no_text_tokenizer(spec):
    """A Gemma tokenizer.json uses a Split pretokenizer QwenBPE cannot parse, so there is no
    honest default -- the caller must pre-tokenize and pass --ids."""
    assert qwen_bpe.text_tokenizer_hint(spec) is None


def test_unknown_spec_is_refused_rather_than_defaulted():
    assert qwen_bpe.text_tokenizer_hint("some-future-model") is None
