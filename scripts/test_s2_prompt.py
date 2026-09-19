#!/usr/bin/env python3
"""Tests for `s2_prompt.py`.

No `ar_prompt_tokens` dump exists yet to gate against (`s2-ar-gate-against-s2cpp-dumps` is open,
not started -- s2.cpp has not been patched to emit it), so this is structural/self-consistency
testing: shape, the zero-codebook invariant on non-semantic rows, and that the id set in play is
s1-mini's own and not S2-Pro's -- not a byte-for-byte match against a real s2.cpp run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_prompt
import s2_tokenizer_convert as stc

MODEL_DIR = stc.DEFAULT_MODEL_DIR
pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(), reason=f"s1-mini snapshot not present at {MODEL_DIR}"
)

IM_START_ID = 151646
IM_END_ID = 151647
VOICE_ID = 151653
SEMANTIC_BEGIN_ID = 151658
S2PRO_IM_END_ID = 151645
S2PRO_SEMANTIC_BEGIN_ID = 151678


@pytest.fixture(scope="module")
def tokenizer_and_ids():
    return s2_prompt.load_tokenizer_and_ids(MODEL_DIR)


def test_prompt_ids_read_by_name_not_hardcoded(tokenizer_and_ids):
    _, ids = tokenizer_and_ids
    assert ids.im_start_id == IM_START_ID
    assert ids.im_end_id == IM_END_ID
    assert ids.voice_id == VOICE_ID
    assert ids.semantic_begin_id == SEMANTIC_BEGIN_ID
    assert ids.num_codebooks == 10
    # the trap: S2-Pro's constants are different, plausible, in-range ids on this vocab too
    assert ids.im_end_id != S2PRO_IM_END_ID
    assert ids.semantic_begin_id != S2PRO_SEMANTIC_BEGIN_ID


def test_no_reference_shape_and_dtype(tokenizer_and_ids):
    tokenizer, ids = tokenizer_and_ids
    matrix = s2_prompt.build_prompt(tokenizer, ids, "Hello, world!")
    assert matrix.dtype == np.int32
    assert matrix.ndim == 2
    assert matrix.shape[1] == ids.num_codebooks + 1 == 11


def test_no_reference_codebook_columns_are_all_zero(tokenizer_and_ids):
    """Every row is a text/control row without a reference -- codebook slots (columns 1..10) are
    masked to zero on all of them, matching `s2_prompt.cpp`'s non-semantic-row convention."""
    tokenizer, ids = tokenizer_and_ids
    matrix = s2_prompt.build_prompt(tokenizer, ids, "Speak this sentence.")
    assert np.all(matrix[:, 1:] == 0)
    assert np.any(matrix[:, 0] != 0), "column 0 must actually carry the encoded prompt"


def test_no_reference_structure_matches_s2_prompt_cpp(tokenizer_and_ids):
    """Reproduces `s2_prompt.cpp`'s no-reference branch token-for-token: system preamble, im_end,
    user turn with the caller's text, im_end, assistant turn, then the voice token -- all in
    column 0, nothing before it."""
    tokenizer, ids = tokenizer_and_ids
    text = "Read this aloud."
    matrix = s2_prompt.build_prompt(tokenizer, ids, text)
    col0 = matrix[:, 0].tolist()

    expected: list[int] = []
    expected += tokenizer.encode("<|im_start|>system")
    expected.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected += tokenizer.encode("convert the provided text to speech")
    expected.append(ids.im_end_id)
    expected.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected += tokenizer.encode("<|im_start|>user")
    expected.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected += tokenizer.encode(text)
    expected.append(ids.im_end_id)
    expected.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected += tokenizer.encode("<|im_start|>assistant")
    expected.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected.append(ids.voice_id)

    assert col0 == expected
    assert col0.count(ids.im_end_id) == 2, "im_end_id closes both the system and the user turn"


def test_reference_audio_shapes_the_codebook_columns(tokenizer_and_ids):
    """A voice profile prepends T reference-code rows, and those rows carry non-zero codebook
    columns -- the case `s2-voice-profiles-reference-audio` will eventually feed with real codec
    output. Synthetic codes here only, since the codec encoder is unbuilt."""
    tokenizer, ids = tokenizer_and_ids
    rng = np.random.default_rng(0)
    t_prompt = 6
    codes = rng.integers(0, 4096, size=(ids.num_codebooks, t_prompt), dtype=np.int32)

    text = "Say this in the reference voice."
    prompt_text = "This is the reference transcript."
    matrix = s2_prompt.build_prompt(tokenizer, ids, text, prompt_text=prompt_text, prompt_codes=codes)

    sys_pre_len = len(tokenizer.encode("<|im_start|>system")) + 1
    sys_pre_len += len(tokenizer.encode("convert the provided text to speech reference to the following:\n\nText:\n"))
    sys_pre_len += len(tokenizer.encode("<|speaker:0|>"))
    sys_pre_len += len(tokenizer.encode(prompt_text))
    sys_pre_len += len(tokenizer.encode("\n\nSpeech:\n"))

    ref_rows = matrix[sys_pre_len : sys_pre_len + t_prompt]
    # cb0's row is double-used: column 0 gets it offset into vocab space, column 1 (cb0) keeps the
    # raw code -- see build_prompt's docstring.
    assert np.array_equal(ref_rows[:, 0], codes[0, :] + ids.semantic_begin_id)
    assert np.array_equal(ref_rows[:, 1:], codes.T)
    assert np.all(ref_rows[:, 1:] != 0) or np.any(codes != 0)  # codebook columns are genuinely populated

    # rows outside the reference span are unaffected: still text ids in column 0, zero codebooks
    assert np.all(matrix[:sys_pre_len, 1:] == 0)
    assert np.all(matrix[sys_pre_len + t_prompt :, 1:] == 0)


def test_reference_requires_all_three_fields(tokenizer_and_ids):
    """`has_reference` in `s2_prompt.cpp` is codes AND T_prompt>0 AND non-empty prompt_text --
    missing any one falls back to the no-reference branch, not a partial one."""
    tokenizer, ids = tokenizer_and_ids
    codes = np.zeros((ids.num_codebooks, 4), dtype=np.int32)

    no_text_matrix = s2_prompt.build_prompt(tokenizer, ids, "hi", prompt_text="", prompt_codes=codes)
    no_codes_matrix = s2_prompt.build_prompt(tokenizer, ids, "hi", prompt_text="a reference")
    plain_matrix = s2_prompt.build_prompt(tokenizer, ids, "hi")

    assert no_text_matrix.shape == plain_matrix.shape
    assert no_codes_matrix.shape == plain_matrix.shape
    assert np.all(no_text_matrix[:, 1:] == 0)
    assert np.all(no_codes_matrix[:, 1:] == 0)


def test_wrong_codebook_count_is_rejected(tokenizer_and_ids):
    tokenizer, ids = tokenizer_and_ids
    bad_codes = np.zeros((ids.num_codebooks + 1, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        s2_prompt.build_prompt(tokenizer, ids, "hi", prompt_text="ref", prompt_codes=bad_codes)


def test_speaker_tag_already_present_is_not_duplicated(tokenizer_and_ids):
    """`s2_prompt.cpp` only injects the default `<|speaker:0|>` when `prompt_text` has no
    `<|speaker:` tag of its own -- check the exact column-0 prefix, not just a length delta."""
    tokenizer, ids = tokenizer_and_ids
    codes = np.zeros((ids.num_codebooks, 2), dtype=np.int32)
    prompt_text = "<|speaker:3|> already tagged."
    matrix = s2_prompt.build_prompt(tokenizer, ids, "hi", prompt_text=prompt_text, prompt_codes=codes)

    expected_pre: list[int] = []
    expected_pre += tokenizer.encode("<|im_start|>system")
    expected_pre.append(s2_prompt.NEWLINE_TOKEN_ID)
    expected_pre += tokenizer.encode("convert the provided text to speech reference to the following:\n\nText:\n")
    expected_pre += tokenizer.encode(prompt_text)  # no injected "<|speaker:0|>" before this
    expected_pre += tokenizer.encode("\n\nSpeech:\n")

    assert matrix[: len(expected_pre), 0].tolist() == expected_pre
