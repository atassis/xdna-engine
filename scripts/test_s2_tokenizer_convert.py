#!/usr/bin/env python3
"""Tests for `s2_tokenizer_convert.py`.

Neither `tokenizers` nor `tiktoken` is installed in `.venv-iron` (checked at collection time
below), so the primary check is functional, not just schema-shaped: this module's own `Tokenizer`
class -- a from-scratch reimplementation of the normalize/split/BPE-merge/vocab-lookup pipeline --
loads the produced file and round-trips real text through it. That does not prove bit-for-bit
parity with the Rust `tokenizers` 0.20 crate the engine actually embeds.

That parity WAS checked out-of-band (not part of this suite, so it runs with no extra
dependencies): an unrelated venv on this box already has the `tokenizers` PyPI package (0.23.2,
same wire format) installed. Loading the produced file there via `Tokenizer.from_file` -- the
identical call `rust/npu-engine/src/llm/config.rs` makes -- round-tripped every string below and
resolved `<|im_end|>`/`<|semantic:0|>`/`<|pad|>` to the same ids this suite asserts. If a
`tokenizers` install is present wherever this suite runs, `test_real_tokenizers_library_if_available`
repeats that check inline instead of skipping it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_tokenizer_convert as stc

MODEL_DIR = stc.DEFAULT_MODEL_DIR
S2PRO_TOKENIZER_JSON = Path(__file__).resolve().parents[2] / "s2.cpp" / "models" / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(), reason=f"s1-mini snapshot not present at {MODEL_DIR}"
)

# Measured facts this converter must reproduce (task spec + independently re-verified against
# special_tokens.json/tokenizer.tiktoken directly -- see report). Do not "fix" a failing test by
# editing these; a mismatch means the converter drifted from the checkpoint's own files.
IM_START_ID = 151646
IM_END_ID = 151647
PAD_ID = 151645
SEMANTIC_BEGIN_ID = 151658
SEMANTIC_END_ID = 155753
NUM_SEMANTIC = 4096
NUM_NON_SEMANTIC_SPECIALS = 15
NUM_SPECIALS = NUM_NON_SEMANTIC_SPECIALS + NUM_SEMANTIC
BASE_VOCAB_SIZE = 151643

# The same ids on S2-Pro (`s2.cpp/models/tokenizer.json`) -- a DIFFERENT vocabulary. Reusing these
# for s1-mini is exactly the defect this task exists to prevent (S2-Pro's im_end is s1-mini's pad).
S2PRO_IM_END_ID = 151645
S2PRO_SEMANTIC_BEGIN_ID = 151678


@pytest.fixture(scope="module")
def tokenizer_json() -> dict:
    return stc.build_tokenizer_json(MODEL_DIR)


@pytest.fixture(scope="module")
def tokenizer(tokenizer_json: dict) -> stc.Tokenizer:
    return stc.Tokenizer(tokenizer_json)


@pytest.fixture(scope="module")
def specials() -> dict[str, int]:
    return stc.load_special_tokens(MODEL_DIR / "special_tokens.json")


def test_tiktoken_ranks_are_dense_and_byte_complete():
    ranks = stc.load_tiktoken_ranks(MODEL_DIR / "tokenizer.tiktoken")
    assert len(ranks) == BASE_VOCAB_SIZE
    assert sorted(ranks.values()) == list(range(BASE_VOCAB_SIZE))
    single_byte = [t for t in ranks if len(t) == 1]
    assert len(single_byte) == 256, "byte-level BPE must cover all 256 raw byte values"


def test_special_tokens_json_matches_measured_facts(specials: dict[str, int]):
    assert len(specials) == NUM_SPECIALS
    assert specials["<|im_start|>"] == IM_START_ID
    assert specials["<|im_end|>"] == IM_END_ID
    assert specials["<|pad|>"] == PAD_ID
    assert specials["<|semantic:0|>"] == SEMANTIC_BEGIN_ID
    assert specials["<|semantic:4095|>"] == SEMANTIC_END_ID

    semantic = {k: v for k, v in specials.items() if k.startswith("<|semantic:")}
    non_semantic = {k: v for k, v in specials.items() if not k.startswith("<|semantic:")}
    assert len(semantic) == NUM_SEMANTIC
    assert len(non_semantic) == NUM_NON_SEMANTIC_SPECIALS
    assert min(semantic.values()) == SEMANTIC_BEGIN_ID
    assert max(semantic.values()) == SEMANTIC_END_ID
    # contiguous, no gaps, no id shared between the base vocab and the added range
    assert sorted(specials.values()) == list(range(BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + NUM_SPECIALS))


def test_reconstructed_bpe_has_no_collision_and_right_merge_count(tokenizer_json: dict):
    model = tokenizer_json["model"]
    assert len(model["vocab"]) == BASE_VOCAB_SIZE
    # every multi-byte token reduces to exactly one merge (reconstruct_bpe raises otherwise);
    # the count is also cross-checked byte-for-byte against S2-Pro's own file below.
    assert len(model["merges"]) == BASE_VOCAB_SIZE - 256
    assert set(model["vocab"]) & set(t["content"] for t in tokenizer_json["added_tokens"]) == set()


def test_added_tokens_match_special_tokens_json_exactly(tokenizer_json: dict, specials: dict[str, int]):
    added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
    assert added == specials
    assert all(t["special"] is True for t in tokenizer_json["added_tokens"])
    assert len(tokenizer_json["added_tokens"]) == len(specials), "no duplicate ids/content"


def test_ids_are_s1_mini_not_s2_pro(tokenizer_json: dict):
    added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
    assert added["<|im_end|>"] == IM_END_ID
    assert added["<|im_end|>"] != S2PRO_IM_END_ID
    assert added["<|semantic:0|>"] == SEMANTIC_BEGIN_ID
    assert added["<|semantic:0|>"] != S2PRO_SEMANTIC_BEGIN_ID
    # the trap named in the task: S2-Pro's im_end id is s1-mini's pad, not its im_end
    assert added["<|pad|>"] == S2PRO_IM_END_ID
    assert added["<|pad|>"] != added["<|im_end|>"]


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n",
        "Testing 123 -- multilingual: héllo wörld, привет мир, こんにちは",
        "<|semantic:42|> is a control token, not plain text.",
        "",
    ],
)
def test_round_trips_text(tokenizer: stc.Tokenizer, text: str):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_encode_to_their_own_single_id(tokenizer: stc.Tokenizer, specials: dict[str, int]):
    for content, tid in specials.items():
        assert tokenizer.encode(content) == [tid], content


def test_convert_writes_a_loadable_file(tmp_path: Path, tokenizer_json: dict):
    out = tmp_path / "tokenizer.json"
    result = stc.convert(MODEL_DIR, out)
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == tokenizer_json == result

    # exercised through the file, not the in-memory dict, since that's what `Tokenizer::from_file`
    # actually reads
    reloaded = stc.Tokenizer.from_file(out)
    assert reloaded.decode(reloaded.encode("round trip through disk")) == "round trip through disk"


@pytest.mark.skipif(not S2PRO_TOKENIZER_JSON.exists(), reason="s2.cpp checkout not present")
def test_base_vocab_matches_s2pro_format_reference_byte_for_byte(tokenizer_json: dict):
    """S2-Pro's tokenizer.json is a FORMAT reference, not a substitute (see module docstring):
    its base BPE table (ids 0..151642) should match ours exactly since both trace to the same
    underlying vocab, but its added tokens (specials/semantic ids) must NOT."""
    s2pro = json.loads(S2PRO_TOKENIZER_JSON.read_text())
    mine = tokenizer_json["model"]
    ref = s2pro["model"]
    assert mine["vocab"] == ref["vocab"]
    assert mine["merges"] == ref["merges"]

    s2pro_added = {t["content"]: t["id"] for t in s2pro["added_tokens"]}
    mine_added = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
    assert s2pro_added["<|im_end|>"] != mine_added["<|im_end|>"]
    assert s2pro_added["<|semantic:0|>"] != mine_added["<|semantic:0|>"]


def test_real_tokenizers_library_if_available(tokenizer_json: dict, specials: dict[str, int], tmp_path: Path):
    """If the `tokenizers` PyPI package happens to be present (it is not in `.venv-iron` as of
    writing -- this project's own venv check), validate against the actual library instead of only
    this module's reimplementation."""
    tokenizers_lib = pytest.importorskip("tokenizers", reason="tokenizers package not installed")
    out = tmp_path / "tokenizer.json"
    out.write_text(json.dumps(tokenizer_json))
    tok = tokenizers_lib.Tokenizer.from_file(str(out))
    assert tok.token_to_id("<|im_end|>") == specials["<|im_end|>"]
    assert tok.token_to_id("<|pad|>") == specials["<|pad|>"]
    text = "Hello, world! <|im_start|>system\nspeak<|im_end|>"
    enc = tok.encode(text)
    assert tok.decode(enc.ids, skip_special_tokens=False) == text
