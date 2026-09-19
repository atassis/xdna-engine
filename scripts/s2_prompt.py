#!/usr/bin/env python3
"""Text (+ optional reference audio) -> the `[n_tokens, 11]` int32 `flat_tokens` matrix the S2
Slow AR consumes. Row t = `[semantic_or_text_id, cb0_id, .., cb9_id]` (11 = num_codebooks+1);
codebook columns are 0 outside a reference-code span, since a text/control row has no codebook
meaning (`docs/s2-ar-graph-map.md` section 2, "INPUT").

Ported from `s2.cpp/src/s2_prompt.cpp::build_prompt` -- same three sections (system preamble,
optional reference-code span, user+assistant preamble) in the same order, same literal strings,
same transpose from that file's `[channel, time]` `PromptTensor` layout to `[time, channel]` here.
That file is S2-Pro's; its ids do not apply to s1-mini (see `s2_tokenizer_convert.py`'s docstring
and `PromptIds` below) -- only the STRUCTURE is reused, the ids come from the checkpoint actually
being run.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_tokenizer_convert as stc

# `s2_prompt.cpp`'s `NEWLINE = { 198 }` literal: the plain single-byte BPE token for '\n'. Not
# re-derived per checkpoint -- a byte-level BPE vocab always assigns every raw byte its own base
# token (verified for s1-mini: `test_s2_tokenizer_convert.py::test_tiktoken_ranks_are_dense...`),
# and 198 is that id in both s1-mini's and S2-Pro's vocab (their base BPE tables are byte-for-byte
# identical -- see s2_tokenizer_convert.py).
NEWLINE_TOKEN_ID = 198


@dataclass(frozen=True)
class PromptIds:
    """The special-token ids `build_prompt` needs, each read BY NAME from a checkpoint's own
    `special_tokens.json` via `from_special_tokens` -- never a literal constant. `im_end_id` is
    kept as its own named field rather than folded away because it is the one this project has
    twice hardcoded wrong: S2-Pro's im_end (151645) is s1-mini's `<|pad|>`, a plausible in-range id
    on both checkpoints, so the wrong constant does not fail to load, it produces a model that
    never stops. Construct from the checkpoint actually being run, not copied from a sibling.
    """

    im_start_id: int
    im_end_id: int
    voice_id: int
    semantic_begin_id: int
    num_codebooks: int = 10

    @classmethod
    def from_special_tokens(cls, specials: dict[str, int], num_codebooks: int = 10) -> "PromptIds":
        return cls(
            im_start_id=specials["<|im_start|>"],
            im_end_id=specials["<|im_end|>"],
            voice_id=specials["<|voice|>"],
            semantic_begin_id=specials["<|semantic:0|>"],
            num_codebooks=num_codebooks,
        )


def build_prompt(
    tokenizer: stc.Tokenizer,
    ids: PromptIds,
    text: str,
    prompt_text: str = "",
    prompt_codes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """`flat_tokens[n_tokens, ids.num_codebooks + 1]`, int32.

    `prompt_codes`, when given, is `[num_codebooks, T_prompt]` raw RVQ code indices from the codec
    ENCODER (0..codebook_size-1) for a reference-audio voice profile -- prepended so the model
    conditions on it (`s2-voice-profiles-reference-audio`; the encoder itself is unbuilt, so this
    always runs the no-reference branch until something supplies codes). A reference is used only
    when codes, a positive `T_prompt`, AND non-empty `prompt_text` are all present, matching
    `s2_prompt.cpp`'s `has_reference` exactly.

    cb0's row supplies BOTH the codebook-0 slot (its raw code, column 1) AND, offset by
    `semantic_begin_id`, column 0's vocab id for that span -- `s2_prompt.cpp` double-uses cb0 the
    same way (`row 0 = prompt_codes[0] + sem_begin`, `row 1 (cb0) = prompt_codes[0]` unchanged),
    not a bug introduced here.
    """
    has_reference = prompt_codes is not None and prompt_codes.shape[1] > 0 and prompt_text != ""
    if prompt_codes is not None and prompt_codes.shape[0] != ids.num_codebooks:
        raise ValueError(f"prompt_codes has {prompt_codes.shape[0]} codebooks, expected {ids.num_codebooks}")
    prompt_has_speaker_tag = "<|speaker:" in prompt_text

    sys_pre: list[int] = []
    sys_post: list[int] = []

    if has_reference:
        sys_pre += tokenizer.encode("<|im_start|>system")
        sys_pre.append(NEWLINE_TOKEN_ID)
        sys_pre += tokenizer.encode(
            "convert the provided text to speech reference to the following:\n\nText:\n"
        )
        if not prompt_has_speaker_tag:
            sys_pre += tokenizer.encode("<|speaker:0|>")
        sys_pre += tokenizer.encode(prompt_text)
        sys_pre += tokenizer.encode("\n\nSpeech:\n")

        sys_post.append(ids.im_end_id)
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode("<|im_start|>user")
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode(text)
        sys_post.append(ids.im_end_id)
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode("<|im_start|>assistant")
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post.append(ids.voice_id)
    else:
        sys_post += tokenizer.encode("<|im_start|>system")
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode("convert the provided text to speech")
        sys_post.append(ids.im_end_id)
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode("<|im_start|>user")
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode(text)
        sys_post.append(ids.im_end_id)
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post += tokenizer.encode("<|im_start|>assistant")
        sys_post.append(NEWLINE_TOKEN_ID)
        sys_post.append(ids.voice_id)

    t_prompt = prompt_codes.shape[1] if has_reference else 0
    n_tokens = len(sys_pre) + t_prompt + len(sys_post)
    flat = np.zeros((n_tokens, ids.num_codebooks + 1), dtype=np.int32)

    pos = 0
    if sys_pre:
        flat[pos : pos + len(sys_pre), 0] = sys_pre
    pos += len(sys_pre)

    if has_reference:
        flat[pos : pos + t_prompt, 0] = prompt_codes[0, :] + ids.semantic_begin_id
        flat[pos : pos + t_prompt, 1:] = prompt_codes.T
        pos += t_prompt

    if sys_post:
        flat[pos : pos + len(sys_post), 0] = sys_post

    return flat


def load_tokenizer_and_ids(model_dir: Path, num_codebooks: int = 10) -> tuple[stc.Tokenizer, PromptIds]:
    specials = stc.load_special_tokens(model_dir / "special_tokens.json")
    tokenizer = stc.Tokenizer(stc.build_tokenizer_json(model_dir))
    return tokenizer, PromptIds.from_special_tokens(specials, num_codebooks=num_codebooks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--model-dir", type=Path, default=stc.DEFAULT_MODEL_DIR)
    ap.add_argument("--num-codebooks", type=int, default=10)
    args = ap.parse_args()

    tokenizer, ids = load_tokenizer_and_ids(args.model_dir, args.num_codebooks)
    matrix = build_prompt(tokenizer, ids, args.text)
    print(f"flat_tokens shape={matrix.shape}")
    print(matrix)


if __name__ == "__main__":
    main()
