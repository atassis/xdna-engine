#!/usr/bin/env python3
"""Convert fishaudio/openaudio-s1-mini's tiktoken vocab into a `tokenizer.json` the engine's
`tokenizers` 0.20 crate can load (`Tokenizer::from_file`, `rust/npu-engine/src/llm/config.rs`).

s1-mini ships `tokenizer.tiktoken` (a bare `<base64 bytes> <rank>` table -- no merges list) and
`special_tokens.json`, never a `tokenizer.json` -- confirmed against the live `fishaudio/s1-mini`
HF repo listing (redirect target of `openaudio-s1-mini`), not just this local snapshot: the
upstream file list is exactly those two files plus config.json/model.pth/codec.pth/README.

Its base BPE table is byte-for-byte identical to the sibling S2-Pro's own `tokenizer.json`
(`s2.cpp/models/tokenizer.json`, itself a verbatim copy of `fishaudio/s2-pro`'s real HF tokenizer
repo, not hand-derived) -- all 151643 ranks, same bytes, same ids. So that file's
normalizer/pre_tokenizer/decoder are reused here as the FORMAT authority (`_S2PRO_SPLIT_PATTERN`
below), while every ID comes from s1-mini's OWN `special_tokens.json`: S2-Pro's ids do not apply
to this checkpoint (its `<|im_end|>` 151645 is s1-mini's `<|pad|>`).

A tiktoken file stores ranks, not merges -- ranks alone don't say which two pieces formed a merged
token. Merges are reconstructed with tiktoken's own published algorithm (`tiktoken/load.py::bpe`,
also `transformers.integrations.tiktoken`'s `TikTokenConverter`): for each token in increasing
rank order, repeatedly merge the lowest-rank adjacent byte pair using only strictly-lower-rank
merges. For a token that came from real BPE training this always bottoms out at exactly its two
immediate constituents. No `tiktoken`/`tokenizers` package is required or used.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import regex

DEFAULT_MODEL_DIR = Path(
    "/mnt/data/cache/huggingface/hub/models--fishaudio--openaudio-s1-mini/snapshots/"
    "f4b445029346701e082b60bb63fcc2d1bb17a0e2"
)
DEFAULT_OUT = Path("artifacts/openaudio-s1-mini/tokenizer/tokenizer.json")

# `s2.cpp/models/tokenizer.json`'s pre_tokenizer `Split` regex, verbatim (a Qwen2-lineage
# GPT-4-style pattern -- note the bare `\p{N}` with no `{1,3}` bound, unlike cl100k_base's own).
# s1-mini ships no pre_tokenizer of its own to read; reused here on the strength of the byte-exact
# base-vocab match above, not independently confirmed for s1-mini specifically -- see
# test_s2_tokenizer_convert.py's docstring for what that leaves unverified.
_S2PRO_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte<->printable-unicode bijection. `tokenizers`' `ByteLevel` pre_tokenizer/decoder
    apply this same mapping, so a BPE vocab string is this mapping applied per raw byte of the
    token -- never the raw bytes themselves."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


def bytes_to_pseudo(b: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[c] for c in b)


def pseudo_to_bytes(s: str) -> bytes:
    return bytes(UNICODE_TO_BYTE[c] for c in s)


def load_tiktoken_ranks(path: Path) -> dict[bytes, int]:
    """Parse a tiktoken `<base64> <rank>` table into `{token_bytes: rank}`."""
    ranks: dict[bytes, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rank = line.split()
            ranks[base64.b64decode(b64)] = int(rank)
    return ranks


def _merge_parts(ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """`tiktoken/load.py::bpe`: repeatedly merge the lowest-rank adjacent byte pair using only
    merges below `max_rank` (i.e. only merges already established at a lower rank than `token`
    itself), until none apply."""
    parts = [bytes([c]) for c in token]
    while True:
        min_idx, min_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (min_rank is None or r < min_rank):
                min_idx, min_rank = i, r
        if min_rank is None or min_rank >= max_rank:
            return parts
        parts[min_idx : min_idx + 2] = [parts[min_idx] + parts[min_idx + 1]]


def reconstruct_bpe(ranks: dict[bytes, int]) -> tuple[dict[str, int], list[list[str]]]:
    """Derive an HF `tokenizers`-format `(vocab, merges)` pair from tiktoken ranks alone. `vocab`
    maps each token's byte-level pseudo-unicode spelling to its rank/id; `merges` is the ordered
    list of pseudo-unicode pairs that produced each multi-byte token, in rank order -- which is
    what makes it a valid priority-ordered BPE merge list, not just a by-product listing.

    Raises if any token doesn't reduce to exactly two constituents, which would mean the rank
    table isn't a real BPE merge sequence.
    """
    by_rank = sorted(ranks.items(), key=lambda kv: kv[1])
    vocab: dict[str, int] = {}
    merges: list[list[str]] = []
    for token, rank in by_rank:
        vocab[bytes_to_pseudo(token)] = rank
        if len(token) == 1:
            continue
        parts = _merge_parts(ranks, token, rank)
        if len(parts) != 2 or parts[0] + parts[1] != token:
            raise ValueError(f"rank {rank} token {token!r} did not reduce to one merge: {parts!r}")
        merges.append([bytes_to_pseudo(parts[0]), bytes_to_pseudo(parts[1])])
    return vocab, merges


def load_special_tokens(path: Path) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


def build_added_tokens(specials: dict[str, int]) -> list[dict]:
    """One `added_tokens` entry per `special_tokens.json` id, marked `special: true` -- every one
    of s1-mini's 4111 (15 control tokens + 4096 `<|semantic:i|>`) is a control/codebook token, none
    a plain-vocabulary word, matching how S2-Pro's own `<|semantic:i|>` entries are marked."""
    return [
        {
            "id": tid,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, tid in sorted(specials.items(), key=lambda kv: kv[1])
    ]


def build_tokenizer_json(model_dir: Path) -> dict:
    ranks = load_tiktoken_ranks(model_dir / "tokenizer.tiktoken")
    vocab, merges = reconstruct_bpe(ranks)
    specials = load_special_tokens(model_dir / "special_tokens.json")
    collision = set(vocab) & set(specials)
    if collision:
        raise ValueError(f"special token(s) collide with the base BPE vocab: {collision}")
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": build_added_tokens(specials),
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": _S2PRO_SPLIT_PATTERN}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }


def convert(model_dir: Path, out_path: Optional[Path] = None) -> dict:
    data = build_tokenizer_json(model_dir)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data))
    return data


class Tokenizer:
    """Encode/decode over a `tokenizer.json` produced by this module, reimplementing the
    normalize -> split -> byte-level -> BPE-merge -> vocab-lookup pipeline `tokenizers::Tokenizer`
    executes, entirely in the vocab/merges/pre_tokenizer/added_tokens the FILE carries -- not off
    the source tiktoken/special_tokens files again. Exists because neither `tokenizers` nor
    `tiktoken` is installed in `.venv-iron` (checked; see test module docstring), so this is how
    the produced file's own content gets exercised rather than merely schema-checked.

    Scope: this is a from-scratch reimplementation, not the Rust `tokenizers` crate, so it does
    not prove bit-for-bit parity with `Tokenizer::from_file` on exotic Unicode -- see the test
    module for what was additionally cross-checked against a real `tokenizers` install.
    """

    def __init__(self, tokenizer_json: dict):
        model = tokenizer_json["model"]
        self.vocab: dict[str, int] = model["vocab"]
        self.id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merge_rank: dict[tuple[str, str], int] = {
            (a, b): i for i, (a, b) in enumerate(model["merges"])
        }
        self.added: dict[str, int] = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
        self.id_to_added: dict[int, str] = {v: k for k, v in self.added.items()}

        split_pattern = next(
            p["pattern"]["Regex"]
            for p in tokenizer_json["pre_tokenizer"]["pretokenizers"]
            if p["type"] == "Split"
        )
        self._split_re = regex.compile(split_pattern)
        self._special_re = (
            re.compile("|".join(re.escape(s) for s in sorted(self.added, key=len, reverse=True)))
            if self.added
            else None
        )

    @classmethod
    def from_file(cls, path: Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text()))

    def _bpe(self, piece: str) -> list[str]:
        parts = list(piece)
        while len(parts) > 1:
            best_idx, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.merge_rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_idx, best_rank = i, r
            if best_idx is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts

    def _encode_ordinary(self, text: str) -> list[int]:
        ids = []
        for chunk in self._split_re.findall(text):
            piece = bytes_to_pseudo(chunk.encode("utf-8"))
            ids.extend(self.vocab[p] for p in self._bpe(piece))
        return ids

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        pos = 0
        if self._special_re is not None:
            for m in self._special_re.finditer(text):
                if m.start() > pos:
                    ids.extend(self._encode_ordinary(text[pos : m.start()]))
                ids.append(self.added[m.group(0)])
                pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_ordinary(text[pos:]))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()

        def flush() -> None:
            if buf:
                out.append(bytes(buf).decode("utf-8"))
                buf.clear()

        for i in ids:
            special = self.id_to_added.get(i)
            if special is not None:
                flush()
                out.append(special)
            else:
                buf.extend(pseudo_to_bytes(self.id_to_token[i]))
        flush()
        return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    convert(args.model_dir, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
