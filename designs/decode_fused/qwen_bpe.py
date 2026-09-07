#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal byte-level BPE encoder for Qwen3, from tokenizer.json alone.

Exists because the iron venv has neither `transformers` nor `tokenizers`, and a perplexity gate
needs real tokenized text rather than the 8 ids the oracle happens to carry. Byte-level BPE is
small enough to implement exactly; the point is that it is CHECKABLE -- self_test() re-derives
the oracle's own prompt_ids, so a wrong tokenizer fails loudly here instead of quietly inflating
a perplexity number that would then be compared against another arm's equally wrong one.

Encode only. Decoding is not needed by any caller and is not implemented.
"""
import functools
import json
import os
try:
    # `regex`, not `re`: the pretokenizer pattern uses \p{L}/\p{N}, which `re` cannot parse. Not
    # in .venv-iron; install it somewhere off to the side rather than into the shared venv:
    #   python -m pip install --target /tmp/pylibs regex   # then PYTHONPATH=/tmp/pylibs
    import regex
except ModuleNotFoundError as e:  # pragma: no cover - environment, not logic
    raise SystemExit(
        "qwen_bpe needs the `regex` module (the pretokenizer pattern uses \\p{L}/\\p{N}, which "
        "the stdlib `re` cannot parse). Install it to a scratch dir and put that on PYTHONPATH:\n"
        "  python -m pip install --target /tmp/pylibs regex && export PYTHONPATH=/tmp/pylibs"
    ) from e


@functools.lru_cache(maxsize=1)
def _byte_encoder():
    """GPT-2's byte->unicode map: every byte gets a printable codepoint so BPE runs over text."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class QwenBPE:
    def __init__(self, tokenizer_json):
        spec = json.load(open(tokenizer_json))
        self.vocab = spec["model"]["vocab"]
        merges = spec["model"]["merges"]
        # tokenizer.json stores merges either as "a b" strings or as ["a", "b"] pairs.
        pairs = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)) for m in merges]
        self.ranks = {p: i for i, p in enumerate(pairs)}
        pat = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pat = regex.compile(pat)
        self.b2u = _byte_encoder()
        # Added/special tokens are matched verbatim BEFORE the pretokenizer regex, exactly as
        # the fast tokenizer does; without this, <|endoftext|> would be split into punctuation.
        self.specials = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        self._special_re = (regex.compile("|".join(regex.escape(s) for s in
                                                   sorted(self.specials, key=len, reverse=True)))
                            if self.specials else None)

    def _bpe(self, token):
        word = list(token)
        if len(word) < 2:
            return word
        while True:
            best, bi = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                return word
            word[bi:bi + 2] = [word[bi] + word[bi + 1]]
            if len(word) == 1:
                return word

    def encode(self, text):
        ids = []
        chunks = ([text] if self._special_re is None
                  else [c for c in self._special_re.split(text) if c is not None])
        # split() drops the delimiters, so re-walk with finditer to keep specials in order.
        pos, parts = 0, []
        if self._special_re is not None:
            for m in self._special_re.finditer(text):
                if m.start() > pos:
                    parts.append((text[pos:m.start()], False))
                parts.append((m.group(0), True))
                pos = m.end()
        if pos < len(text):
            parts.append((text[pos:], False))
        if not parts:
            parts = [(text, False)]
        for chunk, is_special in parts:
            if is_special:
                ids.append(self.specials[chunk])
                continue
            for piece in self.pat.findall(chunk):
                s = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                for sub in self._bpe(s):
                    ids.append(self.vocab[sub])
        return ids


def self_test(tokenizer_json, ref_json):
    """Re-derive a known-good tokenization. Raises if the encoder is wrong."""
    ref = json.load(open(ref_json))
    got = QwenBPE(tokenizer_json).encode(ref["prompt"])
    if got != ref["prompt_ids"]:
        raise SystemExit(f"tokenizer self-test FAILED on {ref['prompt']!r}: "
                         f"got {got}, oracle has {ref['prompt_ids']}")
    return got


if __name__ == "__main__":
    import sys
    tj, rj = sys.argv[1], sys.argv[2]
    ids = self_test(tj, rj)
    ref = json.load(open(rj))
    print(f"self-test PASS: {ref['prompt']!r} -> {ids}")
