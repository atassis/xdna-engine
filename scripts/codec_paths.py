#!/usr/bin/env python3
"""Where the S2 checkpoint lives, without hardcoding anyone's directory layout.

The codec verify scripts all need the S2-Pro GGUF. Naming an absolute path in this repo is wrong
twice over: it only works on one machine, and it links a public file to a local layout, which
`scripts/check_no_private_refs.sh` rejects and the pre-push hook blocks.

Resolution order:
  1. $S2_GGUF                          -- explicit, wins everywhere
  2. <workspace>/s2.cpp/models/...     -- the sibling-checkout layout this was developed in
  3. clear error naming both           -- rather than a stack trace from open()

The model is not redistributable through this repo, so there is no fallback that fetches it.
"""
import os
from pathlib import Path

MODEL_NAME = "s2-pro-q6_k.gguf"
TOKENIZER_NAME = "tokenizer.json"


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _candidates(name):
    """Every plausible location, nearest first.

    Walks ANCESTORS rather than checking only the repo's immediate parent: a git worktree can sit
    arbitrarily deep (an agent worktree lives at <repo>/.claude/worktrees/<id>/, whose parent is
    not the workspace), and the one-level version failed there with a confusing "not found" for a
    model that was present the whole time.
    """
    root = _repo_root()
    out = [root / "models" / name]
    for anc in [root] + list(root.parents)[:6]:
        out.append(anc / "s2.cpp" / "models" / name)
    return out


def _resolve(env_var, name):
    env = os.environ.get(env_var)
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{env_var}={p} does not exist")
        return str(p)
    for c in _candidates(name):
        if c.exists():
            return str(c)
    tried = "\n  ".join(str(c) for c in _candidates(name))
    raise FileNotFoundError(
        f"could not find {name}. Set {env_var}=/path/to/{name}, or place it at one of:\n  {tried}")


def gguf():
    """Path to the S2-Pro GGUF. Override with $S2_GGUF."""
    return _resolve("S2_GGUF", MODEL_NAME)


def tokenizer():
    """Path to tokenizer.json. Override with $S2_TOKENIZER."""
    return _resolve("S2_TOKENIZER", TOKENIZER_NAME)


if __name__ == "__main__":
    for fn in (gguf, tokenizer):
        try:
            print(f"{fn.__name__:10s} {fn()}")
        except FileNotFoundError as e:
            print(f"{fn.__name__:10s} NOT FOUND -- {e}")
