#!/usr/bin/env python3
"""Refuse to install a decode/prefill pair whose shared scratch arena disagrees.

A batched-prefill ELF built with `--decode-meta` carries no weight blobs of its own
(`meta.json`'s `weights_from` names the decode build's `buffers/` dir instead) and shares one
`FusedArena` with decode at runtime -- so every weight/cache buffer must sit at the identical
scratch offset in both. `rust/npu-engine/src/llm/artifact.rs`'s `check_shared_layout_agrees`
enforces this AT LOAD, refusing to bind the pair; this script mirrors that same comparison
(shared buffer name + either side typed `scratch` -> offsets must match) so a mismatched pair
never reaches the load path. `weights_from` is exactly what `arena_shared` gates on
(`designs/decode_fused/gen_llm_prefill.py`: `"arena_shared": bool(dec_meta_path)`); null/absent
means the ELF is not arena-shared and there is nothing here to check.

  check_prefill_arena_pairing.py <decode_dir> <prefill_dir>

Exit 0: not arena-shared, or shared and every offset agrees. Exit 1: disagreement or a shared
pair with a missing meta.json. Exit 2: usage error.
"""
import json
import os
import sys


def loc(entry):
    return (entry.get("type"), entry.get("offset"), entry.get("len"))


def span(entry):
    return f"{entry.get('type')}[{entry['offset']}, {entry['offset'] + entry['len']})"


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    decode_dir, prefill_dir = argv[1], argv[2]
    prefill_meta_path = os.path.join(prefill_dir, "meta.json")

    if not os.path.isfile(prefill_meta_path):
        print(f"no meta.json at {prefill_meta_path} -- cannot tell whether {prefill_dir} "
              "is arena-shared", file=sys.stderr)
        return 1
    prefill_meta = json.load(open(prefill_meta_path))

    weights_from = prefill_meta.get("weights_from")
    if not weights_from:
        print(f"{prefill_dir}: not arena-shared (no weights_from) -- nothing to check")
        return 0

    decode_meta_path = os.path.join(decode_dir, "meta.json")
    if not os.path.isfile(decode_meta_path):
        print(f"{prefill_dir} declares weights_from={weights_from} (arena-shared) but its decode "
              f"pair has no meta.json at {decode_meta_path}", file=sys.stderr)
        return 1
    decode_meta = json.load(open(decode_meta_path))

    dlayout, players = decode_meta.get("layout", {}), prefill_meta.get("layout", {})
    shared = sorted(n for n in dlayout if n in players
                     and (dlayout[n].get("type") == "scratch" or players[n].get("type") == "scratch"))
    for name in shared:
        a, b = dlayout[name], players[name]
        if loc(a) != loc(b):
            print(f"""shared-arena layout mismatch on `{name}`:
  {decode_dir} places it at {span(a)}
  {prefill_dir} places it at {span(b)}
-- the two ELFs cannot share one arena; rebuild prefill against THIS decode build:
   DECODE_META={decode_meta_path} scripts/build_prefill.sh""", file=sys.stderr)
            return 1

    print(f"shared-arena layout agrees: {decode_dir} <-> {prefill_dir} "
          f"({len(shared)} shared scratch buffer(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
