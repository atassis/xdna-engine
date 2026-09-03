#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-token LPDDR byte map of a fused-decode artifact, grouped by role.

First-order: bytes RESIDENT in the scratch arena and touched per dispatch, read off the
generator's own meta.json layout. It does not model on-chip reuse within the dispatch, and
for the self-KV terms it reports the ALLOCATION (S=448), not the position-dependent touch --
both caveats matter when this number becomes a denominator.

Supersedes the hand-derived "~198 MB/token", which is the layer-weight subtotal only.

  python3 scripts/decode_byte_map.py artifacts/fused_decode12/meta.json
"""
import collections
import json
import re
import sys

ROLE_GROUPS = {
    "layer weights": ("Wf1", "Wf2", "Wqkv", "Wso", "Wcq", "Wco",
                      "bias_f1", "bias_qkv", "bias_cq", "bso", "bco", "bf2"),
    "cross-attn K/V": ("Kenc", "Venc"),
    "self KV cache": ("kcache", "vcache"),
    "lm head": ("Wproj",),
}


def main(path):
    meta = json.load(open(path))
    per_role = collections.Counter()
    for name, buf in meta["layout"].items():
        if buf["type"] != "scratch":
            continue
        per_role[re.sub(r"^L\d+_", "", name)] += buf["len"]

    print(f"{'role':24} {'MB/token':>10}")
    for role, nbytes in per_role.most_common():
        print(f"{role:24} {nbytes / 1e6:10.2f}")

    print(f"\n{'group':24} {'MB/token':>10}")
    grouped, claimed = collections.Counter(), set()
    for group, roles in ROLE_GROUPS.items():
        for role in roles:
            if role in per_role:
                grouped[group] += per_role[role]
                claimed.add(role)
    for role in per_role:
        if role not in claimed:
            grouped["ungrouped"] += per_role[role]
    for group, nbytes in grouped.most_common():
        print(f"{group:24} {nbytes / 1e6:10.2f}")

    total = sum(per_role.values())
    print(f"\n{'mapped scratch total':24} {total / 1e6:10.2f} MB/token")
    print(f"{'declared scratch_size':24} {meta['scratch_size'] / 1e6:10.2f} MB")
    print(f"{'npu_logits':24} {str(meta.get('npu_logits')):>10}")
    print(f"\ntransport floor at 52.69 GB/s: {total / 52.69e9 * 1e3:.2f} ms/token")


if __name__ == "__main__":
    main(sys.argv[1])
