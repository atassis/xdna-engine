#!/usr/bin/env python3
"""check_selfdescribing_indexes.py -- a doc that counts what it indexes must agree with it.

Three files here describe the tree by number: ARCHITECTURE.md counts workspace members and crate
rows, scripts/README.md counts scripts. All were written by hand, so all drift the moment the thing
they count changes, and nothing said so. Measured 2026-09-05: the crate count was 15 against 18 real
members. Measured again 2026-09-08 while fixing that: the count had since been corrected, but two
crates still had no table row, and scripts/README.md said 194 against 315 files. The task recording
the drift quoted 307, which had ALSO drifted by the time it was read. A number nobody can check is a
number that is wrong later.

So: derive, never edit. Prints the true values; exits non-zero on disagreement.

Usage: check_selfdescribing_indexes.py [--fix-counts]
  --fix-counts rewrites the bare COUNTS in place. It does NOT fill in table rows: a missing crate
  needs a human-written responsibility line, and inventing one is worse than the gap.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cargo_members():
    s = (REPO / "rust" / "Cargo.toml").read_text()
    mem = re.findall(r'"([^"]+)"', re.search(r"members\s*=\s*\[(.*?)\]", s, re.S).group(1))
    dm = re.search(r"default-members\s*=\s*\[(.*?)\]", s, re.S)
    default = re.findall(r'"([^"]+)"', dm.group(1)) if dm else []
    return mem, default


def script_entries():
    d = REPO / "scripts"
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "README.md")


def main():
    fix = "--fix-counts" in sys.argv
    issues, edits = [], []

    mem, default = cargo_members()
    arch_path = REPO / "ARCHITECTURE.md"
    arch = arch_path.read_text()

    m = re.search(r"(\d+) workspace members", arch)
    if not m:
        issues.append("ARCHITECTURE.md: no 'N workspace members' sentence to check")
    elif int(m.group(1)) != len(mem):
        issues.append(f"ARCHITECTURE.md says {m.group(1)} workspace members; rust/Cargo.toml has {len(mem)}")
        edits.append((arch_path, m.group(0), f"{len(mem)} workspace members"))

    m = re.search(r"builds the (\d+)", arch)
    if m and int(m.group(1)) != len(default):
        issues.append(f"ARCHITECTURE.md says it builds {m.group(1)} default-members; there are {len(default)}")
        edits.append((arch_path, m.group(0), f"builds the {len(default)}"))

    rows = set(re.findall(r"^\|\s*`?(npu-[a-z0-9-]+)`?\s*\|", arch, re.M))
    missing = sorted(set(mem) - rows)
    extra = sorted(rows - set(mem))
    if missing:
        issues.append(f"ARCHITECTURE.md crate table has no row for: {', '.join(missing)}")
    if extra:
        issues.append(f"ARCHITECTURE.md crate table has rows for non-members: {', '.join(extra)}")

    sr_path = REPO / "scripts" / "README.md"
    sr = sr_path.read_text()
    entries = script_entries()
    m = re.search(r"^(\d+) entries", sr, re.M)
    if not m:
        issues.append("scripts/README.md: no 'N entries' line to check")
    elif int(m.group(1)) != len(entries):
        issues.append(f"scripts/README.md says {m.group(1)} entries; scripts/ holds {len(entries)}")
        edits.append((sr_path, m.group(0), f"{len(entries)} entries"))

    # Coverage is REPORTED, never failed: the table is a curated index, and the honest state is
    # "documents M of N", not a demand that every script get a row the moment it lands.
    documented = len(re.findall(r"^\|\s*`", sr, re.M))
    print(f"workspace members {len(mem)} | default-members {len(default)} | crate rows {len(rows)}")
    print(f"scripts/ entries {len(entries)} | README table rows {documented} "
          f"({documented * 100 // max(len(entries), 1)}% documented)")

    if fix and edits:
        for path, old, new in edits:
            t = path.read_text()
            if old in t:
                path.write_text(t.replace(old, new, 1))
                print(f"fixed {path.name}: {old!r} -> {new!r}")
        print("counts rewritten; table rows are NOT auto-filled -- write those by hand")
        return 0

    if issues:
        print("\nDISAGREEMENT:")
        for i in issues:
            print(f"  {i}")
        print("\nRe-run with --fix-counts to rewrite the counts (tables stay manual).")
        return 1
    print("\nOK -- every self-describing count matches what it describes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
