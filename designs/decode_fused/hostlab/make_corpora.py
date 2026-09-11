import os as _os
# Work root: corpora/ and runs/ live here, NOT in the repo -- a 6000-position logprob
# memmap is 3.6 GB and belongs on nvme. Override with QLAB_WORK.
QLAB = _os.environ.get("QLAB_WORK", "/mnt/data/xdna/qlab")
#!/usr/bin/env python3
"""Build the corpus set for the weight-format quality lab. Four genres, so a delta
that is a property of ONE genre cannot hide as a property of the format."""
import os, re, glob

OUT = QLAB + "/corpora"
# repo root, four levels up from designs/decode_fused/hostlab/
REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


def strip_gutenberg(t):
    t = t.replace("\r\n", "\n")
    m = re.search(r"\*\*\* START OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* END OF TH[EIS]+ PROJECT GUTENBERG EBOOK", t)
    if m:
        t = t[:m.start()]
    return t.strip()


def write(name, text):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(text)
    print(f"{name:28s} {len(text):>9,d} chars")


# 1. natural prose -- narrative fiction, four public-domain books.
prose = []
for gid in ("1342", "84", "2701", "1661"):
    with open(f"{OUT}/gb-{gid}.txt", encoding="utf-8", errors="replace") as f:
        prose.append(strip_gutenberg(f.read()))
write("natural-prose.txt", "\n\n".join(prose))

# 2. encyclopedic prose -- wikitext-2 raw test, the standard perplexity corpus.
with open(f"{OUT}/wikitext2-test.txt", encoding="utf-8") as f:
    write("wikitext2.txt", f.read().strip())

# 3. the RECORDED corpus -- this project's own public docs, in the recorded order.
#    Pinned here so the arm that reproduces the device result is reproducible.
docs = ["docs/data-movement-thesis.md", "docs/where-time-goes.md",
        "docs/benchmark-methodology.md"]
write("project-docs.txt",
      "".join(open(os.path.join(REPO, d)).read() for d in docs))

# 4. code -- a fourth genre, from a tree the model never saw as prose.
src = []
for pat in ("rust/npu-engine/src/**/*.rs",
            "designs/**/*.py"):
    for p in sorted(glob.glob(os.path.join(REPO, pat), recursive=True))[:40]:
        try:
            src.append(open(p, encoding="utf-8").read())
        except Exception:
            pass
write("code.txt", "\n\n".join(src)[:1_200_000])
