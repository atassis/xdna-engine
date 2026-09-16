#!/usr/bin/env python3
"""PDI duplication census for a fused decode ELF, and the minimal-PDI re-assembly.

aiecc's `makeFullElfConfigJson` (mlir-aie src/tools/aiecc/SidecarFiles.h) hands EVERY
xrt-kernel the whole `allPdis` array, "so aiebu-asm can resolve any load_pdi reference".
A fused decode emits one xrt-kernel per design plus `main`, so N designs cost N copies of
all N PDIs: on the 48-layer Gemma-4-12B arm, 624 `.pdi.*` sections holding 26 distinct
PDIs, 8091264 B for 337136 B of content.

Only `main` issues load_pdi -- every per-design device has zero `aiex.configure`. So a
kernel needs its own PDI (to stay individually dispatchable) and nothing else, which the
`--rewrite` mode emits. MEASURED at 48 layers: 29772848 -> 22308760 B (-25.1%), every
`.ctrltext.*` byte-identical and the distinct PDI contents unchanged.

  census:  decode_elf_pdi_census.py <decode.elf> [<other.elf> ...]
  rewrite: decode_elf_pdi_census.py --rewrite <full_elf_config.json> <out.json>
           then: aiebu-asm -t aie2_config -j <out.json> -o <out.elf>

`aiebu-asm` does NOT validate PDI references -- a config naming no PDIs at all still
assembles -- so a rewrite is gated by comparing sections against the baseline ELF, never
by the assembler exiting 0.
"""
import collections, hashlib, json, os, re, subprocess, sys

SEC = re.compile(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)")


def sections(path):
    """[(name, sha256, size)] in section-header order."""
    blob = open(path, "rb").read()
    out = subprocess.run(["readelf", "-S", "-W", path], capture_output=True, text=True).stdout
    got = []
    for line in out.splitlines():
        m = SEC.match(line)
        if m:
            off, size = int(m.group(2), 16), int(m.group(3), 16)
            got.append((m.group(1), hashlib.sha256(blob[off:off + size]).hexdigest(), size))
    return got


def census(path):
    secs = sections(path)
    pdi = [s for s in secs if s[0].startswith(".pdi.")]
    ctrl = [s for s in secs if "ctrl" in s[0]]
    uniq = {s[1]: s[2] for s in pdi}
    copies = collections.Counter(s[0] for s in pdi)
    print(f"{os.path.basename(path)}  {os.path.getsize(path)} B")
    print(f"  pdi        {sum(s[2] for s in pdi):10d} B  {len(pdi):4d} sections, "
          f"{len(uniq)} distinct ({sum(uniq.values())} B), copies={sorted(set(copies.values()))}")
    print(f"  duplicated {sum(s[2] for s in pdi) - sum(uniq.values()):10d} B")
    print(f"  control    {sum(s[2] for s in ctrl):10d} B  {len(ctrl):4d} sections")
    return secs


def rewrite(cfg_path, out_path):
    cfg = json.load(open(cfg_path))
    for k in cfg["xrt-kernels"]:
        if k["name"] == "main":
            continue
        own = k["name"] + ".pdi"
        k["PDIs"] = [p for p in k["PDIs"] if os.path.basename(p["PDI_file"]) == own]
    json.dump(cfg, open(out_path, "w"), indent=1)
    kept = [(k["name"], len(k["PDIs"])) for k in cfg["xrt-kernels"]]
    print(f"{out_path}: {len(kept)} kernels, PDI refs "
          f"{sum(n for _, n in kept)} (was {len(kept) * max(n for _, n in kept)})")


def compare(a, b):
    """Every control section identical and every distinct PDI present is the gate."""
    def group(secs, pred):
        d = collections.defaultdict(set)
        for n, h, _ in secs:
            if pred(n):
                d[n].add(h)
        return d
    ca, cb = group(a, lambda n: "ctrl" in n), group(b, lambda n: "ctrl" in n)
    pa, pb = group(a, lambda n: n.startswith(".pdi.")), group(b, lambda n: n.startswith(".pdi."))
    bad = [n for n in ca if ca[n] != cb.get(n)]
    print(f"\ncontrol sections differing: {bad or 'NONE'}")
    print(f"pdi distinct content identical: {pa == pb}")
    return not bad and pa == pb


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rewrite"]:
        rewrite(sys.argv[2], sys.argv[3])
    else:
        got = [census(p) for p in sys.argv[1:]]
        if len(got) == 2:
            sys.exit(0 if compare(*got) else 1)
