"""Pack the control codes of several builds of one image into one full ELF:
`fwd_pack.py <out> <base build> <part build>...`.

A forward ladder is too large to lower as one design under the build memory cap (aiecc's peak is
about 75x the MLIR text), so its rungs are built as separate designs of the same image, each with
`emit=` naming its sequences, and packed here: one PDI, every build's instances, then
`aiebu-asm -t aie2_config` (aiecc's own last step). The builds must share the image: their
gen_args.txt agree except for emit=, and their CDOs (cdo_main) are byte-identical."""
import filecmp
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_paths
sys.path.insert(0, str(rf_paths.REPO / "designs" / "decode_fused"))  # elf_zst.write_elf: the repo's
                                                                      # <name>.elf.zst convention
from elf_zst import write_elf

BUILD = os.environ.get("RF_BUILD", str(rf_paths.BUILD_ROOT))


def gen_args(b):
    return [a for a in open(f"{BUILD}/{b}/gen_args.txt").read().split() if not a.startswith("emit=")]


def commit(d):
    """d's git HEAD, or None if d is unset/not a checkout (e.g. an unrecorded IRON pin)."""
    if not d:
        return None
    r = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def provenance():
    """The source this pack ran from: this file's own repo, IRON (PYTHONPATH's first entry,
    per env.sh), and the toolchain instance (RF_INST) -- what recipes/rf48C.sh now pins."""
    return {"private_commit": commit(os.path.dirname(os.path.abspath(__file__))),
            "iron_commit": commit((os.environ.get("PYTHONPATH", "") or "").split(":")[0]),
            "instance": os.path.basename(os.environ.get("RF_INST", "")) or None}


def main():
    out, builds = sys.argv[1], sys.argv[2:]
    base = builds[0]
    args0 = gen_args(base)
    cdo0 = f"{BUILD}/{base}/cdo_main"
    bdir0 = f"{BUILD}/{base}"
    cfg = json.load(open(f"{bdir0}/full_elf_config.json"))
    kern = cfg["xrt-kernels"][0]
    # aiecc writes these relative to the build dir they're written into
    # (so aiebu-asm can resolve them against the JSON's own directory); this
    # pack merges several builds' JSON into one at `od`, so absolutize each
    # against the build dir it came from -- an already-absolute path (older
    # builds) is left unchanged by os.path.join.
    for pdi in kern.get("PDIs", []):
        pdi["PDI_file"] = os.path.join(bdir0, pdi["PDI_file"])
    inst = {i["id"]: os.path.join(bdir0, i["TXN_ctrl_code_file"]) for i in kern["instance"]}
    src = {k: base for k in inst}
    for b in builds[1:]:
        bdir = f"{BUILD}/{b}"
        assert gen_args(b) == args0, (b, gen_args(b), args0)
        cmp = filecmp.dircmp(cdo0, f"{bdir}/cdo_main")
        _, mism, errs = filecmp.cmpfiles(cdo0, f"{bdir}/cdo_main", os.listdir(cdo0), shallow=False)
        assert not mism and not errs and not cmp.left_only and not cmp.right_only, (b, mism, errs)
        k = json.load(open(f"{bdir}/full_elf_config.json"))["xrt-kernels"][0]
        assert k["arguments"] == kern["arguments"], b
        for i in k["instance"]:
            txn = os.path.join(bdir, i["TXN_ctrl_code_file"])
            if i["id"] in inst:
                assert i["id"] == "boot" or filecmp.cmp(txn, inst[i["id"]], shallow=False), \
                    (b, i["id"], "differs from", src[i["id"]])
                continue
            inst[i["id"]] = txn
            src[i["id"]] = b
        for f in ("params.txt",):
            assert filecmp.cmp(f"{BUILD}/{b}/{f}", f"{BUILD}/{base}/{f}", shallow=False), (b, f)
    od = f"{BUILD}/{out}"
    os.makedirs(od, exist_ok=True)
    kern["instance"] = [{"TXN_ctrl_code_file": v, "id": k} for k, v in inst.items()]
    json.dump(cfg, open(f"{od}/full_elf_config.json", "w"), indent=2)
    subprocess.run(["aiebu-asm", "-t", "aie2_config", "-j", f"{od}/full_elf_config.json", "-o", f"{od}/design.elf"], check=True)
    for f in ("params.txt", "fwd_layout.json", "gen_env.txt"):
        if os.path.exists(f"{BUILD}/{base}/{f}"):
            shutil.copy(f"{BUILD}/{base}/{f}", f"{od}/{f}")
    open(f"{od}/gen_args.txt", "w").write(" ".join(args0) + "\n")
    sizes = {k: os.path.getsize(v) for k, v in inst.items()}
    elf_bytes = open(f"{od}/design.elf", "rb").read()
    elf_prov = write_elf(f"{od}/design.elf", elf_bytes)   # writes design.elf.zst only
    json.dump({"sources": src, "control_code_bytes": sizes, "total_control_code_bytes": sum(sizes.values()),
               "elf_bytes": len(elf_bytes), "elf": elf_prov, "provenance": provenance()},
              open(f"{od}/pack.json", "w"), indent=1)
    print(f"{out}: {len(inst)} sequences, control code {sum(sizes.values()) / 1e6:.2f} MB, "
          f"ELF {len(elf_bytes) / 1e6:.2f} MB, sha256 {elf_prov['sha256']}")


if __name__ == "__main__":
    main()
