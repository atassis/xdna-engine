#!/usr/bin/env python3
# Census of a SHIPPED full-ELF artifact: where its bytes go, and how much of the control stream is
# the same bytes repeated. Device-free; reads only the .elf.
#
# This is the post-link half of scripts/dispatch_control_census.py, which counts ops in the emitted
# MLIR. What ships is a TXN transaction binary per runtime sequence, and the two disagree in ways
# the MLIR cannot show: on the 48-layer Gemma-4 decode ELF (29,772,848 B) 21,206,408 B is the ONE
# `main` sequence, 95.2% of it byte-identical across the 48 layers, and a further 8,091,264 B of
# `.pdi` holds 336,768 B of distinct images.
#
# Opcode encoding follows mlir-aie include/aie/Runtime/TxnEncoding.h. UPDATE_REG (3 words) and
# CREATE_SCRATCHPAD (4 words) carry no size field -- their word[1] is a payload, so a generic
# "size lives at word[1]" decoder walks off the rails at the first one and silently swallows the
# rest of the stream as a single op.
#
#   python3 scripts/elf_control_census.py <artifact.elf> [...]
#   python3 scripts/elf_control_census.py --segments <artifact.elf>   # per-configure breakdown
import argparse
import collections
import hashlib
import os
import re
import struct
import subprocess
import sys

OPC = {0: "WRITE", 1: "BLOCKWRITE", 2: "BLOCKSET", 3: "MASKWRITE", 4: "MASKPOLL", 5: "NOOP",
       6: "PREEMPT", 7: "MASKPOLL_BUSY", 8: "LOADPDI", 9: "LOAD_PM_START",
       10: "CREATE_SCRATCHPAD", 11: "UPDATE_STATE_TABLE", 12: "UPDATE_REG", 13: "UPDATE_SCRATCH",
       14: "CONFIG_SHIMDMA_BD", 15: "CONFIG_SHIMDMA_DMABUF_BD", 128: "TCT", 129: "DDR_PATCH",
       130: "READ_REGS", 131: "RECORD_TIMER", 132: "MERGE_SYNC", 200: "LOAD_PM_END_INTERNAL"}

# Opcodes whose length is fixed by the emitter rather than carried in the instruction.
FIXED_WORDS = {0: 6, 3: 7, 6: 1, 8: 4, 10: 4, 12: 3}
SIZE_AT_WORD3 = {1}  # BLOCKWRITE: 4-word header + payload, byte size at word[3]

SEC_RE = re.compile(r"\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)")


def sections(path):
    out = subprocess.run(["readelf", "-S", "-W", path], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        m = SEC_RE.match(line)
        if m:
            _, name, _, addr, off, size = m.groups()
            res.append(dict(name=name, addr=int(addr, 16), off=int(off, 16), size=int(size, 16)))
    return res


def read_sec(path, sec):
    with open(path, "rb") as f:
        f.seek(sec["off"])
        return f.read(sec["size"])


def parse(buf):
    """Yield (byte_offset, raw_word0, opname, byte_size, words) for each TXN op after the header."""
    w = struct.unpack("<%dI" % (len(buf) // 4), buf[:len(buf) // 4 * 4])
    i, n = 4, len(w)
    while i < n:
        raw = w[i]
        op = raw & 0xFF
        if op in FIXED_WORDS:
            sz = FIXED_WORDS[op] * 4
        elif op in SIZE_AT_WORD3:
            sz = w[i + 3]
        else:
            sz = w[i + 1]
        if sz <= 0 or sz % 4 or (i * 4 + sz) > n * 4:
            raise ValueError("undecodable op %s (0x%08x) at byte %d" % (OPC.get(op, op), raw, i * 4))
        yield (i * 4, raw, OPC.get(op, "UNK_%d" % op), sz, w[i:i + sz // 4])
        i += sz // 4


def header(buf):
    w = struct.unpack("<4I", buf[:16])
    return dict(devgen=(w[0] >> 16) & 0xFF, rows=(w[0] >> 24) & 0xFF, cols=w[1] & 0xFF,
                numops=w[2], txnsize=w[3])


def segments(ops):
    """Split into configure-delimited segments. One LOADPDI starts each aiex.configure region."""
    segs, cur = [], None
    for o in ops:
        if o[2] == "LOADPDI":
            if cur is not None:
                segs.append(cur)
            cur = []
        if cur is None:
            cur = []
        cur.append(o)
    segs.append(cur)
    return segs


def census(path, show_segments=False):
    secs = sections(path)
    ctrl = [s for s in secs if s["name"].startswith(".ctrltext")]
    pdi = [s for s in secs if s["name"].startswith(".pdi")]
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        uniq = {}
        for s in pdi:
            f.seek(s["off"])
            uniq[hashlib.md5(f.read(s["size"])).hexdigest()] = s["size"]
    ctrl_b = sum(s["size"] for s in ctrl)
    pdi_b = sum(s["size"] for s in pdi)
    print("%s  %d B" % (path, total))
    print("  .ctrltext  %12d  %5.1f%%   %d sequences" % (ctrl_b, 100 * ctrl_b / total, len(ctrl)))
    print("  .pdi       %12d  %5.1f%%   %d sections, %d distinct (%d B)"
          % (pdi_b, 100 * pdi_b / total, len(pdi), len(uniq), sum(uniq.values())))

    big = max(ctrl, key=lambda s: s["size"])
    buf = read_sec(path, big)
    ops = list(parse(buf))
    hdr = header(buf)
    print("  largest sequence %s: %d B, %d txn ops" % (big["name"], big["size"], hdr["numops"]))
    by = collections.Counter()
    cnt = collections.Counter()
    for _, _, name, sz, _ in ops:
        by[name] += sz
        cnt[name] += 1
    for k in sorted(by, key=lambda x: -by[x]):
        print("      %-20s n=%8d  %12d B  %5.2f%%" % (k, cnt[k], by[k], 100 * by[k] / big["size"]))

    segs = segments(ops)
    words = struct.unpack("<%dI" % (len(buf) // 4), buf)
    byshape = collections.defaultdict(list)
    for sg in segs:
        nb = sum(o[3] for o in sg)
        sig = hashlib.md5(str([(o[2], o[3]) for o in sg]).encode()).hexdigest()
        byshape[sig].append((sg[0][0] // 4, nb // 4))
    tmpl = var = redundant = 0
    for _, insts in byshape.items():
        n, nw = len(insts), insts[0][1]
        tmpl += nw
        if n == 1:
            continue
        v = sum(1 for j in range(nw)
                if any(words[s + j] != words[insts[0][0] + j] for s, _ in insts[1:]))
        var += n * v
        redundant += (n - 1) * (nw - v)
    print("  %d configures (+1 prologue) in %d distinct shapes;  template %d B + varying %d B"
          " + REDUNDANT %d B (%.1f%%)"
          % (cnt["LOADPDI"], len(byshape), tmpl * 4, var * 4, redundant * 4,
             400.0 * redundant / len(buf)))

    cfg = run = ntct = 0
    for sg in segs:
        k = next((i for i, o in enumerate(sg) if o[2] == "DDR_PATCH"), len(sg))
        cfg += sum(o[3] for o in sg[:k])
        run += sum(o[3] for o in sg[k:])
        ntct += sum(1 for o in sg[k:] if o[2] == "TCT")
    print("  cost model: %d configures x %.0f B + %d syncs x %.0f B  =  %d B"
          % (len(segs), cfg / max(len(segs), 1), ntct, run / max(ntct, 1), cfg + run))
    print("  device-resident: instruction BO = %d B (this section alone); .pdi reaching the device"
          " = the ids its relocations name, not the %d B stored" % (big["size"], pdi_b))
    if show_segments:
        for sig, insts in sorted(byshape.items(), key=lambda kv: -len(kv[1]) * kv[1][0][1]):
            print("      shape %s  x%-5d %8d B each  %10d B total"
                  % (sig[:8], len(insts), insts[0][1] * 4, len(insts) * insts[0][1] * 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("elf", nargs="+")
    ap.add_argument("--segments", action="store_true", help="list configure shapes and their counts")
    args = ap.parse_args()
    for p in args.elf:
        census(p, args.segments)


if __name__ == "__main__":
    sys.exit(main())
