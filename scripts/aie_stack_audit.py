#!/usr/bin/env python3
"""Every built AIE core ELF: deepest call-chain stack use vs its `stack_size` reservation.

WHY. `aie.core`'s stack_size defaults to 0x400, and the generated linker script places the next
objectFIFO buffer at the FIRST BYTE past the reservation with zero clearance -- so an oversized
frame overwrites it silently. Measured 2026-09-02 on relpos: deepest chain 0x480 vs 0x400,
kpv0_cons_buff_0 at 0x70400, and the whole encoder returned NaN. Both sides of that inequality are
printed in artifacts we already build, so this is a static read, not a debugging campaign.

Run it after any kernel build:  scripts/aie_stack_audit.py

CAVEAT -- it UNDER-REPORTS. `.LBB*` labels are attributed as if they were separate functions rather
than folded into their parent, so a chain through a labelled block is undercounted: it scored relpos
at 1024 where hand-reading the disassembly gave 1088-1152. Treat "deepest" as a FLOOR. A design
inside 25% of its reservation is reported `tight` for that reason and wants a hand check, not a
reflex bump -- on dwconv1d, raising the reservation changed the encoder's bit-exact output with NO
accuracy change (identical rel-L2 to three figures), i.e. it perturbed rather than fixed."""
import re, subprocess, sys, pathlib, collections
PEANO = "llvm-objdump"

def frames_and_edges(elf):
    dis = subprocess.run([PEANO,"-d",elf],capture_output=True,text=True).stdout
    cur=None; frames={}; edges=collections.defaultdict(set); addr_of={}
    for l in dis.splitlines():
        m=re.match(r'^([0-9a-f]+) <(.+)>:', l)
        if m: cur=m.group(2); addr_of[int(m.group(1),16)]=cur; continue
        if cur is None: continue
        f=re.search(r'padd(?:a|xm)\s+\[sp\], #(-?0x[0-9a-fA-F]+|-?\d+)', l)
        if f:
            v=int(f.group(1),0)
            if v>0: frames[cur]=max(frames.get(cur,0),v)
        for j in re.finditer(r'\bjl\s+#?(0x[0-9a-fA-F]+)', l):
            edges[cur].add(int(j.group(1),0))
    return frames, edges, addr_of

def deepest(frames, edges, addr_of):
    # labels (.LBBn_m) belong to the enclosing function; fold their edges upward by address order
    def resolve(t): return addr_of.get(t)
    best=0
    def walk(fn, acc, seen):
        nonlocal best
        acc += frames.get(fn,0)
        best=max(best,acc)
        if fn in seen: return
        for t in edges.get(fn,()):
            n=resolve(t)
            if n: walk(n, acc, seen|{fn})
    roots=[f for f in list(frames)+list(edges) if f.startswith(('_main_init','main','core_'))]
    for r in set(roots) or set(frames): walk(r,0,frozenset())
    # labels are reached only via their parent's fallthrough; approximate by also walking them
    for f in list(edges):
        if f.startswith('.LBB'): walk(f,0,frozenset())
    return best

rows=[]
for prj in pathlib.Path(".").glob("mlir-aie/programming_examples/**/*.mlir.prj"):
    ld = sorted(prj.glob("ldScripts_*.ld.script"))
    elfs = sorted(prj.glob("elfs_*/*.elf"))
    if not ld or not elfs: continue
    txt = ld[0].read_text()
    m = re.search(r'_sp_start_value_DM_stack = \.;\s*\n\. \+= (0x[0-9a-fA-F]+);', txt)
    if not m: continue
    res = int(m.group(1),16)
    fr,ed,ao = frames_and_edges(str(elfs[0]))
    need = deepest(fr,ed,ao)
    rows.append((prj.parent.parent.name+"/"+prj.stem, need, res, len(elfs)))
print(f"{'design':52s} {'deepest':>8s} {'reserved':>9s}  verdict")
for n,need,res,k in sorted(rows, key=lambda r:-(r[1]/max(r[2],1))):
    v = "OVERFLOW" if need>res else ("tight (<25% margin)" if need > res*0.75 else "ok")
    print(f"{n[:52]:52s} {need:8d} {res:9d}  {v}")
