"""rattn2's attention columns with the key window issued in segments, past the one-task limits
(K054: 64 blocks in the shim window BD's iteration; 256 block reads a task by the repeat field).

Per pass the window is NB blocks. The per-block tasks (shim K/V windows, MemTile rings, core S^T and
P egress) are issued per segment of at most G blocks: start, await the x V sends (the last tasks of
a segment's dataflow), free the rest, reuse their BD ids. Windows of at most G blocks run
unsegmented, several passes a task, as rattn2 does. The cores read the block count from RTP word 1
and carry the softmax state across segments; a segment is invisible to them.

Sequence w<NB>n<nt>, args as rattn2 except %kv: one K window [NKMAX][256], then one V window,
shared by every column (column c's own Q and widths in %x)."""
import os
import sys
import rattn2_design as A2
from m1_design import M

G = 64                      # blocks per segment: the shim window BD's iteration size (K054)
NKMAX = 262144              # keys in %kv
TCT_ID = 26                 # the MemTile's controller_id (AIETargetModel::getTileToControllerIdMap, row 1)
NO_TCT = os.environ.get("RSEG_NO_TCT") == "1"   # negative control: no MemTile TCT route
BCAST = os.environ.get("RSEG_BCAST") == "1"     # one K and one V read per segment, broadcast to all MemTiles
KS, VS, VIN_BD = 3, 4, 16                        # K and V source shims; V's MemTile S2MM2 BDs (even bank)
PUSH = os.environ.get("RSEG_PUSH") == "1"       # later segments: queue pushes on the configured BDs, no BD rewrite
SENT = 999983               # rattn2 NBW while emitting cores: its last-block constant becomes RTP-derived


def args_sig():
    return (f"%x : memref<{A2.NC * A2.XCOL}xi8>, %kv : memref<{2 * NKMAX * A2.HD}xbf16>, "
            f"%o : memref<{A2.NC * A2.NTMAX * A2.QPASS}xi8>")


class SegBody(A2.Body):
    def c(self, v, I):
        return "%cnbwm1" if v == SENT - 1 else super().c(v, I)


def body(m, t, role):
    """rattn2's core program with the block count from RTP word 1."""
    b = SegBody(m, t)
    m(f"%core_{t} = aie.core(%{t}) {{")
    I = 6
    for v in (0, 1, 2):
        m(f"%c{v} = arith.constant {v} : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%go_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    m(f"%nb32 = memref.load %rtp_{t}[%c1] : memref<4xi32>", I)
    m("%cnbw = arith.index_cast %nb32 : i32 to index", I)
    m("%cnbwm1 = arith.subi %cnbw, %c1 : index", I)
    A2.attn_section(m, b, t, role, I)
    I -= 2
    m("}", I)
    m("aie.end", I)


class SegCfg(A2.Standalone):
    """The shared K/V window, from block koff on (per segment)."""
    koff = 0

    def shim_tasks(self, nt, c):
        X, KV = f"%x : memref<{A2.NC * A2.XCOL}xi8>", f"%kv : memref<{2 * NKMAX * A2.HD}xbf16>"
        nb = A2.NBW
        assert nb <= 64 and nb * nt <= 256, (nb, nt)          # K054, the repeat field
        win = f"sizes = [{nb}, 4, {A2.KEYS}, {A2.SW}] strides = [{A2.KEYS * A2.HD}, {A2.SW}, {A2.HD}, 1]"
        rep = f" {{repeat_count = {nb * nt - 1} : i32}}"
        off = self.koff * A2.KEYS * A2.HD
        return [("sk", 0, f"aie.dma_bd({KV} offset = {off} len = {4 * A2.KEYS * A2.SW} {win})", rep),
                ("sw", 1, f"aie.dma_bd({X} offset = {c * A2.XCOL} len = {A2.WB})", ""),
                ("sq", 1, f"aie.dma_bd({X} offset = {c * A2.XCOL + A2.WB} len = {nt * A2.QPASS})", ""),
                ("sv", 1, f"aie.dma_bd({KV} offset = {NKMAX * A2.HD + off} len = {4 * A2.KEYS * A2.SW} {win})", rep)]


CFG = SegCfg()
PER_BLOCK = {"sk", "sv", "kin", "vin", "sin", "pin", "kout", "vout0", "vout1", "sout", "egs", "egp"}
LAST = ("vout0", "vout1")   # a segment is done when both x V sends are


def segments(nb, nt):
    """(first block, blocks, passes) per segment, in issue order."""
    if nb <= G:
        k = max(1, min(nt, 256 // nb))
        while k > 1 and (nb * k) % 2:
            k -= 1
        out, p = [], 0
        while p < nt:
            n = min(k, nt - p)
            out.append((0, nb, n))
            p += n
        return out
    assert nb % G == 0, nb
    return [(s * G, G, 1) for _ in range(nt) for s in range(nb // G)]


def issue(m, I, nb, passes, task, capture, cols=range(A2.NC), xform=None):
    """The attention's tasks with the window issued in segments. capture(passes, c, nb, koff) returns
    rattn2.attn_tasks' task calls for column c over `passes` passes of an nb-block window whose first
    block is koff; the per-dispatch ones are taken from capture(passes, c, 2, 0). xform(a, c) may
    rewrite a per-block task's arguments or drop it (None)."""
    base = lambda name: name.rsplit("_", 1)[0]
    for c in cols:                             # the per-dispatch tasks (widths, Q, O)
        for a, k in capture(passes, c, 2, 0):
            if base(a[0]) not in PER_BLOCK:
                task(*a, **k)
    first = {}                                 # (column, task) -> the segment whose BDs a push reuses
    for si, (koff, g, np_) in enumerate(segments(nb, passes)):
        names, syncs = [], []
        for c in cols:
            for a, k in capture(np_, c, g, koff):
                if base(a[0]) in PER_BLOCK:
                    a = list(a)
                    if xform and (a := xform(a, c)) is None:
                        continue
                    last = base(a[0]) in LAST
                    key = (c, base(a[0]))
                    if PUSH and not a[1].startswith("shim") and key in first and first[key] == (g, np_):
                        syncs += push(m, I, a, last)       # same BDs as before: queue push only
                        continue
                    first.setdefault(key, (g, np_))
                    a[0] = f"{a[0]}_s{si}"
                    if last:
                        a[5] = a[5].replace(" {repeat_count", " {issue_token = true, repeat_count")
                    task(*a, **k)
                    names.append(a[0])
        for nm in names:
            if base(base(nm)) in LAST:
                m(f"aiex.dma_await_task(%{nm})", I)
        for sy in syncs:
            m(sy, I)
        for nm in names:
            if base(base(nm)) not in LAST:
                m(f"aiex.dma_free_task(%{nm})", I)


_PQ = [0]


def push(m, I, a, token):
    """A task-queue push of task a's first BD with its repeat count (its BDs are still configured from
    an earlier segment); returns the sync that awaits it when token."""
    import re
    name, tile, d, ch, body = a[:5]
    if tile.startswith("t_"):             # the layer image's core tiles, t_<col>_<row>
        col, row = (int(v) for v in tile[2:].split("_"))
    else:
        col = int(tile.rsplit("_", 1)[1])
        row = 1 if tile.startswith("mem") else {v: k for k, v in A2.ROLE.items()}[tile.rsplit("_", 1)[0]]
    bd = int(re.search(r"bd_id = (\d+)", " ".join(body)).group(1))
    rep = int(re.search(r"repeat_count = (\d+)", a[5]).group(1)) if len(a) > 5 and "repeat_count" in a[5] else 0
    _PQ[0] += 1
    n = _PQ[0]
    m(f"%pqb{n} = arith.constant {bd} : i32", I)
    m(f"%pqr{n} = arith.constant {rep} : i32", I)
    m(f"aiex.npu.push_queue ({col}, {row}, {d}:{ch}) bd_id %pqb{n} repeat %pqr{n} {{issue_token = {str(token).lower()}}} : i32, i32", I)
    if not token:
        return []
    v = [f"%pqs{n}_{i}" for i in range(4)]
    for vi, x in zip(v, (col, row, 1 if d == "MM2S" else 0, ch)):
        m(f"{vi} = arith.constant {x} : i32", I)
    m(f"%pqo{n} = arith.constant 1 : i32", I)
    return [f"aiex.npu.sync({v[0]}, {v[1]}, {v[2]}, {v[3]}, %pqo{n}, %pqo{n}) : i32, i32, i32, i32, i32, i32"]


def capture(nt, c, nb, koff):
    """rattn2.attn_tasks' tasks for column c at window nb, nt passes, as (name, args) in its order."""
    out = []
    A2.NBW, CFG.koff = nb, koff
    A2.attn_tasks(None, nt, c, lambda *a, **k: out.append((a, k)), A2_locked, A2_chain, CFG)
    return out


def A2_locked(acq, bd, rel, va="%one", vr="%one"):
    return ([f"aie.use_lock(%{acq}, AcquireGreaterEqual, {va})"] if acq else []) + [bd] + \
           ([f"aie.use_lock(%{rel}, Release, {vr})"] if rel else [])


def A2_chain(parts):
    out = []
    for i, p in enumerate(parts):
        if i:
            out += [f"aie.next_bd ^b{i}", f"^b{i}:"]
        out += p
    return out


def sequence(m, nb, nt):
    I = 6
    assert nt <= A2.NTMAX and nb * A2.KEYS <= NKMAX and nb % 2 == 0
    m(f"aie.runtime_sequence @w{nb}n{nt}({args_sig()}) {{")
    m(f"%ntv = arith.constant {nt} : i32", I)
    m(f"%nbv = arith.constant {nb} : i32", I)
    for c in range(A2.NC):
        for role in A2.ROLE.values():
            m(f"aiex.npu.rtp_write(@rtp_{role}_{c}, 0, %ntv) : i32", I)
            m(f"aiex.npu.rtp_write(@rtp_{role}_{c}, 1, %nbv) : i32", I)
    for c in range(A2.NC):
        for nm, v in (("qe", 1), ("we", 1), ("qf", 0), ("kf", 0), ("ke", 2), ("v0f", 0), ("v0e", 2), ("v1f", 0), ("v1e", 2),
                      ("sf", 0), ("se", 2), ("paf", 0), ("pae", 2), ("pbf", 0), ("pbe", 2), ("wf", 0), ("ctf", 0), ("cte", 2)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
        for role in A2.ROLE.values():
            m(f"aiex.set_lock(%go_{role}_{c}, 1)", I)

    def emit_task(name, tile, d, ch, body, attrs="", pkt=""):
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}{pkt}) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m("%two = arith.constant 2 : i32", I + 2)
        for line in body:
            m(line, I + 2)
        m("aie.end", I + 2)
        m("}" + attrs, I)
        m(f"aiex.dma_start_task(%{name})", I)

    issue(m, I, nb, nt, emit_task, capture, xform=bcast_xform if BCAST else None)
    for c in range(A2.NC):
        m(f"aiex.dma_await_task(%so_{c})", I)
    m("}", 4)


def bcast_xform(a, c):
    """One shim read per segment: K from shim KS MM2S0, V from shim VS MM2S0, both broadcast to every
    MemTile; V lands on MemTile S2MM2 (free), so its BDs move to the even bank."""
    nm = a[0].rsplit("_", 2)[0]
    if nm in ("sk", "sv"):
        if c:
            return None
        a[1], a[3] = f"shim_{KS if nm == 'sk' else VS}", 0
    elif nm == "vin":
        a[3] = 2
        a[4] = [ln.replace(f"bd_id = {26 + i} :", f"bd_id = {VIN_BD + i} :") for ln in a[4] for i in [
            next((i for i in range(4) if f"bd_id = {26 + i} :" in ln), 0)]]
    return a


def bcast_flows(m):
    for c in range(A2.NC):
        m(f"aie.flow(%shim_{KS}, DMA : 0, %mem_{c}, DMA : 0)")
        m(f"aie.flow(%shim_{VS}, DMA : 0, %mem_{c}, DMA : 2)")


def tct_flow(m, mem, shim):
    """The MemTile's task-complete tokens to the controller, so a segment can await a MemTile task:
    aiecc routes only the shim's (route-shim-to-tct=shim-only, K057)."""
    m(f"aie.packet_flow({TCT_ID}) {{")
    m(f"aie.packet_source<{mem}, TileControl : 0>", 6)
    m(f"aie.packet_dest<{shim}, South : 0>", 6)
    m("} {ctrl_pkt_flow = true, keep_pkt_header = true}")


def emit(rungs):
    A2.body = body
    m = M()
    m("module {", 0)
    m("aie.device(npu2) @main {", 2)
    for line in A2.attn_decls():
        m(line)
    A2.NBW = SENT
    for c in range(A2.NC):
        A2.column(m, c)
        if BCAST:
            m.lines.remove(f"    aie.flow(%shim_{c}, DMA : 0, %mem_{c}, DMA : 0)")
        if not NO_TCT:
            tct_flow(m, f"%mem_{c}", f"%shim_{c}")
    if BCAST:
        bcast_flows(m)
    m(f"aie.runtime_sequence @boot({args_sig()}) {{")
    m("aiex.npu.load_pdi {device_ref = @main}", 6)
    m("}", 4)
    for nb, nt in rungs:
        sequence(m, nb, nt)
    m("}", 2)
    m("}", 0)
    return "\n".join(m.lines) + "\n"


def kernels():
    return A2.kernels()


def build_text(args):
    """args: nb:nt rungs, e.g. 64:1 4096:1"""
    return emit([tuple(int(v) for v in a.split(":")) for a in args])


if __name__ == "__main__":
    sys.stdout.write(build_text(sys.argv[1:]))
