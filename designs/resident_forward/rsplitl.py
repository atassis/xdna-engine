"""The global layer's decode (P = 1) with the key range split across the 8 columns, in the layer image
(rlayer_design `split=<nb>`, sequence g1s{nb}). After the head pass and the cache write:

  Q    each column sends its two q heads' row to Qall in DDR (the row-2 -> MemTile links are full,
       K056); every column reads all 16 back: one 16-row pass holds the 16 heads.
  A    column c attends to blocks [c nbc, (c+1) nbc) of the window (nbc = nb / 8, segments of rseg's
       64); its x V workers export raw f32 O and the record tail [l][masks][m] to `partials`.
  M    the softmax cores of columns 0 and 1 merge worker 0's and worker 1's dims of the 8 columns'
       records (tails first, then O units) and export the merged, normalised O to `merged`.
  R    each column reloads its two heads' merged rows (one unit per head and worker), writes their O
       A blocks with rf_pv_finish_a and exports them as the unsplit layer does.

The DDR areas follow the window in %kvr (bf16 elements from SCR0 = nb * 64 * KVROW, or past the
whole cache when SCR_ROWS is set, so every rung of a ladder shares one scratch):
Qall [half 2][head 16][256], partials [column][worker][record WREC B], merged [worker][16384 B, the
worker's O tile layout]; a record is [tail unit][O 16384 B] padded to 11 units."""
import os
import re
import rattn2_design as A2
import rattnh as AH
import rseg_design as RS
import rglobal as G

UNIT = A2.UNIT
KVROW = G.KVROW
WREC = 11 * UNIT                     # tail unit + 10 units of O (16384 B)
CTB = 2 * WREC                       # a column's two workers
TAIL_OFF = 2048 + 64                 # l in the P record (cl after P, l after corr)
QP_B = 8192 + A2.AB                  # two 8640-B Q units read 8192 apart
QALL_E = 2 * 16 * 256
OB = 16384                           # an x V worker's f32 O
SCR_E = QALL_E + 8 * CTB // 2 + OB
GEO = (1, 8, 1, 4, 2, 2, 5, 3, 2048 + 128 + 64 + 64, 1)   # H S NP hb NSK NSV SU PU REC SPL
SCR_ROWS = None                      # rlayer_design `gcap=`: the cache rows the scratch follows


def scr0(nb):
    assert SCR_ROWS is None or nb * 64 <= SCR_ROWS, (nb, SCR_ROWS)
    return (SCR_ROWS or nb * 64) * KVROW


def qall(nb):
    return scr0(nb)


def partials(nb, c):
    return scr0(nb) + QALL_E + c * CTB // 2


def merged(nb):
    return scr0(nb) + QALL_E + 8 * CTB // 2


def kvr_elems(nb):
    return scr0(nb) + SCR_E


def steps(L):
    """The global rows with the split attention row (one pass, P = 1)."""
    g = G.steps(L)
    return [(L.ATTNP,) + GEO if s_[0] == L.ATTNP else s_ for s_ in g]


def widths_x(L, c):
    """%x byte offset of column c's widths record (in the global RoPE region, past the one-block
    table the head pass reads)."""
    return 16 * L.PCAP_T * L.D * 2 + 8192 + c * A2.WB


class Layer:
    """attn_tasks' cfg for one column's share: one pass, Q from DDR, the CT export path."""
    np, nsk, nsv, koff, q_from_host, oa = 1, 2, 2, 0, True, False

    def __init__(self, L, nb):
        self.L, self.nb, self.nbc = L, nb, nb // 8

    def suffix(self, c, role):
        return self.L.LAYER.suffix(c, role)

    def tile(self, c, role):
        return self.L.LAYER.tile(c, role)

    def pkt(self, c, role):
        return self.L.LAYER.pkt(c, role)

    def mem_size(self, nm):
        return {"CT": CTB, "QP": QP_B}.get(nm) or self.L.mem_size(nm)

    def kv(self):
        return f"%kvr : memref<{kvr_elems(self.nb)}xbf16>"

    def shim_tasks(self, nt, c):
        L, KV = self.L, self.kv()
        X = f"%x : memref<{L.xbuf()}xi8>"
        nb = A2.NBW
        assert nb <= 64 and nb <= 256, nb
        win = f"sizes = [{nb}, 8, 64, 64] strides = [{64 * KVROW}, 64, {KVROW}, 1]"
        rep = f" {{repeat_count = {nb - 1} : i32}}"
        off = (c * self.nbc + self.koff) * 64 * KVROW
        return [("sk", 0, f"aie.dma_bd({KV} offset = {off} len = {8 * 64 * 64} {win})", rep),
                ("sw", 1, f"aie.dma_bd({X} offset = {widths_x(L, c)} len = {A2.WB})", ""),
                ("sq", 1, f"aie.dma_bd({KV} offset = {qall(self.nb)} len = {QALL_E})", ""),
                ("sv", 1, f"aie.dma_bd({KV} offset = {off + 512} len = {8 * 64 * 64} {win})", rep)]

    def q_send(self, nt, c):
        return (f"aie.dma_bd(%QP_{c} : memref<{QP_B}xi8> offset = 0 len = {2 * A2.AB} "
                f"sizes = [2, 3, {A2.AB // 3}] strides = [8192, {A2.AB // 3}, 1]) {{bd_id = 12 : i32}}", f"qf_{c}", "")

    def out_bd(self, nt, c):
        return f"aie.dma_bd({self.kv()} offset = {partials(self.nb, c)} len = {CTB // 2})"


def export(a, bd):
    """rsplit's out path for the raw O: a worker sends [tail][O] (ooo ids 44 + h, 32 + h), the MemTile
    takes the column's four packets into CT, then sends CT out in one piece."""
    nm = a[0].rsplit("_", 1)[0]
    if nm in ("ego0", "ego1"):
        h = int(nm[-1])
        t = a[1][2:]
        a[4] = [f"aie.use_lock(%chc_{t}, AcquireGreaterEqual, %one)",
                f"aie.dma_bd(%pc_{t} : memref<{AH.REC_MAX}xi8> offset = {TAIL_OFF} len = 192) {{bd_id = {bd} : i32, out_of_order_id = {44 + h} : i32}}",
                f"aie.use_lock(%cht_{t}, Release, %one)",
                "aie.next_bd ^b1", "^b1:",
                f"aie.use_lock(%cht_{t}, AcquireGreaterEqual, %one)",
                f"aie.dma_bd(%o_{t} : memref<{AH.O_E}xf32> offset = 0 len = {AH.O_E}) {{bd_id = {bd + 1} : i32, out_of_order_id = {32 + h} : i32}}",
                f"aie.use_lock(%chp_{t}, Release, %one)"]
        a[5] = re.sub(r"repeat_count = \d+", "repeat_count = 0", a[5])
    elif nm == "ctin":
        c = a[1].rsplit("_", 1)[1]
        # out of order: a packet's id picks the BD of that id
        parts = [(32 + h, h * WREC + UNIT, AH.O_E * 4) for h in range(2)] + [(44 + h, h * WREC, 192) for h in range(2)]
        a[4] = RS.A2_chain([[f"aie.use_lock(%cte_{c}, AcquireGreaterEqual, %one)",
                             f"aie.dma_bd(%CT_{c} : memref<{CTB}xi8> offset = {o} len = {ln}) {{bd_id = {i} : i32}}",
                             f"aie.use_lock(%ctf_{c}, Release, %one)"] for i, o, ln in parts])
        a[5] = re.sub(r"repeat_count = \d+", "repeat_count = 3", a[5])       # out of order: a packet an execution
    elif nm == "ctout":
        c = a[1].rsplit("_", 1)[1]
        a[4] = ["%four = arith.constant 4 : i32", f"aie.use_lock(%ctf_{c}, AcquireGreaterEqual, %four)",
                f"aie.dma_bd(%CT_{c} : memref<{CTB}xi8> offset = 0 len = {CTB} sizes = [16, {CTB // 16}] strides = [{CTB // 16}, 1]) {{bd_id = 41 : i32}}",
                f"aie.use_lock(%cte_{c}, Release, %four)"]
        a[5] = re.sub(r" \{repeat_count = \d+ : i32\}", "", a[5])
    return a


UPTO = os.environ.get("RF_SPLIT_UPTO", "")      # bisection: end the command after phase q, a, m or r


def attention(L, m, nt, I, task, locked, chain, issued, freed, nb):
    """rlayer_design.attention_tasks for g1s{nb}; True when the command ends here (RF_SPLIT_UPTO)."""
    assert nt == 1 and nb % 16 == 0
    NC, cfg = L.NC, Layer(L, nb)
    KV = cfg.kv()
    # (the cache write is awaited by the caller) Q to DDR: each column's heads 2c, 2c+1, row 0 of both halves
    for c in range(NC):
        QK_ = f"%QK_{c} : memref<{L.QKB}xi8>"
        task(f"qx{c}", f"mem_{c}", "MM2S", 5, [f"aie.dma_bd({QK_} offset = 0 len = 2048 sizes = [2, 2, 512] strides = [16384, 8192, 1]) {{bd_id = 30 : i32}}"])
        task(f"qxs{c}", f"shim_{c}", "S2MM", 0,
             [f"aie.dma_bd({KV} offset = {qall(nb) + 2 * c * 256} len = 1024 sizes = [2, 2, 256] strides = [256, 4096, 1])"],
             " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%qxs{c})", I)
    if UPTO == "q":
        return True
    for name, tile in issued:
        if tile.startswith("mem_") and name not in freed:
            m(f"aiex.dma_free_task(%{name})", I)
            freed.add(name)
    for c in range(NC):
        for nm, v in (("we", 1), ("qe", 1), ("qf", 0), ("kf", 0), ("ke", 2), ("v0f", 0), ("v0e", 2), ("v1f", 0), ("v1e", 2),
                      ("sf", 0), ("se", 2), ("paf", 0), ("pae", 2), ("pbf", 0), ("pbe", 2), ("wf", 0), ("ctf", 0), ("cte", 4),
                      ("oaf", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    # A: the split attention, each column over its nbc blocks in segments
    nbc = nb // 8
    base = lambda name: name.rsplit("_", 1)[0]

    def cap(c, nbw, koff):
        out = []
        A2.NBW, cfg.koff = nbw, koff
        A2.attn_tasks(None, 1, c, lambda *a, **k: out.append((a, k)), RS.A2_locked, RS.A2_chain, cfg)
        return out

    def emit(a, k, bd=5):
        a = export(list(a), bd)
        task(*a, **k)
    for c in range(NC):
        for a, k in cap(c, 2, 0):
            if base(a[0]) not in RS.PER_BLOCK:
                emit(a, k)
    for si, (koff, g_, np_) in enumerate(RS.segments(nbc, 1)):
        names = []
        for c in range(NC):
            for a, k in cap(c, g_, koff):
                if base(a[0]) in RS.PER_BLOCK:
                    a = list(a)
                    a[0] = f"{a[0]}_s{si}"
                    if base(base(a[0])) in RS.LAST:
                        a[5] = a[5].replace(" {repeat_count", " {issue_token = true, repeat_count")
                    task(*a, **k)
                    names.append(a[0])
        for nm in names:
            if base(base(nm)) in RS.LAST:
                m(f"aiex.dma_await_task(%{nm})", I)
        for nm in names:
            if base(base(nm)) not in RS.LAST:
                m(f"aiex.dma_free_task(%{nm})", I)
            freed.add(nm)
    A2.NBW = L.NBW_S
    for c in range(NC):
        m(f"aiex.dma_await_task(%so_{c})", I)
    if UPTO == "a":
        return True
    for name, tile in issued:
        if not tile.startswith("shim_") and name not in freed:
            m(f"aiex.dma_free_task(%{name})", I)
            freed.add(name)
    # M: columns 0 and 1's softmax cores (row 3) merge worker h = column; each gets its worker's 8
    # tails, then the 8 columns' O units, one unit a slot of a 2-slot ring on MemTile MM2S3
    for h in range(2):
        CT_ = f"%CT_{h} : memref<{CTB}xi8>"
        t = L.core_name(h, 3)
        for nm, v in (("ke", 2), ("kf", 0), ("cte", 1), ("ctf", 0)):
            m(f"aiex.set_lock(%{nm}_{h}, {v})", I)
        m(f"aiex.set_lock(%mgp_{t}, 1)", I)
        m(f"aiex.set_lock(%mgc_{t}, 0)", I)
        p0 = partials(nb, 0) + h * WREC // 2
        task(f"mgs{h}", f"shim_{h}", "MM2S", 0,
             chain([[f"aie.dma_bd({KV} offset = {p0} len = {8 * UNIT // 2} sizes = [8, {UNIT // 2}] strides = [{CTB // 2}, 1])"],
                    [f"aie.dma_bd({KV} offset = {p0 + UNIT // 2} len = {80 * UNIT // 2} sizes = [8, 10, {UNIT // 2}] strides = [{CTB // 2}, {UNIT // 2}, 1])"]]))
        task(f"mgi{h}", f"mem_{h}", "S2MM", 0, chain([locked(f"ke_{h}", f"aie.dma_bd({CT_} offset = {i * UNIT} len = {UNIT}) {{bd_id = {i} : i32}}", f"kf_{h}")
                                                     for i in range(2)]), " {repeat_count = 43 : i32}")
        task(f"mgo{h}", f"mem_{h}", "MM2S", 3, chain([locked(f"kf_{h}", f"aie.dma_bd({CT_} offset = {i * UNIT} len = {UNIT}) {{bd_id = {34 + i} : i32}}", f"ke_{h}")
                                                     for i in range(2)]), " {repeat_count = 43 : i32}")
        task(f"mge{h}", f"t_{t}", "MM2S", 0, locked(f"mgc_{t}", f"aie.dma_bd(%acc_{t} : memref<{AH.O_E}xf32> offset = 0 len = {AH.O_E}) {{bd_id = 8 : i32}}", f"mgp_{t}"),
             "", L.LAYER.pkt(h, "sm"))
        task(f"mgl{h}", f"mem_{h}", "S2MM", 4, locked(f"cte_{h}", f"aie.dma_bd({CT_} offset = {2 * UNIT} len = {OB}) {{bd_id = 2 : i32}}", f"ctf_{h}"))
        task(f"mgx{h}", f"mem_{h}", "MM2S", 5, locked(f"ctf_{h}", f"aie.dma_bd({CT_} offset = {2 * UNIT} len = {OB} sizes = [16, {OB // 16}] strides = [{OB // 16}, 1]) {{bd_id = 41 : i32}}", f"cte_{h}"))
        task(f"mgw{h}", f"shim_{h}", "S2MM", 0, [f"aie.dma_bd({KV} offset = {merged(nb) + h * OB // 2} len = {OB // 2})"], " {issue_token = true}")
    for h in range(2):
        m(f"aiex.dma_await_task(%mgw{h})", I)
    if UPTO == "m":
        return True
    for h in range(2):
        for nm in ("mgs", "mgi", "mgo", "mge", "mgl", "mgx"):
            m(f"aiex.dma_free_task(%{nm}{h})", I)
            freed.add(f"{nm}{h}")
    # R: every column reloads its two heads' merged rows (row 2c + hh of the worker's tile layout)
    for c in range(NC):
        for nm, v in (("ke", 2), ("kf", 0), ("v0e", 2), ("v0f", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for c in range(NC):
        CTc = f"%CT_{c} : memref<{CTB}xi8>"
        for h, (e, f, ids_in, ids_out, ch_out) in enumerate((("ke", "kf", (0, 1), (34, 35), 1), ("v0e", "v0f", (26, 27), (8, 9), 2))):
            o0 = merged(nb) + h * OB // 2
            rows = []
            for hh in range(2):
                R_ = 2 * c + hh
                off = o0 + 2 * ((R_ // 8) * 512 + (R_ % 8) * 8)
                rows += [[f"aie.dma_bd({KV} offset = {off} len = 512 sizes = [4, 8, 16] strides = [2048, 128, 1])"],
                         [f"aie.dma_bd({KV} offset = {o0} len = {(UNIT - 1024) // 2})"]]
            task(f"rls{c}_{h}", f"shim_{c}", "MM2S", h, chain(rows))
            task(f"rli{c}_{h}", f"mem_{c}", "S2MM", h, chain([locked(f"{e}_{c}", f"aie.dma_bd({CTc} offset = {(2 * h + i) * UNIT} len = {UNIT}) {{bd_id = {ids_in[i]} : i32}}", f"{f}_{c}")
                                                            for i in range(2)]))
            task(f"rlo{c}_{h}", f"mem_{c}", "MM2S", ch_out, chain([locked(f"{f}_{c}", f"aie.dma_bd({CTc} offset = {(2 * h + i) * UNIT} len = {UNIT}) {{bd_id = {ids_out[i]} : i32}}", f"{e}_{c}")
                                                                 for i in range(2)]))
    # the A blocks out as the unsplit global layer's (2 passes = the 2 heads, OA, the token)
    lay = G.Layer(L.LAYER, L)
    for c in range(NC):
        out = []
        A2.NBW = 2
        A2.attn_tasks(None, 1, c, lambda *a, **k: out.append((a, k)), RS.A2_locked, RS.A2_chain, lay)
        for a, k in out:
            if base(a[0]) in ("ego0", "ego1", "ctin", "tk", "so"):
                a = list(a)
                a[0] = "r" + a[0]
                task(*a, **k)
    A2.NBW = L.NBW_S
    for c in range(NC):
        m(f"aiex.dma_await_task(%rso_{c})", I)
    return UPTO == "r"
