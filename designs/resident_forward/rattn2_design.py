"""P3.3b step B: the attention columns on the layer image's STATIC core topology.

Every core keeps the gemm phases' two rings: S2MM0 a private 1728-B ring from MemTile MM2S r, S2MM1
the 8640-B broadcast from MemTile MM2S4. Attention streams are cut into those units and assembled
or re-tiled on the core (rf_attn_glue.cc):
  qk  (row 2)  Q on the broadcast (2 units a pass, 16 rows of one head each), K on its private ring
               (19 units a block: [4 slices][64][64] slice-major), S^T out to MemTile S2MM3
  pv0 (row 3), pv1 (row 4)  per block P+cl (3 units) then its V half (10 units) on the private
               ring; O out as one packet a pass to MemTile S2MM5 (out-of-order, one BD per half)
  sm  (row 5)  the widths (2 units, all passes), then S^T (5 units a block) on its private ring;
               P+cl out twice (one copy per x V worker) to MemTile S2MM4
Column-local: column c is kv head c, its pass tb the 16 rows of both of its q heads. K and V come
from a host window here (the cache later), Q and the widths from the host (the head pass later).
Sequence a<nt>, per column: %x = widths (int32 [pass][hi 32, lo 32]) in a 3456-B record, then Q
[pass][2 heads][16][256]; %kv = K then V windows [NBW*64][256]; %o = O [pass][half][slice][32][64].
"""
import os
import sys
import rf_paths
from m1_design import M

NC, ROWS, KEYS, HD, SW = 8, 32, 64, 256, 64
UNIT, AB = 1728, 8640
NTMAX = 7
KU, SU, PU, VU = 19, 5, 3, 10                   # units a block: K, S^T, P+cl, a V half
KB, VB, SB, PCB = 4 * KEYS * SW * 2, 2 * KEYS * SW * 2, KEYS * ROWS * 4, ROWS * 64 * 2 + 2 * ROWS * 4
if os.environ.get("RF_PV_EDGE", "1") != "0":
    PCB += 64                                     # P + cl, then the rows' partial-block mask
PCPV = PCB
WB = 2 * UNIT                                   # the widths record (<= 7 passes x 256 B used)
QPASS = 2 * 16 * HD * 2
XCOL = WB + NTMAX * QPASS
NBW = 8                                          # window blocks (a build parameter)
QK_O, PV_O = GQK, GPV = "attn_qk.o", "attn_pv.o"
FA_SRC = rf_paths.iron_kernel("fused_attn.cc")
GLUE_SRC = rf_paths.iron_kernel("rf_attn_glue.cc")
FA_DEFS = ["-D__AIECC__", "-Dbf16_f32_ONLY", "-DROUND_CONV_EVEN", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16", f"-DFA_ROWS={ROWS}"]
ROLE = {2: "qk", 3: "pv0", 4: "pv1", 5: "sm"}
SELECT_BUF = bool(os.environ.get("RF_SELECT_BUF"))   # ring units: select the buffer, one call site
PV_EDGE = os.environ.get("RF_PV_EDGE", "1") != "0"  # native bf16 x V on each row's partial blocks only (K055); 0: bfp16 only
PV_NATIVE = bool(os.environ.get("RF_PV_NATIVE"))  # x V on the native bf16 mmul (per-key exact; bfp16 shares an exponent over 8 keys)
FINISH_A = False         # x V emits O's bfp16 A block for its K sub-pass (the fused layer) instead of bf16 rows
ABO = 16 * 288 * 9 // 8  # one A block: 16 rows x a 288-wide K sub-pass, bfp16 (o_ref.ABO)
TRACE = []               # (col, row) cores to trace (`trace=c.r,...`)
TRACE_BYTES = int(os.environ.get("RF_TRACE_BYTES", 1 << 22))
TRACE_EVENTS = tuple(os.environ.get("RF_TRACE_EVENTS", "LOCK_STALL,MEMORY_STALL,STREAM_STALL,INSTR_VECTOR,"
                                    "INSTR_LOCK_ACQUIRE_REQ,INSTR_LOCK_RELEASE_REQ,INSTR_EVENT_0,INSTR_EVENT_1").split(","))


def kvcol():
    return 2 * NBW * KEYS * HD


def args_sig():
    return (f"%x : memref<{NC * XCOL}xi8>, %kv : memref<{NC * kvcol()}xbf16>, "
            f"%o : memref<{NC * NTMAX * QPASS}xi8>")     # a trace build appends its buffer as arg 3


def mem_bufs():
    return (("QP", NTMAX * QPASS + 448), ("KR", 2 * KU * UNIT), ("V0R", 2 * VU * UNIT), ("V1R", 2 * VU * UNIT),
            ("SR", 2 * SU * UNIT), ("PA", 2 * PU * UNIT), ("PB", 2 * PU * UNIT), ("W", WB), ("CT", QPASS))


def attn_decls():
    B8, F32, BF = "memref<{}xi8>", "memref<{}xf32>", "memref<{}xbf16>"
    sigs = {
        "fa_zero_sT": ([F32.format(KEYS * ROWS)], QK_O),
        "fa_zero_o": ([F32.format(ROWS * HD // 2), "i32"], PV_O),
        "rf_q_rows": ([B8.format(AB), BF.format(HD * ROWS), "i32"], GQK),
        "rf_k_unit": ([B8.format(UNIT), BF.format(KEYS * SW), BF.format(HD * ROWS), F32.format(KEYS * ROWS), "i32"], GQK),
        "rf_w_unit": ([B8.format(UNIT), B8.format(WB), "i32"], GQK),
        "rf_s_unit": ([B8.format(UNIT), F32.format(KEYS * ROWS), "i32"], GQK),
        "rf_sm_init": ([F32.format(2 * ROWS + 4), B8.format(WB), "i32"], GQK),
        "rf_sm_block": ([F32.format(KEYS * ROWS), B8.format(PCB), F32.format(2 * ROWS + 4), B8.format(WB), "i32"], GQK),
        "rf_pc_unit": ([B8.format(UNIT), B8.format(PCPV), "i32"], GPV),
        "rf_pv_begin": ([F32.format(ROWS * HD // 2), B8.format(PCPV)] + ([BF.format(2 * ROWS * SW)] if PV_EDGE else []), GPV),
        "rf_v_unit": ([B8.format(UNIT), BF.format(KEYS * SW), B8.format(PCPV), F32.format(ROWS * HD // 2)]
                      + ([BF.format(2 * ROWS * SW)] if PV_EDGE else []) + ["i32"], GPV),
        "rf_pv_end": ([B8.format(PCPV), F32.format(ROWS)], GPV),
        "rf_pv_finish": ([F32.format(ROWS * HD // 2), F32.format(ROWS), BF.format(2 * ROWS * SW)], GPV),
        "rf_pv_finish_a": ([F32.format(ROWS * HD // 2), F32.format(ROWS), BF.format(2 * ROWS * SW)], GPV),
    }
    return [f'func.func private @{name}({", ".join(args)}) attributes {{link_with = "{obj}"}}'
            for name, (args, obj) in sigs.items()]


def emit(nts):
    m = M()
    m("module {", 0)
    m("aie.device(npu2) @main {", 2)
    for line in attn_decls():
        m(line)
    for c in range(NC):
        column(m, c)
    for k, (c, r) in enumerate(TRACE):
        m(f"aie.trace @tr_{c}_{r}(%{ROLE[r]}_{c}) {{")
        m('aie.trace.mode "Event-Time"', 6)
        m(f"aie.trace.packet id={24 + k} type=core", 6)   # clear of the ctx packet ids 1-16
        for ev in TRACE_EVENTS:
            m(f'aie.trace.event<"{ev}">', 6)
        m("aie.trace.start broadcast=15", 6)
        m("aie.trace.stop broadcast=14", 6)
        m("}")
    m(f"aie.runtime_sequence @boot({args_sig()}) {{")
    m("aiex.npu.load_pdi {device_ref = @main}", 6)
    m("}", 4)
    for nt in nts:
        sequence(m, nt)
    m("}", 2)
    m("}", 0)
    return "\n".join(m.lines) + "\n"


def pid(c, h):
    return 1 + 2 * c + h


def column(m, c):
    n = lambda s: f"{s}_{c}"
    m(f"%{n('shim')} = aie.tile({c}, 0)")
    m(f"%{n('mem')} = aie.tile({c}, 1)")
    for r, role in ROLE.items():
        m(f"%{n(role)} = aie.tile({c}, {r})")
    m(f"aie.flow(%{n('shim')}, DMA : 0, %{n('mem')}, DMA : 0)")
    m(f"aie.flow(%{n('shim')}, DMA : 1, %{n('mem')}, DMA : 1)")
    for k, role in enumerate(ROLE.values()):
        m(f"aie.flow(%{n('mem')}, DMA : {k}, %{n(role)}, DMA : 0)")
        m(f"aie.flow(%{n('mem')}, DMA : 4, %{n(role)}, DMA : 1)")
    m(f"aie.flow(%{n('qk')}, DMA : 0, %{n('mem')}, DMA : 3)")
    m(f"aie.flow(%{n('sm')}, DMA : 0, %{n('mem')}, DMA : 4)")
    for h in range(2):
        m(f"aie.packet_flow({pid(c, h)}) {{")
        m(f"aie.packet_source<%{n('pv' + str(h))}, DMA : 0>", 6)
        m(f"aie.packet_dest<%{n('mem')}, DMA : 5>", 6)
        m("} {keep_pkt_header = true}")
    m(f"aie.flow(%{n('mem')}, DMA : 5, %{n('shim')}, DMA : 0)")
    for nm, sz in mem_bufs():
        m(f"%{n(nm)} = aie.buffer(%{n('mem')}) {{sym_name = \"{n(nm)}\"}} : memref<{sz}xi8>")
    for nm in ("qe", "qf", "kf", "ke", "v0f", "v0e", "v1f", "v1e", "sf", "se", "paf", "pae", "pbf", "pbe", "we", "wf", "ctf", "cte"):
        m(f"%{n(nm)} = aie.lock(%{n('mem')}) {{init = 0 : i32, sym_name = \"{n(nm)}\"}}")
    for role in ROLE.values():
        core(m, c, role)


def attn_bufs(role):
    """The role's attention buffers and locks (the rings, rtp and go are the caller's)."""
    bufs, locks = [], []
    if role == "qk":
        bufs += [("qT", HD * ROWS, "bf16"), ("sT", KEYS * ROWS, "f32"), ("kbuf", KEYS * SW, "bf16")]
        locks += [("sp", 1), ("sc", 0)]
    elif role == "sm":
        bufs += [("s", KEYS * ROWS, "f32"), ("st", 2 * ROWS + 4, "f32"), ("w", WB, "i8"), ("pc0", PCB, "i8"), ("pc1", PCB, "i8")]
        locks += [("pe", 2), ("pf", 0), ("pm", 0)]
    else:
        bufs += [("o", ROWS * HD // 2, "f32"), ("inv", ROWS, "f32"), ("vbuf", KEYS * SW, "bf16"),
                 ("pc", PCPV, "i8"),
                 ("ch", 2 * ROWS * SW, "bf16")]
        locks += [("chp", 1), ("chc", 0)]
    return bufs, locks


def core(m, c, role):
    t = f"{role}_{c}"
    bufs = [("wb0", UNIT, "i8"), ("wb1", UNIT, "i8"), ("ab0", AB, "i8"), ("ab1", AB, "i8"), ("rtp", 4, "i32")]
    locks = [("cwp", 2), ("cwc", 0), ("cap", 2), ("cac", 0), ("go", 0)]
    ab, al = attn_bufs(role)
    bufs, locks = bufs + ab, locks + al
    for nm, sz, ty in bufs:
        m(f"%{nm}_{t} = aie.buffer(%{t}) {{sym_name = \"{nm}_{t}\"}} : memref<{sz}x{ty}>")
    for nm, init in locks:
        m(f"%{nm}_{t} = aie.lock(%{t}) {{init = {init} : i32, sym_name = \"{nm}_{t}\"}}")
    m(f"%cdma_{t} = aie.mem(%{t}) {{")
    m("%one = arith.constant 1 : i32", 6)
    for q, (ch, buf, ln, acq, rel) in enumerate(((0, "wb", UNIT, "cwp", "cwc"), (1, "ab", AB, "cap", "cac"))):
        end = "^q1" if q == 0 else "^qend"
        m(f"%s{q} = aie.dma_start(S2MM, {ch}, ^q{q}a, {end})", 6)
        for i, lab, nxt in ((0, "a", "b"), (1, "b", "a")):
            m(f"^q{q}{lab}:", 4)
            m(f"aie.use_lock(%{acq}_{t}, AcquireGreaterEqual, %one)", 6)
            m(f"aie.dma_bd(%{buf}{i}_{t} : memref<{ln}xi8> offset = 0 len = {ln})", 6)
            m(f"aie.use_lock(%{rel}_{t}, Release, %one)", 6)
            m(f"aie.next_bd ^q{q}{nxt}", 6)
        m("^q1:" if q == 0 else "^qend:", 4)
    m("aie.end", 6)
    m("}")
    body(m, t, role)
    m(f"}} {{stack_size = {2048 if role == 'qk' else 4096} : i32}}")


class Body:
    """Core-program emitter: constants on demand, ring units by running index."""

    def __init__(self, m, t):
        self.m, self.t, self.k = m, t, 0

    def c(self, v, I):
        self.k += 1
        self.m(f"%atk{self.k} = arith.constant {v} : index", I)
        return f"%atk{self.k}"

    def tmp(self):
        self.k += 1
        return f"%atv{self.k}"

    def unit(self, I, ring, idx, calls):
        """Acquire the next unit of ring 'w' (private) or 'a' (broadcast), run calls(buffer) on the
        buffer of parity idx, release."""
        m, t = self.m, self.t
        acq, rel, buf = ("cwc", "cwp", "wb") if ring == "w" else ("cac", "cap", "ab")
        par, ev = self.tmp(), self.tmp()
        m(f"{par} = arith.remui {idx}, %c2 : index", I)
        m(f"{ev} = arith.cmpi eq, {par}, %c0 : index", I)
        m(f"aie.use_lock(%{acq}_{t}, AcquireGreaterEqual, %one)", I)
        if calls is None:
            m(f"aie.use_lock(%{rel}_{t}, Release, %one)", I)
            return
        if SELECT_BUF:     # one call site on the selected buffer (program memory)
            sel = self.tmp()
            ty = f"memref<{UNIT if ring == 'w' else AB}xi8>"
            m(f"{sel} = arith.select {ev}, %{buf}0_{t}, %{buf}1_{t} : {ty}", I)
            for line in calls(sel):
                m(line, I)
        else:
            m(f"scf.if {ev} {{", I)
            for i in range(2):
                for line in calls(f"%{buf}{i}_{t}"):
                    m(line, I + 2)
                if i == 0:
                    m("} else {", I)
            m("}", I)
        m(f"aie.use_lock(%{rel}_{t}, Release, %one)", I)


def body(m, t, role):
    B8 = lambda n_: f"memref<{n_}xi8>"
    F32 = lambda n_: f"memref<{n_}xf32>"
    BF = lambda n_: f"memref<{n_}xbf16>"
    b = Body(m, t)
    m(f"%core_{t} = aie.core(%{t}) {{")
    I = 6
    for v in (0, 1, 2):
        m(f"%c{v} = arith.constant {v} : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m(f"%cnbw = arith.constant {NBW} : index", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%go_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    attn_section(m, b, t, role, I)
    I -= 2
    m("}", I)
    m("aie.end", I)


def attn_section(m, b, t, role, I):
    """One dispatch's attention on core t (buffers %<name>_t): needs %c0, %c1, %c2, %one, %cnbw and
    %nt in scope."""
    B8 = lambda n_: f"memref<{n_}xi8>"
    F32 = lambda n_: f"memref<{n_}xf32>"
    BF = lambda n_: f"memref<{n_}xbf16>"
    if role == "sm":
        for u in range(2):
            cu = b.c(u, I)
            b.unit(I, "w", cu, lambda buf, u=u: [f"%wu{u}_{buf[1:]} = arith.constant {u} : i32",
                                                 f"func.call @rf_w_unit({buf}, %w_{t}, %wu{u}_{buf[1:]}) : ({B8(UNIT)}, {B8(WB)}, i32) -> ()"])
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("%tb32 = arith.index_cast %tb : index to i32", J)
    # Q: two broadcast units a pass, for every core in the column
    c2 = b.c(2, J)
    idx = b.tmp()
    m(f"{idx} = arith.muli %tb, {c2} : index", J)
    m(f"scf.for %qh = %c0 to {c2} step %c1 {{", J)
    idx2 = b.tmp()
    m(f"{idx2} = arith.addi {idx}, %qh : index", J + 2)
    if role == "qk":
        m("%qh32 = arith.index_cast %qh : index to i32", J + 2)
        b.unit(J + 2, "a", idx2, lambda buf: [f"func.call @rf_q_rows({buf}, %qT_{t}, %qh32) : ({B8(AB)}, {BF(HD * ROWS)}, i32) -> ()"])
    else:
        b.unit(J + 2, "a", idx2, None)
    m("}", J)
    if role == "sm":
        m(f"func.call @rf_sm_init(%st_{t}, %w_{t}, %tb32) : ({F32(2 * ROWS + 4)}, {B8(WB)}, i32) -> ()", J)
    if role in ("pv0", "pv1"):
        m(f"%no = arith.constant {ROWS * HD // 2} : i32", J)
        m(f"func.call @fa_zero_o(%o_{t}, %no) : ({F32(ROWS * HD // 2)}, i32) -> ()", J)
    m("scf.for %bk = %c0 to %cnbw step %c1 {", J)
    K = J + 2
    blk = b.tmp()
    m(f"{blk} = arith.muli %tb, %cnbw : index", K)
    blk2 = b.tmp()
    m(f"{blk2} = arith.addi {blk}, %bk : index", K)

    def units(n_units, base, calls):
        """n_units private units of this block; unit u's running index base + blk2 * per + u."""
        per, off = base
        cper = b.c(per, K)
        first = b.tmp()
        m(f"{first} = arith.muli {blk2}, {cper} : index", K)
        if off:
            first2 = b.tmp()
            m(f"{first2} = arith.addi {first}, {b.c(off, K)} : index", K)
            first = first2
        cn = b.c(n_units, K)
        m(f"scf.for %u = %c0 to {cn} step %c1 {{", K)
        L = K + 2
        idx = b.tmp()
        m(f"{idx} = arith.addi {first}, %u : index", L)
        u32 = b.tmp()
        m(f"{u32} = arith.index_cast %u : index to i32", L)
        b.unit(L, "w", idx, lambda buf: calls(buf, u32))
        m("}", K)

    if role == "qk":
        m(f"aie.use_lock(%sp_{t}, AcquireGreaterEqual, %one)", K)
        m(f"func.call @fa_zero_sT(%sT_{t}) : ({F32(KEYS * ROWS)}) -> ()", K)
        units(KU, (KU, 0), lambda buf, u32: [
            f"func.call @rf_k_unit({buf}, %kbuf_{t}, %qT_{t}, %sT_{t}, {u32}) : ({B8(UNIT)}, {BF(KEYS * SW)}, {BF(HD * ROWS)}, {F32(KEYS * ROWS)}, i32) -> ()"])
        m(f"aie.use_lock(%sc_{t}, Release, %one)", K)
    elif role == "sm":
        units(SU, (SU, 2), lambda buf, u32: [
            f"func.call @rf_s_unit({buf}, %s_{t}, {u32}) : ({B8(UNIT)}, {F32(KEYS * ROWS)}, i32) -> ()"])
        par, ev = b.tmp(), b.tmp()
        m(f"{par} = arith.remui %bk, %c2 : index", K)
        m(f"{ev} = arith.cmpi eq, {par}, %c0 : index", K)
        m(f"aie.use_lock(%pe_{t}, AcquireGreaterEqual, %one)", K)
        m(f"scf.if {ev} {{", K)
        for i in range(2):
            m(f"func.call @rf_sm_block(%s_{t}, %pc{i}_{t}, %st_{t}, %w_{t}, %tb32) : ({F32(KEYS * ROWS)}, {B8(PCB)}, {F32(2 * ROWS + 4)}, {B8(WB)}, i32) -> ()", K + 2)
            if i == 0:
                m("} else {", K)
        m("}", K)
        m(f"aie.use_lock(%pf_{t}, Release, %one)", K)
    else:
        units(PU, (PU + VU, 0), lambda buf, u32: [
            f"func.call @rf_pc_unit({buf}, %pc_{t}, {u32}) : ({B8(UNIT)}, {B8(PCPV)}, i32) -> ()"])
        if PV_EDGE:        # the finished-O buffer holds the partial rows' P over the block (K055)
            m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", K)
            m(f"func.call @rf_pv_begin(%o_{t}, %pc_{t}, %ch_{t}) : ({F32(ROWS * HD // 2)}, {B8(PCPV)}, {BF(2 * ROWS * SW)}) -> ()", K)
        else:
            m(f"func.call @rf_pv_begin(%o_{t}, %pc_{t}) : ({F32(ROWS * HD // 2)}, {B8(PCPV)}) -> ()", K)
        units(VU, (PU + VU, PU), lambda buf, u32: [
            f"func.call @rf_v_unit({buf}, %vbuf_{t}, %pc_{t}, %o_{t}, %ch_{t}, {u32}) : ({B8(UNIT)}, {BF(KEYS * SW)}, {B8(PCPV)}, {F32(ROWS * HD // 2)}, {BF(2 * ROWS * SW)}, i32) -> ()"
            if PV_EDGE else
            f"func.call @rf_v_unit({buf}, %vbuf_{t}, %pc_{t}, %o_{t}, {u32}) : ({B8(UNIT)}, {BF(KEYS * SW)}, {B8(PCPV)}, {F32(ROWS * HD // 2)}, i32) -> ()"])
        if PV_EDGE:
            m(f"aie.use_lock(%chp_{t}, Release, %one)", K)
        last = b.tmp()
        m(f"{last} = arith.cmpi eq, %bk, {b.c(NBW - 1, K)} : index", K)
        m(f"scf.if {last} {{", K)
        m(f"func.call @rf_pv_end(%pc_{t}, %inv_{t}) : ({B8(PCPV)}, {F32(ROWS)}) -> ()", K + 2)
        m("}", K)
    m("}", J)
    if role in ("pv0", "pv1"):
        m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", J)
        fin = "rf_pv_finish_a" if FINISH_A else "rf_pv_finish"
        m(f"func.call @{fin}(%o_{t}, %inv_{t}, %ch_{t}) : ({F32(ROWS * HD // 2)}, {F32(ROWS)}, {BF(2 * ROWS * SW)}) -> ()", J)
        m(f"aie.use_lock(%chc_{t}, Release, %one)", J)
    m("}", I)


def sequence(m, nt):
    I = 6
    nbk = NBW * nt
    assert nbk % 2 == 0 and nbk <= 256 and nt <= NTMAX
    m(f"aie.runtime_sequence @a{nt}({args_sig()}) {{")
    if TRACE:
        m(f"aie.trace.host_config {{buffer_size = {TRACE_BYTES} : i32}}", I)
        for c, r in TRACE:
            m(f"aie.trace.start_config @tr_{c}_{r}", I)
    m(f"%ntv = arith.constant {nt} : i32", I)
    for c in range(NC):
        for role in ROLE.values():
            m(f"aiex.npu.rtp_write(@rtp_{role}_{c}, 0, %ntv) : i32", I)
    for c in range(NC):
        for nm, v in (("qe", 1), ("we", 1), ("qf", 0), ("kf", 0), ("ke", 2), ("v0f", 0), ("v0e", 2), ("v1f", 0), ("v1e", 2), ("sf", 0), ("se", 2),
                      ("paf", 0), ("pae", 2), ("pbf", 0), ("pbe", 2), ("wf", 0), ("ctf", 0), ("cte", 2)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
        for role in ROLE.values():
            m(f"aiex.set_lock(%go_{role}_{c}, 1)", I)

    def task(name, tile, d, ch, body, attrs="", pkt=""):
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}{pkt}) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m("%two = arith.constant 2 : i32", I + 2)
        for line in body:
            m(line, I + 2)
        m("aie.end", I + 2)
        m("}" + attrs, I)
        m(f"aiex.dma_start_task(%{name})", I)

    def locked(acq, bd, rel, va="%one", vr="%one"):
        return ([f"aie.use_lock(%{acq}, AcquireGreaterEqual, {va})"] if acq else []) + [bd] + \
               ([f"aie.use_lock(%{rel}, Release, {vr})"] if rel else [])

    def chain(parts):
        out = []
        for i, p in enumerate(parts):
            if i:
                out += [f"aie.next_bd ^b{i}", f"^b{i}:"]
            out += p
        return out

    for c in range(NC):
        attn_tasks(m, nt, c, task, locked, chain, STANDALONE)
    for c in range(NC):
        m(f"aiex.dma_await_task(%so_{c})", I)
    m("}", 4)


class Standalone:
    """rattn2's own column: Q, widths and K/V windows from the host."""
    q_from_host = True
    oa = False
    oa_readout = None

    def suffix(self, c, role):
        return f"{role}_{c}"

    def tile(self, c, role):
        return f"{role}_{c}"

    def mem_size(self, nm):
        return dict(mem_bufs())[nm]

    def pkt(self, c, role):
        return ""

    def shim_tasks(self, nt, c):
        X, KV = f"%x : memref<{NC * XCOL}xi8>", f"%kv : memref<{NC * kvcol()}xbf16>"
        win = f"sizes = [{NBW}, 4, {KEYS}, {SW}] strides = [{KEYS * HD}, {SW}, {HD}, 1]"
        rep = f" {{repeat_count = {NBW * nt - 1} : i32}}"
        return [("sk", 0, f"aie.dma_bd({KV} offset = {c * kvcol()} len = {4 * KEYS * SW} {win})", rep),
                ("sw", 1, f"aie.dma_bd({X} offset = {c * XCOL} len = {WB})", ""),
                ("sq", 1, f"aie.dma_bd({X} offset = {c * XCOL + WB} len = {nt * QPASS})", ""),
                ("sv", 1, f"aie.dma_bd({KV} offset = {c * kvcol() + NBW * KEYS * HD} len = {4 * KEYS * SW} {win})", rep)]

    def q_send(self, nt, c):
        return (f"aie.dma_bd(%QP_{c} : memref<{self.mem_size('QP')}xi8> offset = 0 len = {2 * nt * AB} "
                f"sizes = [{2 * nt}, 3, {AB // 3}] strides = [{QPASS // 2}, {AB // 3}, 1]) {{bd_id = 12 : i32}}", f"qf_{c}", "")

    def out_bd(self, nt, c):
        return (f"aie.dma_bd(%o : memref<{NC * NTMAX * QPASS}xi8> offset = {c * NTMAX * QPASS} len = {nt * QPASS} "
                f"sizes = [{nt * 8}, 2048] strides = [2048, 1])")


STANDALONE = Standalone()


def attn_tasks(m, nt, c, task, locked, chain, cfg):
    """One column's attention tasks: shim streams, MemTile rings and unit sends, core egress, O out.
    cfg.np passes per 16-row block, cfg.nsk K and cfg.nsv V (per x V worker) sub-blocks per key
    block (a global layer: 2, 2, 2)."""
    np_, nsk, nsv = getattr(cfg, "np", 1), getattr(cfg, "nsk", 1), getattr(cfg, "nsv", 1)
    nbk = NBW * nt * np_
    rep2 = f" {{repeat_count = {nbk // 2 - 1} : i32}}"
    repk = f" {{repeat_count = {nbk * nsk // 2 - 1} : i32}}"
    repv = f" {{repeat_count = {nbk * nsv // 2 - 1} : i32}}"
    n = lambda s: f"{s}_{c}"
    buf = lambda nm, off, ln, extra="": f"aie.dma_bd(%{n(nm)} : memref<{cfg.mem_size(nm)}xi8> offset = {off} len = {ln}{extra})"
    units = lambda k: f" sizes = [{k}, {UNIT}] strides = [{UNIT}, 1]"
    # shim: K and V windows per pass (the 4th size is an iteration: repeats re-read it), widths
    for nm, ch, bd, attrs in cfg.shim_tasks(nt, c):
        task(n(nm), n("shim"), "MM2S", ch, bd if isinstance(bd, list) else [bd], attrs)
    # MemTile receives (even channels: BDs 0-23, odd: 24-47)
    task(n("kin"), n("mem"), "S2MM", 0, chain([locked(n("ke"), buf("KR", i * KU * UNIT, KB) + f" {{bd_id = {i} : i32}}", n("kf")) for i in range(2)]), repk)
    task(n("win"), n("mem"), "S2MM", 1, locked(n("we"), buf("W", 0, WB) + " {bd_id = 24 : i32}", n("wf")))
    if cfg.q_from_host:
        task(n("qin"), n("mem"), "S2MM", 1, locked(n("qe"), buf("QP", 0, nt * QPASS) + " {bd_id = 25 : i32}", n("qf")))
    task(n("vin"), n("mem"), "S2MM", 1, chain([locked(n(f"v{h}e"), buf(f"V{h}R", i * VU * UNIT, VB) + f" {{bd_id = {26 + 2 * i + h} : i32}}", n(f"v{h}f"))
                                               for i in range(2) for h in range(2)]), repv)
    task(n("sin"), n("mem"), "S2MM", 3, chain([locked(n("se"), buf("SR", i * SU * UNIT, SB) + f" {{bd_id = {30 + i} : i32}}", n("sf")) for i in range(2)]), rep2)
    task(n("pin"), n("mem"), "S2MM", 4, chain([locked(n(f"p{ab}e"), buf(f"P{ab.upper()}", i * PU * UNIT, PCB) + f" {{bd_id = {2 + 2 * i + k} : i32}}", n(f"p{ab}f"))
                                               for i in range(2) for k, ab in enumerate("ab")]), rep2)
    if cfg.oa:
        # O's A blocks: x V worker h's packet lands in sub-pass h of the pass's record; each BD
        # iterates over the passes
        dims = f" sizes = [{np_ * nt}, 1, 2, {ABO // 2}] strides = [{2 * ABO}, 0, {ABO // 2}, 1]"
        task(n("ctin"), n("mem"), "S2MM", 5, chain([locked(None, buf("OA", h * ABO, ABO, dims) + f" {{bd_id = {32 + h} : i32}}", n("oaf"))
                                                     for h in range(2)]),
             f" {{out_of_order, repeat_count = {2 * np_ * nt - 1} : i32}}", ", <pkt_type = 0, pkt_id = 0>")
    else:
        task(n("ctin"), n("mem"), "S2MM", 5, chain([locked(n("cte"), buf("CT", h * QPASS // 2, QPASS // 2) + f" {{bd_id = {32 + h} : i32}}", n("ctf")) for h in range(2)]),
             f" {{out_of_order, repeat_count = {2 * nt - 1} : i32}}", ", <pkt_type = 0, pkt_id = 0>")
    # MemTile sends, cut into units for the core rings
    task(n("kout"), n("mem"), "MM2S", 0, chain([locked(n("kf"), buf("KR", i * KU * UNIT, KU * UNIT, units(KU)) + f" {{bd_id = {6 + i} : i32}}", n("ke")) for i in range(2)]), repk)
    for h, (ch, ids) in enumerate(((1, (34, 35, 36, 37)), (2, (8, 9, 10, 11)))):
        P = "PA" if h == 0 else "PB"
        pl = "pa" if h == 0 else "pb"
        parts = []
        if nsv == 1:
            for i in range(2):
                parts.append(locked(n(f"{pl}f"), buf(P, i * PU * UNIT, PU * UNIT, units(PU)) + f" {{bd_id = {ids[2 * i]} : i32}}", n(f"{pl}e")))
                parts.append(locked(n(f"v{h}f"), buf(f"V{h}R", i * VU * UNIT, VU * UNIT, units(VU)) + f" {{bd_id = {ids[2 * i + 1]} : i32}}", n(f"v{h}e")))
        else:             # per block: its P record, then both V ring slots (two sub-blocks)
            assert nsv == 2
            vid = (42, 43) if h == 0 else (14, 15)
            for i in range(2):
                parts.append(locked(n(f"{pl}f"), buf(P, i * PU * UNIT, PU * UNIT, units(PU)) + f" {{bd_id = {ids[2 * i]} : i32}}", n(f"{pl}e")))
                for j in range(2):
                    parts.append(locked(n(f"v{h}f"), buf(f"V{h}R", j * VU * UNIT, VU * UNIT, units(VU)) + f" {{bd_id = {(ids[1], ids[3])[j] if i == 0 else vid[j]} : i32}}", n(f"v{h}e")))
        task(n(f"vout{h}"), n("mem"), "MM2S", ch, chain(parts), rep2)
    task(n("wout"), n("mem"), "MM2S", 3, locked(n("wf"), buf("W", 0, WB, units(2)) + " {bd_id = 38 : i32}", n("wf")))
    task(n("sout"), n("mem"), "MM2S", 3, chain([locked(n("sf"), buf("SR", i * SU * UNIT, SU * UNIT, units(SU)) + f" {{bd_id = {39 + i} : i32}}", n("se")) for i in range(2)]), rep2)
    qbuf, qlock, qattrs = cfg.q_send(nt, c)
    task(n("qout"), n("mem"), "MM2S", 4, locked(qlock, qbuf, qlock), qattrs)
    if not cfg.oa:
        task(n("ctout"), n("mem"), "MM2S", 5, locked(n("ctf"), buf("CT", 0, QPASS, f" sizes = [8, 2048] strides = [2048, 1]") + " {bd_id = 41 : i32}", n("cte"), "%two", "%two"),
             f" {{repeat_count = {nt - 1} : i32}}")
    # core egress
    t, tl = cfg.suffix(c, "qk"), cfg.tile(c, "qk")
    task(n("egs"), tl, "MM2S", 0, locked(f"sc_{t}", f"aie.dma_bd(%sT_{t} : memref<{KEYS * ROWS}xf32> offset = 0 len = {KEYS * ROWS}) {{bd_id = 4 : i32}}", f"sp_{t}"),
         f" {{repeat_count = {nbk - 1} : i32}}", cfg.pkt(c, "qk"))
    t, tl = cfg.suffix(c, "sm"), cfg.tile(c, "sm")
    parts = []
    for i in range(2):
        pc = f"aie.dma_bd(%pc{i}_{t} : memref<{PCB}xi8> offset = 0 len = {PCB})"
        parts.append(locked(f"pf_{t}", pc + f" {{bd_id = {4 + 2 * i} : i32}}", f"pm_{t}"))
        parts.append(locked(f"pm_{t}", pc + f" {{bd_id = {5 + 2 * i} : i32}}", f"pe_{t}"))
    task(n("egp"), tl, "MM2S", 0, chain(parts), rep2, cfg.pkt(c, "sm"))
    for h in range(2):
        t, tl = cfg.suffix(c, f"pv{h}"), cfg.tile(c, f"pv{h}")
        ln = ABO // 2 if cfg.oa else 2 * ROWS * SW
        task(n(f"ego{h}"), tl, "MM2S", 0,
             locked(f"chc_{t}", f"aie.dma_bd(%ch_{t} : memref<{2 * ROWS * SW}xbf16> offset = 0 len = {ln}) {{bd_id = 4 : i32, out_of_order_id = {32 + h} : i32}}", f"chp_{t}"),
             f" {{repeat_count = {np_ * nt - 1} : i32}}", f", <pkt_type = 0, pkt_id = {pid(c, h)}>")
    if not cfg.oa:
        task(n("so"), n("shim"), "S2MM", 0, [cfg.out_bd(nt, c)], " {issue_token = true}")
    elif cfg.oa_readout:
        for nm, tile, d, ch, bd, attrs in cfg.oa_readout(nt, c):
            task(n(nm), tile, d, ch, bd, attrs)


def kernels():
    here = os.path.dirname(os.path.abspath(__file__))
    dims = [f"-DDIM_M={ROWS}", f"-DDIM_K={SW}", f"-DDIM_N={SW}"]
    pv_defs = [d for d in FA_DEFS if not (PV_NATIVE and "EMULATE" in d)] + (["-DRF_PV_NATIVE"] if PV_NATIVE else []) + dims
    pv_src = f"{here}/rf_attn_pv.cc"
    if PV_EDGE:
        pv_src, pv_defs = [(pv_src, pv_defs + ["-DRF_PV_EDGE"]), (f"{here}/rf_pv_native.cc", dims)], []
    return [(QK_O, f"{here}/rf_attn_qk.cc", FA_DEFS + (["-DRF_PV_EDGE"] if PV_EDGE else []) + [f"-DDIM_M={KEYS}", f"-DDIM_K={SW}", f"-DDIM_N={ROWS}"]),
            (PV_O, pv_src, pv_defs)]


def build_text(args):
    global NBW, TRACE
    if args and args[0].startswith("nbw="):
        NBW, args = int(args[0][4:]), args[1:]
    if args and args[0].startswith("trace="):
        TRACE = [tuple(int(v) for v in cr.split(".")) for cr in args[0][6:].split(",")]
        args = args[1:]
    return emit([int(a) for a in args])


if __name__ == "__main__":
    sys.stdout.write(build_text(sys.argv[1:]))
