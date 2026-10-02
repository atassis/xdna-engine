"""Attention on the layer image's static rings, 16 query rows per head (fused_attn.cc at FA_ROWS=16,
rf_attn_glue_h.cc). A pass holds H heads sharing one K/V stream, each head S x 64 dims: a sliding
layer H=2, S=4, one pass per 16-row block; a global layer H=1, S=8, one pass per q head (two per
block). K and V blocks travel as sub-blocks of 4 and 2 slices in the sliding sizes' ring slots, so
the MemTile rings and the core buffers are the same for both layer types.

The core program reads the geometry from values in scope (`Geo` SSA names, the layer image's step
table); the control code takes it as Python ints (`GeoI`)."""
import math
import os
import rf_paths
import rattn2_design as A2
from rattn2_design import UNIT, AB, KEYS, SW, NC, ABO, WB

R = 16                                  # query rows per head
KU, VU = 19, 10                         # units per K sub-block (4 slices), V sub-block (2 slices)
KSB, VSB = 4 * KEYS * SW * 2, 2 * KEYS * SW * 2
P_H, CL_H = R * KEYS * 2, 2 * R * 4
ST_H = 48                               # a head's softmax state (2R + 4 floats), 64-B aligned
HMAX, SMAX, HBMAX = 2, 8, 4             # heads per pass, slices per head, slices per x V worker
QT_E = max(2 * 4, 1 * 8) * SW * R       # qT elements (H x S slices of 16 x 64)
O_E = max(2 * 2, 1 * 4) * SW * R        # an x V worker's o, floats
REC_MAX = HMAX * (P_H + CL_H) + 64
QK_O, PV_O = "attnh_qk.o", "attnh_pv.o"
RT_NBW = False                          # the last key block is %cnbw - 1 (a runtime block count), not NBW - 1
SPLIT = False                           # the layer image's key-range split (rsplitl): a 10th geometry field


class GeoI:
    """A layer type's attention geometry (control code)."""

    def __init__(self, H, S):
        self.H, self.S = H, S
        self.NP = 2 // H                  # passes per 16-row block (2 q heads per column)
        self.hb = S // 2                  # slices per x V worker
        self.NSK, self.NSV = S // 4, self.hb // 2
        self.SB = H * KEYS * R * 4        # S^T bytes per block
        self.SU = -(-self.SB // UNIT)
        self.REC = H * (P_H + CL_H) + 64  # P + cl + masks per block
        self.PU = -(-self.REC // UNIT)


SLIDING, GLOBAL = GeoI(2, 4), GeoI(1, 8)


def mem_bufs():
    su, pu = max(SLIDING.SU, GLOBAL.SU), max(SLIDING.PU, GLOBAL.PU)
    return (("KR", 2 * KU * UNIT), ("V0R", 2 * VU * UNIT), ("V1R", 2 * VU * UNIT), ("SR", 2 * su * UNIT),
            ("PA", 2 * pu * UNIT), ("PB", 2 * pu * UNIT), ("W", WB))


def attn_bufs_base(role):
    s_, SPLIT_ = globals()["SPLIT"], None
    globals()["SPLIT"] = False
    try:
        return attn_bufs(role)
    finally:
        globals()["SPLIT"] = s_


def attn_bufs(role):
    """(buffers, locks) of a role, sized for both layer types."""
    if role == "qk":
        return [("qT", QT_E, "bf16"), ("sT", HMAX * KEYS * R, "f32"), ("kbuf", KEYS * SW, "bf16")], [("sp", 1), ("sc", 0)]
    if SPLIT and role.startswith("pv"):     # cht hands the raw-O export's tail BD to its O BD
        bufs, locks = attn_bufs_base(role)
        return bufs, locks + [("cht", 0)]
    if SPLIT and role == "sm":              # the merge's accumulator and its export's locks
        bufs, locks = attn_bufs_base(role)
        return bufs + [("acc", O_E, "f32")], locks + [("mgp", 1), ("mgc", 0)]
    if role == "sm":
        return ([("s", HMAX * KEYS * R, "f32"), ("st", HMAX * ST_H, "f32"), ("w", WB, "i8"),
                 ("pc0", REC_MAX, "i8"), ("pc1", REC_MAX, "i8")], [("pe", 2), ("pf", 0), ("pm", 0)])
    return ([("o", O_E, "f32"), ("inv", HMAX * R, "f32"), ("vbuf", KEYS * SW, "bf16"), ("pc", REC_MAX, "i8"),
             ("ch", 2 * 2 * R * SW, "bf16")], [("chp", 1), ("chc", 0)])


B8, F32, BF = "memref<{}xi8>".format, "memref<{}xf32>".format, "memref<{}xbf16>".format
SIGS = {
    "rf_zero_sT": ([F32(HMAX * KEYS * R), "i32"], QK_O),
    "fa_zero_o": ([F32(O_E), "i32"], PV_O),
    "rf_q_rows": ([B8(AB), BF(QT_E), "i32"], QK_O),
    "rf_k_unit": ([B8(UNIT), BF(KEYS * SW), BF(QT_E), F32(HMAX * KEYS * R), "i32", "i32"], QK_O),
    "rf_s_unit": ([B8(UNIT), F32(HMAX * KEYS * R), "i32", "i32"], QK_O),
    "rf_w_unit": ([B8(UNIT), B8(WB), "i32"], QK_O),
    "rf_sm_init": ([F32(HMAX * ST_H), B8(WB), "i32", "i32"], QK_O),
    "rf_sm_block": ([F32(HMAX * KEYS * R), B8(REC_MAX), F32(HMAX * ST_H), B8(WB), "i32", "i32"], QK_O),
    "rf_pc_unit": ([B8(UNIT), B8(REC_MAX), "i32", "i32"], PV_O),
    "rf_pv_begin": ([F32(O_E), B8(REC_MAX), BF(2 * 2 * R * SW), "i32"], PV_O),
    "rf_v_unit": ([B8(UNIT), BF(KEYS * SW), B8(REC_MAX), F32(O_E), BF(2 * 2 * R * SW), "i32", "i32"], PV_O),
    "rf_pv_end": ([B8(REC_MAX), F32(HMAX * R), "i32"], PV_O),
    "rf_pv_finish_a": ([F32(O_E), F32(HMAX * R), BF(2 * 2 * R * SW), "i32"], PV_O),
    "rf_sm_block_m": ([F32(HMAX * KEYS * R), B8(REC_MAX), F32(HMAX * ST_H), B8(WB), "i32", "i32"], QK_O),
    "rf_merge_tails": ([F32(O_E), B8(WB), B8(UNIT), "i32"], QK_O),
    "rf_merge_ounit": ([F32(O_E), B8(WB), B8(UNIT), "i32", "i32"], QK_O),
    "rf_merge_norm": ([F32(O_E), B8(WB)], QK_O),
    "rf_o_row0": ([B8(UNIT), F32(O_E), F32(HMAX * R)], PV_O),
}


SPLIT_ONLY = ("rf_sm_block_m", "rf_merge_tails", "rf_merge_ounit", "rf_merge_norm", "rf_o_row0")


def attn_decls():
    return [f'func.func private @{nm}({", ".join(a)}) attributes {{link_with = "{o}"}}' for nm, (a, o) in SIGS.items()
            if SPLIT or nm not in SPLIT_ONLY]


def call(nm, args):
    return f"func.call @{nm}({', '.join(args)}) : ({', '.join(SIGS[nm][0])}) -> ()"


def table_row(gi):
    """A layer type's attention fields for the step table (divisions stay on the host: the core's
    index arithmetic is 64-bit and a runtime divide does not lower)."""
    return (gi.H, gi.S, gi.NP, gi.hb, gi.NSK, gi.NSV, gi.SU, gi.PU, gi.REC)


def geo_values(m, b, I, f):
    """SSA values of a layer type's geometry from the step table's fields f[1..9] (index values)."""
    g = dict(zip(("H", "S", "NP", "hb", "NSK", "NSV", "SU", "PU", "REC"), f[1:10]))
    if SPLIT:
        g["SPL"] = b.tmp()
        m(f"{g['SPL']} = arith.cmpi ne, {f[10]}, {b.c(0, I)} : index", I)

    def v(expr, ty="index"):
        n = b.tmp()
        m(f"{n} = {expr} : {ty}", I)
        return n
    c = lambda x: b.c(x, I)
    i32 = lambda x: v(f"arith.index_cast {x}", "index to i32")
    H, S, hb = g["H"], g["S"], g["hb"]
    g["H32"], g["REC32"] = i32(H), i32(g["REC"])
    g["SB32"] = i32(v(f"arith.muli {H}, {c(KEYS * R * 4)}"))
    g["GK"] = i32(v(f"arith.addi {H}, {v(f'arith.muli {S}, {c(16)}')}"))            # H | S << 4
    g["GV"] = i32(v(f"arith.addi {H}, {v(f'arith.muli {hb}, {c(16)}')}"))           # H | hb << 4
    g["NO32"] = i32(v(f"arith.muli {v(f'arith.muli {H}, {hb}')}, {c(SW * R)}"))
    return g


def _bin(m, b, op, x, y, I):
    n = b.tmp()
    m(f"{n} = arith.{op} {x}, {y} : index", I)
    return n


def attn_section(m, b, t, role, I, g):
    """One dispatch's attention on core t: needs %c0, %c1, %c2, %one, %cnbw, %nt in scope and the
    geometry g (geo_values)."""
    c = lambda x, J: b.c(x, J)
    _mul = lambda x, y, J: _bin(m, b, "muli", x, y, J)
    _add = lambda x, y, J: _bin(m, b, "addi", x, y, J)
    if role == "sm":
        for u in range(2):
            cu = c(u, I)
            b.unit(I, "w", cu, lambda buf, u=u: [f"%wu{u}_{buf[1:]} = arith.constant {u} : i32",
                                                 call("rf_w_unit", [buf, f"%w_{t}", f"%wu{u}_{buf[1:]}"])])
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("%tb32 = arith.index_cast %tb : index to i32", J)
    m(f"scf.for %pp = %c0 to {g['NP']} step %c1 {{", J)
    J += 2
    pass_ = b.tmp()
    m(f"{pass_} = arith.addi {_mul('%tb', g['NP'], J)}, %pp : index", J)
    # Q: two broadcast units a pass (a head each when H=2, a head's dims halves when H=1)
    q0 = _mul(pass_, c(2, J), J)
    m(f"scf.for %qh = %c0 to {c(2, J)} step %c1 {{", J)
    qi = b.tmp()
    m(f"{qi} = arith.addi {q0}, %qh : index", J + 2)
    if role == "qk":
        m("%qo = arith.muli %qh, " + c(4 * SW * R, J + 2) + " : index", J + 2)
        m("%qo32 = arith.index_cast %qo : index to i32", J + 2)
        b.unit(J + 2, "a", qi, lambda buf: [call("rf_q_rows", [buf, f"%qT_{t}", "%qo32"])])
    else:
        b.unit(J + 2, "a", qi, None)
    m("}", J)
    if role == "sm":
        m("%rec32 = arith.index_cast %tb : index to i32", J)
        m(call("rf_sm_init", [f"%st_{t}", f"%w_{t}", "%rec32", g["H32"]]), J)
    if role in ("pv0", "pv1"):
        m(call("fa_zero_o", [f"%o_{t}", g["NO32"]]), J)
    m("scf.for %bk = %c0 to %cnbw step %c1 {", J)
    K = J + 2
    blk = b.tmp()
    m(f"{blk} = arith.addi {_mul(pass_, '%cnbw', K)}, %bk : index", K)

    def units(n_units, per, off, calls, nsub=None, slices=0, gbase=None):
        """n_units private units of this block (per a block, first at off). With nsub, n_units is
        per sub-block and calls get gbase | (sub-block x slices) << 8 (its first slice)."""
        first = _mul(blk, per, K)
        if off is not None:
            f2 = b.tmp()
            m(f"{f2} = arith.addi {first}, {off} : index", K)
            first = f2
        if nsub is None:
            m(f"scf.for %u = %c0 to {n_units} step %c1 {{", K)
            L = K + 2
            idx = b.tmp()
            m(f"{idx} = arith.addi {first}, %u : index", L)
            u32 = b.tmp()
            m(f"{u32} = arith.index_cast %u : index to i32", L)
            b.unit(L, "w", idx, lambda buf: calls(buf, u32, None))
            m("}", K)
            return
        m(f"scf.for %sb = %c0 to {nsub} step %c1 {{", K)
        L = K + 2
        sbase = _mul("%sb", n_units, L)
        s0 = _mul("%sb", c(slices * 256, L), L)
        s032, gsub = b.tmp(), b.tmp()
        m(f"{s032} = arith.index_cast {s0} : index to i32", L)
        m(f"{gsub} = arith.addi {gbase}, {s032} : i32", L)
        m(f"scf.for %u = %c0 to {n_units} step %c1 {{", L)
        M_ = L + 2
        idx = b.tmp()
        m(f"{idx} = arith.addi {first}, {_add(sbase, '%u', M_)} : index", M_)
        u32 = b.tmp()
        m(f"{u32} = arith.index_cast %u : index to i32", M_)
        b.unit(M_, "w", idx, lambda buf: calls(buf, u32, gsub))
        m("}", L)
        m("}", K)

    if role == "qk":
        m(f"aie.use_lock(%sp_{t}, AcquireGreaterEqual, %one)", K)
        m(call("rf_zero_sT", [f"%sT_{t}", g["H32"]]), K)
        per = _mul(g["NSK"], c(KU, K), K)
        units(c(KU, K), per, None, lambda buf, u32, gs: [
            call("rf_k_unit", [buf, f"%kbuf_{t}", f"%qT_{t}", f"%sT_{t}", u32, gs])], g["NSK"], 4, g["GK"])
        m(f"aie.use_lock(%sc_{t}, Release, %one)", K)
    elif role == "sm":
        units(g["SU"], g["SU"], c(2, K), lambda buf, u32, sb: [call("rf_s_unit", [buf, f"%s_{t}", u32, g["SB32"]])])
        par, ev = b.tmp(), b.tmp()
        m(f"{par} = arith.remui %bk, %c2 : index", K)
        m(f"{ev} = arith.cmpi eq, {par}, %c0 : index", K)
        m(f"aie.use_lock(%pe_{t}, AcquireGreaterEqual, %one)", K)
        m(f"scf.if {ev} {{", K)
        for i in range(2):
            args_ = [f"%s_{t}", f"%pc{i}_{t}", f"%st_{t}", f"%w_{t}", "%rec32", g["H32"]]
            if SPLIT:        # the split also exports the rows' running max (rf_split_glue)
                m(f"scf.if {g['SPL']} {{", K + 2)
                m(call("rf_sm_block_m", args_), K + 4)
                m("} else {", K + 2)
                m(call("rf_sm_block", args_), K + 4)
                m("}", K + 2)
            else:
                m(call("rf_sm_block", args_), K + 2)
            if i == 0:
                m("} else {", K)
        m("}", K)
        m(f"aie.use_lock(%pf_{t}, Release, %one)", K)
    else:
        per = _add(g["PU"], _mul(g["NSV"], c(VU, K), K), K)
        units(g["PU"], per, None, lambda buf, u32, sb: [call("rf_pc_unit", [buf, f"%pc_{t}", u32, g["REC32"]])])
        m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", K)     # ch holds the partial rows' P (K055)
        m(call("rf_pv_begin", [f"%o_{t}", f"%pc_{t}", f"%ch_{t}", g["GV"]]), K)
        units(c(VU, K), per, g["PU"], lambda buf, u32, gs: [
            call("rf_v_unit", [buf, f"%vbuf_{t}", f"%pc_{t}", f"%o_{t}", f"%ch_{t}", u32, gs])], g["NSV"], 2, g["GV"])
        m(f"aie.use_lock(%chp_{t}, Release, %one)", K)
        last = b.tmp()
        if RT_NBW:
            lastb = _bin(m, b, "subi", "%cnbw", "%c1", K)
        else:
            lastb = c(A2.NBW - 1, K)
        m(f"{last} = arith.cmpi eq, %bk, {lastb} : index", K)
        m(f"scf.if {last} {{", K)
        m(call("rf_pv_end", [f"%pc_{t}", f"%inv_{t}", g["H32"]]), K + 2)
        m("}", K)
    m("}", J)
    if SPLIT and role == "sm" and t.split("_")[0] in ("0", "1"):
        m(f"scf.if {g['SPL']} {{", J)
        sm_split_tail(m, b, t, J + 2, g)
        m("}", J)
    if role in ("pv0", "pv1"):
        m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", J)
        if SPLIT:
            m(f"scf.if {g['SPL']} {{", J)
            m(f"aie.use_lock(%chp_{t}, Release, %one)", J + 2)
            split_tail(m, b, t, J + 2, g)
            m("} else {", J)
            m(call("rf_pv_finish_a", [f"%o_{t}", f"%inv_{t}", f"%ch_{t}", g["GV"]]), J + 2)
            m(f"aie.use_lock(%chc_{t}, Release, %one)", J + 2)
            m("}", J)
        else:
            m(call("rf_pv_finish_a", [f"%o_{t}", f"%inv_{t}", f"%ch_{t}", g["GV"]]), J)
            m(f"aie.use_lock(%chc_{t}, Release, %one)", J)
    J -= 2
    m("}", J)
    m("}", I)


def one_unit(m, b, t, I, idx, call_):
    """A private-ring unit with one call site (the buffer selected by parity: program memory)."""
    par, ev, sel = b.tmp(), b.tmp(), b.tmp()
    m(f"{par} = arith.remui {idx}, %c2 : index", I)
    m(f"{ev} = arith.cmpi eq, {par}, %c0 : index", I)
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    m(f"{sel} = arith.select {ev}, %wb0_{t}, %wb1_{t} : memref<{UNIT}xi8>", I)
    m(call_(sel), I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)


def split_tail(m, b, t, I, g):
    """The split's end of the attention on an x V core (one pass, P = 1): export the raw O and the
    record tail, then reload the column's two heads' merged rows (one unit each; indices continue the
    attention's, K051) and write their O A blocks with rf_pv_finish_a. chp/chc hand o, ch and pc to
    each export in turn."""
    c = lambda x: b.c(x, I)
    m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", I)
    m(f"aie.use_lock(%chc_{t}, Release, %one)", I)               # tail + raw O
    per = _bin(m, b, "addi", g["PU"], _bin(m, b, "muli", g["NSV"], c(VU), I), I)
    base = _bin(m, b, "muli", "%cnbw", per, I)
    hh = b.tmp()
    m(f"scf.for {hh} = {c(0)} to {c(2)} step {c(1)} {{", I)
    m(f"aie.use_lock(%chp_{t}, AcquireGreaterEqual, %one)", I + 2)
    one_unit(m, b, t, I + 2, _bin(m, b, "addi", base, hh, I + 2), lambda buf: call("rf_o_row0", [buf, f"%o_{t}", f"%inv_{t}"]))
    m(call("rf_pv_finish_a", [f"%o_{t}", f"%inv_{t}", f"%ch_{t}", g["GV"]]), I + 2)
    m(f"aie.use_lock(%chc_{t}, Release, %one)", I + 2)
    m("}", I)


def sm_split_tail(m, b, t, I, g):
    """Columns 0 and 1's softmax cores merge worker 0's and worker 1's dims of the 8 columns' records
    (8 tails, then each column's 10 O units, from the private ring after the attention's units) into
    acc, normalise it, and export it (mgc/mgp)."""
    c = lambda x: b.c(x, I)
    base = _bin(m, b, "addi", _bin(m, b, "muli", "%cnbw", g["SU"], I), c(2), I)
    j = b.tmp()
    m(f"scf.for {j} = {c(0)} to {c(8)} step {c(1)} {{", I)
    j32 = b.tmp()
    m(f"{j32} = arith.index_cast {j} : index to i32", I + 2)
    one_unit(m, b, t, I + 2, _bin(m, b, "addi", base, j, I + 2), lambda buf: call("rf_merge_tails", [f"%acc_{t}", f"%w_{t}", buf, j32]))
    m("}", I)
    jj, uu = b.tmp(), b.tmp()
    m(f"scf.for {jj} = {c(0)} to {c(8)} step {c(1)} {{", I)
    m(f"scf.for {uu} = {c(0)} to {c(10)} step {c(1)} {{", I + 2)
    jj32, uu32 = b.tmp(), b.tmp()
    m(f"{jj32} = arith.index_cast {jj} : index to i32", I + 4)
    m(f"{uu32} = arith.index_cast {uu} : index to i32", I + 4)
    j10 = _bin(m, b, "addi", _bin(m, b, "muli", jj, c(10), I + 4), uu, I + 4)
    idx = _bin(m, b, "addi", _bin(m, b, "addi", base, c(8), I + 4), j10, I + 4)
    one_unit(m, b, t, I + 4, idx, lambda buf: call("rf_merge_ounit", [f"%acc_{t}", f"%w_{t}", buf, jj32, uu32]))
    m("}", I + 2)
    m("}", I)
    m(call("rf_merge_norm", [f"%acc_{t}", f"%w_{t}"]), I)
    m(f"aie.use_lock(%mgp_{t}, AcquireGreaterEqual, %one)", I)
    m(f"aie.use_lock(%mgc_{t}, Release, %one)", I)


def kernels():
    here = os.path.dirname(os.path.abspath(__file__))
    iron_i = f"-I{rf_paths.iron_kernel_dir()}"
    iron_root_i = f"-I{rf_paths.iron_kernels_root()}"
    fa_i = f"-I{rf_paths.iron_kernel_dir()}"
    fa = ["-D__AIECC__", "-Dbf16_f32_ONLY", "-DROUND_CONV_EVEN", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16", f"-DFA_ROWS={R}"]
    pv_dims = [f"-DDIM_M={R}", f"-DDIM_K={SW}", f"-DDIM_N={SW}"]
    qk_dims = [f"-DDIM_M={KEYS}", f"-DDIM_K={SW}", f"-DDIM_N={R}"]
    # kernels at -O2, glue at -Oz (column 7's program memory), partially linked into one object each
    glue = f"{here}/rf_split_glue.cc"
    qs = [(glue, fa + qk_dims + ["-DRF_GLUE_QK", "-Oz"]), (glue, fa + qk_dims + ["-DRF_GLUE_MERGE"])] if SPLIT else []
    ps = [(glue, fa + pv_dims + ["-DRF_GLUE_ROW0", "-Oz"])] if SPLIT else []
    return [(QK_O, [(f"{here}/rf_attn_qk_k.cc", fa + qk_dims + [fa_i]),
                     (f"{here}/rf_attn_qk_g.cc", fa + qk_dims + ["-Oz", iron_i])] + qs, []),
            (PV_O, [(f"{here}/rf_attn_pv_k.cc", fa + pv_dims + [fa_i, iron_i]),
                     (f"{here}/rf_attn_pv_g.cc", fa + pv_dims + ["-Oz", iron_i]),
                     (f"{here}/rf_pv_native.cc", pv_dims + [iron_root_i])] + ps, [])]
