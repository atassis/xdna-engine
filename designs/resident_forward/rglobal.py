"""The global (full-attention) layer as rows of the layer image's step table and runtime sequences
g1, g2 (P_cap 32), beside the sliding layer's (rlayer_design `g`).

Geometry (attn_layout.GLOBAL with the K split pair-aligned, so a RoPE pair stays in one column):
MemTile m holds q heads 2m, 2m+1 (512 each) and K dims [32m, 32m+32) + [256+32m, 256+32m+32):
1088 columns, 34 N blocks per chain row (68 down-form blocks). V = rms(K) without gain (k_eq_v).

MemTile QK region, bytes, for pieces of up to NTG 16-row blocks (sized by `size_for`):
  OUT  [0, NTG*TB)     per 16-row block [q 2m half 0 | half 1 | q 2m+1 half 0 | half 1] 4 x 8192,
                       then [K 16 x 64 | V 16 x 64] 4096: 36864 a block, the attention's Q units
  IN   [NTG*TB, ..)    the landed q planes [16*NTG][512] x 2 and the raw K plane [16*NTG][64]
Global cache row: [K 512 | V 512 in slice order 0 1 4 5 2 3 6 7] (each x V worker's two sub-blocks
adjacent in the window stream)."""
import os
import numpy as np
import rattn2_design as A2
import rattnh as AH

NQG = 34
NTS = tuple(int(v) for v in os.environ.get("RF_GNTS", "1,2").split(","))   # per-layer global sequences g{nt}
TB = 36864                  # one 16-row block of OUT
KPB = 16 * 64 * 2           # one 16-row block of the raw K plane
NTG = IN = PLANE = KPL = None


def size_for(ntg):
    """Place IN, the q planes and the K plane for pieces of up to ntg blocks; returns the bytes used."""
    global NTG, IN, PLANE, KPL
    NTG, IN, PLANE = ntg, ntg * TB, ntg * 16 * 512 * 2
    KPL = IN + 2 * PLANE
    return KPL + ntg * KPB


size_for(max(NTS))
KVROW = 1024
VSLOT = (0, 1, 4, 5, 2, 3, 6, 7)       # V slice s is stored at slot VSLOT[s] (an involution)
KN_BITS = 0x3B000000        # bits of 1/512 (RF_FAST_RSQRT's argument)
HEADG = 8

# the attention's ring flow is the sliding layer's (units per block), so the table row takes the
# sliding S^T and P+cl unit counts; the core reads only its own bytes of them
GEO = AH.GeoI(1, 8)
GEO.SU, GEO.PU = AH.SLIDING.SU, AH.SLIDING.PU


def steps(L):
    """The global layer's rows (L = rlayer_design with the global geometry set)."""
    s = [(L.GAINS,), (L.NORM_PRE, 0), (L.DOWN, 2 * NQG, 1, L.NSUB, L.KB_Q), (L.HEADP, 1), (L.ATTNP,) + AH.table_row(GEO)]
    s += [(L.DOWN, L.NDBO, L.NSPO, L.NSUBO, L.KB_O), (L.GAINS,), (L.NORM_PRE, 1), (L.SYNC, L.SY_CX_DRAIN)]
    s += [(L.GAINS,), (L.NORM_PRE, 0), (L.GATE, L.NBLK, 1), (L.SYNC, L.SY_CYP_DRAIN), (L.DOWN, L.NDBM, L.NSPM, L.NSUB, L.KB_Q),
          (L.GAINS,), (L.NORM_PRE, 1), (L.SYNC, L.SY_CX_DRAIN)] + ([(L.SYNC, L.SY_W_DRAIN, L.NW_MP - L.NW_M)] if L.NW_MP > L.NW_M else [])
    return s


def head_pass(L, m, c, r, I, glob):
    """Both layer types' head pass in one body (`glob` an i1 value, the row's f1). Per 16-row block:
    sliding, 5 units (rope, 4 q/k/v units of 4 rows); global, 10 units: rope [16][cos 64 | sin 64],
    8 q units (4 rows of one head), the raw K [16][64]. On K the head core runs the cross-column norm
    on the row-0 cascade (rstd at (7,0), to the head cores on the core stream), then V and roped K."""
    t = L.core_name(c, r)
    A, W, E = f"memref<{L.ABLK}xi8>", f"memref<{L.WB}xi8>", f"memref<{L.ELEM}xi8>"
    work = L.head_core(c, r)
    last = c == L.NC - 1
    drain = L.scale_core(c, r) and not work        # (7,3) takes the K rstd broadcast too
    m("%gqo = arith.constant 30720 : i32", I)
    m("%gko = arith.constant 32768 : i32", I)
    m("%g1k = arith.constant 1024 : i32", I)
    m("%g8k = arith.constant 8192 : i32", I)
    m("%g0 = arith.constant 0 : i32", I)
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    m(f"func.call @rf_scr_copy_e(%wb0_{t}, %wblk_{t}, %gqo, %g1k) : ({E}, {W}, i32, i32) -> ()", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    if work:
        m(f"scf.if {glob} {{", I)
        m(f"func.call @rf_scr_copy_e(%wb1_{t}, %wblk_{t}, %gko, %g1k) : ({E}, {W}, i32, i32) -> ()", I + 2)
        m("}", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)
    m(f"%gg = arith.extui {glob} : i1 to i32", I)
    m("%gkw = arith.constant 64 : i32", I)
    m(f"%gkn = arith.constant {KN_BITS} : i32", I)
    m(f"%gcc = arith.constant {c} : i32", I)
    m(f"%gnu = arith.select {glob}, %c10, %c5 : index", I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("%gtu = arith.muli %tb, %gnu : index", J)
    m("scf.for %hk = %c0 to %gnu step %c1 {", J)
    K = J + 2
    m("%hi = arith.addi %gtu, %hk : index", K)
    m("%hp = arith.remui %hi, %c2 : index", K)
    m("%he = arith.cmpi eq, %hp, %c0 : index", K)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", K)

    def both(lines, J_):
        m("scf.if %he {", J_)
        for x in lines(0):
            m(x, J_ + 2)
        m("} else {", J_)
        for x in lines(1):
            m(x, J_ + 2)
        m("}", J_)
    if work or drain:
        m("%hr0 = arith.cmpi eq, %hk, %c0 : index", K)
        m("%hrk = arith.cmpi eq, %hk, %c9 : index", K)
        m("scf.if %hr0 {", K)
        if work:
            both(lambda i: [f"func.call @rf_scr_copy(%ab{i}_{t}, %wblk_{t}, %g0, %g8k) : ({A}, {W}, i32, i32) -> ()"], K + 2)
        m("} else {", K)
        m("scf.if %hrk {", K + 2)
        K2 = K + 4
        if work:
            if c == 0:
                both(lambda i: [f"func.call @rf_ss_first_w(%ab{i}_{t}, %gkw) : ({A}, i32) -> ()"], K2)
            elif not last:
                both(lambda i: [f"func.call @rf_ss_mid_w(%ab{i}_{t}, %gkw) : ({A}, i32) -> ()"], K2)
            else:
                both(lambda i: [f"func.call @rfz_ss_last_w(%wblk_{t}, %ab{i}_{t}, %gkw, %gkn) : ({W}, {A}, i32, i32) -> ()"], K2)
            if not last:
                m(f"func.call @rf_rstd_recv(%wblk_{t}) : ({W}) -> ()", K2)
            m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", K2)
            both(lambda i: [f"func.call @rf_head_k_g(%ab{i}_{t}, %wblk_{t}, %gcc) : ({A}, {W}, i32) -> ()"], K2)
            m(f"aie.use_lock(%cxc_{t}, Release, %one)", K2)
        else:
            m(f"func.call @rf_rstd_recv(%wblk_{t}) : ({W}) -> ()", K2)
        m("} else {", K + 2)
        if work:
            m("%hk1 = arith.subi %hk, %c1 : index", K2)
            m("%hk4 = arith.remui %hk1, %c4 : index", K2)
            m("%hu = arith.index_cast %hk4 : index to i32", K2)
            m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", K2)
            both(lambda i: [f"func.call @rf_head_unit_rt(%ab{i}_{t}, %wblk_{t}, %hu, %gg) : ({A}, {W}, i32, i32) -> ()"], K2)
            m(f"aie.use_lock(%cxc_{t}, Release, %one)", K2)
        m("}", K + 2)
        m("}", K)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", K)
    m("}", J)
    m("}", I)
    # sliding: 71 units a block, so an odd dispatch drains one pad unit (K051); global: 80
    m("%ntodd = arith.remui %nt, %c2 : index", I)
    m("%odd0 = arith.cmpi ne, %ntodd, %c0 : index", I)
    m("%gtrue = arith.constant true", I)
    m(f"%gsl = arith.xori {glob}, %gtrue : i1", I)
    m("%odd = arith.andi %odd0, %gsl : i1", I)
    m("scf.if %odd {", I)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", I + 2)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", I + 2)
    m("}", I)
    if work:
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", I)
        m(f"aie.use_lock(%cxp_{t}, Release, %one)", I)


def decls(L):
    A, W, E = f"memref<{L.ABLK}xi8>", f"memref<{L.WB}xi8>", f"memref<{L.ELEM}xi8>"
    return [("rf_ss_first_w", f"{A}, i32", L.NORM), ("rf_ss_mid_w", f"{A}, i32", L.NORM),
            ("rfz_ss_last_w", f"{W}, {A}, i32, i32", L.NORMZ),
            ("rf_head_unit_rt", f"{A}, {W}, i32, i32", L.HEADN), ("rf_head_k_g", f"{A}, {W}, i32", L.HEADN),
            ("rf_scr_copy", f"{A}, {W}, i32, i32", L.HEADN), ("rf_scr_copy_e", f"{E}, {W}, i32, i32", L.HEADN)]


# ---- control code
def landing(L, task, locked, chain, c, nt):
    """The chain end's rows into the IN planes: q 2m (16 down blocks of 32 columns), q 2m+1, K (2)."""
    QK_ = f"%QK_{c} : memref<{L.QKB}xi8>"
    parts = []
    for i, (off, nb, row) in enumerate(((IN, 16, 1024), (IN + PLANE, 16, 1024), (KPL, 2, 128))):
        parts.append(locked(f"qf_{c}", f"aie.dma_bd({QK_} offset = {off} len = {nb * 16 * nt * 64} "
                                       f"sizes = [{nb}, {16 * nt}, 64] strides = [64, {row}, 1]) {{bd_id = {20 + i} : i32}}", f"qr_{c}"))
    task(f"ql{c}", f"mem_{c}", "S2MM", 2, chain(parts))


def head_sends(L, task, locked, chain, c, nt):
    """Per 16-row block: rope, q 2m units 0-3, q 2m+1 units 0-3, K; each an 8640-B read."""
    QK_, RP_ = f"%QK_{c} : memref<{L.QKB}xi8>", f"%RP_{c} : memref<{L.PCAP_T * L.ROPE_T + L.SLACK}xi8>"
    hh = locked(f"rr_{c}", f"aie.dma_bd({RP_} offset = 0 len = 8640 sizes = [{nt}, 1, 4, 2160] strides = [4096, 0, 2160, 1]) {{bd_id = 14 : i32}}", f"rr_{c}")
    for i, off in enumerate((IN, IN + PLANE)):
        q = f"aie.dma_bd({QK_} offset = {off} len = {4 * 8640} sizes = [{nt}, 4, 4, 2160] strides = [16384, 4096, 2160, 1]) {{bd_id = {15 + i} : i32}}"
        hh += [f"aie.next_bd ^h{i + 1}", f"^h{i + 1}:"] + (locked(f"qr_{c}", q, f"qr_{c}", "%three", "%three") if i == 0 else [q])
    hh += ["aie.next_bd ^h3", "^h3:",
           f"aie.dma_bd({QK_} offset = {KPL} len = 8640 sizes = [{nt}, 1, 4, 2160] strides = [2048, 0, 2160, 1]) {{bd_id = 19 : i32}}"]
    task(f"hh{c}", f"mem_{c}", "MM2S", 4, hh, f" {{repeat_count = {nt - 1} : i32}}")


def head_land(L, task, locked, chain, c, nt):
    """The head cores' outputs into OUT: a q unit is [half][4 rows x 256], then [K | V] a block."""
    QK_ = f"%QK_{c} : memref<{L.QKB}xi8>"
    hl = []
    for i, off in enumerate((0, 16384)):
        hl += ([f"aie.next_bd ^l{i}", f"^l{i}:"] if i else []) + \
              [f"aie.dma_bd({QK_} offset = {off} len = 16384 sizes = [{nt}, 4, 2, 2048] strides = [{TB}, 2048, 8192, 1]) {{bd_id = {34 + i} : i32}}"]
    hl += ["aie.next_bd ^l2", "^l2:", f"aie.use_lock(%hw_{c}, AcquireGreaterEqual, %one)",
           f"aie.dma_bd({QK_} offset = 32768 len = 4096 sizes = [{nt}, 1, 2, 2048] strides = [{TB}, 0, 2048, 1]) {{bd_id = 36 : i32}}",
           f"aie.use_lock(%hr_{c}, Release, %one)"]
    hbq = 3 if c < L.NC - 1 or L.head_core(c, 0) else 2
    task(f"hl{c}", f"mem_{c}", "S2MM", hbq, hl, f" {{repeat_count = {nt - 1} : i32}}")


def cache_send(L, task, locked, c, nt):
    """The new rows' [K | V] blocks out of OUT once all nt have landed, K of every block then V (the
    shim scatters them)."""
    QK_ = f"%QK_{c} : memref<{L.QKB}xi8>"
    v = "%one" if nt == 1 else f"%v{nt}"
    task(f"kw{c}", f"mem_{c}", "MM2S", 5, locked(f"hr_{c}", f"aie.dma_bd({QK_} offset = 32768 len = {2 * nt * 2048} "
                                                  f"sizes = [2, {nt}, 2048] strides = [2048, {TB}, 1]) {{bd_id = 29 : i32}}", f"hr_{c}", v, v))


def cache_write(L, task, chain, c, nt):
    """This column's K and V dims of the new rows into the cache rows."""
    j = c // 2
    kv = f"%kvw : memref<{L.kvbuf()}xbf16>"
    parts = [[f"aie.dma_bd({kv} offset = {32 * c} len = {16 * nt * 64} sizes = [{16 * nt}, 2, 32] strides = [{KVROW}, 256, 1])"],
             [f"aie.dma_bd({kv} offset = {512 + VSLOT[j] * 64 + 32 * (c % 2)} len = {16 * nt * 64} "
              f"sizes = [{16 * nt}, 2, 32] strides = [{KVROW}, {(VSLOT[4 + j] - VSLOT[j]) * 64}, 1])"]]
    task(f"kws{c}", f"shim_{c}", "S2MM", 0, chain(parts), " {issue_token = true}")


class Layer:
    """rlayer_design.Layer for the global attention: one kv head read by every column, H=1, S=8;
    koff the first key block of a segment's window."""
    np, nsk, nsv, koff = 2, 2, 2, 0

    def __init__(self, base, L):
        self.b, self.L = base, L
        self.q_from_host, self.oa = base.q_from_host, base.oa

    def __getattr__(self, k):
        return getattr(self.b, k)

    def shim_tasks(self, nt, c):
        L = self.L
        X, KV = f"%x : memref<{L.xbuf()}xi8>", f"%kvr : memref<{L.kvr_elems()}xbf16>"
        assert A2.NBW <= 64 and A2.NBW * nt * self.np <= 256, (A2.NBW, nt, self.np)   # K054, repeats
        win = f"sizes = [{A2.NBW}, 8, 64, 64] strides = [{64 * KVROW}, 64, {KVROW}, 1]"
        rep = f" {{repeat_count = {A2.NBW * nt * self.np - 1} : i32}}"
        off = self.koff * 64 * KVROW
        return [("sk", 0, f"aie.dma_bd({KV} offset = {off} len = {8 * 64 * 64} {win})", rep),
                ("sw", 1, f"aie.dma_bd({X} offset = {L.xbuf() - A2.WB} len = {A2.WB})", ""),
                ("sv", 1, f"aie.dma_bd({KV} offset = {off + 512} len = {8 * 64 * 64} {win})", rep)]

    def oa_readout(self, nt, c):
        """rlayer_design.Layer.oa_readout with 2 passes a block (4 A blocks a block)."""
        return self.b.oa_readout(self.np * nt, c)

    def q_send(self, nt, c):
        L = self.L
        return (f"aie.dma_bd(%QK_{c} : memref<{L.QKB}xi8> offset = 0 len = {4 * A2.AB} "
                f"sizes = [{nt}, 4, 4, 2160] strides = [{TB}, 8192, 2160, 1]) {{bd_id = 12 : i32}}",
                f"hr_{c}", f" {{repeat_count = {nt - 1} : i32}}")


# ---- the segmented key window (rseg_design): the global window from key 0, 64 blocks a segment
SEG_RUNGS = ()              # window blocks of the g{nt}w{nb} codes (rlayer_design `seg=`)


PUSH = os.environ.get("RF_GSEG_PUSH", "1") == "1"     # later segments as queue pushes (rseg_design.PUSH)


def issue(L, m, I, nb, nt, task):
    """The global attention's tasks with the window issued in segments. Per-command tasks as
    attn_tasks makes them (2 passes a block); per-block tasks per pass (np 1), each pass's window
    cut in segments of rseg_design.G blocks, every segment awaited on its x V sends. With PUSH a
    MemTile or core task whose last configuration had the same shape is a bare queue push."""
    import rseg_design as RS
    base = lambda name: name.rsplit("_", 1)[0]
    lay = Layer(L.LAYER, L)

    def cap(c, passes, np_, nbw, koff):
        out = []
        A2.NBW, lay.np, lay.koff = nbw, np_, koff
        A2.attn_tasks(None, passes, c, lambda *a, **k: out.append((a, k)), RS.A2_locked, RS.A2_chain, lay)
        return out
    for c in range(L.NC):
        for a, k in cap(c, nt, 2, 2, 0):
            if base(a[0]) not in RS.PER_BLOCK:
                task(*a, **k)
    done, shape = [], {}
    for si, (koff, g, np_) in enumerate(RS.segments(nb, 2 * nt)):
        names, syncs = [], []
        for c in range(L.NC):
            for a, k in cap(c, np_, 1, g, koff):
                if base(a[0]) in RS.PER_BLOCK:
                    a = list(a)
                    last = base(a[0]) in RS.LAST
                    key = (c, base(a[0]))
                    if PUSH and not a[1].startswith("shim") and shape.get(key) == (g, np_):
                        syncs += RS.push(m, I, a, last)
                        continue
                    shape[key] = (g, np_)
                    a[0] = f"{a[0]}_s{si}"
                    if last:
                        a[5] = a[5].replace(" {repeat_count", " {issue_token = true, repeat_count")
                    task(*a, **k)
                    names.append(a[0])
        for nm in names:
            if base(base(nm)) in RS.LAST:
                m(f"aiex.dma_await_task(%{nm})", I)
        for sy in syncs:
            m(sy, I)
        for nm in names:
            if base(base(nm)) not in RS.LAST:
                m(f"aiex.dma_free_task(%{nm})", I)
        done += names
    A2.NBW = nb
    return done
