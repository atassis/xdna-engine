"""Phase 3 P3.3b: the attention half of a sliding layer in one image, built up from rqkv_design.

Step A (this file's base): pre-attention norm, QKV and the head pass as rqkv_design with HEAD, but
the MemTile keeps q/k/v HEAD-MAJOR ([4 heads][P_cap rows][256], so a head's rows are contiguous for
the attention's Q and the K/V cache write) and the weight distribution is runtime tasks (its
channels carry attention streams later). From rqkv_design's docstring:

x enters row-major bf16 (MemTile c holds columns c*480 .. +479, as in rmlp_design). The norm pass is
rmlp_design's (input_layernorm gain from the weight stream); the chain runs 32 N blocks of 64 per
row, and the chain end writes each (N block, 16-row block) as bf16 rows [16][64] from its GELU
scratch. N blocks 0-15 of row r land in MemTile 2r, 16-31 in 2r+1, so MemTile m holds q heads 2m
and 2m+1, k head m and v head m, 1024 columns row-major ([t][16 rows][1024]). For the gate the
q/k/v go to DDR.
"""
import os
import re
import sys
import numpy as np
import rf_paths
import rattn2_design as A2
import rattnh as AH
import rmlp_design as R
from kvring import DEFAULT_WINDOW_ROWS
from rmlp_design import (M, core_name, NC, NR, ELEM, ABLK, PCAP_T, KERN, NORM, XROW, XB, UNITB, XRB,
                         WB, WGAIN, OUT_OFF, convert_loop, act_call, norm_pass, scale_core, gains)
from o_ref import NDB as NDBO, NSPO, KCO
from mlp_ref import SUB_K
from m1_design import OBLK, NBLK, NSUB, HT
from m2_design import NDB as NDBM, NSP as NSPM, WCOL as WCOL_MLP

NQ = 32                                   # N blocks of 64 per chain row
QROW = 1024 * 2                           # one row of a MemTile's q/k/v, bf16
QT = 16 * QROW                            # one 16-row block
QB = 16 * 64 * 2                          # one (N block, 16-row block) chunk from the chain end
HEAD = True              # the head pass is always on here
HEADN = "head.o"
ROPE_T = 16 * 256 * 2    # one 16-row block of [cos | sin]
SLACK = 448              # an 8640-B unit read over a 8192-B record spills this far
HS_B = PCAP_T * 16 * 256 * 2               # one head plane of the head-major q/k/v, bytes
HSPILL = 112             # a head-pass unit reads 2160 B per plane: 4 rows (2048 B) and this spill
QKB = 4 * HS_B + HSPILL
NW = 2 + NQ * 10 + 2     # weight elements per core per dispatch: gains, QKV, head gains
ATTN = False             # step C: the attention phase in the same image (`attn`, `nbw=N`)
ROLE_OF = {0: "qk", 1: "pv0", 2: "pv1", 3: "sm"}
ROW_OF = {v: k for k, v in ROLE_OF.items()}
KVROW = 8 * 2 * 256      # cache row: [8 kv heads][K, V][256] bf16 elements
GLOB = False             # `g`: the global layer's table rows and sequences g1, g2 (rglobal)
HEADSEG = False          # `h`: the LM head's table rows and sequence h1 (rhead)
SEG = ()                 # `seg=nb,..`: the global window segmented (rseg_design), codes g{nt}w{nb}
ARENA = False            # `arena`: sequence args in FusedArena's order [input x, output o, scratch w, kvw, kvr]
FWD = None               # `fwd=lo-hi[+h]`: the whole forward in one command, f1 (+ head) and f2 (rforward)
SPLIT_NB = 0             # `split=nb[,nb..]`: the global decode with the key range split, g1s{nb} (rsplitl);
SPLITS = ()              # SPLIT_NB is the smallest rung, SPLITS every rung (f1's ladder with `fwd=`)
GCAP = None              # `gcap=rows`: the global cache rows in the forward's scratch (rforward.g_cap)
FSEG = None              # `fseg=nb,..`: f2's rungs (default SEG); larger g2w{nb} serve prefill per layer
PRUNE = os.environ.get("RF_PRUNE_CONSTS", "1") == "1"   # drop a task's unused constants
S64 = False              # `s64`: the forward's scratch in i64 words even below 8 GiB (as the served build)
EMIT = None              # `emit=name,..`: only these runtime sequences (and boot); rung builds for fwd_pack
TRACE = []               # `trace=c.row,..`: core tiles traced in the per-layer sequences (packet ids 27-31)
TRACE_BYTES = int(os.environ.get("RF_TRACE_BYTES", 1 << 22))
TRACE_EVENTS = tuple(os.environ.get("RF_TRACE_EVENTS", "LOCK_STALL,MEMORY_STALL,STREAM_STALL,INSTR_VECTOR,"
                                    "INSTR_LOCK_ACQUIRE_REQ,INSTR_LOCK_RELEASE_REQ,ACTIVE,DISABLED").split(","))
SRING_FWD = None         # `sring=C`: the forward's sliding caches as one blocked ring of C rows (kvring)
FNTS = ((2, None),)      # `fnt=nt[:nb],..`: prefill families f{nt} (16*nt rows), each on f2_rungs() up to nb blocks
SRING = None             # set while rforward emits a sliding layer: the cache BDs are kvring's
CUR_SPLIT = False


def argv(x, w, o, *kv):
    """The runtime-sequence arguments in the build's order (hosts pass them through this)."""
    return (x, o, w, *kv) if ARENA else (x, w, o, *kv)
NBW_S = None             # the sliding layer's window blocks (the build's nbw)
CUR_NB = None


def kvr_elems():
    """%kvr: the global cache up to rseg_design.NKMAX keys when segmented, else the window."""
    if GEO_G and CUR_SPLIT:
        import rsplitl
        return rsplitl.kvr_elems(CUR_NB)
    if GEO_G and SEG:
        import rseg_design
        return rseg_design.NKMAX * KVROW
    return A2.NBW * 64 * KVROW
GEO_G = False            # the geometry in effect is the global layer's (set_geo)
# pinned MemTile layout (bytes): x and q/k/v persist; the gemm staging and the attention rings
# overlay (the rings start at q/k/v plane 2, free once the new K/V rows are in the cache); W and CT
# in the tail are written before that
MT_XR, MT_QK = 0, 108480
MT_GEMM = (MT_QK + 4 * PCAP_T * 16 * 256 * 2 + 112 + 63) // 64 * 64
MT_ATT = MT_QK + 2 * PCAP_T * 16 * 256 * 2
OA_B = PCAP_T * 2 * A2.ABO + (A2.AB - A2.ABO)        # O's A blocks, [pass][sub-pass][5184 B] + read spill
D = R.D
ATTN_H = bool(os.environ.get("RF_ATTN_H"))   # attention at 16 rows per head (rattnh), both layer types
QKV_DOWN = bool(os.environ.get("RF_QKV_DOWN", "1" if ATTN_H else ""))  # QKV on the chain's down form (32-column blocks, the O/MLP-down entries): no
                         # bf16-rows chain end on column 7 (program memory)


def gelu_off():
    """RF_GELU_OFF=1: gate/up's GELU x up on (6,r) (the epilogue core) from bf16 that (7,r) parks in
    (6,r)'s memory (its west neighbour). Does not route on this pin: (6,r)'s h joins the landing packet
    flows 17-24, and every column's row-2 -> MemTile link is already 4 of 4 channels."""
    return MPH and QKV_DOWN and os.environ.get("RF_GELU_OFF", "0") != "0"


EPI = NC - 2             # the epilogue column


def qkv_downform(stream):
    """wstream_h per column ([2 gain][32 n][5 sub][2 half][4 row] elements, then 8 head-gain elements)
    -> the down form's order ([2 gain][32 n][2 half][5 sub][4 row], the head gains)."""
    a = stream.reshape(NC, -1, ELEM)
    g, q = 2 * NR, NQ * NSUB * 2 * NR
    out = a.copy()
    out[:, g:g + q] = a[:, g:g + q].reshape(NC, NQ, NSUB, 2, NR, ELEM).transpose(0, 1, 3, 2, 4, 5).reshape(NC, q, ELEM)
    return out.reshape(-1)
HEAD_PID7 = 25           # column 7's head rows ride (7,0)'s S^T packet flow to MemTile 7 S2MM3
NORMZ = "normz.o"        # rf_norm.cc at -Oz, symbols prefixed rfz_ (column 7's program memory)


HEAD7_ROW0 = os.environ.get("RF_HEAD7_ROW0", "1") == "1"


def head_core(c, r):
    """The head pass's work core: the scale core, except column 7's on row 0 (D3). RF_HEAD7_ROW0=0
    moves it to (7,3), which overflows (7,3)'s program memory by 816 B: the head unit brings
    rf_intmath's sqrt_rn and div_rn with it (+3136 B there, -2032 B on (7,0))."""
    if c == NC - 1 and ATTN and HEAD7_ROW0:
        return r == 0
    return r == 0 if c < NC - 1 else scale_core(c, r)


def nz(c, name):
    """Column 7 links rf_norm's per-unit entries from the -Oz build (program memory)."""
    return name.replace("rf_", "rfz_") if ATTN and c == NC - 1 else name


def gains_c(m, c, t, I):
    """rmlp_design.gains with the column's rf_gain build (GLOB: the shared scratch copy)."""
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    if GLOB:
        m("%gso = arith.constant 30720 : i32", I)
        m("%gsl = arith.constant 1024 : i32", I)
        m(f"func.call @rf_scr_copy_e(%wb0_{t}, %wblk_{t}, %gso, %gsl) : (memref<{ELEM}xi8>, memref<{WB}xi8>, i32, i32) -> ()", I)
    else:
        m(f"func.call @{nz(c, 'rf_gain')}(%wb0_{t}, %wblk_{t}) : (memref<{ELEM}xi8>, memref<{WB}xi8>) -> ()", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)


def own_xlocks(c, r):
    return scale_core(c, r) or head_core(c, r)
OPH = False              # step D2: O projection + post-attention norm + residual as phase 2 (`o`)
KB_Q, KB_O = R.KC // 8, KCO // 8       # K blocks per chain position: QKV (and MLP), O
NSUBO = KCO // SUB_K
NW_O = NDBO * NSPO * NSUBO + 2           # O weight elements per core, then the post-attention gains
WCOL_O = NW_O * NR * ELEM
CT_O = NSPO * A2.ABO                     # one block of O's A blocks in OA
NSPO_S = NSPO


def set_geo(glob):
    """Swap the geometry constants the sequences read between the sliding and global layers."""
    global NQ, NW, NSPO, NW_O, WCOL_O, CT_O, KVROW, GEO_G
    import rglobal
    NQ = rglobal.NQG if glob else 32
    NW = 2 + NQ * 10 + 2
    NSPO = 4 if glob else NSPO_S
    NW_O = NDBO * NSPO * NSUBO + 2
    WCOL_O = NW_O * NR * ELEM
    CT_O = NSPO * A2.ABO
    KVROW = rglobal.KVROW if glob else 8 * 2 * 256
    GEO_G = glob
MPH = False              # step D3: the MLP block (rmlp_design) as phase 3 (`m`)
NW_M = 2 + NBLK * NSUB * 2 + NDBM * NSPM * NSUB + 2   # per core: gains, gate/up, down, gains
WCOL_M = NW_M * NR * ELEM
assert WCOL_M == WCOL_MLP + 2 * R.WGAIN
# a MemTile BD moves GW consecutive elements of one row (one task's 255 repeats cannot cover the
# MLP's element count one at a time); the row count pads to whole 2-slot rounds, the core drains it
GW = 4
NW_MP = -(-NW_M // (2 * GW)) * 2 * GW
WCOL_MP = NW_MP * NR * ELEM


def mlp_grouped(stream):
    """rmlp_ref.stream ([c][element][row] of 1728 B) -> the fused layer's order: per column, per
    group of GW elements, per row, the group's elements (zero elements pad each row to NW_MP). The
    shim then streams it contiguously (a 4-D gather would need an iteration size over 64)."""
    a = stream.reshape(NC, NW_M, NR, ELEM)
    out = np.zeros((NC, NW_MP, NR, ELEM), np.uint8)
    out[:, :NW_M] = a
    out = out.reshape(NC, NW_MP // GW, GW, NR, ELEM).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(out).reshape(-1)


def wcol():
    return WGAIN + NQ * 5 * 2 * NR * ELEM + (WGAIN if HEAD else 0)


def xbuf():
    return 16 * PCAP_T * D * 2 + (PCAP_T * ROPE_T if HEAD else 0) + (A2.WB if ATTN else 0)


def obuf():
    return NC * PCAP_T * A2.QPASS if ATTN else NC * 4 * HS_B


def kvbuf():
    return 16 * PCAP_T * KVROW          # elements per cache window argument


def wbytes():
    return NC * (wcol() + (WCOL_O if OPH else 0) + (WCOL_MP if MPH else 0))


def args_sig():
    xs, ws, os_ = f"%x : memref<{xbuf()}xi8>", f"%w : memref<{wbytes()}xi8>", f"%o : memref<{obuf()}xi8>"
    base = ", ".join((xs, os_, ws) if ARENA else (xs, ws, os_))
    if ATTN:
        base += (f", %kvw : memref<{kvbuf()}xbf16>, %kvr : memref<{kvr_elems()}xbf16>")
    return base


def emit(nts):
    R.SS_LAST = "rfz_ss_last" if ATTN else "rf_ss_last"
    m = M()
    m("module {", 0)
    if FWD:
        import rforward
        for line in rforward.decls(sys.modules[__name__]):
            m(line, 2)
    m("aie.device(npu2) @main {", 2)
    for c in range(NC):
        m(f"%shim_{c} = aie.tile({c}, 0)")
        m(f"%mem_{c} = aie.tile({c}, 1)")
        for r in range(NR):
            m(f"%t_{core_name(c, r)} = aie.tile({c}, {2 + r})")
    for c in range(NC):
        m(f"aie.flow(%shim_{c}, DMA : 0, %mem_{c}, DMA : 0)")
        m(f"aie.flow(%shim_{c}, DMA : 1, %mem_{c}, DMA : 1)")
        for r in range(NR):
            m(f"aie.flow(%mem_{c}, DMA : {r}, %t_{core_name(c, r)}, DMA : 0)")
            m(f"aie.flow(%mem_{c}, DMA : 4, %t_{core_name(c, r)}, DMA : 1)")
        m(f"aie.flow(%mem_{c}, DMA : 5, %shim_{c}, DMA : 0)")
        if c < NC - 1:
            m(f"aie.flow(%t_{core_name(c, 0)}, DMA : 0, %mem_{c}, DMA : 3)")
            m(f"aie.flow(%t_{core_name(NC - 1, 0)}, Core : 0, %t_{core_name(c, 0)}, Core : 0)")
    m(f"aie.flow(%t_{core_name(NC - 1, 0)}, Core : 0, %t_{core_name(NC - 1, NR - 1)}, Core : 0)")
    for r in range(NR):
        for j in range(2):
            if ATTN:     # column 7's ports also carry its attention egress: packet-switched
                m(f"aie.packet_flow({17 + 2 * r + j}) {{")
                m(f"aie.packet_source<%t_{core_name(7, r)}, DMA : {j}>", 6)
                if gelu_off():
                    m(f"aie.packet_source<%t_{core_name(EPI, r)}, DMA : 1>", 6)
                m(f"aie.packet_dest<%mem_{2 * r + j}, DMA : 2>", 6)
                m("}")
            else:
                m(f"aie.flow(%t_{core_name(7, r)}, DMA : {j}, %mem_{2 * r + j}, DMA : 2)")
        for c in range(NC - 1):
            m(f"aie.cascade_flow(%t_{core_name(c, r)}, %t_{core_name(c + 1, r)})")
    if SEG:            # MemTile task-complete tokens to the controller (K057): segments await them
        import rseg_design
        for c in range(NC):
            rseg_design.tct_flow(m, f"%mem_{c}", f"%shim_{c}")
    if ATTN:
        for c in range(NC):
            if c < NC - 1:
                m(f"aie.flow(%t_{core_name(c, 3)}, DMA : 0, %mem_{c}, DMA : 4)")
            else:
                for pid_, r, ch in ((25, 0, 3), (26, 3, 4)):
                    m(f"aie.packet_flow({pid_}) {{")
                    m(f"aie.packet_source<%t_{core_name(c, r)}, DMA : 0>", 6)
                    m(f"aie.packet_dest<%mem_{c}, DMA : {ch}>", 6)
                    m("}")
            for h in range(2):
                m(f"aie.packet_flow({A2.pid(c, h)}) {{")
                m(f"aie.packet_source<%t_{core_name(c, 1 + h)}, DMA : 0>", 6)
                m(f"aie.packet_dest<%mem_{c}, DMA : 5>", 6)
                m("} {keep_pkt_header = true}")
        for line in (AH.attn_decls() if ATTN_H else A2.attn_decls()):
            m(line)
    A, W, G = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", "memref<1024xi16>"
    for name, sig, obj in (
            ("chain_convert_half_k", f"memref<{ELEM}xi8>, {W}, i32, i32, i32", KERN),
            ("chain_mm_first_r2", f"{A}, {W}", KERN), ("chain_mm_mid_r2", f"{A}, {W}", KERN),
            ("chain_mm_last_bf16rows_r2", f"{A}, {W}, {G}", KERN),
            ("rf_gain", f"memref<{ELEM}xi8>, {W}", NORM), ("rf_unit", f"{A}, {W}, i32", NORM),
            ("rf_ss_first", f"{W}, i32", NORM), ("rf_ss_mid", f"{W}, i32", NORM),
            ("rf_ss_last" if not ATTN else "rfz_ss_last", f"{W}, i32", NORM if not ATTN else NORMZ), ("rf_rstd_recv", W, NORM),
            ("rf_pre_scale", W, NORM)) + ((
            ("rf_ss_first_w", f"{A}, i32", NORM), ("rf_ss_mid_w", f"{A}, i32", NORM),
            ("rfz_ss_last_w", f"{W}, {A}, i32, i32", NORMZ), ("rf_head_unit_rt", f"{A}, {W}, i32, i32", HEADN),
            ("rf_head_k_g", f"{A}, {W}, i32", HEADN)) if KNORM_PROBE else ()) + ((
            ("rfz_gain", f"memref<{ELEM}xi8>, {W}", NORMZ), ("rfz_unit", f"{A}, {W}, i32", NORMZ)) if ATTN else ()) + ((
            ("rf_head_rope", f"{A}, {W}", HEADN), ("rf_head_unit_hm", f"{A}, {W}, i32", HEADN)) if HEAD else ()) + ((
            ("chain_down_first_k", f"{A}, {W}, i32", KERN), ("chain_down_mid_k", f"{A}, {W}, i32", KERN),
            ("chain_down_last_k", f"{A}, {W}, i32, i32, i32", KERN),
            ("chain_down_emit_rows_k", f"{W}, {G}, i32, i32, i32", KERN)) if OPH or QKV_DOWN else ()) + ((
            ("rf_post_scale", W, NORM),) if ATTN else ()) + ((
            ("chain_mm_last_geluup_r2", f"{A}, {W}, memref<{OBLK}xi8>, {G}", KERN),) if MPH and not gelu_off() else ()) + ((
            ("chain_mm_last_park_r2", f"{A}, {W}, {G}", KERN),
            ("chain_gelu_up_r2", f"{G}, memref<{OBLK}xi8>", KERN)) if gelu_off() else ()) + (
            tuple(RG().decls(sys.modules[__name__])) if GLOB and not KNORM_PROBE else ()) + (
            (("rf_head_unit_rt", f"{A}, {W}, i32, i32", HEADN),) if KNORM_PROBE and not GLOB else ()):
        m(f'func.func private @{name}({sig}) attributes {{link_with = "{obj}"}}')
    if ATTN:           # (the head rows need the attention image)
        rows = ", ".join("[" + ", ".join(str(v) for v in st) + "]" for st in all_steps())
        m(f'memref.global "private" constant @steps : memref<{len(all_steps())}x{NF}xi32> = dense<[{rows}]>')
    for c in range(NC):
        memtile(m, c)
    for c in range(NC):
        for r in range(NR):
            core(m, c, r, 4096 if ATTN else (3072 if c == NC - 1 else 1024))
    for k, (c, row) in enumerate(TRACE):
        m(f"aie.trace @tr_{c}_{row}(%t_{core_name(c, row - 2)}) {{")
        m('aie.trace.mode "Event-Time"', 6)
        assert 27 + k <= 31, "trace packet ids 27-31: 1-16 attention egress, 17-26 the image's own flows"
        m(f"aie.trace.packet id={27 + k} type=core", 6)
        for ev in TRACE_EVENTS:
            m(f'aie.trace.event<"{ev}">', 6)
        m("aie.trace.start broadcast=15", 6)
        m("aie.trace.stop broadcast=14", 6)
        m("}")
    m(f"aie.runtime_sequence @boot({args_sig()}) {{")
    m("aiex.npu.load_pdi {device_ref = @main}", 6)
    m("}", 4)
    want = lambda name: EMIT is None or name in EMIT
    for nt in nts:
        if want(f"p{nt}"):
            sequence(m, nt)
    if GLOB:
        set_geo(True)
        for nt in RG().NTS:
            for nb in (SEG or (None,)):
                if want(f"g{nt}{f'w{nb}' if nb else ''}"):
                    sequence(m, nt, nb)
        for nb in SPLITS:
            if want(f"g1s{nb}"):
                sequence(m, 1, nb, split=True)
        set_geo(False)
        A2.NBW = NBW_S
        globals()["CUR_SPLIT"] = False
    if HEADSEG and want("h1"):
        import rhead
        rhead.sequence(sys.modules[__name__], m)
    if FWD:
        import rforward
        for nt, rungs in f_rungs():
            for i, nb in enumerate(rungs or (None,)):
                if want(rforward.rung_name(nt, nb, i == 0)):
                    rforward.sequence(sys.modules[__name__], m, nt, *FWD, nb=nb, first=i == 0)
    m("}", 2)
    m("}", 0)
    return "\n".join(mul32(m.lines) if os.environ.get("RF_MUL32") == "1" else m.lines) + "\n"


MULI = re.compile(r"^(\s*)(%[\w.]+) = arith\.muli (%[\w.]+), (%[\w.]+) : index$")


def mul32(lines):
    """Core-body index multiplies in i32: index lowers to i64, whose multiply is a __muldi3 call."""
    out, core = [], False
    for ln in lines:
        core = core or "= aie.core(" in ln
        g = MULI.match(ln) if core else None
        if g:
            ind, r, a, b = g.groups()
            n = r[1:]
            out += [f"{ind}%{n}_a32 = arith.index_cast {a} : index to i32", f"{ind}%{n}_b32 = arith.index_cast {b} : index to i32",
                    f"{ind}%{n}_32 = arith.muli %{n}_a32, %{n}_b32 : i32", f"{ind}{r} = arith.index_cast %{n}_32 : i32 to index"]
            continue
        out.append(ln)
        if core and ln.startswith("    }"):
            core = False
    return out


def mt_layout():
    """(name, size, pinned address or None, alloc_group or None) of one MemTile's buffers."""
    gemm = [("wst", 2 * NR * ELEM), ("XY", PCAP_T * ABLK), ("RP", PCAP_T * ROPE_T + SLACK)]
    if not ATTN:
        return [("wst", 2 * NR * ELEM, None, None), ("XR", XRB, None, None), ("XY", PCAP_T * ABLK, None, None),
                ("QK", QKB, None, None), ("RP", PCAP_T * ROPE_T + SLACK, None, None)]
    # q/k/v is in the gemm group only so the attention rings may overlay its k/v planes; its q planes
    # are outside every ring's pinned range and are read, by DMA, in the attention phase
    out, a = [("XR", XRB, MT_XR, None), ("QK", QKB, MT_QK, "gemm")], MT_GEMM
    for nm, sz in gemm:
        out.append((nm, sz, a, "gemm"))
        a += (sz + 63) // 64 * 64
    tail = a
    a = MT_ATT
    for nm, sz in A2.mem_bufs():
        if nm in ("QP", "W", "CT"):
            continue
        out.append((nm, sz, a, "attn"))
        a += (sz + 63) // 64 * 64
    if OPH:
        # O's y rows over the q planes (free once the attention has read them); O's weights reuse wst
        out.append(("XYO", XRB, MT_QK, "o"))
    if MPH:
        # the MLP after O: everything but x is free
        a_ = XRB
        for nm, sz in (("XYM", R.XYB), ("HM", PCAP_T * HT), ("wsm", 2 * NR * GW * ELEM)):
            out.append((nm, sz, a_, "mlp"))
            a_ += (sz + 63) // 64 * 64
        assert a_ <= 524288, a_
    if LAYER.oa:
        # O's A blocks: written by the attention, read by O; free of the rings and the q planes
        out.append(("OA", OA_B, a, "oa"))
        a += (OA_B + 63) // 64 * 64
        tail = max(tail, a)
        out.append(("W", A2.WB, tail, "w" if MPH else None))
        tail += (A2.WB + 63) // 64 * 64
    else:
        for nm, sz in A2.mem_bufs():
            if nm in ("W", "CT"):
                out.append((nm, sz, tail, None))
                tail += (sz + 63) // 64 * 64
    assert a <= 524288 and tail <= 524288, (a, tail)
    if HEADSEG:
        import rhead
        out += rhead.mem_bufs()
    if SPLIT_NB:        # the split's CT and Q over O's A-block buffer (free until the reload writes it)
        import rsplitl
        oa = next(a_ for n_, _, a_, _ in out if n_ == "OA")
        out += [("CT", rsplitl.CTB, oa, "spct"), ("QP", rsplitl.QP_B, oa + rsplitl.CTB, "spqp")]
        assert rsplitl.CTB + rsplitl.QP_B <= OA_B
    return out


def mem_size(nm):
    return {n_: sz for n_, sz, _, _ in mt_layout()}[nm]


def memtile(m, c):
    for nm, sz, addr, grp in mt_layout():
        extra = (f", alloc_group = \"{grp}\"" if grp else "") + (f", address = {addr} : i32" if addr is not None else "")
        m(f"%{nm}_{c} = aie.buffer(%mem_{c}) {{sym_name = \"{nm}_{c}\"{extra}}} : memref<{sz}xi8>")
    if ATTN:
        for nm in ("kf", "ke", "v0f", "v0e", "v1f", "v1e", "sf", "se", "paf", "pae", "pbf", "pbe", "we", "wf", "ctf", "cte", "oaf", "oae") + (("qe",) if SPLIT_NB else ()):
            m(f"%{nm}_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"{nm}_{c}\"}}")
    for r in range(NR):
        m(f"%wp_{c}_{r} = aie.lock(%mem_{c}) {{init = 2 : i32, sym_name = \"wp_{c}_{r}\"}}")
        m(f"%wc_{c}_{r} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"wc_{c}_{r}\"}}")
    for nm in ("xf", "xp", "xw", "xr", "qf", "qr", "rf", "rr", "hw", "hr") + (("yf", "yr", "ow", "od") if OPH else ()) + (("mhf", "mhr") if MPH else ()):
        m(f"%{nm}_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"{nm}_{c}\"}}")


def core(m, c, r, stack):
    t = core_name(c, r)
    last = c == NC - 1
    g = ', alloc_group = "gemm"' if ATTN else ""
    m(f"%wblk_{t} = aie.buffer(%t_{t}) {{sym_name = \"wblk_{t}\"{g}}} : memref<{WB}xi8>")
    m(f"%rtp_{t} = aie.buffer(%t_{t}) {{sym_name = \"rtp_{t}\"}} : memref<4xi32>")
    if ATTN:
        bufs, locks = (AH.attn_bufs if ATTN_H else A2.attn_bufs)(ROLE_OF[r])
        for nm, sz, ty in bufs:
            m(f"%{nm}_{t} = aie.buffer(%t_{t}) {{sym_name = \"{nm}_{t}\", alloc_group = \"attn\"}} : memref<{sz}x{ty}>")
        for nm, init in locks:
            m(f"%{nm}_{t} = aie.lock(%t_{t}) {{init = {init} : i32, sym_name = \"{nm}_{t}\"}}")
    for i in range(2):
        m(f"%wb{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"wb{i}_{t}\"}} : memref<{ELEM}xi8>")
        m(f"%ab{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ab{i}_{t}\"}} : memref<{ABLK}xi8>")
    for nm, init in (("wp", 2), ("wc", 0), ("ap", 2), ("ac", 0), ("go", 0)):
        m(f"%c{nm}_{t} = aie.lock(%t_{t}) {{init = {init} : i32, sym_name = \"c{nm}_{t}\"}}")
    if own_xlocks(c, r):
        m(f"%cxp_{t} = aie.lock(%t_{t}) {{init = 1 : i32, sym_name = \"cxp_{t}\"}}")
        m(f"%cxc_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"cxc_{t}\"}}")
    if last:
        m(f"%gscr_{t} = aie.buffer(%t_{t}) {{sym_name = \"gscr_{t}\"{g}}} : memref<1024xi16>")
        if MPH and not gelu_off():
            for i in range(2):
                m(f"%ob{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ob{i}_{t}\"{g}}} : memref<{OBLK}xi8>")
        m(f"%cyp_{t} = aie.lock(%t_{t}) {{init = {2 if QKV_DOWN else 1} : i32, sym_name = \"cyp_{t}\"}}")
        for j in range(2):
            m(f"%cyc{j}_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"cyc{j}_{t}\"}}")
    if c == EPI and gelu_off():
        for i in range(2):
            m(f"%gs{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"gs{i}_{t}\"{g}}} : memref<1024xi16>")
            m(f"%ob{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ob{i}_{t}\"{g}}} : memref<{OBLK}xi8>")
        for nm, init in (("cgE", 2), ("cgF", 0), ("coE", 2), ("coF", 0)):
            m(f"%{nm}_{t} = aie.lock(%t_{t}) {{init = {init} : i32, sym_name = \"{nm}_{t}\"}}")
    m(f"%cdma_{t} = aie.mem(%t_{t}) {{")
    m("%one = arith.constant 1 : i32", 6)
    chans = [("S2MM", 0, "wb", "cwp", "cwc", ELEM), ("S2MM", 1, "ab", "cap", "cac", ABLK)]
    for q, (d, ch, buf, acq, rel, ln) in enumerate(chans):
        end = f"^q{q + 1}" if q < len(chans) - 1 else "^qend"
        m(f"%s{q} = aie.dma_start({d}, {ch}, ^q{q}a, {end})", 6)
        for i, lab, nxt in ((0, "a", "b"), (1, "b", "a")):
            m(f"^q{q}{lab}:", 4)
            m(f"aie.use_lock(%{acq}_{t}, AcquireGreaterEqual, %one)", 6)
            m(f"aie.dma_bd(%{buf}{i}_{t} : memref<{ln}xi8> offset = 0 len = {ln})", 6)
            m(f"aie.use_lock(%{rel}_{t}, Release, %one)", 6)
            m(f"aie.next_bd ^q{q}{nxt}", 6)
        m(f"^q{q + 1}:" if q < len(chans) - 1 else "^qend:", 4)
    m("aie.end", 6)
    m("}")
    core_body(m, c, r, last)
    m(f"}} {{stack_size = {stack} : i32}}")


def core_body(m, c, r, last):
    t = core_name(c, r)
    A, W, G = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", "memref<1024xi16>"
    mm = "chain_mm_first_r2" if c == 0 else "chain_mm_mid_r2"
    m(f"%core_{t} = aie.core(%t_{t}) {{")
    I = 6
    for v in sorted({0, 1, 2, 3, 4, 5, 10, 16, 32} | set(range(NF))):
        m(f"%c{v} = arith.constant {v} : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%two = arith.constant 2 : i32", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%cgo_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    if ATTN:
        core_steps(m, c, r, I)
        I -= 2
        m("}", I)
        m("aie.end", I)
        return
    gains(m, t, I)
    R.ATT = False
    norm_pass(m, c, r, I, post=False)
    # QKV: 32 N blocks; the chain end in two halves of 16, one per output channel
    for oj, cnt in ([(None, NQ)] if not last else [(0, 16), (1, 16)]):
        m(f"scf.for %n = %c0 to %c{cnt} step %c1 {{", I)
        I += 2
        m("%n10 = arith.muli %n, %c10 : index", I)
        convert_loop(m, t, I, "%n10", 10, True, fn="chain_convert_half_k", kb=KB_Q)
        m("%nnt = arith.muli %n, %nt : index", I)
        m("scf.for %tb = %c0 to %nt step %c1 {", I)
        I += 2
        m("%idx = arith.addi %nnt, %tb : index", I)
        if last:
            m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %one)", I)

        def call(i):
            if last:
                return [f"func.call @chain_mm_last_bf16rows_r2(%ab{i}_{t}, %wblk_{t}, %gscr_{t}) : ({A}, {W}, {G}) -> ()"]
            return [f"func.call @{mm}(%ab{i}_{t}, %wblk_{t}) : ({A}, {W}) -> ()"]
        act_call(m, t, I, "%idx", call)
        if last:
            m(f"aie.use_lock(%cyc{oj}_{t}, Release, %one)", I)
        I -= 2
        m("}", I)
        I -= 2
        m("}", I)
    if HEAD:
        head_pass(m, c, r, I)
    I -= 2
    m("}", I)
    m("aie.end", I)


def head_pass(m, c, r, I):
    """Per 16-row block: the rope unit, then four 4-row q/k/v units; the scaling core of the
    column (the norm pass's) processes them and hands each result to its write-back DMA, the
    other cores only drain the broadcast."""
    t = core_name(c, r)
    A, W = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>"
    work = head_core(c, r)
    gains_c(m, c, t, I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("%tb5 = arith.muli %tb, %c5 : index", J)
    # unit 0 of each 16-row block is the rope table, units 1-4 the q/k/v rows (one loop body)
    m("scf.for %hk = %c0 to %c5 step %c1 {", J)
    K = J + 2
    m("%hi = arith.addi %tb5, %hk : index", K)
    m("%hp = arith.remui %hi, %c2 : index", K)
    m("%he = arith.cmpi eq, %hp, %c0 : index", K)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", K)
    if work:
        m("%hr0 = arith.cmpi eq, %hk, %c0 : index", K)
        m("%hk1 = arith.subi %hk, %c1 : index", K)
        m("%hu = arith.index_cast %hk1 : index to i32", K)
        m("scf.if %hr0 {", K)
        for i in range(2):
            m(f"scf.if %he {{" if i == 0 else "} else {", K + 2)
            m(f"func.call @rf_head_rope(%ab{i}_{t}, %wblk_{t}) : ({A}, {W}) -> ()", K + 4)
        m("}", K + 2)
        m("} else {", K)
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", K + 2)
        for i in range(2):
            m(f"scf.if %he {{" if i == 0 else "} else {", K + 2)
            if KNORM_PROBE or GLOB:      # the runtime-geometry unit both layer types share
                m(f"func.call @rf_head_unit_rt(%ab{i}_{t}, %wblk_{t}, %hu, %zero32) : ({A}, {W}, i32, i32) -> ()", K + 4)
            else:
                m(f"func.call @rf_head_unit_hm(%ab{i}_{t}, %wblk_{t}, %hu) : ({A}, {W}, i32) -> ()", K + 4)
        m("}", K + 2)
        m(f"aie.use_lock(%cxc_{t}, Release, %one)", K + 2)
        m("}", K)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", K)
    m("}", J)
    m("}", I)
    # 2 + 32 + 5 units per 16-row block: odd for odd nt, and the ab ping-pong must start every
    # dispatch on ab0, so an odd dispatch drains one pad unit (K051)
    m("%ntodd = arith.remui %nt, %c2 : index", I)
    m("%odd = arith.cmpi ne, %ntodd, %c0 : index", I)
    m("scf.if %odd {", I)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", I + 2)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", I + 2)
    m("}", I)
    if work:
        # the last unit may still be leaving through the write-back DMA (K050)
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", I)
        m(f"aie.use_lock(%cxp_{t}, Release, %one)", I)


def down_loop(m, c, r, I, ndb, nsp, nsub, kb, half, nsp1):
    """The chain's down-projection form (Nb = 32, nsp K sub-passes of kb K blocks accumulated at the
    chain end in f32), then column 7 emits bf16 rows into gscr for the MemTile's y landing. The
    extents are SSA values (index; kb i32)."""
    t = core_name(c, r)
    last = c == NC - 1
    A, W, G = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", "memref<1024xi16>"
    dmm = "chain_down_" + ("first_k" if c == 0 else ("last_k" if last else "mid_k"))
    m(f"scf.for %n = %c0 to {ndb} step %c1 {{", I)
    I += 2
    m(f"scf.for %s = %c0 to {nsp} step %c1 {{", I)
    I += 2
    m(f"%ns = arith.muli %n, {nsp} : index", I)
    m("%nsi = arith.addi %ns, %s : index", I)
    m(f"%nse = arith.muli %nsi, {nsub} : index", I)
    convert_loop(m, t, I, "%nse", nsub, False, fn="chain_convert_half_k", kb=kb)
    m("%s32 = arith.index_cast %s : index to i32", I)
    m("%nsnt = arith.muli %nsi, %nt : index", I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    I += 2
    m("%idx = arith.addi %nsnt, %tb : index", I)
    m("%tb32 = arith.index_cast %tb : index to i32", I)

    def dn(i):
        if last:
            return [f"func.call @{dmm}(%ab{i}_{t}, %wblk_{t}, %tb32, %s32, {kb}) : ({A}, {W}, i32, i32, i32) -> ()"]
        return [f"func.call @{dmm}(%ab{i}_{t}, %wblk_{t}, {kb}) : ({A}, {W}, i32) -> ()"]
    act_call(m, t, I, "%idx", dn)
    I -= 2
    m("}", I)
    if last:
        m(f"%sl = arith.cmpi eq, %s, {nsp1} : index", I)
        m("scf.if %sl {", I)
        J = I + 2
        m(f"%sw = arith.cmpi eq, %n, {half} : index", J)
        m("scf.if %sw {", J)
        m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %two)", J + 2)
        m(f"aie.use_lock(%cyp_{t}, Release, %two)", J + 2)
        m("}", J)
        m(f"%lo = arith.cmpi ult, %n, {half} : index", J)
        m(f"%nh = arith.subi %n, {half} : index", J)
        m("%nl = arith.select %lo, %n, %nh : index", J)
        m("%nlnt = arith.muli %nl, %nt : index", J)
        m("scf.for %tb = %c0 to %nt step %c1 {", J)
        K = J + 2
        m("%e = arith.addi %nlnt, %tb : index", K)
        m("%ep = arith.remui %e, %c2 : index", K)
        m("%ep32 = arith.index_cast %ep : index to i32", K)
        m("%tb32 = arith.index_cast %tb : index to i32", K)
        m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %one)", K)
        m(f"func.call @chain_down_emit_rows_k(%wblk_{t}, %gscr_{t}, %tb32, %ep32, {kb}) : ({W}, {G}, i32, i32, i32) -> ()", K)
        m("scf.if %lo {", K)
        m(f"aie.use_lock(%cyc0_{t}, Release, %one)", K + 2)
        m("} else {", K)
        m(f"aie.use_lock(%cyc1_{t}, Release, %one)", K + 2)
        m("}", K)
        m("}", J)
        m("}", I)
    I -= 2
    m("}", I)
    I -= 2
    m("}", I)


def gate_loop(m, c, r, I, nblk, half, gelu):
    """The chain's gate/up form (Nb = 64, 2 x NSUB weight elements per N block): QKV (column 7's
    sink bf16 rows into gscr) or gate/up (GELU(gate) x up into ob, `gelu` an i1 value). Column 7
    splits the N blocks into two halves, one per output channel, in one loop body."""
    t = core_name(c, r)
    last = c == NC - 1
    A, W, G, O = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", "memref<1024xi16>", f"memref<{OBLK}xi8>"
    mm = "chain_mm_first_r2" if c == 0 else "chain_mm_mid_r2"
    if last:
        m("scf.for %oj = %c0 to %c2 step %c1 {", I)
        I += 2
        m("%oj1 = arith.cmpi ne, %oj, %c0 : index", I)
        if MPH and not gelu_off():        # gate/up: both ob buffers drained before channel 1 starts
            m(f"%ojw = arith.andi %oj1, {gelu} : i1", I)
            m("scf.if %ojw {", I)
            m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %two)", I + 2)
            m(f"aie.use_lock(%cyp_{t}, Release, %two)", I + 2)
            m("}", I)
    m(f"scf.for %n = %c0 to {half if last else nblk} step %c1 {{", I)
    I += 2
    m(f"%n10 = arith.muli %n, %c{2 * NSUB} : index", I)
    convert_loop(m, t, I, "%n10", 2 * NSUB, True, fn="chain_convert_half_k", kb="%kbq")
    m("%nnt = arith.muli %n, %nt : index", I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    I += 2
    m("%idx = arith.addi %nnt, %tb : index", I)
    off, te = gelu_off(), core_name(EPI, r)
    if last:
        m(f"aie.use_lock(%{'cgE_' + te if off else 'cyp_' + t}, AcquireGreaterEqual, %one)", I)

    def call(i):
        if not last:
            return [f"func.call @{mm}(%ab{i}_{t}, %wblk_{t}) : ({A}, {W}) -> ()"]
        if off:
            pk = lambda b: f"func.call @chain_mm_last_park_r2(%ab{i}_{t}, %wblk_{t}, %gs{b}_{te}) : ({A}, {W}, {G}) -> ()"
            return [f"%op{i} = arith.remui %idx, %c2 : index", f"%oe{i} = arith.cmpi eq, %op{i}, %c0 : index",
                    f"scf.if %oe{i} {{", "  " + pk(0), "} else {", "  " + pk(1), "}"]
        rows = f"func.call @chain_mm_last_bf16rows_r2(%ab{i}_{t}, %wblk_{t}, %gscr_{t}) : ({A}, {W}, {G}) -> ()"
        if not MPH:
            return [rows]
        gu = lambda b: f"func.call @chain_mm_last_geluup_r2(%ab{i}_{t}, %wblk_{t}, %ob{b}_{t}, %gscr_{t}) : ({A}, {W}, {O}, {G}) -> ()"
        if QKV_DOWN:       # the gate form serves gate/up only
            return [f"%op{i} = arith.remui %idx, %c2 : index", f"%oe{i} = arith.cmpi eq, %op{i}, %c0 : index",
                    f"scf.if %oe{i} {{", "  " + gu(0), "} else {", "  " + gu(1), "}"]
        return [f"scf.if {gelu} {{",
                f"  %op{i} = arith.remui %idx, %c2 : index", f"  %oe{i} = arith.cmpi eq, %op{i}, %c0 : index",
                f"  scf.if %oe{i} {{", "    " + gu(0), "  } else {", "    " + gu(1), "  }",
                "} else {", "  " + rows, "}"]
    act_call(m, t, I, "%idx", call)
    if last and off:
        m(f"aie.use_lock(%cgF_{te}, Release, %one)", I)
    elif last:
        m("scf.if %oj1 {", I)
        m(f"aie.use_lock(%cyc1_{t}, Release, %one)", I + 2)
        m("} else {", I)
        m(f"aie.use_lock(%cyc0_{t}, Release, %one)", I + 2)
        m("}", I)
    if c == EPI and off:     # the epilogue for the previous item, overlapping (7,r)'s chain end
        m("%gp = arith.cmpi ugt, %idx, %c0 : index", I)
        m("scf.if %gp {", I)
        m("%gj = arith.subi %idx, %c1 : index", I + 2)
        epi_gelu(m, t, I + 2, "%gj", "p")
        m("}", I)
    I -= 2
    m("}", I)
    I -= 2
    m("}", I)
    if last:
        I -= 2
        m("}", I)
    if c == EPI and off:
        m(f"%gtot = arith.muli {nblk}, %nt : index", I)
        m("%glast = arith.subi %gtot, %c1 : index", I)
        epi_gelu(m, t, I, "%glast", "l")


def epi_gelu(m, t, I, j, sfx):
    """(6,r): GELU x up of parked item `j` from gs into ob, for its MM2S 1 to the h landing."""
    G, O = "memref<1024xi16>", f"memref<{OBLK}xi8>"
    m(f"aie.use_lock(%cgF_{t}, AcquireGreaterEqual, %one)", I)
    m(f"aie.use_lock(%coE_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%gq{sfx} = arith.remui {j}, %c2 : index", I)
    m(f"%ge{sfx} = arith.cmpi eq, %gq{sfx}, %c0 : index", I)
    m(f"scf.if %ge{sfx} {{", I)
    m(f"func.call @chain_gelu_up_r2(%gs0_{t}, %ob0_{t}) : ({G}, {O}) -> ()", I + 2)
    m("} else {", I)
    m(f"func.call @chain_gelu_up_r2(%gs1_{t}, %ob1_{t}) : ({G}, {O}) -> ()", I + 2)
    m("}", I)
    m(f"aie.use_lock(%cgE_{t}, Release, %one)", I)
    m(f"aie.use_lock(%coF_{t}, Release, %one)", I)


def norm_rt(m, c, r, I, post):
    """rmlp_design.norm_pass for both forms in one body, `post` an i1 value: units in (pre x0 x1;
    post x0 y0 x1 y1), the row-0 cascade sum of squares, rstd to the scaling cores, scale."""
    t = core_name(c, r)
    W, A = f"memref<{WB}xi8>", f"memref<{ABLK}xi8>"
    partial, scale = r == 0, scale_core(c, r)
    m(f"%nu = arith.select {post}, %c4, %c2 : index", I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("scf.for %u = %c0 to %nu step %c1 {", J)
    K = J + 2
    if scale:
        m("%u0 = arith.cmpi eq, %u, %c0 : index", K)
        m(f"%ux = arith.andi %u0, {post} : i1", K)
        m("scf.if %ux {", K)
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", K + 2)
        m("}", K)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", K)
    if partial or scale:
        # post slots 0 2 1 3: (u & 1) * 2 + (u >> 1); pre slots u
        m("%ub = arith.andi %u, %c1 : index", K)
        m("%uh = arith.shrui %u, %c1 : index", K)
        m("%ub2 = arith.muli %ub, %c2 : index", K)
        m("%upo = arith.addi %ub2, %uh : index", K)
        m(f"%us = arith.select {post}, %upo, %u : index", K)
        m("%us32 = arith.index_cast %us : index to i32", K)
        m("%ue = arith.cmpi eq, %ub, %c0 : index", K)
        m("scf.if %ue {", K)
        m(f"func.call @{nz(c, 'rf_unit')}(%ab0_{t}, %wblk_{t}, %us32) : ({A}, {W}, i32) -> ()", K + 2)
        m("} else {", K)
        m(f"func.call @{nz(c, 'rf_unit')}(%ab1_{t}, %wblk_{t}, %us32) : ({A}, {W}, i32) -> ()", K + 2)
        m("}", K)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", K)
    m("}", J)
    if partial:
        fn_ = "rf_ss_first" if c == 0 else (R.SS_LAST if c == NC - 1 else "rf_ss_mid")
        m(f"%src = arith.select {post}, %two, %zero32 : i32", J)
        m(f"func.call @{fn_}(%wblk_{t}, %src) : ({W}, i32) -> ()", J)
    if scale:
        m(f"func.call @rf_rstd_recv(%wblk_{t}) : ({W}) -> ()", J)
        m(f"scf.if {post} {{", J)
        m(f"func.call @rf_post_scale(%wblk_{t}) : ({W}) -> ()", J + 2)
        m("} else {", J)
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", J + 2)
        m(f"func.call @rf_pre_scale(%wblk_{t}) : ({W}) -> ()", J + 2)
        m("}", J)
        m(f"aie.use_lock(%cxc_{t}, Release, %one)", J)
    m("}", I)


# the core program is a step table (one row per step: kind, then up to four extents) over one
# emitted body per kind; kinds:
GAINS, NORM_PRE, GATE, DOWN, HEADP, ATTNP, SYNC, KNORM = range(8)
KNORM_PROBE = os.environ.get("RF_KNORM_PROBE") == "1"   # sizing only: the global K norm's step, never run
KW, KN_BITS = 64, 0x44000000                            # K columns per column; bits of 512.0f
NF = 10                  # step table fields: kind, then extents     # NORM_PRE: f1 selects the post form
SY_CYP_UP, SY_CYP_DRAIN, SY_CX_DRAIN, SY_W_DRAIN = range(4)      # SYNC operations (W_DRAIN: f2 elements)


def steps():
    """The layer as data: QKV, attention, O, MLP (the enabled phases), in order."""
    qkv = (DOWN, 2 * NQ, 1, NSUB, KB_Q) if QKV_DOWN else (GATE, NQ, 0)
    s = [(GAINS,), (NORM_PRE, 0), qkv, (HEADP,)] + ([(KNORM,)] if KNORM_PROBE else []) + [(ATTNP,) + AH.table_row(AH.SLIDING)]
    if OPH:
        s += ([] if QKV_DOWN else [(SYNC, SY_CYP_UP)]) + [(DOWN, NDBO, NSPO, NSUBO, KB_O), (GAINS,), (NORM_PRE, 1), (SYNC, SY_CX_DRAIN)]
    if MPH:
        s += [(GAINS,), (NORM_PRE, 0), (GATE, NBLK, 1), (SYNC, SY_CYP_DRAIN), (DOWN, NDBM, NSPM, NSUB, KB_Q),
              (GAINS,), (NORM_PRE, 1), (SYNC, SY_CX_DRAIN)] + ([(SYNC, SY_W_DRAIN, NW_MP - NW_M)] if NW_MP > NW_M else [])
    return [tuple(x) + (0,) * (NF - len(x)) for x in s]


def knorm(m, c, r, I):
    """The global layer's cross-column K norm on the row-0 cascade: per 16-row block, the column's
    KW-wide K slice sums, rstd at (7,0) with n = 512, to the head cores on the core stream."""
    if r != 0:
        return
    t = core_name(c, r)
    A, W = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>"
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    m("%kq = arith.remui %tb, %c2 : index", J)
    m("%ke = arith.cmpi eq, %kq, %c0 : index", J)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", J)
    for i in range(2):
        m("scf.if %ke {" if i == 0 else "} else {", J)
        if c == 0:
            m(f"func.call @rf_ss_first_w(%ab{i}_{t}, %kw) : ({A}, i32) -> ()", J + 2)
        elif c < NC - 1:
            m(f"func.call @rf_ss_mid_w(%ab{i}_{t}, %kw) : ({A}, i32) -> ()", J + 2)
        else:
            m(f"func.call @rfz_ss_last_w(%wblk_{t}, %ab{i}_{t}, %kw, %knb) : ({W}, {A}, i32, i32) -> ()", J + 2)
    m("}", J)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", J)
    if c < NC - 1:
        m(f"func.call @rf_rstd_recv(%wblk_{t}) : ({W}) -> ()", J)
    m(f"%kc{c} = arith.constant {c} : i32", J)
    m(f"func.call @rf_head_k_g(%ab0_{t}, %wblk_{t}, %kc{c}) : ({A}, {W}, i32) -> ()", J)
    m("}", I)


def RG():
    import rglobal
    return rglobal


def all_steps():
    """The step table: the sliding layer's rows, then (GLOB) the global layer's."""
    was = GEO_G
    set_geo(False)
    s = steps()
    if GLOB:
        set_geo(True)
        g = RG().steps(sys.modules[__name__])
        s = s + [tuple(x) + (0,) * (NF - len(x)) for x in g]
    set_geo(False)
    if HEADSEG:
        import rhead
        s = s + [tuple(x) + (0,) * (NF - len(x)) for x in rhead.steps(sys.modules[__name__])]
    if SPLIT_NB:
        import rsplitl
        set_geo(True)
        s = s + [tuple(x) + (0,) * (NF - len(x)) for x in rsplitl.steps(sys.modules[__name__])]
    set_geo(was)
    return s


def step_range(kind):
    """kind: False (sliding), True (global), "head" or "split" (the global decode, key range split)."""
    was, head_, spl_ = GEO_G, HEADSEG, SPLIT_NB
    set_geo(False)
    n = len(steps())
    globals()["HEADSEG"], globals()["SPLIT_NB"] = False, 0
    ng = len(all_steps())
    globals()["HEADSEG"] = head_
    nh = len(all_steps())
    globals()["SPLIT_NB"] = spl_
    set_geo(was)
    if kind == "head":
        return ng, nh
    if kind == "split":
        return nh, len(all_steps())
    return (n, ng) if kind else (0, n)


def core_steps(m, c, r, I):
    """One dispatch of the core program: the step table, one index_switch case per kind (the
    alloc_group overlays need the phases in exclusive branches of one selector)."""
    t = core_name(c, r)
    last = c == NC - 1
    n = len(all_steps())
    if SEG:            # the window's block count from RTP word 3 (a segmented window's codes differ)
        m(f"%cnbw32 = memref.load %rtp_{t}[%c3] : memref<4xi32>", I)
        m("%cnbw = arith.index_cast %cnbw32 : i32 to index", I)
    else:
        m(f"%cnbw = arith.constant {A2.NBW} : index", I)
    m(f"%cnst = arith.constant {n} : index", I)
    m(f"%kbq = arith.constant {KB_Q} : i32", I)
    m("%zero32 = arith.constant 0 : i32", I)
    if KNORM_PROBE:
        m(f"%kw = arith.constant {KW} : i32", I)
        m(f"%knb = arith.constant {KN_BITS} : i32", I)
    m(f"%stp = memref.get_global @steps : memref<{n}x{NF}xi32>", I)
    if GLOB or HEADSEG:           # the layer type's rows: [rtp 1, rtp 2)
        for i, nm in ((1, "kb"), (2, "ke")):
            m(f"%{nm}32 = memref.load %rtp_{t}[%c{i}] : memref<4xi32>", I)
            m(f"%{nm} = arith.index_cast %{nm}32 : i32 to index", I)
        m("scf.for %k = %kb to %ke step %c1 {", I)
    else:
        m("scf.for %k = %c0 to %cnst step %c1 {", I)
    J = I + 2
    for f in range(NF):
        m(f"%f{f}_32 = memref.load %stp[%k, %c{f}] : memref<{n}x{NF}xi32>", J)
        m(f"%f{f} = arith.index_cast %f{f}_32 : i32 to index", J)
    m("scf.index_switch %f0", J)
    kinds = sorted({s_[0] for s_ in all_steps()})
    for kd in kinds:
        m(f"case {kd} {{", J)
        K = J + 2
        if kd == GAINS:
            gains_c(m, c, t, K)
        elif kd == NORM_PRE:       # both forms: f1 = 1 for the post form
            m("%post = arith.cmpi ne, %f1, %c0 : index", K)
            norm_rt(m, c, r, K, "%post")
        elif kd == GATE:
            m("%ghalf = arith.divui %f1, %c2 : index", K)
            m("%gelu = arith.cmpi ne, %f2, %c0 : index", K)
            gate_loop(m, c, r, K, "%f1", "%ghalf", "%gelu")
        elif kd == DOWN:
            m("%dhalf = arith.divui %f1, %c2 : index", K)
            m("%nsp1 = arith.subi %f2, %c1 : index", K)
            down_loop(m, c, r, K, "%f1", "%f2", "%f3", "%f4_32", "%dhalf", "%nsp1")
        elif kd == HEADP:
            if GLOB:         # one body for both layer types, f1 = 1 global
                m("%hglob = arith.cmpi ne, %f1, %c0 : index", K)
                RG().head_pass(sys.modules[__name__], m, c, r, K, "%hglob")
            else:
                head_pass(m, c, r, K)
        elif kd == KNORM:
            knorm(m, c, r, K)
        elif kd == ATTNP:
            if ATTN_H:
                b_ = A2.Body(m, t)
                AH.attn_section(m, b_, t, ROLE_OF[r], K, AH.geo_values(m, b_, K, [f"%f{i}" for i in range(NF)]))
            else:
                A2.attn_section(m, A2.Body(m, t), t, ROLE_OF[r], K)
        elif kd == SYNC:
            ops = {SY_CYP_UP: [f"aie.use_lock(%cyp_{t}, Release, %one)"] if last else [],
                   SY_CYP_DRAIN: [f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %two)",
                                  f"aie.use_lock(%cyp_{t}, Release, %two)"] if last else [],
                   SY_CX_DRAIN: [f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)",
                                 f"aie.use_lock(%cxp_{t}, Release, %one)"] if scale_core(c, r) else [],
                   SY_W_DRAIN: ["scf.for %e = %c0 to %f2 step %c1 {",
                                f"  aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)",
                                f"  aie.use_lock(%cwp_{t}, Release, %one)", "}"]}
            live = [(op, body) for op, body in ops.items() if body and any(s_[0] == SYNC and s_[1] == op for s_ in all_steps())]
            if live:
                m("scf.index_switch %f1", K)
                for op, body in live:
                    m(f"case {op} {{", K)
                    for line in body:
                        m(line, K + 2)
                    m("scf.yield", K + 2)
                    m("}", K)
                m("default {", K)
                m("scf.yield", K + 2)
                m("}", K)
        m("scf.yield", K)
        m("}", J)
    m("default {", J)
    m("scf.yield", J + 2)
    m("}", J)
    m("}", I)


def o_tasks(m, nt, I, task, locked, chain, issued, freed):
    """After the attention's done token: its tasks and the QKV phase's core tasks freed (their BD ids
    are reused), then ro_design's tasks with the context from OA and x still resident in XR."""
    X = nt * XB
    for name, tile in issued:
        if not tile.startswith("shim_") and name not in freed:
            m(f"aiex.dma_free_task(%{name})", I)
            freed.add(name)
    for c in range(NC):
        for nm, v in (("yf", 15), ("yr", 0), ("ow", 1), ("od", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
        for r in range(NR):
            m(f"aiex.set_lock(%wp_{c}_{r}, 2)", I)
            m(f"aiex.set_lock(%wc_{c}_{r}, 0)", I)
    for c in range(NC):
        task(f"osw{c}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{wbytes()}xi8> offset = {NC * wcol() + c * WCOL_O} len = {WCOL_O})"])
    n2 = f"%v{NSPO * nt}"

    def units(buf, n, bd):
        return (f"aie.dma_bd({buf} offset = 0 len = {UNITB + XROW} sizes = [{n}, 3, 2, {(UNITB + XROW) // 6}] "
                f"strides = [{UNITB}, {(UNITB + XROW) // 3}, {(UNITB + XROW) // 6}, 1]) {{bd_id = {bd} : i32}}")
    for c in range(NC):
        W_ = f"%wst_{c} : memref<{2 * NR * ELEM}xi8>"
        task(f"owi{c}", f"mem_{c}", "S2MM", 0, chain([locked(f"wp_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * ELEM} len = {ELEM}) {{bd_id = {s_ * NR + r} : i32}}", f"wc_{c}_{r}")
                                                      for s_ in range(2) for r in range(NR)]),
             f" {{repeat_count = {NW_O * NR // 8 - 1} : i32}}")
        for r in range(NR):
            ids = ((8, 9), (24, 25), (10, 11), (26, 27))[r]
            task(f"owo{c}_{r}", f"mem_{c}", "MM2S", r, chain([locked(f"wc_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * ELEM} len = {ELEM}) {{bd_id = {ids[s_]} : i32}}", f"wp_{c}_{r}")
                                                             for s_ in range(2)]),
                 f" {{repeat_count = {NW_O // 2 - 1} : i32}}")
        XR_, XY_, OA_ = f"%XR_{c} : memref<{XRB}xi8>", f"%XYO_{c} : memref<{XRB}xi8>", f"%OA_{c} : memref<{OA_B}xi8>"
        wbq = (3, 31) if c < NC - 1 else (2, 23)
        # per (N block, sub-pass): nt A blocks, each an 8640-B read of its 5184-B record
        ob = [locked(f"oaf_{c}", f"aie.dma_bd({OA_} offset = {sp * A2.ABO} len = {nt * ABLK} sizes = [{nt}, 3, {ABLK // 3}] strides = [{CT_O}, {ABLK // 3}, 1]) {{bd_id = {14 + sp} : i32}}", f"oaf_{c}", n2, n2)
              for sp in range(NSPO)]
        task(f"oob{c}", f"mem_{c}", "MM2S", 4, chain(ob), f" {{repeat_count = {NDBO - 1} : i32}}")
        task(f"oly{c}", f"mem_{c}", "S2MM", 2, locked(f"yf_{c}",
             f"aie.dma_bd({XY_} offset = 0 len = {nt * 16 * 64} sizes = [{NDBO // 2}, {nt}, 16, 64] strides = [64, {XB}, {XROW}, 1]) {{bd_id = 21 : i32}}", f"yr_{c}"),
             f" {{repeat_count = {NDBO // 2 - 1} : i32}}")
        task(f"opp{c}", f"mem_{c}", "MM2S", 4, chain([locked(f"xp_{c}", units(XR_, 2 * nt, 18), f"xp_{c}"),
                                                      locked(f"yr_{c}", units(XY_, 2 * nt, 19), f"yr_{c}", f"%v{NDBO // 2}", f"%v{NDBO // 2}")]),
             f" {{repeat_count = {2 * nt - 1} : i32}}")
        task(f"ool{c}", f"mem_{c}", "S2MM", wbq[0], locked(f"ow_{c}", f"aie.dma_bd({XR_} offset = 0 len = {X}) {{bd_id = {wbq[1]} : i32}}", f"od_{c}"))

    def wo(t, ch, bd, pkt=""):
        task(f"owb_{t}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = 0 len = {XB}) {{bd_id = {bd} : i32}}", f"cxp_{t}"),
             f" {{repeat_count = {nt - 1} : i32}}", pkt)
    for c in range(NC - 1):
        wo(core_name(c, 0), 0, 9)
    for r in range(NR):
        t = core_name(NC - 1, r)
        for j in range(2):
            task(f"oyd{j}_{r}", f"t_{t}", "MM2S", j,
                 locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = 0 len = 512 sizes = [2, 2, 2, 128] strides = [512, 256, 128, 1]) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                 f" {{repeat_count = {NDBO // 2 * nt - 1} : i32}}", c7pkt(r, j))
    wo(core_name(NC - 1, NR - 1), 1, 15, c7pkt(NR - 1, 1))
    readout(m, nt, I, task, locked, "o", token=MPH)


def readout(m, nt, I, task, locked, pfx, token):
    """x_out from XR to %o once the write-back has landed, or (token) 64 B of it: the sync point
    before the next phase's tasks."""
    X = nt * XB
    for c in range(NC):
        bd = f"aie.dma_bd(%XR_{c} : memref<{XRB}xi8> offset = 0 len = {64 if token else X}) {{bd_id = 29 : i32}}"
        task(f"{pfx}xo{c}", f"mem_{c}", "MM2S", 5, locked(f"od_{c}", bd, f"od_{c}" if token else f"ow_{c}"))
    for c in range(NC):
        dst = f"offset = {c * XROW} len = 64" if token else \
              f"offset = {c * XROW} len = {X} sizes = [{16 * nt}, {XROW}] strides = [{D * 2}, 1]"
        task(f"{pfx}so{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{obuf()}xi8> {dst})"], " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%{pfx}so{c})", I)


def mlp_tasks(m, nt, I, task, locked, chain, issued, freed):
    """After O's x_out token: O's tasks freed, then rmlp_design's block tasks on the resident x,
    the weights GW elements per MemTile BD."""
    X = nt * XB
    for name, tile in issued:
        if not tile.startswith("shim_") and name not in freed:
            m(f"aiex.dma_free_task(%{name})", I)
            freed.add(name)
    for c in range(NC):
        for nm, v in (("xw", 1), ("xr", 0), ("mhf", 1), ("mhr", 0), ("yf", NDBM // 2), ("yr", 0), ("ow", 1), ("od", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
        for r in range(NR):
            m(f"aiex.set_lock(%wp_{c}_{r}, 2)", I)
            m(f"aiex.set_lock(%wc_{c}_{r}, 0)", I)
    for c in range(NC):     # the weights in mlp_grouped order
        task(f"msw{c}", f"shim_{c}", "MM2S", 0,
             [f"aie.dma_bd(%w : memref<{wbytes()}xi8> offset = {NC * (wcol() + WCOL_O) + c * WCOL_MP} len = {WCOL_MP})"])

    def units(buf, n, bd):
        return (f"aie.dma_bd({buf} offset = 0 len = {UNITB + XROW} sizes = [{n}, 3, 2, {(UNITB + XROW) // 6}] "
                f"strides = [{UNITB}, {(UNITB + XROW) // 3}, {(UNITB + XROW) // 6}, 1]) {{bd_id = {bd} : i32}}")
    for c in range(NC):
        GE = GW * ELEM
        W_ = f"%wsm_{c} : memref<{2 * NR * GE}xi8>"
        task(f"mwi{c}", f"mem_{c}", "S2MM", 0, chain([locked(f"wp_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * GE} len = {GE}) {{bd_id = {s_ * NR + r} : i32}}", f"wc_{c}_{r}")
                                                     for s_ in range(2) for r in range(NR)]), f" {{repeat_count = {NW_MP // (2 * GW) - 1} : i32}}")
        for r in range(NR):
            ids = ((8, 9), (24, 25), (10, 11), (26, 27))[r]
            task(f"mwo{c}_{r}", f"mem_{c}", "MM2S", r, chain([locked(f"wc_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * GE} len = {GE}) {{bd_id = {ids[s_]} : i32}}", f"wp_{c}_{r}")
                                                            for s_ in range(2)]), f" {{repeat_count = {NW_MP // (2 * GW) - 1} : i32}}")
        XR_, XY_, H_ = f"%XR_{c} : memref<{XRB}xi8>", f"%XYM_{c} : memref<{R.XYB}xi8>", f"%HM_{c} : memref<{PCAP_T * HT}xi8>"
        wbq = (3, 30, 31) if c < NC - 1 else (2, 22, 23)
        task(f"mnb{c}", f"mem_{c}", "MM2S", 4, locked(f"xp_{c}",
             f"aie.dma_bd({XR_} offset = 0 len = {2 * nt * (UNITB + XROW)} sizes = [{2 * nt}, 3, {(UNITB + XROW) // 3}] "
             f"strides = [{UNITB}, {(UNITB + XROW) // 3}, 1]) {{bd_id = 12 : i32}}", f"xp_{c}"))
        task(f"mxl{c}", f"mem_{c}", "S2MM", wbq[0], locked(f"xw_{c}", f"aie.dma_bd({XY_} offset = 0 len = {nt * ABLK}) {{bd_id = {wbq[1]} : i32}}", f"xr_{c}"))
        task(f"mbc{c}", f"mem_{c}", "MM2S", 4, locked(f"xr_{c}", f"aie.dma_bd({XY_} offset = 0 len = {nt * ABLK}) {{bd_id = 13 : i32}}", f"xr_{c}"),
             f" {{repeat_count = {NBLK - 1} : i32}}")
        hb = [locked(f"mhr_{c}", f"aie.dma_bd({H_} offset = {sp * ABLK} len = {nt * ABLK} sizes = [{nt}, 3, {ABLK // 3}] strides = [{HT}, {ABLK // 3}, 1]) {{bd_id = {14 + sp} : i32}}", f"mhr_{c}")
              for sp in range(NSPM)]
        task(f"mhb{c}", f"mem_{c}", "MM2S", 4, chain(hb), f" {{repeat_count = {NDBM - 1} : i32}}")
        task(f"mld{c}", f"mem_{c}", "S2MM", 2, locked(f"mhf_{c}", f"aie.dma_bd({H_} offset = 0 len = {NBLK // 2 * nt * OBLK} sizes = [{NBLK // 2}, {nt}, {OBLK}] strides = [{OBLK}, {HT}, 1]) {{bd_id = 20 : i32}}", f"mhr_{c}"))
        task(f"mly{c}", f"mem_{c}", "S2MM", 2, locked(f"yf_{c}",
             f"aie.dma_bd({XY_} offset = 0 len = {nt * 16 * 64} sizes = [{NDBM // 2}, {nt}, 16, 64] strides = [64, {XB}, {XROW}, 1]) {{bd_id = 21 : i32}}", f"yr_{c}"),
             f" {{repeat_count = {NDBM // 2 - 1} : i32}}")
        task(f"mpp{c}", f"mem_{c}", "MM2S", 4, chain([locked(f"xp_{c}", units(XR_, 2 * nt, 18), f"xp_{c}"),
                                                      locked(f"yr_{c}", units(XY_, 2 * nt, 19), f"yr_{c}", f"%v{NDBM // 2}", f"%v{NDBM // 2}")]),
             f" {{repeat_count = {2 * nt - 1} : i32}}")
        task(f"mol{c}", f"mem_{c}", "S2MM", wbq[0], locked(f"ow_{c}", f"aie.dma_bd({XR_} offset = 0 len = {X}) {{bd_id = {wbq[2]} : i32}}", f"od_{c}"))

    def wb(t, ch, bds, pkt=""):
        W_ = f"%wblk_{t} : memref<{WB}xi8>"
        task(f"mwx_{t}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd({W_} offset = {OUT_OFF} len = {ABLK}) {{bd_id = {bds[0]} : i32}}", f"cxp_{t}"),
             f" {{repeat_count = {nt - 1} : i32}}", pkt)
        return lambda: task(f"mwb_{t}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd({W_} offset = 0 len = {XB}) {{bd_id = {bds[1]} : i32}}", f"cxp_{t}"),
                            f" {{repeat_count = {nt - 1} : i32}}", pkt)
    later = [wb(core_name(c, 0), 0, (8, 9)) for c in range(NC - 1)]
    for r in range(NR):
        t = core_name(NC - 1, r)
        if r == NR - 1:
            later.append(wb(t, 1, (14, 15), c7pkt(r, 1)))
        for j in range(2):
            O_ = lambda i: f"%ob{i}_{t} : memref<{OBLK}xi8>"
            if gelu_off():     # the epilogue core's h, halves in order on its MM2S 1
                te = core_name(EPI, r)
                task(f"mgo{j}_{r}", f"t_{te}", "MM2S", 1,
                     chain([locked(f"coF_{te}", f"aie.dma_bd(%ob{i}_{te} : memref<{OBLK}xi8> offset = 0 len = {OBLK}) {{bd_id = {12 + 2 * j + i} : i32}}", f"coE_{te}") for i in range(2)]),
                     f" {{repeat_count = {NBLK // 2 * nt // 2 - 1} : i32}}", c7pkt(r, j))
            else:
                task(f"mgo{j}_{r}", f"t_{t}", "MM2S", j,
                    chain([locked(f"cyc{j}_{t}", f"aie.dma_bd({O_(i)} offset = 0 len = {OBLK}) {{bd_id = {8 + 2 * j + i} : i32}}", f"cyp_{t}") for i in range(2)]),
                    f" {{repeat_count = {NBLK // 2 * nt // 2 - 1} : i32}}", c7pkt(r, j))
            task(f"myd{j}_{r}", f"t_{t}", "MM2S", j,
                 locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = 0 len = 512 sizes = [2, 2, 2, 128] strides = [512, 256, 128, 1]) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                 f" {{repeat_count = {NDBM // 2 * nt - 1} : i32}}", c7pkt(r, j))
    for f in later:
        f()
    readout(m, nt, I, task, locked, "m", token=False)


def sequence(m, nt, nb=None, split=False):
    global CUR_NB, CUR_SPLIT
    CUR_NB, CUR_SPLIT = nb, split
    X = nt * XB
    name = f"g{nt}s{nb}" if split else f"{'g' if GEO_G else 'p'}{nt}{f'w{nb}' if nb else ''}"
    m(f"aie.runtime_sequence @{name}({args_sig()}) {{")
    I = 6
    if TRACE:            # host_config appends the trace buffer as the sequence's last argument
        eg = os.environ.get("RF_TRACE_SHIM")        # the shim column the trace leaves through
        m(f"aie.trace.host_config {{buffer_size = {TRACE_BYTES} : i32" + (f", egress_shim_col = {eg} : i32" if eg else "") + "}", I)
        for c, row in TRACE:
            m(f"aie.trace.start_config @tr_{c}_{row}", I)
    m(f"%ntv = arith.constant {nt} : i32", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 0, %ntv) : i32", I)
    if GLOB or HEADSEG:
        kb_, ke_ = step_range("split" if CUR_SPLIT else GEO_G)
        m(f"%kbv = arith.constant {kb_} : i32", I)
        m(f"%kev = arith.constant {ke_} : i32", I)
        for c in range(NC):
            for r in range(NR):
                m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 1, %kbv) : i32", I)
                m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 2, %kev) : i32", I)
    if SEG:
        m(f"%nbwv = arith.constant {(nb // 8 if CUR_SPLIT else nb) or NBW_S} : i32", I)
        for c in range(NC):
            for r in range(NR):
                m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 3, %nbwv) : i32", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.set_lock(%wp_{c}_{r}, 2)", I)
            m(f"aiex.set_lock(%wc_{c}_{r}, 0)", I)
        for nm, v in (("xf", 1), ("xp", 0), ("xw", 1), ("xr", 0), ("qf", 3 if GEO_G else 4), ("qr", 0),
                      ("rf", 1), ("rr", 0), ("hw", nt if GEO_G else 1), ("hr", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for c in range(NC):
        for r in range(NR):
            t = core_name(c, r)
            if own_xlocks(c, r):
                m(f"aiex.set_lock(%cxp_{t}, 1)", I)
                m(f"aiex.set_lock(%cxc_{t}, 0)", I)
    for r in range(NR):
        t = core_name(NC - 1, r)
        m(f"aiex.set_lock(%cyp_{t}, {2 if QKV_DOWN else 1})", I)
        for j in range(2):
            m(f"aiex.set_lock(%cyc{j}_{t}, 0)", I)
        if gelu_off():
            for nm, v in (("cgE", 2), ("cgF", 0), ("coE", 2), ("coF", 0)):
                m(f"aiex.set_lock(%{nm}_{core_name(EPI, r)}, {v})", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.set_lock(%cgo_{core_name(c, r)}, 1)", I)

    issued = []

    # rglobal.Layer.oa_readout calls the base with nt*np (np=2), so a GLOB design's n2 = %v{2*nt}
    # needs constants up to 2*PCAP_T*2, not just 2*PCAP_T -- GEO_G is the per-layer toggle inside
    # this loop, wrong to gate on here since consts is built once before the loop runs.
    nt_cap = PCAP_T * (2 if GLOB else 1)
    consts = [("one", 1), ("v16", 16), ("v4", 4), ("two", 2)] + ([("three", 3)] if GLOB else []) + [
        (f"v{v_}", v_) for v_ in sorted(({2 * k for k in range(1, nt_cap + 1)} | set(range(3, PCAP_T + 1))) - {4, 16} | ({NDBO // 2} if OPH else set()) | ({NDBM // 2} if MPH else set()))]
    tok = re.compile(r"%(\w+)")

    def task(name, tile, d, ch, body, attrs="", pkt=""):
        issued.append((name, tile))
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}{pkt}) {{", I)
        used = {t_ for line in body for t_ in tok.findall(line)} if PRUNE else None
        for cn, cv in consts:            # only the ones the body names (they are 40% of a forward's text)
            if used is None or cn in used:
                m(f"%{cn} = arith.constant {cv} : i32", I + 2)
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
        # the weight distribution: 8 staging slots round robin in, two per row out, one element each
        W_ = f"%wst_{c} : memref<{2 * NR * ELEM}xi8>"
        task(f"wi{c}", f"mem_{c}", "S2MM", 0, chain([locked(f"wp_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * ELEM} len = {ELEM}) {{bd_id = {s_ * NR + r} : i32}}", f"wc_{c}_{r}")
                                                     for s_ in range(2) for r in range(NR)]),
             f" {{repeat_count = {NW * NR // 8 - 1} : i32}}")
        for r in range(NR):
            ids = ((8, 9), (24, 25), (10, 11), (26, 27))[r]
            task(f"wo{c}_{r}", f"mem_{c}", "MM2S", r, chain([locked(f"wc_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NR + r) * ELEM} len = {ELEM}) {{bd_id = {ids[s_]} : i32}}", f"wp_{c}_{r}")
                                                            for s_ in range(2)]),
                 f" {{repeat_count = {NW // 2 - 1} : i32}}")
    for c in range(NC):
        task(f"sw{c}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{wbytes()}xi8> offset = {c * wcol()} len = {wcol()})"])
        task(f"sx{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{xbuf()}xi8> offset = {c * XROW} len = {X} sizes = [{16 * nt}, {XROW}] strides = [{D * 2}, 1])"])
        if HEAD:
            task(f"sr{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{xbuf()}xi8> offset = {16 * PCAP_T * D * 2} len = {nt * (4096 if GEO_G else ROPE_T)})"])
    for c in range(NC):
        XR_, XY_, QK_ = f"%XR_{c} : memref<{XRB}xi8>", f"%XY_{c} : memref<{PCAP_T * ABLK}xi8>", f"%QK_{c} : memref<{QKB}xi8>"
        wbq = (3, 30) if c < NC - 1 else (2, 17)
        task(f"xin{c}", f"mem_{c}", "S2MM", 1, locked(f"xf_{c}", f"aie.dma_bd({XR_} offset = 0 len = {X}) {{bd_id = 28 : i32}}", f"xp_{c}"))
        RP_ = f"%RP_{c} : memref<{PCAP_T * ROPE_T + SLACK}xi8>"
        if HEAD:
            task(f"rin{c}", f"mem_{c}", "S2MM", 1, locked(f"rf_{c}", f"aie.dma_bd({RP_} offset = 0 len = {nt * (4096 if GEO_G else ROPE_T)}) {{bd_id = 33 : i32}}", f"rr_{c}"))
        task(f"nb{c}", f"mem_{c}", "MM2S", 4, locked(f"xp_{c}",
             f"aie.dma_bd({XR_} offset = 0 len = {2 * nt * (UNITB + XROW)} sizes = [{2 * nt}, 3, {(UNITB + XROW) // 3}] "
             f"strides = [{UNITB}, {(UNITB + XROW) // 3}, 1]) {{bd_id = 12 : i32}}", f"xp_{c}"))
        task(f"xl{c}", f"mem_{c}", "S2MM", wbq[0], locked(f"xw_{c}", f"aie.dma_bd({XY_} offset = 0 len = {nt * ABLK}) {{bd_id = {wbq[1]} : i32}}", f"xr_{c}"))
        task(f"bc{c}", f"mem_{c}", "MM2S", 4, locked(f"xr_{c}", f"aie.dma_bd({XY_} offset = 0 len = {nt * ABLK}) {{bd_id = 13 : i32}}", f"xr_{c}"),
             f" {{repeat_count = {(2 * NQ if QKV_DOWN else NQ) - 1} : i32}}")     # A blocks once per N block
        if GEO_G:
            G_, L_ = RG(), sys.modules[__name__]
            G_.landing(L_, task, locked, chain, c, nt)
            G_.head_sends(L_, task, locked, chain, c, nt)
            G_.head_land(L_, task, locked, chain, c, nt)
            G_.cache_send(L_, task, locked, c, nt)
            continue
        # N blocks 4h .. 4h+3 of this MemTile are head h: [4][rows][64 cols] into its plane
        qlp = (f"sizes = [8, {16 * nt}, 64] strides = [64, 512, 1]" if QKV_DOWN      # 32-column blocks
               else f"sizes = [4, {16 * nt}, 128] strides = [128, 512, 1]")
        task(f"ql{c}", f"mem_{c}", "S2MM", 2, chain([locked(f"qf_{c}",
             f"aie.dma_bd({QK_} offset = {h * HS_B} len = {4 * nt * QB} {qlp}) {{bd_id = {20 + h} : i32}}", f"qr_{c}")
             for h in range(4)]))
        if HEAD:
            # per 16-row block: the rope unit, then 4 q/k/v units of 4 rows, each an 8640-B read
            # of an 8192-B record (the spill is ignored by the cores)
            hh = locked(f"rr_{c}", f"aie.dma_bd({RP_} offset = 0 len = 8640 sizes = [{nt}, 3, 2, 1440] strides = [{ROPE_T}, 2880, 1440, 1]) {{bd_id = 14 : i32}}", f"rr_{c}") + \
                ["aie.next_bd ^h1", "^h1:"] + \
                locked(f"qr_{c}", f"aie.dma_bd({QK_} offset = 0 len = {4 * 8640} sizes = [{nt}, 4, 4, 2160] strides = [8192, 2048, {HS_B}, 1]) {{bd_id = 15 : i32}}", f"qr_{c}", "%v4", "%v4")
            task(f"hh{c}", f"mem_{c}", "MM2S", 4, hh, f" {{repeat_count = {nt - 1} : i32}}")
            if nt % 2:
                task(f"hp{c}", f"mem_{c}", "MM2S", 4, locked(f"rr_{c}", f"aie.dma_bd({RP_} offset = 0 len = 8640) {{bd_id = 16 : i32}}", f"rr_{c}"))
            hbq = (3, 31) if c < NC - 1 or head_core(c, 0) else (2, 18)
            task(f"hl{c}", f"mem_{c}", "S2MM", hbq[0], locked(f"hw_{c}", f"aie.dma_bd({QK_} offset = 0 len = {nt * QT} sizes = [{4 * nt}, 4, 2048] strides = [2048, {HS_B}, 1]) {{bd_id = {hbq[1]} : i32}}", f"hr_{c}"))
            if ATTN:     # the new K/V rows (planes 2, 3) to the cache
                task(f"kw{c}", f"mem_{c}", "MM2S", 5, locked(f"hr_{c}", f"aie.dma_bd({QK_} offset = {2 * HS_B} len = {2 * 16 * nt * 512} sizes = [2, {16 * nt}, 512] strides = [{HS_B}, 512, 1]) {{bd_id = 29 : i32}}", f"hr_{c}"))
            else:
                task(f"qo{c}", f"mem_{c}", "MM2S", 5, locked(f"hr_{c}", f"aie.dma_bd({QK_} offset = 0 len = {4 * 16 * nt * 512} sizes = [4, {16 * nt}, 512] strides = [{HS_B}, 512, 1]) {{bd_id = 29 : i32}}", f"hw_{c}"))
        else:
            task(f"qo{c}", f"mem_{c}", "MM2S", 5, locked(f"qr_{c}", f"aie.dma_bd({QK_} offset = 0 len = {nt * QT}) {{bd_id = 29 : i32}}", f"qf_{c}", "%v16", "%v16"))
    for c in range(NC - 1):
        t = core_name(c, 0)
        task(f"wx_{t}", f"t_{t}", "MM2S", 0, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = {OUT_OFF} len = {ABLK}) {{bd_id = 8 : i32}}", f"cxp_{t}"),
             f" {{repeat_count = {nt - 1} : i32}}")
        if HEAD:
            task(f"wh_{t}", f"t_{t}", "MM2S", 0, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = 8192 len = {WH()}) {{bd_id = 9 : i32}}", f"cxp_{t}"),
                 f" {{repeat_count = {WHN() * nt - 1} : i32}}")
    for r in range(NR):
        t = core_name(NC - 1, r)
        if r == NR - 1:
            task(f"wx_{t}", f"t_{t}", "MM2S", 1, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = {OUT_OFF} len = {ABLK}) {{bd_id = 14 : i32}}", f"cxp_{t}"),
                 f" {{repeat_count = {nt - 1} : i32}}", c7pkt(r, 1))
        for j in range(2):
            if QKV_DOWN:     # the down form's rows: [16][32] from alternating gscr halves
                task(f"qd{j}_{r}", f"t_{t}", "MM2S", j,
                     locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = 0 len = 512 sizes = [2, 2, 2, 128] strides = [512, 256, 128, 1]) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                     f" {{repeat_count = {NQ * nt - 1} : i32}}", c7pkt(r, j))
            else:
                task(f"qd{j}_{r}", f"t_{t}", "MM2S", j,
                     locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = 0 len = 1024) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                     f" {{repeat_count = {16 * nt - 1} : i32}}", c7pkt(r, j))
        if HEAD and head_core(NC - 1, r):
            ch, pkt = (0, f", <pkt_type = 0, pkt_id = {HEAD_PID7}>") if r == 0 else (1, c7pkt(r, 1))
            task(f"wh_{t}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = 8192 len = {WH()}) {{bd_id = 15 : i32}}", f"cxp_{t}"),
                 f" {{repeat_count = {WHN() * nt - 1} : i32}}", pkt)
    if ATTN:
        freed = set()
        if attention_tasks(m, nt, I, task, locked, chain, issued, freed):
            m("}", 4)          # bisection: the command ends after the split's phase (RF_SPLIT_UPTO)
            return
        if OPH:
            o_tasks(m, nt, I, task, locked, chain, issued, freed)
            if os.environ.get("RF_GSTOP") == "o":      # bisection: end the command after the O projection
                m("}", 4)
                return
        if MPH:
            mlp_tasks(m, nt, I, task, locked, chain, issued, freed)
        m("}", 4)
        return
    for c in range(NC):
        task(f"so{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{obuf()}xi8> offset = {c * 4 * HS_B} len = {4 * 16 * nt * 512} sizes = [4, {16 * nt}, 512] strides = [{HS_B}, 512, 1])"],
             " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%so{c})", I)
    m("}", 4)


def WH():
    """A head core's write-back unit: 4 rows x 4 heads x 256 (sliding), a q unit or [K | V] (global)."""
    return 4096 if GEO_G else 8192


def WHN():
    return 9 if GEO_G else 4


def c7pkt(r, j):
    """Column 7's MM2S ports are packet-switched in the attention image (landing ids 17-24)."""
    return f", <pkt_type = 0, pkt_id = {17 + 2 * r + j}>" if ATTN else ""


class Layer:
    """rattn2's attention column inside this image: Q from the q/k/v planes, K/V windows from the
    cache argument, the widths from the tail of %x (shared by every column)."""
    q_from_host = False
    oa = False               # O's A blocks into the MemTile (the fused layer), else O rows to %o

    def oa_readout(self, nt, c):
        """Step D1's gate: the A blocks straight out after the column's attention tasks."""
        oab, n2 = f"%OA_{c} : memref<{OA_B}xi8>", f"%v{2 * nt}"
        if OPH:
            # step D2: a 64-B token once every A block has landed, the sync point before O's tasks
            return [("tk", f"mem_{c}", "MM2S", 5,
                     [f"aie.use_lock(%oaf_{c}, AcquireGreaterEqual, {n2})",
                      f"aie.dma_bd({oab} offset = 0 len = 64) {{bd_id = 41 : i32}}",
                      f"aie.use_lock(%oaf_{c}, Release, {n2})"], ""),
                    ("so", f"shim_{c}", "S2MM", 0,
                     [f"aie.dma_bd(%o : memref<{obuf()}xi8> offset = {c * XROW} len = 64)"], " {issue_token = true}")]
        return [("oo", f"mem_{c}", "MM2S", 5,
                 [f"aie.use_lock(%oaf_{c}, AcquireGreaterEqual, {n2})",
                  f"aie.dma_bd({oab} offset = 0 len = {nt * 2 * A2.ABO} sizes = [{nt * 4}, {A2.ABO // 2}] strides = [{A2.ABO // 2}, 1]) {{bd_id = 41 : i32}}",
                  f"aie.use_lock(%oae_{c}, Release, {n2})"], ""),
                ("so", f"shim_{c}", "S2MM", 0,
                 [f"aie.dma_bd(%o : memref<{obuf()}xi8> offset = {c * PCAP_T * 2 * A2.ABO} len = {nt * 2 * A2.ABO} sizes = [{nt * 4}, {A2.ABO // 2}] strides = [{A2.ABO // 2}, 1])"],
                 " {issue_token = true}")]

    def suffix(self, c, role):
        return core_name(c, ROW_OF[role])

    def tile(self, c, role):
        return f"t_{core_name(c, ROW_OF[role])}"

    def mem_size(self, nm):
        return mem_size(nm)

    def pkt(self, c, role):
        if c < NC - 1:
            return ""
        return {"qk": ", <pkt_type = 0, pkt_id = 25>", "sm": ", <pkt_type = 0, pkt_id = 26>"}[role]

    def shim_tasks(self, nt, c):
        X, KV = f"%x : memref<{xbuf()}xi8>", f"%kvr : memref<{A2.NBW * 64 * KVROW}xbf16>"
        if SRING:        # the window as two spans of whole blocks, once per pass (kvring)
            import kvring
            rd = lambda kv: [kvring.bd_text(KV, kvring.read_bd(c, kv, 0)), "aie.next_bd ^b1", "^b1:",
                             kvring.bd_text(KV, kvring.read_bd(c, kv, 1))]
            rep = f" {{repeat_count = {nt - 1} : i32}}"
            return [("sk", 0, rd(0), rep), ("sw", 1, f"aie.dma_bd({X} offset = {xbuf() - A2.WB} len = {A2.WB})", ""),
                    ("sv", 1, rd(1), rep)]
        win = f"sizes = [{A2.NBW}, 4, 64, 64] strides = [{64 * KVROW}, 64, {KVROW}, 1]"
        rep = f" {{repeat_count = {A2.NBW * nt - 1} : i32}}"
        return [("sk", 0, f"aie.dma_bd({KV} offset = {c * 512} len = {4 * 64 * 64} {win})", rep),
                ("sw", 1, f"aie.dma_bd({X} offset = {xbuf() - A2.WB} len = {A2.WB})", ""),
                ("sv", 1, f"aie.dma_bd({KV} offset = {c * 512 + 256} len = {4 * 64 * 64} {win})", rep)]

    def q_send(self, nt, c):
        # pass tb: plane 0 rows 16 tb .. (head 2c), then plane 1 (head 2c + 1), each an 8640-B span
        return (f"aie.dma_bd(%QK_{c} : memref<{QKB}xi8> offset = 0 len = {2 * A2.AB} "
                f"sizes = [{nt}, 2, 3, {A2.AB // 3}] strides = [8192, {HS_B}, {A2.AB // 3}, 1]) {{bd_id = 12 : i32}}",
                f"hr_{c}", f" {{repeat_count = {nt - 1} : i32}}")

    def out_bd(self, nt, c):
        return (f"aie.dma_bd(%o : memref<{obuf()}xi8> offset = {c * PCAP_T * A2.QPASS} len = {nt * A2.QPASS} "
                f"sizes = [{nt * 8}, 2048] strides = [2048, 1])")


LAYER = Layer()


def attention_tasks(m, nt, I, task, locked, chain, issued, freed):
    """After the head pass: the new K/V rows to the cache (awaited, since the attention rings overlay
    their planes and the K/V windows read them back), the gemm phases' MemTile tasks freed (their BD
    ids are reused), then rattn2's tasks per column."""
    for c in range(NC):
        if GEO_G:
            RG().cache_write(sys.modules[__name__], task, chain, c, nt)
            continue
        if SRING:        # K then V, each in two spans that never cross a block (kvring)
            import kvring
            kw_ = [[kvring.bd_text(f"%kvw : memref<{kvbuf()}xbf16>", kvring.write_bd(c, kv, sp))] for kv in range(2) for sp in range(2)]
        else:
            kw_ = [[f"aie.dma_bd(%kvw : memref<{kvbuf()}xbf16> offset = {c * 512} len = {2 * 16 * nt * 256} "
                    f"sizes = [2, {16 * nt}, 256] strides = [256, {KVROW}, 1])"]]
        task(f"kws{c}", f"shim_{c}", "S2MM", 0, chain(kw_), " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%kws{c})", I)
    if os.environ.get("RF_GSTOP") == "kv":    # bisection: end the command after the new K/V rows land
        return True
    for name, tile in issued:
        if tile.startswith("mem_") and not name.startswith("kws"):
            m(f"aiex.dma_free_task(%{name})", I)
            freed.add(name)
    for c in range(NC):
        for nm, v in (("we", 1), ("kf", 0), ("ke", 2), ("v0f", 0), ("v0e", 2), ("v1f", 0), ("v1e", 2), ("sf", 0), ("se", 2),
                      ("paf", 0), ("pae", 2), ("pbf", 0), ("pbe", 2), ("wf", 0), ("ctf", 0), ("cte", 2)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for c in range(NC):
        m(f"aiex.set_lock(%oaf_{c}, 0)", I)
    if GEO_G and CUR_SPLIT:
        import rsplitl
        return rsplitl.attention(sys.modules[__name__], m, nt, I, task, locked, chain, issued, freed, CUR_NB)
    if GEO_G and SEG:
        freed.update(RG().issue(sys.modules[__name__], m, I, CUR_NB, nt, task))
    else:
        for c in range(NC):
            A2.attn_tasks(m, nt, c, task, locked, chain, RG().Layer(LAYER, sys.modules[__name__]) if GEO_G else LAYER)
    for c in range(NC):
        m(f"aiex.dma_await_task(%so_{c})", I)
    return os.environ.get("RF_GSTOP") == "att"    # bisection: end the command after attention


def extra_files():
    if not FWD:
        return {}
    import json
    import rforward
    return {"fwd_layout.json": json.dumps(rforward.layout(sys.modules[__name__]), indent=1) + "\n"}


def aiecc_options():
    return ["--get-scratchpad-parameters"] if FWD else []


def kernels():
    import os
    # program memory (rf_textbudget.py): rf_head at -Oz (head unit 3136 B at -O2, 1696 at -Oz), and
    # column 7's rf_ss_last from an -Oz build (2400 B -> 464)
    here = os.path.dirname(os.path.abspath(__file__))
    # RF_FAST=1: the SFU inverse square root for the head and column 7's norm (drops rf_intmath's
    # exact sqrt/div from (7,0)); the global layer's K norm needs it
    fast = ["-DRF_FAST_RSQRT"] if os.environ.get("RF_FAST") == "1" else []
    return R.kernels() + [(HEADN, rf_paths.iron_kernel("rf_head.cc"), [os.environ.get("RF_HEAD_OPT", "-Oz")] + fast),
                          (NORMZ, os.path.join(here, "rf_norm_z.cc"), ["-Oz", f"-I{rf_paths.iron_kernel_dir()}"] + fast)] + [
        (o, src, d + (["-Os"] if o == A2.QK_O else [])) for o, src, d in (AH.kernels() if ATTN_H else A2.kernels())]


def build_text(args):
    return emit(configure(args))


def f2_rungs():
    return SEG if FSEG is None else FSEG


def f_rungs():
    """(nt, rungs) of every forward family: f1's split ladder, then each prefill family's."""
    return ((1, SPLITS),) + tuple((nt, tuple(nb for nb in f2_rungs() if cap is None or nb <= cap)) for nt, cap in FNTS)


def configure(args):
    """Set the module from the build flags; returns the p-sequence nts. Runners call this with the
    flags the build recorded (rf_build's gen_args.txt), so the host's layout is the build's."""
    global ATTN, OPH, MPH, GLOB, HEADSEG, SEG, NBW_S, ARENA, FWD, SPLIT_NB, SPLITS, GCAP, FSEG, SRING_FWD, EMIT, FNTS, OA_B, TRACE
    while args and not args[0].isdigit():
        if args[0] == "attn":
            ATTN = True
        elif args[0] == "oa":
            ATTN, LAYER.oa, A2.FINISH_A = True, True, True
        elif args[0] in ("o", "m"):
            ATTN, LAYER.oa, A2.FINISH_A, OPH = True, True, True, True
            MPH = args[0] == "m"
        elif args[0].startswith("nbw="):
            A2.NBW = int(args[0][4:])
        elif args[0] == "h":
            HEADSEG = True
        elif args[0] == "arena":
            ARENA = True
        elif args[0].startswith("split="):
            import rattnh
            SPLITS = tuple(sorted(int(v) for v in args[0][6:].split(",")))
            SPLIT_NB, rattnh.SPLIT = SPLITS[0], True
            globals()["NF"] = 11
        elif args[0].startswith("gcap="):
            import rsplitl
            GCAP = rsplitl.SCR_ROWS = int(args[0][5:])
        elif args[0].startswith("fwd="):
            r_, hd_ = args[0][4:].split("+")[0], args[0].endswith("+h")
            FWD = (int(r_.split("-")[0]), int(r_.split("-")[1]), hd_)
        elif args[0] == "s64":
            globals()["S64"] = True
        elif args[0].startswith("emit="):
            EMIT = set(args[0][5:].split(","))
        elif args[0].startswith("trace="):
            TRACE = [tuple(int(v) for v in cr.split(".")) for cr in args[0][6:].split(",")]
        elif args[0].startswith("sring="):
            SRING_FWD = int(args[0][6:])
        elif args[0].startswith("fnt="):
            FNTS = tuple(sorted((int(v.split(":")[0]), int(v.split(":")[1]) if ":" in v else None)
                                for v in args[0][4:].split(",")))
        elif args[0].startswith("fseg="):
            FSEG = tuple(sorted(int(v) for v in args[0][5:].split(",")))
        elif args[0].startswith("seg="):
            import rattnh
            SEG = tuple(sorted(int(v) for v in args[0][4:].split(",")))
            rattnh.RT_NBW = True
        elif args[0] == "g":
            assert os.environ.get("RF_FAST") == "1", "the global K norm passes 1/n (RF_FAST=1)"
            GLOB = True
        args = args[1:]
    NBW_S = A2.NBW
    # OA holds np*nt passes: a sliding layer's PCAP_T (np=1), a global layer's 2*nt for the largest nt
    # a global layer is emitted at (its own sequences and every forward family)
    OA_B = max(PCAP_T, 2 * max(RG().NTS + tuple(nt for nt, _ in FNTS)) if GLOB else 0) * 2 * A2.ABO + (A2.AB - A2.ABO)
    assert GCAP is None or all(nb * 64 <= GCAP for nb in SPLITS + SEG), (GCAP, SPLITS, SEG)
    assert set(f2_rungs()) <= set(SEG or f2_rungs()), (FSEG, SEG)
    # K008: a piece's rows live in every MemTile's resident buffers, sized for PCAP_T blocks (mt_layout)
    assert all(2 <= nt <= min(PCAP_T, A2.NTMAX) for nt, _ in FNTS), (FNTS, PCAP_T, A2.NTMAX)
    if GLOB:             # K008: the global layer's QK layout holds its largest piece
        ntg = max(RG().NTS + tuple(nt for nt, _ in FNTS))
        need = RG().size_for(ntg)
        if need > QKB:
            per = need // ntg
            sys.exit(f"error[K008]: the global layer's QK region needs {need} B for {16 * ntg}-row pieces "
                     f"({per} B a 16-row block: OUT {RG().TB}, two q planes, K plane); QK holds {QKB} B\n"
                     f"  --> fnt={','.join(str(nt) for nt, _ in FNTS)} with `g`\n"
                     f"  = note: the largest global piece that fits is {QKB // per * 16} rows")
    if SRING_FWD:        # K059 over every parameter value a host can pass, on the region rforward sizes
        import kvring
        import rforward
        kvring.check(SRING_FWD, NBW_S, (1,) + tuple(nt for nt, _ in FNTS), DEFAULT_WINDOW_ROWS,
                     rforward.s_rows(sys.modules[__name__]) * KVROW)
    return [int(a) for a in args]


if __name__ == "__main__":
    sys.stdout.write(emit([int(a) for a in sys.argv[1:]] or [1]))
