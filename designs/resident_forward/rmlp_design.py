"""M2 design, the whole MLP block in one dispatch: pre-FFN RMSNorm -> gate/up + GELU*up -> down ->
post-FFN RMSNorm -> residual add -> layer scalar, x resident in the MemTiles, weights streaming.

x arrives row-major bf16 (MemTile c holds columns c*480 .. +479) and leaves the same way. The
norm passes run on the idle weight blocks (rf_norm.cc): MemTile c broadcasts 8-row units up its
column (8640 B, the 9th row a spill the cores ignore); row-0 cores sum squares and reduce them
across columns on the row-0 cascade; core (0, 7) computes rstd and sends it on its core stream to
the scaling cores, (0, c) for c < 7 and (3, 7), which write back through their own DMA: xn2 as
the gate/up A blocks, then x_out in place of x. y_d lands row-major, so the post norm reads it
like x. Gains ride the weight stream (2 elements per core before gate/up and after down).
"""
import os
import sys
import rf_paths
from m1_design import M, core_name, NC, NR, KC, NB, ELEM, ABLK, OBLK, NBLK, NSUB, PCAP_T, HT, KERN
from m2_design import convert_loop, act_call, NDB, NSP, WB, DOWN_W, WCOL_GU, WCOL_D

NORM = "norm.o"
STANDIN = "standin.o"
D = 3840
XROW = 480 * 2                            # one row of a column slice, bf16
XB = 16 * XROW                            # one 16-row block
UNITB = 8 * XROW
XRB = (16 * PCAP_T + 1) * XROW            # resident x (+1 row: the last unit's spill)
XYB = max(PCAP_T * ABLK, XRB)             # xn2 blocks, then y_d rows
WGAIN = 2 * NR * ELEM
WCOL = WGAIN + WCOL_GU + WCOL_D + WGAIN
OUT_OFF = 2 * UNITB                       # rf_norm.cc: xn2 block, over the y slots


SS_LAST = "rf_ss_last"     # the chain end's sum-of-squares entry (a design may link another build)


def scale_core(c, r):
    return (r == 0 and c < NC - 1) or (c == NC - 1 and r == NR - 1)


def emit(nts, stack_mid=1024, stack_end=3072):
    m = M()
    m("module {", 0)
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
            m(f"aie.flow(%t_{core_name(7, r)}, DMA : {j}, %mem_{2 * r + j}, DMA : 2)")
        for c in range(NC - 1):
            m(f"aie.cascade_flow(%t_{core_name(c, r)}, %t_{core_name(c + 1, r)})")
    A, W, O, G = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", f"memref<{OBLK}xi8>", "memref<1024xi16>"
    for name, sig, obj in (
            ("chain_convert_half", f"memref<{ELEM}xi8>, {W}, i32, i32", KERN),
            ("chain_mm_first_r2", f"{A}, {W}", KERN), ("chain_mm_mid_r2", f"{A}, {W}", KERN),
            ("chain_mm_last_geluup_r2", f"{A}, {W}, {O}, {G}", KERN),
            ("chain_down_first_r2", f"{A}, {W}", KERN), ("chain_down_mid_r2", f"{A}, {W}", KERN),
            ("chain_down_last_r2", f"{A}, {W}, i32, i32", KERN),
            ("chain_down_emit_rows", f"{W}, {G}, i32, i32", KERN),
            ("rf_gain", f"memref<{ELEM}xi8>, {W}", NORM), ("rf_unit", f"{A}, {W}, i32", NORM),
            ("rf_ss_first", f"{W}, i32", NORM), ("rf_ss_mid", f"{W}, i32", NORM),
            ("rf_ss_last", f"{W}, i32", NORM), ("rf_rstd_recv", W, NORM),
            ("rf_pre_scale", W, NORM), ("rf_post_scale", W, NORM)) + ((
            ("attn_standin_qk", "memref<45056xi8>, i32", STANDIN), ("attn_standin_pv", "memref<41088xi8>, i32", STANDIN),
            ("attn_standin_sm", "memref<26048xi8>, i32", STANDIN)) if ATT else ()):
        m(f'func.func private @{name}({sig}) attributes {{link_with = "{obj}"}}')
    for c in range(NC):
        memtile(m, c)
    for c in range(NC):
        for r in range(NR):
            core(m, c, r, stack_end if c == NC - 1 else stack_mid)
    for k, (c, r) in enumerate(TRACE):
        m(f"aie.trace @tr_{c}_{r}(%t_{core_name(c, r)}) {{")
        m('aie.trace.mode "Event-Time"', 6)
        m(f"aie.trace.packet id={k + 1} type=core", 6)
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
    for nt in BLOCKS2:
        sequence(m, nt, 2)
    for nt in SWITCH_NTS:
        for k in (2, 4):
            sequence(m, nt, 2, switches=k)
    for nt in ATT_NTS:
        sequence(m, nt, 2, attn=True)
    for nt in RES_NTS:
        for r in ("in", "mid", "out"):
            sequence(m, nt, res=r)
    m("}", 2)
    if SWITCH_NTS:
        m("aie.device(npu2) @dummy {", 2)
        m("%t02 = aie.tile(0, 2)")
        m("%core = aie.core(%t02) {")
        m("aie.end", 6)
        m("}")
        m("}", 2)
    m("}", 0)
    return "\n".join(m.lines) + "\n"


CAN = 64                 # R6 canary word block per MemTile (pinned, ungrouped)
CAN_ADDR = 472704        # the first free byte after the pinned layout's larger group


def XBUF():
    return 16 * PCAP_T * D * 2 + (NC * CAN if RES_NTS else 0)


def OB():
    return 16 * PCAP_T * D * 2 + (NC * ATT_MT if ATT else 0) + (NC * CAN if RES_NTS else 0)


def args_sig():
    return f"%x : memref<{XBUF()}xi8>, %w : memref<{NC * WCOL}xi8>, %o : memref<{OB()}xi8>"


def memtile(m, c):
    # with ATT the MemTile layout is pinned (R6): resident x first, then the gemm and attn phases
    # overlaid from one base (objectFIFO allocation's capacity check sums buffers and ignores
    # alloc_group, so unpinned groups are refused as over capacity)
    g = ', alloc_group = "gemm"' if ATT else ""
    at = (lambda a: f", address = {a} : i32") if ATT else (lambda a: "")
    base = XRB
    m(f"%wst_{c} = aie.buffer(%mem_{c}) {{sym_name = \"wst_{c}\"{g}{at(base)}}} : memref<{2 * NR * ELEM}xi8>")
    m(f"%XR_{c} = aie.buffer(%mem_{c}) {{sym_name = \"XR_{c}\"{at(0)}}} : memref<{XRB}xi8>")
    m(f"%XY_{c} = aie.buffer(%mem_{c}) {{sym_name = \"XY_{c}\"{g}{at(base + 2 * NR * ELEM)}}} : memref<{XYB}xi8>")
    m(f"%H_{c} = aie.buffer(%mem_{c}) {{sym_name = \"H_{c}\"{g}{at(base + 2 * NR * ELEM + XYB)}}} : memref<{PCAP_T * HT}xi8>")
    if ATT:
        m(f"%ATT_{c} = aie.buffer(%mem_{c}) {{sym_name = \"ATT_{c}\", alloc_group = \"attn\"{at(base)}}} : memref<{ATT_MT}xi8>")
        m(f"%af_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"af_{c}\"}}")
        m(f"%aw_{c} = aie.lock(%mem_{c}) {{init = 1 : i32, sym_name = \"aw_{c}\"}}")
        if RES_NTS:
            m(f"%CAN_{c} = aie.buffer(%mem_{c}) {{sym_name = \"CAN_{c}\", address = {CAN_ADDR} : i32}} : memref<{CAN}xi8>")
            m(f"%cq_{c} = aie.lock(%mem_{c}) {{init = 1 : i32, sym_name = \"cq_{c}\"}}")
            m(f"%cf_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"cf_{c}\"}}")
    for r in range(NR):
        m(f"%wp_{c}_{r} = aie.lock(%mem_{c}) {{init = 2 : i32, sym_name = \"wp_{c}_{r}\"}}")
        m(f"%wc_{c}_{r} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"wc_{c}_{r}\"}}")
    for nm in ("xf", "xp", "xw", "xr", "hf", "hr", "yf", "yr", "ow", "od"):
        m(f"%{nm}_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"{nm}_{c}\"}}")
    m(f"%mdma_{c} = aie.memtile_dma(%mem_{c}) {{")
    m("%one = arith.constant 1 : i32", 6)
    m("%s0 = aie.dma_start(S2MM, 0, ^w0, ^m0)", 6)
    k = 0
    for s in range(2):
        for r in range(NR):
            nxt = f"^w{k + 1}" if k < 2 * NR - 1 else "^w0"
            m(f"^w{k}:", 4)
            m(f"aie.use_lock(%wp_{c}_{r}, AcquireGreaterEqual, %one)", 6)
            m(f"aie.dma_bd(%wst_{c} : memref<{2 * NR * ELEM}xi8> offset = {(s * NR + r) * ELEM} len = {ELEM})", 6)
            m(f"aie.use_lock(%wc_{c}_{r}, Release, %one)", 6)
            m(f"aie.next_bd {nxt}", 6)
            k += 1
    for r in range(NR):
        end = f"^m{r + 1}" if r < NR - 1 else "^end"
        m(f"^m{r}:", 4)
        m(f"%d{r} = aie.dma_start(MM2S, {r}, ^r{r}a, {end})", 6)
        for s, lab, nxt in ((0, "a", "b"), (1, "b", "a")):
            m(f"^r{r}{lab}:", 4)
            m(f"aie.use_lock(%wc_{c}_{r}, AcquireGreaterEqual, %one)", 6)
            m(f"aie.dma_bd(%wst_{c} : memref<{2 * NR * ELEM}xi8> offset = {(s * NR + r) * ELEM} len = {ELEM})", 6)
            m(f"aie.use_lock(%wp_{c}_{r}, Release, %one)", 6)
            m(f"aie.next_bd ^r{r}{nxt}", 6)
    m("^end:", 4)
    m("aie.end", 6)
    m("}")


def core_rings(m, t):
    """The static S2MM rings: weights (one element per BD) and activations (one unit per BD)."""
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


def core(m, c, r, stack):
    t = core_name(c, r)
    last = c == NC - 1
    g = ', alloc_group = "gemm"' if ATT else ""
    m(f"%wblk_{t} = aie.buffer(%t_{t}) {{sym_name = \"wblk_{t}\"{g}}} : memref<{WB}xi8>")
    m(f"%rtp_{t} = aie.buffer(%t_{t}) {{sym_name = \"rtp_{t}\"}} : memref<4xi32>")
    for i in range(2):
        m(f"%wb{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"wb{i}_{t}\"{g}}} : memref<{ELEM}xi8>")
        m(f"%ab{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ab{i}_{t}\"{g}}} : memref<{ABLK}xi8>")
    if ATT:
        m(f"%ATTL_{t} = aie.buffer(%t_{t}) {{sym_name = \"ATTL_{t}\", alloc_group = \"attn\"}} : memref<{ATT_L1[r][1]}xi8>")
    for nm, init in (("wp", 2), ("wc", 0), ("ap", 2), ("ac", 0), ("go", 0)):
        m(f"%c{nm}_{t} = aie.lock(%t_{t}) {{init = {init} : i32, sym_name = \"c{nm}_{t}\"}}")
    if scale_core(c, r):
        m(f"%cxp_{t} = aie.lock(%t_{t}) {{init = 1 : i32, sym_name = \"cxp_{t}\"}}")
        m(f"%cxc_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"cxc_{t}\"}}")
    if last:
        m(f"%gscr_{t} = aie.buffer(%t_{t}) {{sym_name = \"gscr_{t}\"{g}}} : memref<1024xi16>")
        for j in range(2):
            for i in range(2):
                m(f"%ob{j}{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ob{j}{i}_{t}\"{g}}} : memref<{OBLK}xi8>")
            m(f"%cop{j}_{t} = aie.lock(%t_{t}) {{init = 2 : i32, sym_name = \"cop{j}_{t}\"}}")
            m(f"%coc{j}_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"coc{j}_{t}\"}}")
            m(f"%cyc{j}_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"cyc{j}_{t}\"}}")
        m(f"%cyp_{t} = aie.lock(%t_{t}) {{init = 2 : i32, sym_name = \"cyp_{t}\"}}")
    if not TASKRING:
        core_rings(m, t)
    if HOLLOW:      # reload attribution: same tiles and DMA/lock/stream config, no program
        m(f"%core_{t} = aie.core(%t_{t}) {{")
        m("aie.end", 6)
        m("}")
        return
    core_body(m, c, r, last)
    m(f"}} {{stack_size = {stack} : i32}}")


def gains(m, t, I):
    """Two weight elements: the first carries the gain slice and the layer scalar."""
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    m(f"func.call @rf_gain(%wb0_{t}, %wblk_{t}) : (memref<{ELEM}xi8>, memref<{WB}xi8>) -> ()", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)


def norm_pass(m, c, r, I, post):
    """Per 16-row block: units in (pre x0 x1; post x0 y0 x1 y1), sum of squares on the row-0
    cascade, rstd to the scaling cores, scale and hand the block to the write-back DMA."""
    t = core_name(c, r)
    W, A = f"memref<{WB}xi8>", f"memref<{ABLK}xi8>"
    partial, scale = r == 0, scale_core(c, r)
    slots = (0, 2, 1, 3) if post else (0, 1)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    J = I + 2
    for k, slot in enumerate(slots):
        if post and scale and k == 0:
            m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", J)
        m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", J)
        if partial or scale:
            m(f"%sl{k} = arith.constant {slot} : i32", J)
            m(f"func.call @rf_unit(%ab{k % 2}_{t}, %wblk_{t}, %sl{k}) : ({A}, {W}, i32) -> ()", J)
        m(f"aie.use_lock(%cap_{t}, Release, %one)", J)
    if partial:
        fn = "rf_ss_first" if c == 0 else (SS_LAST if c == NC - 1 else "rf_ss_mid")
        m(f"%src = arith.constant {2 if post else 0} : i32", J)
        m(f"func.call @{fn}(%wblk_{t}, %src) : ({W}, i32) -> ()", J)
    if scale:
        m(f"func.call @rf_rstd_recv(%wblk_{t}) : ({W}) -> ()", J)
        if not post:
            m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", J)
        m(f"func.call @{'rf_post_scale' if post else 'rf_pre_scale'}(%wblk_{t}) : ({W}) -> ()", J)
        m(f"aie.use_lock(%cxc_{t}, Release, %one)", J)
    m("}", I)
    if post and scale and ATT:
        # the last x_out block may still be leaving through this core's DMA; the next phase
        # overlays that storage, so wait for it (K050)
        m(f"aie.use_lock(%cxp_{t}, AcquireGreaterEqual, %one)", I)
        m(f"aie.use_lock(%cxp_{t}, Release, %one)", I)


def core_body(m, c, r, last):
    t = core_name(c, r)
    A, W, O, G = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", f"memref<{OBLK}xi8>", "memref<1024xi16>"
    mm = "chain_mm_first_r2" if c == 0 else ("chain_mm_last_geluup_r2" if last else "chain_mm_mid_r2")
    m(f"%core_{t} = aie.core(%t_{t}) {{")
    I = 6
    for v in (0, 1, 2, 3, 4, 5, 10, 15, 30, 60, 120):
        m(f"%c{v} = arith.constant {v} : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%two = arith.constant 2 : i32", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%cgo_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    if ATT:
        # one go per dispatch; rtp[1] is the phase mask (bit k: phase k is attention), rtp[2] the
        # phase count. alloc_group needs the two phases in exclusive branches of one selector.
        m(f"%sched = memref.load %rtp_{t}[%c1] : memref<4xi32>", I)
        m(f"%nph32 = memref.load %rtp_{t}[%c2] : memref<4xi32>", I)
        m("%nph = arith.index_cast %nph32 : i32 to index", I)
        m("scf.for %k = %c0 to %nph step %c1 {", I)
        m("%k32 = arith.index_cast %k : index to i32", I + 2)
        m("%sh = arith.shrui %sched, %k32 : i32", I + 2)
        m("%bit = arith.andi %sh, %one : i32", I + 2)
        m("%ph = arith.index_cast %bit : i32 to index", I + 2)
        m("scf.index_switch %ph", I + 2)
        m("case 1 {", I + 2)
        role, size = ATT_L1[r]
        m(f"func.call @attn_standin_{role}(%ATTL_{t}, %k32) : (memref<{size}xi8>, i32) -> ()", I + 4)
        m("scf.yield", I + 4)
        m("}", I + 2)
        m("default {", I + 2)
        I += 4
    gains(m, t, I)
    norm_pass(m, c, r, I, post=False)
    # gate/up (element and activation parities are unchanged: both passes above consume evens)
    for oj, cnt in ([(None, 120)] if not last else [(0, 60), (1, 60)]):
        m(f"scf.for %n = %c0 to %c{cnt} step %c1 {{", I)
        I += 2
        m("%n10 = arith.muli %n, %c10 : index", I)
        convert_loop(m, t, I, "%n10", 10, True)
        m("%nnt = arith.muli %n, %nt : index", I)
        m("scf.for %tb = %c0 to %nt step %c1 {", I)
        I += 2
        m("%idx = arith.addi %nnt, %tb : index", I)
        if last:
            m(f"aie.use_lock(%cop{oj}_{t}, AcquireGreaterEqual, %one)", I)

        def gu(i, oj=oj):
            args, sig = f"%ab{i}_{t}, %wblk_{t}", f"{A}, {W}"
            if last:
                args += f", %ob{oj}{i}_{t}, %gscr_{t}"
                sig += f", {O}, {G}"
            return [f"func.call @{mm}({args}) : ({sig}) -> ()"]
        act_call(m, t, I, "%idx", gu)
        if last:
            m(f"aie.use_lock(%coc{oj}_{t}, Release, %one)", I)
        I -= 2
        m("}", I)
        I -= 2
        m("}", I)
    # down
    dmm = "chain_down_first_r2" if c == 0 else ("chain_down_last_r2" if last else "chain_down_mid_r2")
    m("scf.for %n = %c0 to %c30 step %c1 {", I)
    I += 2
    m("scf.for %s = %c0 to %c4 step %c1 {", I)
    I += 2
    m("%ns = arith.muli %n, %c4 : index", I)
    m("%nsi = arith.addi %ns, %s : index", I)
    m("%ns5 = arith.muli %nsi, %c5 : index", I)
    convert_loop(m, t, I, "%ns5", 5, False)
    m("%s32 = arith.index_cast %s : index to i32", I)
    m("%nsnt = arith.muli %nsi, %nt : index", I)
    m("scf.for %tb = %c0 to %nt step %c1 {", I)
    I += 2
    m("%idx = arith.addi %nsnt, %tb : index", I)
    m("%tb32 = arith.index_cast %tb : index to i32", I)

    def dn(i):
        if last:
            return [f"func.call @{dmm}(%ab{i}_{t}, %wblk_{t}, %tb32, %s32) : ({A}, {W}, i32, i32) -> ()"]
        return [f"func.call @{dmm}(%ab{i}_{t}, %wblk_{t}) : ({A}, {W}) -> ()"]
    act_call(m, t, I, "%idx", dn)
    I -= 2
    m("}", I)
    if last:
        # y_d rows: two 1024 B buffers in gscr, alternating per emitted block; channel 1 starts
        # its own alternation at n = 15, so both buffers drain first
        m("%s3 = arith.cmpi eq, %s, %c3 : index", I)
        m("scf.if %s3 {", I)
        J = I + 2
        m("%sw = arith.cmpi eq, %n, %c15 : index", J)
        m("scf.if %sw {", J)
        m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %two)", J + 2)
        m(f"aie.use_lock(%cyp_{t}, Release, %two)", J + 2)
        m("}", J)
        m("%lo = arith.cmpi ult, %n, %c15 : index", J)
        m("%n15 = arith.subi %n, %c15 : index", J)
        m("%nl = arith.select %lo, %n, %n15 : index", J)
        m("%nlnt = arith.muli %nl, %nt : index", J)
        m("scf.for %tb = %c0 to %nt step %c1 {", J)
        K = J + 2
        m("%e = arith.addi %nlnt, %tb : index", K)
        m("%ep = arith.remui %e, %c2 : index", K)
        m("%ep32 = arith.index_cast %ep : index to i32", K)
        m("%tb32 = arith.index_cast %tb : index to i32", K)
        m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %one)", K)
        m(f"func.call @chain_down_emit_rows(%wblk_{t}, %gscr_{t}, %tb32, %ep32) : ({W}, {G}, i32, i32) -> ()", K)
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
    gains(m, t, I)
    norm_pass(m, c, r, I, post=True)
    if ATT:
        m("scf.yield", I)
        I -= 4
        m("}", I + 2)
        m("}", I)
    I -= 2
    m("}", I)
    m("aie.end", I)


def sequence(m, nt, blocks=1, part=None, switches=0, attn=False, res=None):
    """Control code for nt 16-row blocks; blocks=2 runs the MLP block twice in one dispatch, the
    second on the resident x_out (arm B), with an await between the two so no queue exceeds four
    tasks. part="b1" / "b2" split that pair into two sequences for aiex.run (M3's configure
    sweep): b1 takes x in and ends on the block's token, b2 starts from the resident x."""
    X = nt * XB
    name = part if part else ("p" if blocks == 1 else ("a" if attn else ("q" if not switches else f"s{switches}_")))
    if res:
        name = f"r{res}_"
    m(f"aie.runtime_sequence @{name}{'_' if part else ''}{nt}({args_sig()}) {{")
    I = 6
    if TRACE:
        m(f"aie.trace.host_config {{buffer_size = {TRACE_BYTES} : i32}}", I)
        for c, r in TRACE:
            m(f"aie.trace.start_config @tr_{c}_{r}", I)
    m(f"%ntv = arith.constant {nt} : i32", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 0, %ntv) : i32", I)
    if ATT:
        sched, nph = (0b010, 3) if attn else (0, blocks)
        m(f"%sched = arith.constant {sched} : i32", I)
        m(f"%nph = arith.constant {nph} : i32", I)
        for c in range(NC):
            for r in range(NR):
                m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 1, %sched) : i32", I)
                m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 2, %nph) : i32", I)
    for k in range(PAD_WRITES if blocks == 1 else 0):     # control-code size probe
        m(f"aiex.npu.rtp_write(@rtp_{core_name(k % NC, 0)}, 1, %ntv) : i32", I)
    for c in range(NC):
        for nm, v in (("xf", 1), ("xp", 1 if part == "b2" or res in ("mid", "out") else 0), ("xw", 1), ("xr", 0), ("hf", 1), ("hr", 0),
                      ("yf", 15), ("yr", 0), ("ow", 1), ("od", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for c in range(NC):
        for r in range(NR):
            t = core_name(c, r)
            if scale_core(c, r):
                m(f"aiex.set_lock(%cxp_{t}, 1)", I)
                m(f"aiex.set_lock(%cxc_{t}, 0)", I)
    for r in range(NR):
        t = core_name(NC - 1, r)
        for j in range(2):
            m(f"aiex.set_lock(%cop{j}_{t}, 2)", I)
            m(f"aiex.set_lock(%coc{j}_{t}, 0)", I)
            m(f"aiex.set_lock(%cyc{j}_{t}, 0)", I)
        m(f"aiex.set_lock(%cyp_{t}, 2)", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.set_lock(%cgo_{core_name(c, r)}, {1 if ATT else blocks})", I)

    issued = []

    def task(name, tile, d, ch, body, attrs=""):
        issued.append(name)
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m("%v15 = arith.constant 15 : i32", I + 2)
        for line in body:
            m(line, I + 2)
        m("aie.end", I + 2)
        m("}" + attrs, I)
        m(f"aiex.dma_start_task(%{name})", I)

    def locked(acq, bd, rel, v="%one"):
        return [f"aie.use_lock(%{acq}, AcquireGreaterEqual, {v})", bd, f"aie.use_lock(%{rel}, Release, {v})"]

    def chain(parts):
        out = []
        for i, p in enumerate(parts):
            if i:
                out += [f"aie.next_bd ^b{i}", f"^b{i}:"]
            out += p
        return out

    def units(buf, n, bd):
        return (f"aie.dma_bd({buf} offset = 0 len = {UNITB + XROW} sizes = [{n}, 3, 2, {(UNITB + XROW) // 6}] "
                f"strides = [{UNITB}, {(UNITB + XROW) // 3}, {(UNITB + XROW) // 6}, 1]) {{bd_id = {bd} : i32}}")

    def block_tasks(b):
        sfx = f"_{b}"
        for c in range(NC):
            XR, XY, H = f"%XR_{c} : memref<{XRB}xi8>", f"%XY_{c} : memref<{XYB}xi8>", f"%H_{c} : memref<{PCAP_T * HT}xi8>"
            wbq = (3, 30, 31) if c < NC - 1 else (2, 22, 23)   # write-back channel, xn2 and x_out BDs
            if b == 0 and part != "b2" and res not in ("mid", "out"):
                task(f"xin{c}", f"mem_{c}", "S2MM", 1, locked(f"xf_{c}", f"aie.dma_bd({XR} offset = 0 len = {X}) {{bd_id = 28 : i32}}", f"xp_{c}"))
            task(f"nb{c}{sfx}", f"mem_{c}", "MM2S", 4, locked(f"xp_{c}",
                 f"aie.dma_bd({XR} offset = 0 len = {2 * nt * (UNITB + XROW)} sizes = [{2 * nt}, 3, {(UNITB + XROW) // 3}] "
                 f"strides = [{UNITB}, {(UNITB + XROW) // 3}, 1]) {{bd_id = 12 : i32}}", f"xp_{c}"))
            task(f"xl{c}{sfx}", f"mem_{c}", "S2MM", wbq[0], locked(f"xw_{c}", f"aie.dma_bd({XY} offset = 0 len = {nt * ABLK}) {{bd_id = {wbq[1]} : i32}}", f"xr_{c}"))
            task(f"bc{c}{sfx}", f"mem_{c}", "MM2S", 4, locked(f"xr_{c}", f"aie.dma_bd({XY} offset = 0 len = {nt * ABLK}) {{bd_id = 13 : i32}}", f"xr_{c}"),
                 f" {{repeat_count = {NBLK - 1} : i32}}")
            hb = [locked(f"hr_{c}", f"aie.dma_bd({H} offset = {sp * ABLK} len = {nt * ABLK} sizes = [{nt}, 3, {ABLK // 3}] strides = [{HT}, {ABLK // 3}, 1]) {{bd_id = {14 + sp} : i32}}", f"hr_{c}")
                  for sp in range(NSP)]
            task(f"hb{c}{sfx}", f"mem_{c}", "MM2S", 4, chain(hb), f" {{repeat_count = {NDB - 1} : i32}}")
            task(f"ld{c}{sfx}", f"mem_{c}", "S2MM", 2, locked(f"hf_{c}", f"aie.dma_bd({H} offset = 0 len = {60 * nt * OBLK} sizes = [60, {nt}, {OBLK}] strides = [{OBLK}, {HT}, 1]) {{bd_id = 20 : i32}}", f"hr_{c}"))
            task(f"ly{c}{sfx}", f"mem_{c}", "S2MM", 2, locked(f"yf_{c}",
                 f"aie.dma_bd({XY} offset = 0 len = {nt * 16 * 64} sizes = [15, {nt}, 16, 64] strides = [64, {XB}, {XROW}, 1]) {{bd_id = 21 : i32}}", f"yr_{c}"),
                 " {repeat_count = 14 : i32}")
            task(f"pp{c}{sfx}", f"mem_{c}", "MM2S", 4, chain([locked(f"xp_{c}", units(XR, 2 * nt, 18), f"xp_{c}"),
                                                              locked(f"yr_{c}", units(XY, 2 * nt, 19), f"yr_{c}", "%v15")]),
                 f" {{repeat_count = {2 * nt - 1} : i32}}")
            task(f"ol{c}{sfx}", f"mem_{c}", "S2MM", wbq[0], locked(f"ow_{c}", f"aie.dma_bd({XR} offset = 0 len = {X}) {{bd_id = {wbq[2]} : i32}}", f"od_{c}"))
            if b < blocks - 1 or part == "b1" or res in ("in", "mid"):
                # the end of a block, seen at the shim: 64 B of x_out once it has landed
                task(f"tk{c}{sfx}", f"mem_{c}", "MM2S", 5, locked(f"od_{c}", f"aie.dma_bd({XR} offset = 0 len = 64) {{bd_id = 27 : i32}}", f"od_{c}"))
                task(f"st{c}{sfx}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{OB()}xi8> offset = {c * XROW} len = 64)"],
                     " {issue_token = true}")

        def wb_tasks(t, ch, bds):
            W_ = f"%wblk_{t} : memref<{WB}xi8>"
            task(f"wx_{t}{sfx}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd({W_} offset = {OUT_OFF} len = {ABLK}) {{bd_id = {bds[0]} : i32}}", f"cxp_{t}"),
                 f" {{repeat_count = {nt - 1} : i32}}")
            return lambda: task(f"wo_{t}{sfx}", f"t_{t}", "MM2S", ch, locked(f"cxc_{t}", f"aie.dma_bd({W_} offset = 0 len = {XB}) {{bd_id = {bds[1]} : i32}}", f"cxp_{t}"),
                                f" {{repeat_count = {nt - 1} : i32}}")

        later = []
        for c in range(NC - 1):
            later.append(wb_tasks(core_name(c, 0), 0, (8, 9)))
        for r in range(NR):
            t = core_name(NC - 1, r)
            if r == NR - 1:
                later.append(wb_tasks(t, 1, (14, 15)))
            for j in range(2):
                o0, o1 = f"%ob{j}0_{t} : memref<{OBLK}xi8>", f"%ob{j}1_{t} : memref<{OBLK}xi8>"
                body = chain([locked(f"coc{j}_{t}", f"aie.dma_bd({o0} offset = 0 len = {OBLK}) {{bd_id = {8 + 2 * j} : i32}}", f"cop{j}_{t}"),
                              locked(f"coc{j}_{t}", f"aie.dma_bd({o1} offset = 0 len = {OBLK}) {{bd_id = {9 + 2 * j} : i32}}", f"cop{j}_{t}")])
                task(f"go{j}_{r}{sfx}", f"t_{t}", "MM2S", j, body, f" {{repeat_count = {30 * nt - 1} : i32}}")
                task(f"yd{j}_{r}{sfx}", f"t_{t}", "MM2S", j,
                     locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = 0 len = 512 sizes = [2, 2, 2, 128] strides = [512, 256, 128, 1]) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                     f" {{repeat_count = {15 * nt - 1} : i32}}")
        for f in later:
            f()

    # the shim inputs go first: an await between blocks would otherwise wait on a block that has
    # no weights or x yet
    xb = 16 * PCAP_T * D * 2
    for c in range(NC):
        task(f"sw{c}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{NC * WCOL}xi8> offset = {c * WCOL} len = {WCOL})"],
             f" {{repeat_count = {blocks - 1} : i32}}" if blocks > 1 and not switches and not attn else "")
        if part != "b2" and res not in ("mid", "out"):
            task(f"sx{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{XBUF()}xi8> offset = {c * XROW} len = {X} sizes = [{16 * nt}, {XROW}] strides = [{D * 2}, 1])"])
    if res:
        # R6: the first thing each dispatch does to the MemTile is read the canary the previous
        # dispatch left; the last is to leave its own (host-chosen, from the tail of %x)
        for c in range(NC):
            m(f"aiex.set_lock(%cq_{c}, 1)", I)
            m(f"aiex.set_lock(%cf_{c}, {0 if res in ('mid', 'out') else 1})", I)
        if res in ("mid", "out"):
            for c in range(NC):
                CN = f"%CAN_{c} : memref<{CAN}xi8>"
                task(f"cr{c}", f"mem_{c}", "MM2S", 5, locked(f"cq_{c}", f"aie.dma_bd({CN} offset = 0 len = {CAN}) {{bd_id = 34 : i32}}", f"cf_{c}"))
                task(f"cs{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{OB()}xi8> offset = {OB() - NC * CAN + c * CAN} len = {CAN})"])
    for b in range(blocks):
        if b:
            for c in range(NC):
                m(f"aiex.dma_await_task(%st{c}_{b - 1})", I)
            for name in issued:
                if name.endswith(f"_{b - 1}") and not name.startswith("st"):
                    m(f"aiex.dma_free_task(%{name})", I)
            if switches:
                # M3 configure sweep: reload through the dummy design and back; the reload restarts
                # the cores, so block 2 gets its own weight stream and a fresh go
                for name in issued:
                    if name.startswith("sw") and "_" not in name:
                        m(f"aiex.dma_free_task(%{name})", I)
                for _ in range(switches // 2):
                    m("aiex.npu.load_pdi {device_ref = @dummy}", I)
                    m("aiex.npu.load_pdi {device_ref = @main}", I)
                for c in range(NC):
                    for r in range(NR):
                        m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 0, %ntv) : i32", I)
                        m(f"aiex.set_lock(%cgo_{core_name(c, r)}, 1)", I)
                for c in range(NC):
                    task(f"sw{c}_{b}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{NC * WCOL}xi8> offset = {c * WCOL} len = {WCOL})"])
            if attn:
                # the stand-in attention phase's MemTile traffic: K/V-sized data through the
                # overlaid ATT region and back out, so the overlay is exercised on both levels
                ob = OB()
                for c in range(NC):
                    A_ = f"%ATT_{c} : memref<{ATT_MT}xi8>"
                    m(f"aiex.set_lock(%aw_{c}, 1)", I)
                    m(f"aiex.set_lock(%af_{c}, 0)", I)
                    task(f"ai{c}", f"mem_{c}", "S2MM", 1, locked(f"aw_{c}", f"aie.dma_bd({A_} offset = 0 len = {ATT_MT}) {{bd_id = 32 : i32}}", f"af_{c}"))
                    task(f"ao{c}", f"mem_{c}", "MM2S", 5, locked(f"af_{c}", f"aie.dma_bd({A_} offset = 0 len = {ATT_MT}) {{bd_id = 33 : i32}}", f"aw_{c}"))
                    task(f"sa{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%w : memref<{NC * WCOL}xi8> offset = {c * WCOL} len = {ATT_MT})"])
                    task(f"sb{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{ob}xi8> offset = {16 * PCAP_T * D * 2 + c * ATT_MT} len = {ATT_MT})"],
                         " {issue_token = true}")
                for c in range(NC):
                    m(f"aiex.dma_await_task(%sb{c})", I)
                for c in range(NC):
                    for nm in ("ai", "ao", "sa"):
                        m(f"aiex.dma_free_task(%{nm}{c})", I)
                # block 2's weights only after the attention phase: its first elements land in core
                # buffers the attention phase overlays (a streaming prefetch here corrupted them)
                for name in issued:
                    if name.startswith("sw") and "_" not in name:
                        m(f"aiex.dma_free_task(%{name})", I)
                for c in range(NC):
                    task(f"sw{c}_{b}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{NC * WCOL}xi8> offset = {c * WCOL} len = {WCOL})"])
            for c in range(NC):
                for nm, v in (("xp", 1), ("xw", 1), ("xr", 0), ("hf", 1), ("hr", 0),
                              ("yf", 15), ("yr", 0), ("ow", 1), ("od", 0)):
                    m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
        block_tasks(b)
    if res in ("in", "mid"):
        for c in range(NC):
            CN = f"%CAN_{c} : memref<{CAN}xi8>"
            task(f"cw{c}", f"mem_{c}", "S2MM", 1, locked(f"cf_{c}", f"aie.dma_bd({CN} offset = 0 len = {CAN}) {{bd_id = 35 : i32}}", f"cq_{c}"))
            task(f"cx{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{XBUF()}xi8> offset = {XBUF() - NC * CAN + c * CAN} len = {CAN})"])
        for c in range(NC):
            m(f"aiex.dma_await_task(%st{c}_0)", I)
        m("}", 4)
        return
    if part == "b1":
        for c in range(NC):
            m(f"aiex.dma_await_task(%st{c}_0)", I)
        for name in issued:
            if not name.startswith("st"):
                m(f"aiex.dma_free_task(%{name})", I)
        m("}", 4)
        return
    if TASKRING:
        ring_tasks(m, nt, I, task, locked, chain)
    for c in range(NC):
        XR = f"%XR_{c} : memref<{XRB}xi8>"
        task(f"xo{c}", f"mem_{c}", "MM2S", 5, locked(f"od_{c}", f"aie.dma_bd({XR} offset = 0 len = {X}) {{bd_id = 29 : i32}}", f"ow_{c}"))
    for c in range(NC):
        task(f"so{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%o : memref<{OB()}xi8> offset = {c * XROW} len = {X} sizes = [{16 * nt}, {XROW}] strides = [{D * 2}, 1])"],
             " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%so{c})", I)
    m("}", 4)


RING_CHUNK = 512          # 2 BDs x (repeat_count 255 + 1): the most one task moves


def ring_tasks(m, nt, I, task, locked, chain):
    """The S2MM rings as runtime tasks. Per core and channel the dispatch's element / unit count is
    cut into tasks of at most RING_CHUNK, alternating two BD pairs; a pair is reconfigured only
    after the task that last used it has completed (awaited), round robin over the cores."""
    assert not ATT and not RES_NTS and not TRACE
    n_w = 2 + NBLK * 2 * NSUB + NDB * NSP * NSUB + 2
    n_a = (2 + NBLK + NDB * NSP + 4) * nt
    plan = []
    for ch, buf, acq, rel, ln, n, ids in ((0, "wb", "cwp", "cwc", ELEM, n_w, ((0, 1), (2, 3))),
                                          (1, "ab", "cap", "cac", ABLK, n_a, ((4, 5), (6, 7)))):
        chunks = [RING_CHUNK] * (n // RING_CHUNK) + ([n % RING_CHUNK] if n % RING_CHUNK else [])
        assert all(k % 2 == 0 for k in chunks)
        plan.append((ch, buf, acq, rel, ln, chunks, ids))
    steps = max(len(p[5]) for p in plan)
    for k in range(steps):
        for c in range(NC):
            for r in range(NR):
                t = core_name(c, r)
                for ch, buf, acq, rel, ln, chunks, ids in plan:
                    if k >= len(chunks):
                        continue
                    if k >= 2:
                        m(f"aiex.dma_await_task(%rg{ch}_{k - 2}_{t})", I)
                    a, b = ids[k % 2]
                    bds = [locked(f"{acq}_{t}", f"aie.dma_bd(%{buf}{i}_{t} : memref<{ln}xi8> offset = 0 len = {ln}) {{bd_id = {bd} : i32}}", f"{rel}_{t}")
                           for i, bd in ((0, a), (1, b))]
                    task(f"rg{ch}_{k}_{t}", f"t_{t}", "S2MM", ch, chain(bds),
                         f" {{issue_token = true, repeat_count = {chunks[k] // 2 - 1} : i32}}")
    for k in range(max(0, steps - 2), steps):
        for c in range(NC):
            for r in range(NR):
                for ch, *_ , chunks, ids in plan:
                    if k < len(chunks):
                        m(f"aiex.dma_await_task(%rg{ch}_{k}_{core_name(c, r)})", I)


def kernels():
    return [(KERN, os.environ.get("RF_CHAIN_SRC", rf_paths.iron_kernel("chain_mm_bfp16.cc")), ["-DKC=480", "-DNB=64", "-DSUB_K=96"]),
            (NORM, os.environ.get("RF_NORM_SRC", rf_paths.iron_kernel("rf_norm.cc")), os.environ.get("RF_NORM_DEFS", "").split())] + (
            [(STANDIN, os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_standin.cc"), [])])


def aiecc_options():
    """fabric-differential-reconfig: --expand-load-pdis so a load_pdi reload is patchable ctrltext
    bytes (only if RF_EXPAND_PDIS=1 -- other callers of this module must not change behavior)."""
    import os as _os
    return ["--expand-load-pdis"] if _os.environ.get("RF_EXPAND_PDIS") else []


HOLLOW = False
TASKRING = False      # the S2MM rings as runtime tasks (the one-image core-DMA probe)
TRACE = []            # step 4: (col, row) cores to trace
TRACE_BYTES = int(os.environ.get("RF_TRACE_BYTES", 1 << 20))
TRACE_EVENTS = tuple(os.environ.get("RF_TRACE_EVENTS", "ACTIVE,CASCADE_STALL,LOCK_STALL,STREAM_STALL,"
                     "MEMORY_STALL,INSTR_VECTOR,INSTR_CASCADE_GET,INSTR_CASCADE_PUT").split(","))
ATT = False           # rf-one-image-overlay: stand-in attention phase, alloc_group overlay
ATT_MT = 147456       # FusedAttnX2 hd 256 MemTile staging (K, Q, V0, V1)
ATT_L1 = (("qk", 45056), ("pv", 41088), ("pv", 41088), ("sm", 26048))   # by core row: x V on rows 3-4
BLOCKS2 = []
SWITCH_NTS = []
ATT_NTS = []
RES_NTS = []
PAD_WRITES = 0


def build_text(args):
    global BLOCKS2, PAD_WRITES, SWITCH_NTS, ATT_NTS, ATT, TRACE, RES_NTS, TASKRING
    if args and args[0] == "tr":
        TASKRING, args = True, args[1:]
    if "res" in args:
        i = args.index("res")
        RES_NTS = [int(a) for a in args[i + 1:]]
        args = args[:i]
    if args and args[0].startswith("trace="):
        TRACE = [tuple(int(v) for v in cr.split(".")) for cr in args[0][6:].split(",")]
        args = args[1:]
    if "att" in args:
        i = args.index("att")
        ATT, ATT_NTS = True, [int(a) for a in args[i + 1:]]
        args = args[:i]
    if "sw" in args:
        i = args.index("sw")
        SWITCH_NTS = [int(a) for a in args[i + 1:]]
        args = args[:i]
    if args and args[0].startswith("pad"):
        PAD_WRITES = int(args[0][3:])
        args = args[1:]
    if "x2" in args:
        i = args.index("x2")
        BLOCKS2 = [int(a) for a in args[i + 1:]]
        args = args[:i]
    return emit([int(a) for a in args])


if __name__ == "__main__":
    sys.stdout.write(emit([int(a) for a in sys.argv[1:]] or [1]))
