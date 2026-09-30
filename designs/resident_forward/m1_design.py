"""M1 design: the chain GEMM at the gate/up shape on all 8 x 4 cores, x resident in the MemTiles.

Static (in the PDI): core DMAs (weight elements and activation blocks, ping-pong), the MemTile weight
split, cascades, routes. Per class (one named control code per 16-row block count nt): the MemTile
tasks whose extent depends on the rows -- x in, the broadcast of x 120 times, the landing of h, h out
-- and the shim tasks. Cores read nt from an RTP word written by the control code.
"""
import sys
import rf_paths

NC, NR = 8, 4
KC, NB = 480, 64
ELEM = 1728                 # weight element: 96 K rows x 32 columns (int4 g32, bf16 scales)
ABLK = 16 * KC * 9 // 8     # one 16-row bfp16 activation block of a K slice: 8640
OBLK = 576                  # chain-end h output per (N block, 16-row block): 8 subtiles
NBLK = 120                  # N blocks per chain row (gate/up interleaved, Nb = 64)
NSUB = KC // 96             # weight sub-blocks per K slice
PCAP_T = 7                  # 16-row blocks at P_cap = 112
WCOL = NBLK * NSUB * 2 * NR * ELEM        # one column's weight stream per dispatch
HT = 240 * 2 * 72                         # one 16-row block of a MemTile's h (1920 cols)
KERN = "kern.o"


class M:
    def __init__(self):
        self.lines = []
        self.n = 0

    def __call__(self, s, ind=4):
        self.lines.append(" " * ind + s)

    def tmp(self, p="v"):
        self.n += 1
        return f"%{p}{self.n}"


def core_name(c, r):
    return f"{c}_{2 + r}"


PARTS = "all"


def emit(nts, stack_mid=1024, stack_end=3072):
    m = M()
    m("module {", 0)
    m("aie.device(npu2) @main {", 2)
    for c in range(NC):
        m(f"%shim_{c} = aie.tile({c}, 0)")
        m(f"%mem_{c} = aie.tile({c}, 1)")
        for r in range(NR):
            m(f"%t_{core_name(c, r)} = aie.tile({c}, {2 + r})")
    # routes
    for c in range(NC):
        m(f"aie.flow(%shim_{c}, DMA : 0, %mem_{c}, DMA : 0)")
        m(f"aie.flow(%shim_{c}, DMA : 1, %mem_{c}, DMA : 1)")
        for r in range(NR):
            m(f"aie.flow(%mem_{c}, DMA : {r}, %t_{core_name(c, r)}, DMA : 0)")
            m(f"aie.flow(%mem_{c}, DMA : 4, %t_{core_name(c, r)}, DMA : 1)")
        m(f"aie.flow(%mem_{c}, DMA : 5, %shim_{c}, DMA : 0)")
    for r in range(NR):
        for j in range(2):
            m(f"aie.flow(%t_{core_name(7, r)}, DMA : {j}, %mem_{2 * r + j}, DMA : 2)")
        for c in range(NC - 1):
            m(f"aie.cascade_flow(%t_{core_name(c, r)}, %t_{core_name(c + 1, r)})")
    # kernels
    m(f'func.func private @chain_convert_half(memref<{ELEM}xi8>, memref<{KC * NB * 9 // 8}xi8>, i32, i32) attributes {{link_with = "{KERN}"}}')
    m(f'func.func private @chain_mm_first_r2(memref<{ABLK}xi8>, memref<{KC * NB * 9 // 8}xi8>) attributes {{link_with = "{KERN}"}}')
    m(f'func.func private @chain_mm_mid_r2(memref<{ABLK}xi8>, memref<{KC * NB * 9 // 8}xi8>) attributes {{link_with = "{KERN}"}}')
    m(f'func.func private @chain_mm_last_geluup_r2(memref<{ABLK}xi8>, memref<{KC * NB * 9 // 8}xi8>, memref<{OBLK}xi8>, memref<1024xi16>) attributes {{link_with = "{KERN}"}}')
    # MemTile: weight split (static), resident x and h, locks for the per-class tasks
    for c in range(NC):
        m(f"%wst_{c} = aie.buffer(%mem_{c}) {{sym_name = \"wst_{c}\"}} : memref<{2 * NR * ELEM}xi8>")
        m(f"%X_{c} = aie.buffer(%mem_{c}) {{sym_name = \"X_{c}\"}} : memref<{PCAP_T * ABLK}xi8>")
        m(f"%H_{c} = aie.buffer(%mem_{c}) {{sym_name = \"H_{c}\"}} : memref<{PCAP_T * HT}xi8>")
        for r in range(NR):
            m(f"%wp_{c}_{r} = aie.lock(%mem_{c}) {{init = 2 : i32, sym_name = \"wp_{c}_{r}\"}}")
            m(f"%wc_{c}_{r} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"wc_{c}_{r}\"}}")
        for nm in ("xf", "xr", "hf", "hr"):
            m(f"%{nm}_{c} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"{nm}_{c}\"}}")
        m(f"%mdma_{c} = aie.memtile_dma(%mem_{c}) {{")
        m("%one = arith.constant 1 : i32", 6)
        m(f"%s0 = aie.dma_start(S2MM, 0, ^w0, ^m0)", 6)
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
    # cores
    WB = KC * NB * 9 // 8
    for c in range(NC):
        for r in range(NR):
            t = core_name(c, r)
            last = c == NC - 1
            m(f"%wblk_{t} = aie.buffer(%t_{t}) {{sym_name = \"wblk_{t}\"}} : memref<{WB}xi8>")
            m(f"%rtp_{t} = aie.buffer(%t_{t}) {{sym_name = \"rtp_{t}\"}} : memref<4xi32>")
            for i in range(2):
                m(f"%wb{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"wb{i}_{t}\"}} : memref<{ELEM}xi8>")
                m(f"%ab{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ab{i}_{t}\"}} : memref<{ABLK}xi8>")
            for nm, init in (("wp", 2), ("wc", 0), ("ap", 2), ("ac", 0), ("go", 0)):
                m(f"%c{nm}_{t} = aie.lock(%t_{t}) {{init = {init} : i32, sym_name = \"c{nm}_{t}\"}}")
            if last:
                m(f"%gscr_{t} = aie.buffer(%t_{t}) {{sym_name = \"gscr_{t}\"}} : memref<1024xi16>")
                for j in range(2):
                    for i in range(2):
                        m(f"%ob{j}{i}_{t} = aie.buffer(%t_{t}) {{sym_name = \"ob{j}{i}_{t}\"}} : memref<{OBLK}xi8>")
                    m(f"%cop{j}_{t} = aie.lock(%t_{t}) {{init = 2 : i32, sym_name = \"cop{j}_{t}\"}}")
                    m(f"%coc{j}_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"coc{j}_{t}\"}}")
            # DMA
            m(f"%cdma_{t} = aie.mem(%t_{t}) {{")
            m("%one = arith.constant 1 : i32", 6)
            chans = [("S2MM", 0, "wb", "cwp", "cwc", ELEM), ("S2MM", 1, "ab", "cap", "cac", ABLK)]
            if last:
                chans += [("MM2S", j, f"ob{j}", f"coc{j}", f"cop{j}", OBLK) for j in range(2)]
            for q, (d, ch, buf, acq, rel, ln) in enumerate(chans):
                end = f"^q{q + 1}" if q < len(chans) - 1 else "^qend"
                m(f"%s{q} = aie.dma_start({d}, {ch}, ^q{q}a, {end})", 6)
                for i, lab, nxt in ((0, "a", "b"), (1, "b", "a")):
                    m(f"^q{q}{lab}:", 4)
                    m(f"aie.use_lock(%{acq}_{t}, AcquireGreaterEqual, %one)", 6)
                    bn = f"%{buf}{i}_{t}" if not buf.startswith("ob") else f"%{buf}{i}_{t}"
                    m(f"aie.dma_bd({bn} : memref<{ln}xi8> offset = 0 len = {ln})", 6)
                    m(f"aie.use_lock(%{rel}_{t}, Release, %one)", 6)
                    m(f"aie.next_bd ^q{q}{nxt}", 6)
                m(f"^q{q + 1}:" if q < len(chans) - 1 else "^qend:", 4)
            m("aie.end", 6)
            m("}")
            # program
            core_body(m, c, r, last)
            m(f"}} {{stack_size = {stack_end if last else stack_mid} : i32}}")
    # control codes: the PDI loads once, from `boot`; a load_pdi in every class code reloads it on
    # each switch between codes (K046)
    xb, wb, hb = NC * PCAP_T * ABLK, NC * WCOL, NC * PCAP_T * HT
    m(f"aie.runtime_sequence @boot(%x : memref<{xb}xi8>, %w : memref<{wb}xi8>, %h : memref<{hb}xi8>) {{")
    m("aiex.npu.load_pdi {device_ref = @main}", 6)
    m("}", 4)
    for nt in nts:
        sequence(m, nt)
    m("}", 2)
    m("}", 0)
    return "\n".join(m.lines) + "\n"


def core_body(m, c, r, last):
    t = core_name(c, r)
    WB = KC * NB * 9 // 8
    mm = "chain_mm_first_r2" if c == 0 else ("chain_mm_last_geluup_r2" if last else "chain_mm_mid_r2")
    m(f"%core_{t} = aie.core(%t_{t}) {{")
    I = 6
    m("%c0 = arith.constant 0 : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%c1 = arith.constant 1 : index", I)
    m("%c2 = arith.constant 2 : index", I)
    m(f"%cns = arith.constant {NSUB} : index", I)
    m("%c60 = arith.constant 60 : index", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m("%z32 = arith.constant 0 : i32", I)
    m("%two32 = arith.constant 2 : i32", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%cgo_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    halves = [(0, 60)] if not last else [(0, 60), (1, 60)]
    n_loops = [(None, 120)] if not last else [(0, 60), (1, 60)]
    for oj, cnt in n_loops:
        m(f"%cn{cnt}_{oj} = arith.constant {cnt} : index", I)
        m(f"scf.for %n = %c0 to %cn{cnt}_{oj} step %c1 {{", I)
        I += 2
        m("scf.for %s = %c0 to %cns step %c1 {", I)
        I += 2
        m("%s32 = arith.index_cast %s : index to i32", I)
        for i, pair in ((0, "%z32"), (1, "%two32")):
            m(f"aie.use_lock(%ccwc_{t}, AcquireGreaterEqual, %one)" if False else f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", I)
            m(f"func.call @chain_convert_half(%wb{i}_{t}, %wblk_{t}, %s32, {pair}) : (memref<{ELEM}xi8>, memref<{WB}xi8>, i32, i32) -> ()", I)
            m(f"aie.use_lock(%cwp_{t}, Release, %one)", I)
        I -= 2
        m("}", I)
        m("%nnt = arith.muli %n, %nt : index", I)
        m("scf.for %tb = %c0 to %nt step %c1 {", I)
        I += 2
        m("%idx = arith.addi %nnt, %tb : index", I)
        m("%par = arith.remui %idx, %c2 : index", I)
        m("%even = arith.cmpi eq, %par, %c0 : index", I)
        m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", I)
        if last:
            m(f"aie.use_lock(%cop{oj}_{t}, AcquireGreaterEqual, %one)", I)
        m("scf.if %even {", I)
        for i in range(2):
            args = f"%ab{i}_{t}, %wblk_{t}"
            sig = f"memref<{ABLK}xi8>, memref<{WB}xi8>"
            if last:
                args += f", %ob{oj}{i}_{t}, %gscr_{t}"
                sig += f", memref<{OBLK}xi8>, memref<1024xi16>"
            m(f"func.call @{mm}({args}) : ({sig}) -> ()", I + 2)
            if i == 0:
                m("} else {", I)
        m("}", I)
        if last:
            m(f"aie.use_lock(%coc{oj}_{t}, Release, %one)", I)
        m(f"aie.use_lock(%cap_{t}, Release, %one)", I)
        I -= 2
        m("}", I)
        I -= 2
        m("}", I)
    I -= 2
    m("}", I)
    m("aie.end", I)


def sequence(m, nt):
    X, H = nt * ABLK, nt * HT
    xb, wb, hb = NC * PCAP_T * ABLK, NC * WCOL, NC * PCAP_T * HT
    m(f"aie.runtime_sequence @p{nt}(%x : memref<{xb}xi8>, %w : memref<{wb}xi8>, %h : memref<{hb}xi8>) {{")
    I = 6
    m(f"%ntv = arith.constant {nt} : i32", I)
    for c in range(NC):
        for r in range(NR):
            t = core_name(c, r)
            m(f"aiex.npu.rtp_write(@rtp_{t}, 0, %ntv) : i32", I)
    for c in range(NC):
        m(f"aiex.set_lock(%xf_{c}, 1)", I)
        m(f"aiex.set_lock(%xr_{c}, 0)", I)
        m(f"aiex.set_lock(%hf_{c}, 1)", I)
        m(f"aiex.set_lock(%hr_{c}, 0)", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.set_lock(%cgo_{core_name(c, r)}, 1)", I)
    if PARTS == "min":
        m("}", 4)
        return
    if PARTS in ("w", "w3d", "ws", "wa0"):
        for c in range(NC):
            m(f"%sw{c} = aiex.dma_configure_task(%shim_{c}, MM2S, 0) {{", I)
            if PARTS == "w":
                m(f"aie.dma_bd(%w : memref<{wb}xi8> offset = {c * WCOL} len = {WCOL})", I + 2)
            elif PARTS == "wa0":
                m(f"aie.dma_bd(%x : memref<{xb}xi8> offset = {c * PCAP_T * ABLK} len = {16 * ELEM})", I + 2)
            elif PARTS == "ws":
                m(f"aie.dma_bd(%w : memref<{wb}xi8> offset = {c * WCOL} len = {64 * ELEM})", I + 2)
            else:
                m(f"aie.dma_bd(%w : memref<{wb}xi8> offset = {c * WCOL} len = {WCOL} sizes = [120, 20, 3456] strides = [69120, 3456, 1])", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%sw{c})", I)
        m("}", 4)
        return
    if PARTS == "xbc":
        for c in range(NC):
            m(f"%xin{c} = aiex.dma_configure_task(%mem_{c}, S2MM, 1) {{", I)
            m("%one = arith.constant 1 : i32", I + 2)
            m(f"aie.use_lock(%xf_{c}, AcquireGreaterEqual, %one)", I + 2)
            m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 28 : i32}}", I + 2)
            m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%xin{c})", I)
            m(f"%bc{c} = aiex.dma_configure_task(%mem_{c}, MM2S, 4) {{", I)
            m("%one = arith.constant 1 : i32", I + 2)
            m(f"aie.use_lock(%xr_{c}, AcquireGreaterEqual, %one)", I + 2)
            m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 12 : i32}}", I + 2)
            m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
            m("aie.end", I + 2)
            m(f"}} {{repeat_count = {NBLK - 1} : i32}}", I)
            m(f"aiex.dma_start_task(%bc{c})", I)
            m(f"%sx{c} = aiex.dma_configure_task(%shim_{c}, MM2S, 1) {{", I)
            m(f"aie.dma_bd(%x : memref<{xb}xi8> offset = {c * PCAP_T * ABLK} len = {X})", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%sx{c})", I)
        m("}", 4)
        return
    if PARTS == "xonly":
        for c in range(NC):
            m(f"%xin{c} = aiex.dma_configure_task(%mem_{c}, S2MM, 1) {{", I)
            m("%one = arith.constant 1 : i32", I + 2)
            m(f"aie.use_lock(%xf_{c}, AcquireGreaterEqual, %one)", I + 2)
            m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 28 : i32}}", I + 2)
            m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%xin{c})", I)
            m(f"%xo{c} = aiex.dma_configure_task(%mem_{c}, MM2S, 5) {{", I)
            m("%one = arith.constant 1 : i32", I + 2)
            m(f"aie.use_lock(%xr_{c}, AcquireGreaterEqual, %one)", I + 2)
            m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 29 : i32}}", I + 2)
            m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%xo{c})", I)
            m(f"%sx{c} = aiex.dma_configure_task(%shim_{c}, MM2S, 1) {{", I)
            m(f"aie.dma_bd(%x : memref<{xb}xi8> offset = {c * PCAP_T * ABLK} len = {X})", I + 2)
            m("aie.end", I + 2)
            m("}", I)
            m(f"aiex.dma_start_task(%sx{c})", I)
            m(f"%sh{c} = aiex.dma_configure_task(%shim_{c}, S2MM, 0) {{", I)
            m(f"aie.dma_bd(%h : memref<{hb}xi8> offset = {c * PCAP_T * HT} len = {X})", I + 2)
            m("aie.end", I + 2)
            m("} {issue_token = true}", I)
            m(f"aiex.dma_start_task(%sh{c})", I)
        for c in range(NC):
            m(f"aiex.dma_await_task(%sh{c})", I)
        m("}", 4)
        return
    for c in range(NC):
        m(f"%xin{c} = aiex.dma_configure_task(%mem_{c}, S2MM, 1) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m(f"aie.use_lock(%xf_{c}, AcquireGreaterEqual, %one)", I + 2)
        m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 28 : i32}}", I + 2)
        m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
        m("aie.end", I + 2)
        m("}", I)
        m(f"aiex.dma_start_task(%xin{c})", I)
        m(f"%bc{c} = aiex.dma_configure_task(%mem_{c}, MM2S, 4) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m(f"aie.use_lock(%xr_{c}, AcquireGreaterEqual, %one)", I + 2)
        m(f"aie.dma_bd(%X_{c} : memref<{PCAP_T * ABLK}xi8> offset = 0 len = {X}) {{bd_id = 12 : i32}}", I + 2)
        m(f"aie.use_lock(%xr_{c}, Release, %one)", I + 2)
        m("aie.end", I + 2)
        m(f"}} {{repeat_count = {NBLK - 1} : i32}}", I)
        m(f"aiex.dma_start_task(%bc{c})", I)
        m(f"%ld{c} = aiex.dma_configure_task(%mem_{c}, S2MM, 2) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m(f"aie.use_lock(%hf_{c}, AcquireGreaterEqual, %one)", I + 2)
        m(f"aie.dma_bd(%H_{c} : memref<{PCAP_T * HT}xi8> offset = 0 len = {60 * nt * OBLK} sizes = [60, {nt}, {OBLK}] strides = [{OBLK}, {HT}, 1]) {{bd_id = 13 : i32}}", I + 2)
        m(f"aie.use_lock(%hr_{c}, Release, %one)", I + 2)
        m("aie.end", I + 2)
        m("}", I)
        m(f"aiex.dma_start_task(%ld{c})", I)
        m(f"%ho{c} = aiex.dma_configure_task(%mem_{c}, MM2S, 5) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        m(f"aie.use_lock(%hr_{c}, AcquireGreaterEqual, %one)", I + 2)
        m(f"aie.dma_bd(%H_{c} : memref<{PCAP_T * HT}xi8> offset = 0 len = {H}) {{bd_id = 29 : i32}}", I + 2)
        m(f"aie.use_lock(%hf_{c}, Release, %one)", I + 2)
        m("aie.end", I + 2)
        m("}", I)
        m(f"aiex.dma_start_task(%ho{c})", I)
    for c in range(NC):
        m(f"%sw{c} = aiex.dma_configure_task(%shim_{c}, MM2S, 0) {{", I)
        m(f"aie.dma_bd(%w : memref<{wb}xi8> offset = {c * WCOL} len = {WCOL})", I + 2)
        m("aie.end", I + 2)
        m("}", I)
        m(f"aiex.dma_start_task(%sw{c})", I)
        m(f"%sx{c} = aiex.dma_configure_task(%shim_{c}, MM2S, 1) {{", I)
        m(f"aie.dma_bd(%x : memref<{xb}xi8> offset = {c * PCAP_T * ABLK} len = {X})", I + 2)
        m("aie.end", I + 2)
        m("}", I)
        m(f"aiex.dma_start_task(%sx{c})", I)
        m(f"%sh{c} = aiex.dma_configure_task(%shim_{c}, S2MM, 0) {{", I)
        m(f"aie.dma_bd(%h : memref<{hb}xi8> offset = {c * PCAP_T * HT} len = {H})", I + 2)
        m("aie.end", I + 2)
        m("} {issue_token = true}", I)
        m(f"aiex.dma_start_task(%sh{c})", I)
    for c in range(NC):
        if PARTS != "noawait":
            m(f"aiex.dma_await_task(%sh{c})", I)
    m("}", 4)


if __name__ == "__main__":
    nts = [int(a) for a in sys.argv[1:]] or [1, 3]
    sys.stdout.write(emit(nts))


def kernels():
    return [(KERN, rf_paths.iron_kernel("chain_mm_bfp16.cc"),
             ["-DKC=480", "-DNB=64", "-DSUB_K=96"])]


def build_text(args):
    global PARTS
    if args and not args[0].isdigit():
        PARTS = args[0]
        args = args[1:]
    return emit([int(a) for a in args])
