"""M2a design: M1's gate/up chain followed by the down projection in the same dispatch, h resident.

Down (Nb = 32): per chain row 30 N blocks, per N block 4 K sub-passes of 480 per column; MemTile c
broadcasts h[:, c*1920 + s*480 ..] per sub-pass; the chain end accumulates the sub-passes in f32 in
the unused half of its weight block and emits y_d bf16 per N block, n < 15 to MemTile 2r, the rest
to 2r+1 (MemTile m holds y_d columns m*480 .. m*480+479, the residual's slice). y_d goes to DDR.

The column-7 output channels are runtime tasks (gate/up h, then y_d), since their BD length
differs by phase.
"""
import sys
import rf_paths
from m1_design import M, core_name, NC, NR, KC, NB, ELEM, ABLK, OBLK, NBLK, NSUB, PCAP_T, HT, KERN

NDB, NSP = 30, 4
WB = KC * NB * 9 // 8                     # gate/up W block; down uses its first DOWN_W bytes
DOWN_W = KC * 32 * 9 // 8
YB = 1024                                 # one y_d N block of 16 rows, bf16
YT = 15 * YB                              # one 16-row block of a MemTile's y_d
WCOL_GU = NBLK * NSUB * 2 * NR * ELEM
WCOL_D = NDB * NSP * NSUB * NR * ELEM
WCOL = WCOL_GU + WCOL_D
HB_BDS = (14, 16, 17, 18)                # even bank, clear of the static 0-11 (K045)


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
    for r in range(NR):
        for j in range(2):
            m(f"aie.flow(%t_{core_name(7, r)}, DMA : {j}, %mem_{2 * r + j}, DMA : 2)")
        for c in range(NC - 1):
            m(f"aie.cascade_flow(%t_{core_name(c, r)}, %t_{core_name(c + 1, r)})")
    A, W, O = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", f"memref<{OBLK}xi8>"
    for name, sig in (("chain_convert_half", f"memref<{ELEM}xi8>, {W}, i32, i32"),
                      ("chain_mm_first_r2", f"{A}, {W}"), ("chain_mm_mid_r2", f"{A}, {W}"),
                      ("chain_mm_last_geluup_r2", f"{A}, {W}, {O}, memref<1024xi16>"),
                      ("chain_down_first_r2", f"{A}, {W}"), ("chain_down_mid_r2", f"{A}, {W}"),
                      ("chain_down_last_r2", f"{A}, {W}, i32, i32"),
                      ("chain_down_emit", f"{W}, i32")):
        m(f'func.func private @{name}({sig}) attributes {{link_with = "{KERN}"}}')
    for c in range(NC):
        memtile(m, c)
    for c in range(NC):
        for r in range(NR):
            core(m, c, r, stack_end if c == NC - 1 else stack_mid)
    xb, wb, yb = NC * PCAP_T * ABLK, NC * WCOL, NC * PCAP_T * YT
    m(f"aie.runtime_sequence @boot(%x : memref<{xb}xi8>, %w : memref<{wb}xi8>, %y : memref<{yb}xi8>) {{")
    m("aiex.npu.load_pdi {device_ref = @main}", 6)
    m("}", 4)
    for nt in nts:
        sequence(m, nt)
    m("}", 2)
    m("}", 0)
    return "\n".join(m.lines) + "\n"


def memtile(m, c):
    m(f"%wst_{c} = aie.buffer(%mem_{c}) {{sym_name = \"wst_{c}\"}} : memref<{2 * NR * ELEM}xi8>")
    m(f"%X_{c} = aie.buffer(%mem_{c}) {{sym_name = \"X_{c}\"}} : memref<{PCAP_T * ABLK}xi8>")
    m(f"%H_{c} = aie.buffer(%mem_{c}) {{sym_name = \"H_{c}\"}} : memref<{PCAP_T * HT}xi8>")
    m(f"%Y_{c} = aie.buffer(%mem_{c}) {{sym_name = \"Y_{c}\"}} : memref<{PCAP_T * YT}xi8>")
    for r in range(NR):
        m(f"%wp_{c}_{r} = aie.lock(%mem_{c}) {{init = 2 : i32, sym_name = \"wp_{c}_{r}\"}}")
        m(f"%wc_{c}_{r} = aie.lock(%mem_{c}) {{init = 0 : i32, sym_name = \"wc_{c}_{r}\"}}")
    for nm in ("xf", "xr", "hf", "hr", "yf", "yr"):
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


def core(m, c, r, stack):
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
            m(f"%cyc{j}_{t} = aie.lock(%t_{t}) {{init = 0 : i32, sym_name = \"cyc{j}_{t}\"}}")
        m(f"%cyp_{t} = aie.lock(%t_{t}) {{init = 1 : i32, sym_name = \"cyp_{t}\"}}")
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


def convert_loop(m, t, I, base, count, gateup, fn="chain_convert_half", kb=None):
    """`count` weight elements per outer step; buffer parity is the running element index
    `base + e`. Gate/up element e = 2 * sub + half (pair 2 * half); down element e = sub (pair 0)."""
    m(f"scf.for %e = %c0 to {count if isinstance(count, str) else f'%c{count}'} step %c1 {{", I)
    J = I + 2
    m(f"%eg = arith.addi {base}, %e : index", J)
    m("%ep = arith.remui %eg, %c2 : index", J)
    m("%eev = arith.cmpi eq, %ep, %c0 : index", J)
    if gateup:
        m("%sub = arith.divui %e, %c2 : index", J)
        m("%half = arith.remui %e, %c2 : index", J)
        m("%hp = arith.muli %half, %c2 : index", J)
        m("%pair = arith.index_cast %hp : index to i32", J)
    else:
        m("%sub = arith.addi %e, %c0 : index", J)
        m("%pair = arith.constant 0 : i32", J)
    m("%sub32 = arith.index_cast %sub : index to i32", J)
    xa, xt = ("", "")
    if isinstance(kb, str):   # the runtime-K entry, kb an i32 value in scope
        xa, xt = f", {kb}", ", i32"
    elif kb is not None:      # the runtime-K entry: kb 8-row K blocks per chain position
        m(f"%kb = arith.constant {kb} : i32", J)
        xa, xt = ", %kb", ", i32"
    m(f"aie.use_lock(%cwc_{t}, AcquireGreaterEqual, %one)", J)
    m("scf.if %eev {", J)
    for i in range(2):
        m(f"func.call @{fn}(%wb{i}_{t}, %wblk_{t}, %sub32, %pair{xa}) : (memref<{ELEM}xi8>, memref<{WB}xi8>, i32, i32{xt}) -> ()", J + 2)
        if i == 0:
            m("} else {", J)
    m("}", J)
    m(f"aie.use_lock(%cwp_{t}, Release, %one)", J)
    m("}", I)


def act_call(m, t, I, idx, calls):
    """Acquire an activation block, run `calls(i)` on ab{parity}, release."""
    m(f"%par = arith.remui {idx}, %c2 : index", I)
    m("%even = arith.cmpi eq, %par, %c0 : index", I)
    m(f"aie.use_lock(%cac_{t}, AcquireGreaterEqual, %one)", I)
    m("scf.if %even {", I)
    for i in range(2):
        for line in calls(i):
            m(line, I + 2)
        if i == 0:
            m("} else {", I)
    m("}", I)
    m(f"aie.use_lock(%cap_{t}, Release, %one)", I)


def core_body(m, c, r, last):
    t = core_name(c, r)
    A, W, O = f"memref<{ABLK}xi8>", f"memref<{WB}xi8>", f"memref<{OBLK}xi8>"
    mm = "chain_mm_first_r2" if c == 0 else ("chain_mm_last_geluup_r2" if last else "chain_mm_mid_r2")
    m(f"%core_{t} = aie.core(%t_{t}) {{")
    I = 6
    for v in (0, 1, 2, 3, 4, 5, 10, 15, 30, 60, 120):
        m(f"%c{v} = arith.constant {v} : index", I)
    m("%one = arith.constant 1 : i32", I)
    m("%cbig = arith.constant 4294967295 : index", I)
    m("scf.for %it = %c0 to %cbig step %c1 {", I)
    I += 2
    m(f"aie.use_lock(%cgo_{t}, AcquireGreaterEqual, %one)", I)
    m(f"%nt32 = memref.load %rtp_{t}[%c0] : memref<4xi32>", I)
    m("%nt = arith.index_cast %nt32 : i32 to index", I)
    # gate/up: 120 N blocks (the chain end in two halves of 60, one per output channel)
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
                sig += f", {O}, memref<1024xi16>"
            return [f"func.call @{mm}({args}) : ({sig}) -> ()"]
        act_call(m, t, I, "%idx", gu)
        if last:
            m(f"aie.use_lock(%coc{oj}_{t}, Release, %one)", I)
        I -= 2
        m("}", I)
        I -= 2
        m("}", I)
    # down: 30 N blocks x 4 K sub-passes
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
    if last:
        m("%s0 = arith.cmpi eq, %s, %c0 : index", I)
        m("scf.if %s0 {", I)
        m(f"aie.use_lock(%cyp_{t}, AcquireGreaterEqual, %one)", I + 2)
        m("}", I)
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
        m("%s3 = arith.cmpi eq, %s, %c3 : index", I)
        m("scf.if %s3 {", I)
        m(f"func.call @chain_down_emit(%wblk_{t}, %nt32) : ({W}, i32) -> ()", I + 2)
        m("%lo = arith.cmpi ult, %n, %c15 : index", I + 2)
        m("scf.if %lo {", I + 2)
        m(f"aie.use_lock(%cyc0_{t}, Release, %one)", I + 4)
        m("} else {", I + 2)
        m(f"aie.use_lock(%cyc1_{t}, Release, %one)", I + 4)
        m("}", I + 2)
        m("}", I)
    I -= 2
    m("}", I)
    I -= 2
    m("}", I)
    I -= 2
    m("}", I)
    m("aie.end", I)


def sequence(m, nt):
    X, H, Yn = nt * ABLK, nt * HT, nt * YT
    xb, wb, yb = NC * PCAP_T * ABLK, NC * WCOL, NC * PCAP_T * YT
    m(f"aie.runtime_sequence @p{nt}(%x : memref<{xb}xi8>, %w : memref<{wb}xi8>, %y : memref<{yb}xi8>) {{")
    I = 6
    m(f"%ntv = arith.constant {nt} : i32", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.npu.rtp_write(@rtp_{core_name(c, r)}, 0, %ntv) : i32", I)
    for c in range(NC):
        for nm, v in (("xf", 1), ("xr", 0), ("hf", 1), ("hr", 0), ("yf", 1), ("yr", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for r in range(NR):
        t = core_name(NC - 1, r)
        for j in range(2):
            m(f"aiex.set_lock(%cop{j}_{t}, 2)", I)
            m(f"aiex.set_lock(%coc{j}_{t}, 0)", I)
            m(f"aiex.set_lock(%cyc{j}_{t}, 0)", I)
        m(f"aiex.set_lock(%cyp_{t}, 1)", I)
    for c in range(NC):
        for r in range(NR):
            m(f"aiex.set_lock(%cgo_{core_name(c, r)}, 1)", I)

    def task(name, tile, d, ch, body, attrs=""):
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        for line in body:
            m(line, I + 2)
        m("aie.end", I + 2)
        m("}" + attrs, I)
        m(f"aiex.dma_start_task(%{name})", I)

    def locked(acq, bd, rel):
        return [f"aie.use_lock(%{acq}, AcquireGreaterEqual, %one)", bd, f"aie.use_lock(%{rel}, Release, %one)"]

    for c in range(NC):
        Xm, Hm, Ym = f"%X_{c} : memref<{PCAP_T * ABLK}xi8>", f"%H_{c} : memref<{PCAP_T * HT}xi8>", f"%Y_{c} : memref<{PCAP_T * YT}xi8>"
        task(f"xin{c}", f"mem_{c}", "S2MM", 1, locked(f"xf_{c}", f"aie.dma_bd({Xm} offset = 0 len = {X}) {{bd_id = 28 : i32}}", f"xr_{c}"))
        task(f"bc{c}", f"mem_{c}", "MM2S", 4, locked(f"xr_{c}", f"aie.dma_bd({Xm} offset = 0 len = {X}) {{bd_id = 12 : i32}}", f"xr_{c}"),
             f" {{repeat_count = {NBLK - 1} : i32}}")
        # one BD per K sub-pass; the 8640 B block is split by hand into 3 x 2880 (d0 wrap is
        # 1023 words), and a fourth [s] dimension would be the BD iteration, not a wrap (K047)
        hb = []
        for sp, bd in enumerate(HB_BDS):
            if sp:
                hb += [f"aie.next_bd ^h{sp}", f"^h{sp}:"]
            hb += locked(f"hr_{c}", f"aie.dma_bd({Hm} offset = {sp * ABLK} len = {nt * ABLK} sizes = [{nt}, 3, {ABLK // 3}] strides = [{HT}, {ABLK // 3}, 1]) {{bd_id = {bd} : i32}}", f"hr_{c}")
        task(f"hb{c}", f"mem_{c}", "MM2S", 4, hb, f" {{repeat_count = {NDB - 1} : i32}}")
        task(f"ld{c}", f"mem_{c}", "S2MM", 2, locked(f"hf_{c}", f"aie.dma_bd({Hm} offset = 0 len = {60 * nt * OBLK} sizes = [60, {nt}, {OBLK}] strides = [{OBLK}, {HT}, 1]) {{bd_id = 13 : i32}}", f"hr_{c}"))
        task(f"ly{c}", f"mem_{c}", "S2MM", 2, locked(f"yf_{c}", f"aie.dma_bd({Ym} offset = 0 len = {15 * nt * YB} sizes = [15, {nt}, {YB}] strides = [{YB}, {YT}, 1]) {{bd_id = 15 : i32}}", f"yr_{c}"))
        task(f"yo{c}", f"mem_{c}", "MM2S", 5, locked(f"yr_{c}", f"aie.dma_bd({Ym} offset = 0 len = {Yn}) {{bd_id = 29 : i32}}", f"yf_{c}"))
    for r in range(NR):
        t = core_name(NC - 1, r)
        for j in range(2):
            o0, o1 = f"%ob{j}0_{t} : memref<{OBLK}xi8>", f"%ob{j}1_{t} : memref<{OBLK}xi8>"
            body = [f"aie.use_lock(%coc{j}_{t}, AcquireGreaterEqual, %one)",
                    f"aie.dma_bd({o0} offset = 0 len = {OBLK}) {{bd_id = {8 + 2 * j} : i32}}",
                    f"aie.use_lock(%cop{j}_{t}, Release, %one)",
                    f"aie.next_bd ^b1",
                    "^b1:",
                    f"aie.use_lock(%coc{j}_{t}, AcquireGreaterEqual, %one)",
                    f"aie.dma_bd({o1} offset = 0 len = {OBLK}) {{bd_id = {9 + 2 * j} : i32}}",
                    f"aie.use_lock(%cop{j}_{t}, Release, %one)"]
            task(f"go{j}_{r}", f"t_{t}", "MM2S", j, body, f" {{repeat_count = {30 * nt - 1} : i32}}")
            task(f"yd{j}_{r}", f"t_{t}", "MM2S", j,
                 locked(f"cyc{j}_{t}", f"aie.dma_bd(%wblk_{t} : memref<{WB}xi8> offset = {DOWN_W} len = {nt * YB}) {{bd_id = {12 + j} : i32}}", f"cyp_{t}"),
                 f" {{repeat_count = 14 : i32}}")
    for c in range(NC):
        task(f"sw{c}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{wb}xi8> offset = {c * WCOL} len = {WCOL})"])
        task(f"sx{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{xb}xi8> offset = {c * PCAP_T * ABLK} len = {X})"])
        task(f"sy{c}", f"shim_{c}", "S2MM", 0, [f"aie.dma_bd(%y : memref<{yb}xi8> offset = {c * PCAP_T * YT} len = {Yn})"],
             " {issue_token = true}")
    for c in range(NC):
        m(f"aiex.dma_await_task(%sy{c})", I)
    m("}", 4)


def kernels():
    return [(KERN, rf_paths.iron_kernel("chain_mm_bfp16.cc"),
             ["-DKC=480", "-DNB=64", "-DSUB_K=96"])]


def build_text(args):
    return emit([int(a) for a in args])


if __name__ == "__main__":
    sys.stdout.write(emit([int(a) for a in sys.argv[1:]] or [1]))
