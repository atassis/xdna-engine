"""The LM head as rows of the layer image and a runtime sequence h1 (rlayer_design `h`): the final
RMSNorm (the pre-norm form with final_norm's gain), then the tied int4 head pack as a down-form chain
GEMM, 2048 blocks of 32 vocab columns per chain row (262144 = 4 rows x 65536). One 16-row block (the
host passes the row it wants logits for as row 0). Column 7's rows relay through a 2-slot MemTile
ring to the logits buffer [16][262144] bf16.

No core code of its own: GAINS, NORM_PRE and DOWN are the layer's. A task sends at most 256 times,
so the A-block broadcast and column 7's emits are chains of identical BDs, and the weights go 21
elements a MemTile BD (GW) so the distribution fits one task."""
import os
import numpy as np
import rf_paths

V, KH = 262144, 3840
NROW, NCOL, ELEM, SUBK = 4, 8, 1728, 96
NDB = V // NROW // 32               # 2048 down blocks per chain row
NSUB = KH // NCOL // SUBK           # 5
GW = 21
NW = 2 + NDB * NSUB                 # per core: final-norm gains, head
NWP = -(-NW // (2 * GW)) * 2 * GW
WCOL = NWP * NROW * ELEM
BC_CHAIN, QD_CHAIN = 8, 4            # identical BDs a chain: 8 x 256 >= NDB, 4 x 256 >= NDB / 2
RL_BLK = 16                          # blocks a relay slot
OUTB = 16 * V * 2                    # logits [16][262144] bf16
STREAM = os.environ.get("RF_STACK_OUT", str(rf_paths.BUILD_ROOT / "scratch/stack")) + "/w_head.npy"
# head-phase MemTile buffers (alloc_group "head"), after x
XYH, WSH, RL = 108480, 117184, 117184 + 2 * NROW * GW * ELEM
assert RL + 2 * RL_BLK * 1024 <= 524288


def pack_block(q, sc):
    """q int8 [NB*32, 3840], sc f32 [NB*32, 120] -> elements uint8 [NB, 8 columns, 5 subs, 1728]."""
    from chain_ref import bf16_bits
    NB = q.shape[0] // 32
    Q = q.reshape(NB, 32, NCOL, NSUB, SUBK).transpose(0, 2, 3, 1, 4)          # [NB][c][sub][32 n][96 k]
    t = Q.reshape(NB, NCOL, NSUB, 4, 8, SUBK // 8, 8).transpose(0, 1, 2, 3, 5, 4, 6)
    lanes = (t.reshape(NB, NCOL, NSUB, 4, SUBK // 8, 64) & 15).astype(np.uint8)
    nib = (lanes[..., 0::2] | (lanes[..., 1::2] << 4)).reshape(NB, NCOL, NSUB, -1)
    S = sc.reshape(NB, 32, NCOL, NSUB, 3).transpose(0, 2, 3, 4, 1)             # [NB][c][sub][g][n]
    s = np.ascontiguousarray(bf16_bits(S)).view(np.uint8).reshape(NB, NCOL, NSUB, -1)
    return np.concatenate([s, nib], -1)


def head_stream():
    """[column][group NWP/GW][row][GW elements] of 1728 B, cached (566 MB)."""
    if os.path.exists(STREAM):
        return np.load(STREAM, mmap_mode="r")
    import head_ref
    from rmlp_ref import gain_elements
    man = head_ref.load_manifest()
    e, K, Vv, group, rg = head_ref.head_meta(man)
    assert (K, Vv) == (KH, V)
    raw = np.memmap(head_ref.blob_path(e["blob"]), dtype=np.uint8, mode="r", shape=(e["length"],))
    el = np.lib.format.open_memmap(STREAM + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(NCOL, NWP, NROW, ELEM))
    el[:] = 0
    g = head_ref.load_norm_weight(man)
    for c in range(NCOL):
        ge = gain_elements(g, np.float32(1.0), c).reshape(2, NROW, ELEM)
        el[c, 0:2] = ge
    CH = 64                                                   # down blocks a chunk
    for r in range(NROW):
        for n0 in range(0, NDB, CH):
            v0 = r * (V // NROW) + n0 * 32
            q, sc = head_ref.unplanar_chunk(raw, v0, v0 + CH * 32, K, rg)
            pb = pack_block(q, sc)                                # [CH][c][sub][1728]
            for c in range(NCOL):
                el[c, 2 + n0 * NSUB:2 + (n0 + CH) * NSUB, r] = pb[:, c].reshape(-1, ELEM)
    out = el.reshape(NCOL, NWP // GW, GW, NROW, ELEM).transpose(0, 1, 3, 2, 4)
    fin = np.lib.format.open_memmap(STREAM + ".fin.npy", mode="w+", dtype=np.uint8, shape=out.shape)
    for c in range(NCOL):
        fin[c] = out[c]
    fin.flush()
    del fin, el
    os.remove(STREAM + ".tmp.npy")
    os.replace(STREAM + ".fin.npy", STREAM)
    return np.load(STREAM, mmap_mode="r")


def steps(L):
    s = [(L.GAINS,), (L.NORM_PRE, 0), (L.DOWN, NDB, 1, NSUB, L.KB_Q)]
    return s + ([(L.SYNC, L.SY_W_DRAIN, NWP - NW)] if NWP > NW else [])


def mem_bufs():
    return [("XYH", 8640, XYH, "head"), ("WSH", 2 * NROW * GW * ELEM, WSH, "head"), ("RL", 2 * RL_BLK * 1024, RL, "head")]


def sequence(L, m):
    """h1: x rows from %x (row 0 the one wanted), logits to %o [16][262144] bf16."""
    nt = 1
    X = nt * L.XB
    I = 6
    xs, ws, os_ = f"%x : memref<{L.xbuf()}xi8>", f"%w : memref<{NCOL * WCOL}xi8>", f"%o : memref<{OUTB}xi8>"
    sig = ", ".join((xs, os_, ws) if L.ARENA else (xs, ws, os_))
    m(f"aie.runtime_sequence @h1({sig}) {{")
    m("%ntv = arith.constant 1 : i32", I)
    kb_, ke_ = L.step_range("head")
    m(f"%kbv = arith.constant {kb_} : i32", I)
    m(f"%kev = arith.constant {ke_} : i32", I)
    for c in range(NCOL):
        for r in range(NROW):
            t = L.core_name(c, r)
            m(f"aiex.npu.rtp_write(@rtp_{t}, 0, %ntv) : i32", I)
            m(f"aiex.npu.rtp_write(@rtp_{t}, 1, %kbv) : i32", I)
            m(f"aiex.npu.rtp_write(@rtp_{t}, 2, %kev) : i32", I)
    for c in range(NCOL):
        for r in range(NROW):
            m(f"aiex.set_lock(%wp_{c}_{r}, 2)", I)
            m(f"aiex.set_lock(%wc_{c}_{r}, 0)", I)
        for nm, v in (("xf", 1), ("xp", 0), ("xw", 1), ("xr", 0), ("yf", 2), ("yr", 0)):
            m(f"aiex.set_lock(%{nm}_{c}, {v})", I)
    for c in range(NCOL):
        for r in range(NROW):
            t = L.core_name(c, r)
            if L.own_xlocks(c, r):
                m(f"aiex.set_lock(%cxp_{t}, 1)", I)
                m(f"aiex.set_lock(%cxc_{t}, 0)", I)
    for r in range(NROW):
        t = L.core_name(NCOL - 1, r)
        m(f"aiex.set_lock(%cyp_{t}, 2)", I)
        for j in range(2):
            m(f"aiex.set_lock(%cyc{j}_{t}, 0)", I)
    for c in range(NCOL):
        for r in range(NROW):
            m(f"aiex.set_lock(%cgo_{L.core_name(c, r)}, 1)", I)

    def task(name, tile, d, ch, body, attrs="", pkt=""):
        m(f"%{name} = aiex.dma_configure_task(%{tile}, {d}, {ch}{pkt}) {{", I)
        m("%one = arith.constant 1 : i32", I + 2)
        for line in body:
            m(line, I + 2)
        m("aie.end", I + 2)
        m("}" + attrs, I)
        m(f"aiex.dma_start_task(%{name})", I)

    def locked(acq, bd, rel):
        return [f"aie.use_lock(%{acq}, AcquireGreaterEqual, %one)", bd, f"aie.use_lock(%{rel}, Release, %one)"]

    def chain(parts):
        out = []
        for i, p in enumerate(parts):
            if i:
                out += [f"aie.next_bd ^b{i}", f"^b{i}:"]
            out += p
        return out

    GE = GW * ELEM
    for c in range(NCOL):
        W_ = f"%WSH_{c} : memref<{2 * NROW * GE}xi8>"
        task(f"hwi{c}", f"mem_{c}", "S2MM", 0, chain([locked(f"wp_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NROW + r) * GE} len = {GE}) {{bd_id = {s_ * NROW + r} : i32}}", f"wc_{c}_{r}")
                                                      for s_ in range(2) for r in range(NROW)]), f" {{repeat_count = {NWP // (2 * GW) - 1} : i32}}")
        for r in range(NROW):
            ids = ((8, 9), (24, 25), (10, 11), (26, 27))[r]
            task(f"hwo{c}_{r}", f"mem_{c}", "MM2S", r, chain([locked(f"wc_{c}_{r}", f"aie.dma_bd({W_} offset = {(s_ * NROW + r) * GE} len = {GE}) {{bd_id = {ids[s_]} : i32}}", f"wp_{c}_{r}")
                                                             for s_ in range(2)]), f" {{repeat_count = {NWP // (2 * GW) - 1} : i32}}")
    for c in range(NCOL):
        task(f"hsw{c}", f"shim_{c}", "MM2S", 0, [f"aie.dma_bd(%w : memref<{NCOL * WCOL}xi8> offset = {c * WCOL} len = {WCOL})"])
        task(f"hsx{c}", f"shim_{c}", "MM2S", 1, [f"aie.dma_bd(%x : memref<{L.xbuf()}xi8> offset = {c * L.XROW} len = {X} sizes = [{16 * nt}, {L.XROW}] strides = [{L.D * 2}, 1])"])
    for c in range(NCOL):
        XR_, XY_, RL_ = f"%XR_{c} : memref<{L.XRB}xi8>", f"%XYH_{c} : memref<8640xi8>", f"%RL_{c} : memref<{2 * RL_BLK * 1024}xi8>"
        wbq = (3, 30) if c < NCOL - 1 else (2, 17)
        task(f"hxin{c}", f"mem_{c}", "S2MM", 1, locked(f"xf_{c}", f"aie.dma_bd({XR_} offset = 0 len = {X}) {{bd_id = 28 : i32}}", f"xp_{c}"))
        task(f"hnb{c}", f"mem_{c}", "MM2S", 4, locked(f"xp_{c}",
             f"aie.dma_bd({XR_} offset = 0 len = {2 * nt * (L.UNITB + L.XROW)} sizes = [{2 * nt}, 3, {(L.UNITB + L.XROW) // 3}] "
             f"strides = [{L.UNITB}, {(L.UNITB + L.XROW) // 3}, 1]) {{bd_id = 12 : i32}}", f"xp_{c}"))
        task(f"hxl{c}", f"mem_{c}", "S2MM", wbq[0], locked(f"xw_{c}", f"aie.dma_bd({XY_} offset = 0 len = 8640) {{bd_id = {wbq[1]} : i32}}", f"xr_{c}"))
        task(f"hbc{c}", f"mem_{c}", "MM2S", 4, chain([locked(f"xr_{c}", f"aie.dma_bd({XY_} offset = 0 len = 8640) {{bd_id = {(13, 14, 15, 16, 18, 19, 20, 23)[i]} : i32}}", f"xr_{c}")
                                                      for i in range(BC_CHAIN)]), f" {{repeat_count = {NDB // BC_CHAIN - 1} : i32}}")
        # column 7's rows relay: 16 blocks a slot in, out to the logits at this MemTile's 32768 columns
        task(f"hrin{c}", f"mem_{c}", "S2MM", 2, chain([locked(f"yf_{c}", f"aie.dma_bd({RL_} offset = {i * RL_BLK * 1024} len = {RL_BLK * 1024}) {{bd_id = {21 + i} : i32}}", f"yr_{c}")
                                                       for i in range(2)]), f" {{repeat_count = {NDB // 2 // RL_BLK // 2 - 1} : i32}}")
        task(f"hrout{c}", f"mem_{c}", "MM2S", 5, chain([locked(f"yr_{c}", f"aie.dma_bd({RL_} offset = {i * RL_BLK * 1024} len = {RL_BLK * 1024}) {{bd_id = {32 + i} : i32}}", f"yf_{c}")
                                                        for i in range(2)]), f" {{repeat_count = {NDB // 2 // RL_BLK // 2 - 1} : i32}}")
        base = (c // 2) * (V // NROW) + (c % 2) * (V // NROW // 2)
        task(f"hso{c}", f"shim_{c}", "S2MM", 0,
             [f"aie.dma_bd(%o : memref<{OUTB}xi8> offset = {2 * base} len = {NDB // 4 * 1024} "
              f"sizes = [2, {NDB // 4}, 16, 64] strides = [{NDB // 4 * 64}, 64, {2 * V}, 1])"], " {repeat_count = 1 : i32, issue_token = true}")
    for c in range(NCOL - 1):
        t = L.core_name(c, 0)
        task(f"hwx_{t}", f"t_{t}", "MM2S", 0, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{L.WB}xi8> offset = {L.OUT_OFF} len = {L.ABLK}) {{bd_id = 8 : i32}}", f"cxp_{t}"))
    for r in range(NROW):
        t = L.core_name(NCOL - 1, r)
        if r == NROW - 1:
            task(f"hwx_{t}", f"t_{t}", "MM2S", 1, locked(f"cxc_{t}", f"aie.dma_bd(%wblk_{t} : memref<{L.WB}xi8> offset = {L.OUT_OFF} len = {L.ABLK}) {{bd_id = 14 : i32}}", f"cxp_{t}"),
                 "", L.c7pkt(r, 1))
        for j in range(2):
            task(f"hqd{j}_{r}", f"t_{t}", "MM2S", j,
                 chain([locked(f"cyc{j}_{t}", f"aie.dma_bd(%gscr_{t} : memref<1024xi16> offset = {512 * (i % 2)} len = 512) {{bd_id = {8 + 4 * j + i if r < NROW - 1 or j == 0 else (4, 5, 6, 7)[i]} : i32}}", f"cyp_{t}")
                        for i in range(QD_CHAIN)]),
                 f" {{repeat_count = {NDB // 2 // QD_CHAIN - 1} : i32}}", L.c7pkt(r, j))
    for c in range(NCOL):
        m(f"aiex.dma_await_task(%hso{c})", I)
    m("}", 4)
