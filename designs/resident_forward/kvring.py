"""The forward's sliding cache (`sring=C`): one ring of C rows per sliding layer, in 64-row blocks.

Block b holds positions p with p % C in [64b, 64b + 64) as [kv head 8][K, V][slice 4][key 64][64]:
one (block, head, K|V) is a contiguous slab of SLAB elements in the order the attention streams it,
so a window read is whole slabs one block stride apart. A command moves that data in two spans per
stream, each a runtime offset plus a runtime count on a one-granule BD (length_parameter):

  write  16*nt rows from position s: span A to the end of s's block (at most 16*nt - 1 rows),
         span B the rest at the next position's slot; a span never crosses a block, so never the ring
  read   the window's NBW blocks from block (first % C) / 64: span A to the ring end (at most
         NBW - 1 blocks), span B the rest from block 0

Both spans are at least one granule, which a length_parameter BD needs. `params` is the one rule
for the values; `check` walks every value a host can pass and proves each BD stays in its slab and
in the region, from the same region size that sizes the region (rforward.s_rows)."""

BLK = 64                         # rows per block: the attention's key block
DEFAULT_WINDOW_ROWS = 1024
HEADS, HD, SW = 8, 256, 64
SLAB = 4 * BLK * SW              # elements of one (block, head, K|V), [slice][key][64]
BLOCK_E = HEADS * 2 * SLAB       # elements of one block: BLK rows of [8][2][256]
WRITE = ("kvw_a", "kvw_an", "kvw_b", "kvw_bn")
READ = ("kvr_a", "kvr_an", "kvr_b", "kvr_bn")
PARAMS = WRITE + READ            # (offset, extra granules) of span A, then of span B
COUNTS = ("kvw_an", "kvw_bn", "kvr_an", "kvr_bn")


def slot_elem(slot):
    """Element offset of a ring slot's key within its block's slabs."""
    return slot // BLK * BLOCK_E + slot % BLK * SW


def write_spans(s, rows, C):
    """[(first slot, rows)] of span A and span B for rows positions from s."""
    r = s % C
    a = min(rows - 1, BLK - r % BLK)
    return [(r, a), ((s + a) % C, rows - a)]


def read_spans(first, nbw, C):
    """[(first block, blocks)] of span A and span B for the nbw-block window from first."""
    nb, b0 = C // BLK, first % C // BLK
    na = min(nbw, nb - b0)
    if na == nbw:
        na = nbw - 1
        return [(b0, na), ((b0 + na) % nb, 1)]
    return [(b0, na), (0, nbw - na)]


def params(s, rows, first, nbw, C):
    """Every ring parameter for a command: offsets in elements of the region, counts in granules
    past the BD's static one."""
    (wa, na), (wb, nb_) = write_spans(s, rows, C)
    (ra, ma), (rb, mb) = read_spans(first, nbw, C)
    return {"kvw_a": slot_elem(wa), "kvw_an": na - 1, "kvw_b": slot_elem(wb), "kvw_bn": nb_ - 1,
            "kvr_a": ra * BLOCK_E, "kvr_an": ma - 1, "kvr_b": rb * BLOCK_E, "kvr_bn": mb - 1}


def write_bd(c, kv, span):
    """(static offset, len, sizes, strides, granule, params) of column c's K (kv 0) or V write BD:
    a granule is one row, its 4 slices of 64 one slab-slice apart; rows step one key."""
    po, pn = WRITE[2 * span:2 * span + 2]
    return (c * 2 * SLAB + kv * SLAB, HD, [1, 1, 4, SW], [0, SW, BLK * SW, 1], HD, po, pn)


def read_bd(c, kv, span):
    """The window read BD: a granule is one whole slab, granules one block apart."""
    po, pn = READ[2 * span:2 * span + 2]
    return (c * 2 * SLAB + kv * SLAB, SLAB, [1, 1, BLK, HD], [0, BLOCK_E, HD, 1], SLAB, po, pn)


def bd_text(buf, bd):
    off, ln, sz, st, gr, po, pn = bd
    return (f"aie.dma_bd({buf} offset = {off} len = {ln} sizes = [{', '.join(map(str, sz))}] "
            f"strides = [{', '.join(map(str, st))}]) {{offset_parameter = @{po}, length_parameter = @{pn}, "
            f"length_granule = {gr} : i32}}")


def extent(bd, p):
    """[lo, hi) elements a BD touches with parameters p, and the slab of every granule."""
    off, ln, sz, st, gr, po, pn = bd
    inner = sum((n - 1) * s for n, s in zip(sz[2:], st[2:])) + 1
    assert sz[2] * sz[3] == ln == gr and sz[:2] == [1, 1], bd
    lo = off + p[po]
    starts = [lo + g * st[1] for g in range(p[pn] + 1)]
    return lo, starts[-1] + inner, starts, inner


def check(C, nbw, nts, win, region_elems):
    """K059 at build time: for every ring slot s % C, every piece family nt and every window block,
    each write granule lies in one slab of the region and each read granule is a whole slab; the
    padding rows a command writes past its valid ones are never in a query's window."""
    assert C % BLK == 0 and region_elems == C // BLK * BLOCK_E, (C, region_elems)
    assert 2 <= nbw <= C // BLK and nbw * BLK >= win + BLK - 1 + 16 * max(nts), (nbw, C, win, nts)
    for nt in nts:
        rows = 16 * nt
        assert 2 <= rows <= BLK + 1 and rows <= C - win + 1, (nt, C, win)
        for s in range(C):
            p = params(s, rows, 0, nbw, C)
            got = 0
            for span in range(2):
                for c in range(HEADS):
                    for kv in range(2):
                        lo, hi, starts, inner = extent(write_bd(c, kv, span), p)
                        assert 0 <= lo and hi <= region_elems, (s, nt, span, lo, hi)
                        assert lo // SLAB == (hi - 1) // SLAB, (s, nt, span, "write crosses a slab")
                got += p[WRITE[2 * span + 1]] + 1
            assert got == rows, (s, nt, got)
    for b in range(C // BLK):
        p = params(0, 16, b * BLK, nbw, C)
        blocks = []
        for span in range(2):
            for c in range(HEADS):
                for kv in range(2):
                    lo, hi, starts, inner = extent(read_bd(c, kv, span), p)
                    assert 0 <= lo and hi <= region_elems, (b, span, lo, hi)
                    assert all((g - c * 2 * SLAB - kv * SLAB) % BLOCK_E == 0 for g in starts), (b, span)
            blocks += [(p[READ[2 * span]] // BLOCK_E + g) for g in range(p[READ[2 * span + 1]] + 1)]
        assert blocks == [(b + j) % (C // BLK) for j in range(nbw)], (b, blocks)


def describe(C, nbw, region_elems, elem_bytes=2):
    """What a host needs to compute and bound the ring parameters, in bytes (fwd_layout.json)."""
    def d(bd_of):
        bds = [bd_of(c, kv, span) for c in range(HEADS) for kv in range(2) for span in range(2)]
        assert len({b[1:5] for b in [(o, ln, tuple(sz), tuple(st), gr) for o, ln, sz, st, gr, _, _ in bds]}) == 1
        off, ln, sz, st, gr, _, _ = bds[0]
        inner = sum((n - 1) * s for n, s in zip(sz[2:], st[2:])) + 1
        return {"min_off": min(b[0] for b in bds) * elem_bytes, "max_off": max(b[0] for b in bds) * elem_bytes,
                "extent": inner * elem_bytes, "granule_stride": st[1] * elem_bytes}
    return {"block_rows": BLK, "blocks": C // BLK, "window_blocks": nbw, "slab_bytes": SLAB * elem_bytes,
            "block_bytes": BLOCK_E * elem_bytes, "row_bytes": SW * elem_bytes, "region_bytes": region_elems * elem_bytes,
            "write": d(write_bd), "read": d(read_bd), "params": list(PARAMS), "counts": list(COUNTS)}


def rows(region_u16, slots):
    """Ring slots as cache rows [n][8 heads][K, V][256] (uint16), from a region view."""
    import numpy as np
    blk = region_u16.reshape(-1, HEADS, 2, 4, BLK, SW)
    slots = np.asarray(slots)
    out = blk[slots // BLK, :, :, :, slots % BLK, :]        # [n][8][2][4][64]
    return out.reshape(len(slots), HEADS * 2 * HD)


def fits(desc, p):
    """A host's K059 check of ring parameters p (offsets in bytes, counts in granules) against the
    build's own layout record (describe, via fwd_layout.json): every span inside the region, a write
    span inside one slab, a read span on whole blocks. Returns the failures (empty when it fits)."""
    bad = []
    for kind, pre in (("write", "kvw"), ("read", "kvr")):
        d = desc[kind]
        for sp in ("a", "b"):
            off, n = p[f"{pre}_{sp}"], p[f"{pre}_{sp}n"]
            hi = d["max_off"] + off + n * d["granule_stride"] + d["extent"]
            if off < 0 or n < 0 or hi > desc["region_bytes"]:
                bad.append(f"{pre}_{sp}: [{d['min_off'] + off}, {hi}) outside the {desc['region_bytes']}-byte region")
            if kind == "write" and off % desc["slab_bytes"] + n * d["granule_stride"] + d["extent"] > desc["slab_bytes"]:
                bad.append(f"{pre}_{sp}: {n + 1} rows from byte {off} cross a slab")
            if kind == "read" and (off % desc["block_bytes"] or n + 1 > desc["blocks"]):
                bad.append(f"{pre}_{sp}: {n + 1} blocks from byte {off} are not whole blocks of the ring")
    return bad


def put_rows(region_u16, slots, rows_u16):
    """Write cache rows [n][8 heads][K, V][256] at ring slots (the inverse of rows)."""
    import numpy as np
    blk = region_u16.reshape(-1, HEADS, 2, 4, BLK, SW)
    slots = np.asarray(slots)
    blk[slots // BLK, :, :, :, slots % BLK, :] = rows_u16.reshape(len(slots), HEADS, 2, 4, SW)
