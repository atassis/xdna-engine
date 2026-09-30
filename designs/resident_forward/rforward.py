"""The whole forward in one command (rlayer_design `fwd=lo-hi[+h]`): sequence f{nt} runs layers
lo..hi, then (+h, f1 only) the LM head, by splicing each layer's own control code (the p/g sequences'
bodies) into one sequence and re-targeting its BDs onto three arguments in FusedArena's order:

  %x input   [x rows 16*PCAP_T x 3840 bf16][RoPE sliding][RoPE global][widths sliding][widths global]
  %o output  [x slot 0][x slot 1][logits 16 x 262144 bf16]; layer i reads slot i%2 (layer lo: %x's
             rows) and writes slot (i+1)%2
  %s weights: each layer's weight stream in order, then the head's; every BD into it is static
  %k caches: each layer's K/V cache in order (a sliding layer's ring, a global layer's rows and
     its split scratch); every BD whose address moves with the position lands here

The K/V addresses that move with the position are scratchpad parameters added to the BDs at run
time, in %k elements: kvw_g the global layer's first new row (global windows start at key 0). A
sliding cache is linear (kvw_s, kvr_s the first new row and the window's first key row) or, with
`sring=C`, one ring of C rows in 64-row blocks whose BDs and parameters are kvring's. %s and %k are
i32 words, or i64 when either passes 8 GiB (the BD offset is an I32 element count); Plan.S_UNIT is
the element size in bytes.

Rungs: the global layers' window is a sequence property. f1 is the first `split=` rung, f1s{nb} the
others; f2 the first `seg=` rung, f2w{nb} the others. A host picks the smallest rung whose window
(nb * 64 keys) holds n_past + P."""
import json
import os
import re
import numpy as np
import rattn2_design as A2
import rf_paths

STORE = os.environ.get("RF_STORE", str(rf_paths.ARTIFACTS / "store")) + "/manifest.json"
S_CAP = 2048                 # sliding cache rows (rld_run's linear cache)
G_CAP_DEFAULT = 4096         # global cache rows without `gcap=` (the w64 rung's window)
I32_REACH = (2 ** 31 - 1) * 4


def g_cap(L):
    return L.GCAP or G_CAP_DEFAULT


def s_rows(L):
    """A sliding cache's rows: the ring's C, else the linear S_CAP."""
    return L.SRING_FWD or S_CAP


def params(L):
    """The scratchpad parameters, in declaration order."""
    import kvring
    return ("kvw_g",) + kvring.PARAMS if L.SRING_FWD else ("kvw_s", "kvw_g", "kvr_s")


def rung_name(nt, nb, first):
    return f"f{nt}" if first else (f"f1s{nb}" if nt == 1 else f"f{nt}w{nb}")


def global_layers():
    return set(json.load(open(STORE))["full_attention_layers"])


class Plan:
    """Byte layout of the three arguments for layers lo..hi (+ head)."""

    def __init__(self, L, lo, hi, head):
        import rhead
        self.lo, self.hi, self.head = lo, hi, head
        gl = global_layers()
        self.types = {li: li in gl for li in range(lo, hi + 1)}
        L.set_geo(False)
        self.XROWS = 16 * L.PCAP_T * L.D * 2
        self.ROPE = L.PCAP_T * L.ROPE_T
        self.xbuf_l = L.xbuf()
        self.OB = L.obuf()
        wb = {False: L.wbytes()}
        L.set_geo(True)
        wb[True] = L.wbytes()
        L.set_geo(False)
        self.RS, self.RG = self.XROWS, self.XROWS + self.ROPE
        self.WS, self.WG = self.XROWS + 2 * self.ROPE, self.XROWS + 2 * self.ROPE + A2.WB
        self.XF = self.WG + A2.WB
        self.LOGITS = 2 * self.OB
        self.OF = self.LOGITS + (rhead.OUTB if head else 0)
        self.wbase, self.kvbase, a, k = {}, {}, 0, 0
        for li, g in self.types.items():
            self.wbase[li] = a
            a += wb[g]
            self.kvbase[li] = k
            if g and L.SPLITS:          # the largest split rung's window, then its DDR scratch (rsplitl)
                import rsplitl
                k += max(g_cap(L) * 1024, rsplitl.kvr_elems(max(L.SPLITS))) * 2
            else:
                k += (g_cap(L) * 1024 if g else s_rows(L) * L.KVROW) * 2
        self.hbase = a
        self.SF = -(-(a + (rhead.NCOL * rhead.WCOL if head else 0)) // 8) * 8
        self.KF = -(-k // 8) * 8
        self.S_UNIT = 4 if max(self.SF, self.KF) <= I32_REACH and not L.S64 else 8
        self.S_TY = f"i{8 * self.S_UNIT}"
        assert max(self.SF, self.KF) <= (2 ** 31 - 1) * self.S_UNIT, (self.SF, self.KF)
        assert all(v % 8 == 0 for v in list(self.wbase.values()) + list(self.kvbase.values()) + [self.hbase])


BD = re.compile(r"aie\.dma_bd\((%\w+) : memref<(\d+)x(\w+)> offset = (\d+) len = (\d+)"
                r"(?: sizes = \[([^\]]*)\] strides = \[([^\]]*)\])?\)(.*)$")


def retarget(line, plan, li, glob, kind):
    """One BD line of layer li (kind "layer" or "head") onto %x/%o/%s."""
    mm = BD.search(line)
    if not mm:
        return line
    name, n, ty, off, ln, sizes, strides, rest = mm.groups()
    off, ln = int(off), int(ln)
    scale, param = 1, None
    if name == "%x":
        if off < plan.XROWS:                               # x rows: the input or the previous layer's slot
            if li == plan.lo and kind == "layer":
                new, noff = "%x", off
            else:
                src = (li - plan.lo) % 2 if kind == "layer" else (plan.hi + 1 - plan.lo) % 2
                new, noff = "%o", src * plan.OB + off
        elif off >= plan.xbuf_l - A2.WB:                   # widths
            new, noff = "%x", (plan.WG if glob else plan.WS) + off - (plan.xbuf_l - A2.WB)
        else:                                              # RoPE
            new, noff = "%x", (plan.RG if glob else plan.RS) + off - plan.XROWS
    elif name == "%o":
        if kind == "head":
            new, noff = "%o", plan.LOGITS + off
        else:
            new, noff = "%o", ((li - plan.lo + 1) % 2) * plan.OB + off
    elif name == "%w":
        new, noff = "%s", (plan.hbase if kind == "head" else plan.wbase[li]) + off
    elif name in ("%kvw", "%kvr"):
        scale = 2
        new, noff = "%k", plan.kvbase[li] + 2 * off
        param = ("kvw_g" if glob else "kvw_s") if name == "%kvw" else (None if glob else "kvr_s")
        if "offset_parameter" in rest:                   # the BD names its own (kvring)
            param = None
        if os.environ.get("RF_FWD_NOPARAM") == "1":      # bisection: static K/V addresses (position 0)
            param = None
    else:
        return line
    ln *= scale
    sz = st = None
    if sizes is not None:
        sz = [int(v) for v in sizes.split(",")]
        st = [int(v) for v in strides.split(",")]
        if scale != 1:
            sz[-1] *= scale
            st = [v * scale for v in st[:-1]] + [st[-1]]
    ty = "i8"
    if new in ("%s", "%k"):             # bytes -> scratch elements
        u, ty = plan.S_UNIT, plan.S_TY
        assert noff % u == 0 and ln % u == 0, (line, noff, ln)
        noff, ln = noff // u, ln // u
        if sz is not None:
            assert sz[-1] % u == 0 and all(v % u == 0 for v in st[:-1]) and st[-1] == 1, line
            sz[-1] //= u
            st = [v // u for v in st[:-1]] + [1]
        gm = re.search(r"length_granule = (\d+) : i32", rest)
        if gm:                          # a granule of source elements, in scratch elements
            g_ = int(gm.group(1)) * scale
            assert g_ % u == 0, line
            rest = rest[:gm.start(1)] + str(g_ // u) + rest[gm.end(1):]
    size = {"%x": plan.XF, "%o": plan.OF, "%s": plan.SF // plan.S_UNIT, "%k": plan.KF // plan.S_UNIT}[new]
    body = f"aie.dma_bd({new} : memref<{size}x{ty}> offset = {noff} len = {ln}"
    if sz is not None:
        body += f" sizes = [{', '.join(map(str, sz))}] strides = [{', '.join(map(str, st))}]"
    body += ")"
    rest = rest.strip()
    if param:
        if rest.startswith("{"):
            rest = "{offset_parameter = @" + param + ", " + rest[1:]
        else:
            rest = ("{offset_parameter = @" + param + "} " + rest).strip()
    return line[:mm.start()] + body + ((" " + rest) if rest else "")


def splice(lines, plan, li, glob, kind, tag):
    """A sequence's lines (header and closing brace dropped) -> the spliced body."""
    body = lines[1:-1]
    top = {m_.group(1) for ln in body if (m_ := re.match(r"^      %(\w+) = ", ln))}
    pat = re.compile(r"%(\w+)")
    ren = lambda g: f"%{g.group(1)}_{tag}" if g.group(1) in top else g.group(0)
    out = [retarget(pat.sub(ren, ln), plan, li, glob, kind) for ln in body]
    # A layer's control code ends at its x_out readout while some of its shim and MemTile tasks (the
    # weight distribution's tail) can still run; a separate command gets host latency before the
    # next one resets their locks and reuses their BD ids, a spliced layer does not. So every such
    # task left over is given a token and awaited; core-tile tasks have no token route (K057) and
    # all finish before the write-back the readout waits on.
    done = {m_.group(1) for ln in out if (m_ := re.match(r"^      aiex\.dma_(?:free|await)_task\(%(\w+)\)", ln))}
    left, tile_of, i = [], {}, 0
    while i < len(out):
        m_ = re.match(r"^      %(\w+) = aiex\.dma_configure_task\(%(\w+),", out[i])
        if m_ and m_.group(1) not in done:
            name, tile = m_.groups()
            left.append(name)
            tile_of[name] = tile
            if tile.startswith(("shim_", "mem_")):
                j = i + 1
                while not out[j].startswith("      }"):
                    j += 1
                cl = out[j]
                if "issue_token" not in cl:
                    out[j] = "      } {issue_token = true}" if cl.strip() == "}" else cl.replace("} {", "} {issue_token = true, ", 1)
        i += 1
    # RF_FWD_AWAIT: none (default: 48-layer f1 exact without them), shim, all
    mode = os.environ.get("RF_FWD_AWAIT", "none")
    kinds = {"all": ("shim_", "mem_"), "shim": ("shim_",), "none": ()}[mode]
    aw = [t for t in left if tile_of[t].startswith(kinds)]
    if mode != "all":           # no token on a task that is freed
        for k, ln in enumerate(out):
            m_ = re.match(r"^      %(\w+) = aiex\.dma_configure_task", ln)
            if m_ and m_.group(1) in left and m_.group(1) not in aw and tile_of[m_.group(1)].startswith(("shim_", "mem_")):
                j = k + 1
                while not out[j].startswith("      }"):
                    j += 1
                if out[j] == "      } {issue_token = true}":
                    out[j] = "      }"
                else:
                    out[j] = out[j].replace("{issue_token = true, ", "{", 1)
    out += [f"      aiex.dma_await_task(%{t})" for t in aw]
    out += [f"      aiex.dma_free_task(%{t})" for t in left if t not in aw]
    return out


def sequence(L, m, nt, lo, hi, head, nb=None, first=True):
    """f{nt} (or the rung's name): layers lo..hi, global layers on rung nb (f1: the split's blocks,
    f2: the segmented window's; None: the w64 rung), then the head when asked (f1)."""
    import rhead
    from m1_design import M
    plan = Plan(L, lo, hi, head)
    name = rung_name(nt, nb, first)
    m(f"aie.runtime_sequence @{name}(%x : memref<{plan.XF}xi8>, %o : memref<{plan.OF}xi8>, "
      f"%s : memref<{plan.SF // plan.S_UNIT}x{plan.S_TY}>, %k : memref<{plan.KF // plan.S_UNIT}x{plan.S_TY}>) {{")
    for li, g in plan.types.items():
        if li > lo and os.environ.get("RF_FWD_RELOAD") == "1":      # the per-command boot, between layers
            m("aiex.npu.load_pdi {device_ref = @main}", 6)
        t = M()
        A2.NBW = L.NBW_S         # rglobal.issue leaves the global rung here
        L.SRING = None if g else L.SRING_FWD
        L.set_geo(g)
        if g and nt == 1 and L.SPLITS:            # decode: the key range split
            L.sequence(t, 1, nb or L.SPLIT_NB, split=True)
        else:
            L.sequence(t, nt, (nb or 64) if g and L.SEG else None)
        L.set_geo(False)
        L.SRING = None
        for ln in splice(t.lines, plan, li, g, "layer", f"L{li}"):
            m.lines.append(ln)
    if head and nt == 1:
        t = M()
        rhead.sequence(L, t)
        for ln in splice(t.lines, plan, plan.hi, False, "head", "H"):
            m.lines.append(ln)
    m("}", 4)
    A2.NBW = L.NBW_S
    return plan


def decls(L):
    return [f"aiex.scratchpad_parameter @{p} : i32" for p in params(L)]


def layout(L):
    """What a host needs to drive the forward sequences, as JSON (rf_build writes fwd_layout.json)."""
    lo, hi, head = L.FWD
    p = Plan(L, lo, hi, head)
    rungs = []
    for nt, rs in L.f_rungs():
        for i, nb in enumerate(rs or (None,)):
            blocks = nb or 64
            rungs.append({"name": rung_name(nt, nb, i == 0), "nt": nt, "blocks": blocks, "keys": blocks * 64,
                          "kind": "split" if nt == 1 and L.SPLITS else "seg"})
    out = {"range": [lo, hi], "head": head, "XF": p.XF, "OF": p.OF, "SF": p.SF, "KF": p.KF, "s_unit": p.S_UNIT,
           "params": list(params(L)), "param_unit_bytes": p.S_UNIT, "g_cap": g_cap(L), "s_cap": S_CAP, "s_ring": L.SRING_FWD,
           "s_rows": s_rows(L), "wbase": p.wbase, "kvbase": p.kvbase, "hbase": p.hbase,
           "global_layers": sorted(li for li, g in p.types.items() if g), "rungs": rungs}
    if L.SRING_FWD:
        import kvring
        out["s_ring_layout"] = kvring.describe(L.SRING_FWD, L.NBW_S, s_rows(L) * L.KVROW)
    return out
