"""The global layer (layer 5) of the layer image on device: `grun.py <build> <n_past> <P,...> [reps]`
(RF_ATTN_H=1, a build of `rlayer_design m g`). Inputs, the cache before the dispatch and the
references are glob_ref's harness files; the check is a smoke check (no NaN/Inf, rel-L2 of x_out
and of the new K/V cache rows against glob_ref's stages)."""
import os
import sys
import time
import numpy as np
import pyxrt
import rf_paths
import rlayer_design as L
import rattn2_design as A2
import rglobal as G
import attn_layout as AL
from chain_ref import bf16_bits, bf16_val

REF = os.environ.get("RF_GREF", str(rf_paths.BUILD_ROOT / "scratch/globref"))
BUILD = os.environ.get("RF_BUILD", str(rf_paths.BUILD_ROOT))
WCACHE = os.environ.get("RF_GRUN_WCACHE", str(rf_paths.BUILD_ROOT / "scratch/grun/w5.npy"))
CAP = 4096
NC = L.NC


def o_k_index_device():
    """O's K order as the image produces it: sub-pass 2h + w is q head 2c+h, x V worker w's 256 dims."""
    kp = 4 * AL.KC_O
    idx = np.full((NC, kp), -1, np.int64)
    for c in range(NC):
        for j in range(kp):
            sp, jj = divmod(j, AL.KC_O)
            if jj < 256:
                h, w = divmod(sp, 2)
                idx[c, j] = (2 * c + h) * 512 + w * 256 + jj
    return idx.reshape(-1)


def weights():
    """[NC x attn_in (down form)][NC x attn_out][NC x MLP (grouped)] for layer 5, cached."""
    if os.path.exists(WCACHE):
        return np.load(WCACHE)
    import weight_store as ws
    import mlp_ref
    import rmlp_ref
    W = AL.load_global(G_LAYER)
    AL.GLOBAL.kpair = True
    AL.GLOBAL.o_k_index = o_k_index_device
    L.set_geo(True)
    a_in = L.qkv_downform(AL.attn_in_stream(AL.GLOBAL, W))
    a_out = AL.attn_out_stream(AL.GLOBAL, W)
    Wm = ws.load_mlp(G_LAYER)
    gu = np.stack([mlp_ref.gateup_stream(Wm, c) for c in range(NC)])
    mlp = L.mlp_grouped(rmlp_ref.stream(Wm, gu))
    assert a_in.size == NC * L.wcol() and a_out.size == NC * L.WCOL_O, (a_in.size, a_out.size)
    L.set_geo(False)
    w = np.concatenate([a_in, a_out, mlp])
    os.makedirs(os.path.dirname(WCACHE), exist_ok=True)
    np.save(WCACHE, w)
    return w


G_LAYER = 5


def cache_rows(c):
    """glob_ref's [rows][K, V][512] bits -> the image's rows [K 512 | V slice-permuted 512]."""
    out = np.empty((c.shape[0], 1024), np.uint16)
    out[:, :512] = c[:, 0]
    v = c[:, 1].reshape(-1, 8, 64)
    vp = np.empty_like(v)
    for s in range(8):
        vp[:, G.VSLOT[s]] = v[:, s]
    out[:, 512:] = vp.reshape(-1, 512)
    return out


def uncache_v(rows):
    vp = rows[:, 512:].reshape(-1, 8, 64)
    v = np.empty_like(vp)
    for s in range(8):
        v[:, s] = vp[:, G.VSLOT[s]]
    return v.reshape(-1, 512)


def rel(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    build, n_past = sys.argv[1], int(sys.argv[2])
    ps = [int(v) for v in sys.argv[3].split(",")]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    L.ATTN, L.OPH, L.MPH, L.GLOB = True, True, True, True
    L.LAYER.oa = A2.FINISH_A = True
    L.ARENA = os.environ.get("RF_ARENA") == "1"
    A2.NBW = int(os.environ.get("RF_NBW", "20"))
    seg = int(os.environ.get("RF_SEG_NB", "0"))        # a segmented build's rung (window blocks)
    if seg:
        import rattnh
        L.SEG, rattnh.RT_NBW = (seg,), True
    cap = max(CAP, 64 * seg)
    wv_all = weights()
    L.set_geo(True)
    xbuf, wbytes, obuf, kvb = L.xbuf(), L.wbytes(), L.obuf(), L.kvbuf()
    L.set_geo(False)
    assert wv_all.size == wbytes, (wv_all.size, wbytes)
    dev = pyxrt.device(0)
    elf = pyxrt.elf(f"{BUILD}/{build}/design.elf")
    ctx = pyxrt.hw_context(dev, elf)
    names = sorted({f"g{-(-p // 16)}" + (f"w{seg}" if seg else "") for p in ps})
    kern = {n: pyxrt.ext.kernel(ctx, f"main:{n}") for n in names + ["boot"]}
    flags = pyxrt.bo.host_only
    xb = pyxrt.bo(dev, xbuf, flags, 0)
    wbo = pyxrt.bo(dev, wbytes, flags, 0)
    ob = pyxrt.bo(dev, obuf, flags, 0)
    cache = pyxrt.bo(dev, cap * G.KVROW * 2, flags, 0)
    kvw = pyxrt.bo(cache, kvb * 2, n_past * G.KVROW * 2)
    kvr = pyxrt.bo(cache, (64 * seg if seg else A2.NBW * 64) * G.KVROW * 2, 0)
    view = lambda b, n: np.frombuffer(b.map(), dtype=np.uint8, count=n)
    up = lambda b: b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    down = lambda b: b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    view(wbo, wbytes)[:] = wv_all
    up(wbo)
    old = cache_rows(np.load(f"{REF}/cache_{n_past}.npy"))

    def run(name):
        r = pyxrt.run(kern[name])
        for i, b in enumerate(L.argv(xb, wbo, ob, kvw, kvr)):
            r.set_arg(i, b)
        r.start()
        st = r.wait(4000)
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise TimeoutError(f"{name}: {st}")

    ok = True
    for p in ps:
        nt = -(-p // 16)
        f = np.load(f"{REF}/in_{n_past}_{p}.npz")
        assert int(f["nbw"]) <= (seg or A2.NBW), ("window", int(f["nbw"]), A2.NBW)
        cv = view(cache, cap * G.KVROW * 2).view(np.uint16).reshape(cap, G.KVROW)
        cv[:] = 0
        cv[:old.shape[0]] = old
        up(cache)
        buf = np.zeros(xbuf, np.uint8)
        xr = 16 * L.PCAP_T * L.D * 2
        buf[:16 * nt * L.D * 2] = f["x"].view(np.uint8).reshape(-1)
        rope = f["rope"].reshape(16 * nt, 2, 256)[:, :, :64].reshape(16 * nt, 128)
        buf[xr:xr + rope.nbytes] = np.ascontiguousarray(rope).view(np.uint8).reshape(-1)
        w = f["widths"]                                       # [nt][hi, lo][16]
        wr = np.ascontiguousarray(np.stack([w, w], axis=1)).astype(np.int32)
        buf[xbuf - A2.WB:xbuf - A2.WB + wr.nbytes] = wr.view(np.uint8).reshape(-1)
        view(xb, xbuf)[:] = buf
        up(xb)
        view(ob, obuf)[:] = 0xA5
        up(ob)
        run("boot")
        run(f"g{nt}" + (f"w{seg}" if seg else ""))
        down(ob)
        down(cache)
        got = view(ob, xr).view(np.uint16).reshape(16 * L.PCAP_T, L.D)[:p].copy()
        rows = cv[n_past:n_past + p].copy()
        st = np.load(f"{REF}/stages_{n_past}_{p}.npz")
        ref = np.load(f"{REF}/ref_m_{n_past}_{p}.npy")
        gv, rv = bf16_val(got), bf16_val(ref)
        k_rel = rel(bf16_val(rows[:, :512]), bf16_val(st["k"][:p]))
        v_rel = rel(bf16_val(uncache_v(rows)), bf16_val(st["v"][:p]))
        x_rel = rel(gv, rv)
        bad = int((~np.isfinite(gv)).sum()) + int((got == 0xA5A5).sum())
        good = bad == 0 and x_rel < 0.05 and k_rel < 0.05 and v_rel < 0.05
        ok &= good
        print(f"n_past={n_past} P={p} nt={nt}: x_out rel {x_rel:.3e}, K rows rel {k_rel:.3e}, V rows rel {v_rel:.3e}, "
              f"non-finite/unwritten {bad}" + ("" if good else "  <-- FAIL"), flush=True)
        np.save(f"{rf_paths.BUILD_ROOT}/scratch/grun/got_{build}_{n_past}_{p}.npy", got)
    if reps:
        for n in names:
            ts = []
            for _ in range(reps):
                run("boot")
                t0 = time.perf_counter()
                run(n)
                ts.append(time.perf_counter() - t0)
            print(f"{n}: median {1e3 * np.median(ts):.3f} ms over {reps}", flush=True)
    print("GLOBAL SMOKE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
