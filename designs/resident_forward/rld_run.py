"""(rld_run: step D1, the `oa` build: O's bfp16 A blocks read back and compared with f32_to_bfp16 of the
model's O in the fused layer's K order [sub-pass h: head 0 dims h*128.., head 1 dims h*128.., 32 zero].)
P3.3b step C on device: rlayer_design with `attn` (pre-attention norm, QKV, head pass, the new K/V
rows into the cache, attention), O bit-exact against the composed model: qkv_ref's head pass (the
P3.2 reference rows) feeding attn_ref.attention over the cache window, per column and pass.
`rlc_run.py <build> <nbw> <n_past> <P,...> [reps]`.

The cache is one host buffer, [position][8 kv heads][K, V][256] bf16; the dispatch gets two views
of it: %kvw at n_past (where the new rows land) and %kvr at the window's first key (the block
holding the lowest key any row of the dispatch can see). Positions before n_past hold seeded
random rows, the rest zeros."""
import os
import sys
import numpy as np
import pyxrt
import rf_paths
import rlayer_design as L
import rattn2_design as A2
from rfrun import BUILD
from chain_ref import bf16, bf16_bits
from attn_ref import attention
import attn_ref
from bfp16_model import f32_to_bfp16
from m1_design import PCAP_T, NC
from m1_run_helpers import nt_of

WIN = 1024
CAP = 2048                    # cache positions


def widths(n_past, P, nt, first):
    w = np.zeros((nt, 2, 32), np.int32)
    for tb in range(nt):
        for r in range(16):
            p = n_past + 16 * tb + r
            q = min(p, n_past + P - 1)       # padding rows take the last real row's window
            hi, lo = q - first + 1, max(0, q - (WIN - 1) - first)
            w[tb, 0, [r, 16 + r]] = hi
            w[tb, 1, [r, 16 + r]] = lo
    return w


def main():
    build, nbw, n_past = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ps = [int(v) for v in sys.argv[4].split(",")]
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    L.ATTN, A2.NBW = True, nbw
    L.LAYER.oa = A2.FINISH_A = True
    attn_ref.PV_EDGE = A2.PV_EDGE
    nk = nbw * 64
    first = max(0, n_past - (WIN - 1)) // 64 * 64
    X = np.load(str(rf_paths.BUILD_ROOT / "scratch/rmlp/ref.npz"))["x"]
    R = np.load(str(rf_paths.BUILD_ROOT / f"scratch/qkv/ref_h_{n_past}.npz"))
    heads = R["out"]                                        # [8][112][1024] bits: q, q, k, v
    dev = pyxrt.device(0)
    elf = pyxrt.elf(str(BUILD / build / "design.elf"))
    ctx = pyxrt.hw_context(dev, elf)
    names = sorted({f"p{nt_of(p)}" for p in ps})
    kern = {n: pyxrt.ext.kernel(ctx, f"main:{n}") for n in names + ["boot"]}
    flags = pyxrt.bo.host_only
    xb = pyxrt.bo(dev, L.xbuf(), flags, 0)
    wbo = pyxrt.bo(dev, NC * L.wcol(), flags, 0)
    ob = pyxrt.bo(dev, L.obuf(), flags, 0)
    cache = pyxrt.bo(dev, CAP * L.KVROW * 2, flags, 0)
    kvw = pyxrt.bo(cache, L.kvbuf() * 2, n_past * L.KVROW * 2)
    kvr = pyxrt.bo(cache, nk * L.KVROW * 2, first * L.KVROW * 2)
    view = lambda b, n: np.frombuffer(b.map(), dtype=np.uint8, count=n)
    up = lambda b: b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    down = lambda b: b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    view(wbo, NC * L.wcol())[:] = np.load(str(rf_paths.BUILD_ROOT / "scratch/qkv/wstream_h.npy"))
    up(wbo)

    def run(name):
        r = pyxrt.run(kern[name])
        for i, b in enumerate((xb, wbo, ob, kvw, kvr)):
            r.set_arg(i, b)
        r.start()
        st = r.wait(4000)
        if st != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise TimeoutError(f"{name}: {st}")

    run("boot")
    rng = np.random.default_rng(7)
    old = bf16(rng.standard_normal((CAP, 8, 2, 256)).astype(np.float32))
    old[n_past:] = 0
    ok = True
    for p in ps:
        nt = nt_of(p)
        cv = view(cache, CAP * L.KVROW * 2).view(np.uint16).reshape(CAP, 8, 2, 256)
        cv[:] = bf16_bits(old)
        up(cache)
        buf = np.zeros(L.xbuf(), np.uint8)
        x = np.zeros((16 * PCAP_T, L.D), np.uint16)
        x[:p] = bf16_bits(X[:p])
        xr = 16 * PCAP_T * L.D * 2
        buf[:xr] = x.view(np.uint8).reshape(-1)
        rp = np.zeros((16 * PCAP_T, 256), np.uint16)
        rp[:p] = R["rope"][:p]
        buf[xr:xr + PCAP_T * L.ROPE_T] = rp.view(np.uint8).reshape(-1)
        w = widths(n_past, p, nt, first)
        wr = w if not L.ATTN_H else np.ascontiguousarray(np.stack([w[:, :, :16], w[:, :, 16:]], axis=1))
        buf[L.xbuf() - A2.WB:L.xbuf() - A2.WB + wr.nbytes] = wr.view(np.uint8).reshape(-1)
        view(xb, L.xbuf())[:] = buf
        up(xb)
        view(ob, L.obuf())[:] = 0xA5
        up(ob)
        # the context loses the resident design after a few idle seconds (the reference computation
        # between dispatches is longer), so reload it first
        run("boot")
        run(f"p{nt}")
        down(ob)
        o = view(ob, NC * PCAP_T * 2 * A2.ABO).reshape(NC, PCAP_T, 2 * A2.ABO)
        full = view(ob, L.obuf())
        wr_ = np.nonzero(full != 0xA5)[0]
        print("written ranges of %o:", [(int(wr_[0]), int(wr_[-1]))] if wr_.size else [], "count", wr_.size, "per-column stride", PCAP_T * 2 * A2.ABO)
        # the reference cache: the old rows, then the head pass's k/v for rows < P (zero past P)
        new = heads.reshape(NC, 112, 4, 256).astype(np.uint16).copy()
        new[:, p:] = 0
        ref_cache = bf16_bits(old).copy()
        ref_cache[n_past:n_past + 16 * nt, :, 0] = new[:, :16 * nt, 2].transpose(1, 0, 2)
        ref_cache[n_past:n_past + 16 * nt, :, 1] = new[:, :16 * nt, 3].transpose(1, 0, 2)
        down(cache)
        got_cache = view(cache, CAP * L.KVROW * 2).view(np.uint16).reshape(CAP, 8, 2, 256)
        cache_bad = int((got_cache[n_past:n_past + p] != ref_cache[n_past:n_past + p]).sum())
        from chain_ref import bf16_val
        win = bf16_val(ref_cache[first:first + nk]).astype(np.float32)
        bad = 0
        for c in range(NC):
            for tb in range(nt):
                q = bf16_val(np.concatenate([new[c, 16 * tb:16 * tb + 16, 0], new[c, 16 * tb:16 * tb + 16, 1]])).astype(np.float32)
                ctxv = bf16_val(bf16_bits(attention(q, win[:, c, 0], win[:, c, 1], w[tb, 0], lows=w[tb, 1]))).astype(np.float32)
                for r in range(32):          # rows past P: the device's are finished from padding rows
                    if (r % 16) + 16 * tb >= p:
                        ctxv[r] = bf16_val(bf16_bits(attention(q, win[:, c, 0], win[:, c, 1], w[tb, 0], lows=w[tb, 1])))[r]
                kp = np.zeros((2, 16, 288), np.float32)                 # [sub-pass][row][k]
                for h in range(2):
                    for hh in range(2):
                        kp[h, :, hh * 128:(hh + 1) * 128] = ctxv[hh * 16:(hh + 1) * 16, h * 128:(h + 1) * 128]
                blocks = kp.reshape(2, 2, 8, 36, 8).transpose(0, 3, 1, 2, 4)   # [h][kb][r2][row][8]
                ref = f32_to_bfp16(np.ascontiguousarray(blocks).reshape(-1))
                d = (o[c, tb] != ref).reshape(2, -1)                      # [worker][ABO]
                if d.any() and c < 2 and tb == 0:
                    np.save(str(rf_paths.BUILD_ROOT / f"scratch/rldh_c{c}.npy"), np.stack([o[c, tb], ref]))
                    print(f"  c{c} tb{tb}: worker0 head0 {int(d[0, :2304].sum())} head1 {int(d[0, 2304:4608].sum())} "
                          f"pad {int(d[0, 4608:].sum())} | worker1 head0 {int(d[1, :2304].sum())} head1 {int(d[1, 2304:4608].sum())}")
                bad += int((o[c, tb] != ref).sum())
        if bad:
            for c in range(NC):
                for tb in range(nt):
                    pass
        ok &= bad == 0 and cache_bad == 0
        print(f"n_past={n_past} P={p} nt={nt}: cache rows bad {cache_bad}, O bad {bad}", flush=True)
    print("D1 O A BLOCKS ALL EXACT:", ok)
    if reps:
        import time
        t = {n: [] for n in names}
        for i in range(reps):
            for n in (names if i % 2 == 0 else names[::-1]):
                run(n)
                for _ in range(3):
                    t0 = time.perf_counter()
                    run(n)
                    t[n].append((time.perf_counter() - t0) * 1e3)
        for n in names:
            print(f"{n}: hot median {np.median(t[n]):.3f} ms (n={len(t[n])})")


if __name__ == "__main__":
    main()
