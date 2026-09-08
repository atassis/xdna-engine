#!/usr/bin/env python3
"""
Userspace objectFIFO re-arm probe via the xdna-driver AIE_TILE_READ/WRITE ioctls.

WHAT THIS TESTS
---------------
The resident acquire-once objectFIFO is not re-armed across dispatches on one live hw_context
(lock init runs only in the CDO config phase, never per-dispatch). The mlir-air toolchain works
around it with a full PDI reload (coarse). We want a *scoped* driver-side re-arm. The open
xdna-driver can't touch AIE tiles directly, but it already exposes a production, live-context-scoped
tile poke: DRM_AMDXDNA_AIE_TILE_WRITE / _READ -> firmware MSG_OP_AIE_RW_ACCESS (0x203). This script
issues those ioctls from a SECOND fd in this same process (the driver resolves the hw_context by
(context_id, pid) device-globally, and permits "root OR owns the context" -> no rebuild, no root),
to answer: does poking the objectFIFO lock registers between dispatch 1 and 2 re-arm a live resident
context? If yes -> the Tier-1 open-driver re-arm ioctl is viable and we've prototyped its core here.

  observe : dispatch A, sweep-read lock regs; dispatch B (no re-arm), sweep again; print the DELTA
            + the A/B correctness. Pure diagnostic (no writes) -- register-level picture of the bug.
  rearm   : dispatch A, read locks; write the specified lock regs back to their armed values;
            read back to confirm the write LANDED; dispatch B; report re-armed vs still-stale.

Requires: the BROKEN acquire-once build loaded (else dispatch-2 B is already correct and there is
nothing to re-arm -- the baseline check tells you which build is loaded). FW >= 6.24 on NPU4-class
(the 0x203 gate); otherwise the ioctl returns EOPNOTSUPP and we say so.

Device hygiene: single-tenant NPU -> announce + `fuser`, stop npu-asr, serialize before running.

Usage:
  python run_bd_rearm_probe.py --dry-run                 # just bind ctx + confirm 0x203 works
  python run_bd_rearm_probe.py --mode observe            # find the stale locks (start here)
  python run_bd_rearm_probe.py --mode rearm --locks 0:2:0=2,0:2:1=0
     (--locks = comma list of  COL:ROW:LOCKID=VALUE  in partition-relative COL, absolute ROW,
      values from the `armed` column of the observe run. core lock -> 0x1F000+0x10*id,
      mem tile (row 1) lock -> 0xC0000+0x10*id, value field is 6-bit.)
"""
import os, sys, ctypes, struct, errno, argparse, glob
import numpy as np, pyxrt
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# harness setup (mirrors run_bd_AB.py exactly)
# ---------------------------------------------------------------------------
EX = os.path.join(os.path.dirname(__file__), "build")
TQ=8; T=176; DK=128; N_QT=22; H=4; SCALE=0.08838835; NQ=N_QT*TQ; P=2*T-1
def bf(x): return x.astype(bfloat16).astype(np.float32)
def gen(seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((NQ,DK)).astype(np.float32)
    k=bf(rng.standard_normal((T,DK))); v=bf(rng.standard_normal((T,DK)))
    p=bf(rng.standard_normal((P,DK))); bv=bf(0.1*rng.standard_normal((DK,)))
    q=bf(q/((q@k.T).std()+1e-6)); qv=bf(q+bv); AC=q@k.T; BD=qv@p.T
    BDs=np.stack([BD[i,(T-1-i):(T-1-i)+T] for i in range(NQ)]); sc=(AC+bf(BDs))*SCALE
    e=np.exp(sc-sc.max(1,keepdims=True)); ctx=(e/e.sum(1,keepdims=True))@v
    qpv=np.concatenate([q.reshape(N_QT,TQ*DK),qv.reshape(N_QT,TQ*DK)],axis=1).reshape(-1)
    return qpv,p.reshape(-1),k.reshape(-1),v.reshape(-1),ctx
def pack(seeds):
    hs=[gen(s) for s in seeds]
    return (np.concatenate([h[0] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[1] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[2] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[3] for h in hs]).astype(bfloat16).view(np.uint16),
            np.concatenate([h[4] for h in hs],axis=0))
def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))

# ---------------------------------------------------------------------------
# raw DRM / amdxdna ioctl plumbing (via libc ioctl; a 2nd fd on the accel node)
# ---------------------------------------------------------------------------
DRM_TYPE = ord('d')            # 0x64
DRM_COMMAND_BASE = 0x40
_IOC_WRITE, _IOC_READ = 1, 2
def _IOWR(nr, size): return ((_IOC_READ|_IOC_WRITE)<<30)|((size&0x3FFF)<<16)|(DRM_TYPE<<8)|(nr&0xFF)
SET_STATE_NR  = DRM_COMMAND_BASE + 8    # DRM_AMDXDNA_SET_STATE
GET_ARRAY_NR  = DRM_COMMAND_BASE + 10   # DRM_AMDXDNA_GET_ARRAY
P_AIE_TILE_WRITE = 7    # amdxdna_drm_set_state.param
P_AIE_TILE_READ  = 9    # amdxdna_drm_get_array.param
P_HW_CONTEXT_ALL = 0    # amdxdna_drm_get_array.param
ACCESS_REG = 0          # amdxdna_aie_tile_access_type

class AieTileAccess(ctypes.Structure):
    _fields_ = [("pid",ctypes.c_uint64),("context_id",ctypes.c_uint32),
                ("col",ctypes.c_uint32),("row",ctypes.c_uint32),
                ("addr",ctypes.c_uint32),("size",ctypes.c_uint32),
                ("type",ctypes.c_uint8),("pad",ctypes.c_uint8*3)]
class SetState(ctypes.Structure):
    _fields_ = [("param",ctypes.c_uint32),("buffer_size",ctypes.c_uint32),("buffer",ctypes.c_uint64)]
class GetArray(ctypes.Structure):
    _fields_ = [("param",ctypes.c_uint32),("element_size",ctypes.c_uint32),
                ("num_element",ctypes.c_uint32),("pad",ctypes.c_uint32),("buffer",ctypes.c_uint64)]
class HwctxEntry(ctypes.Structure):
    _fields_ = [("context_id",ctypes.c_uint32),("start_col",ctypes.c_uint32),
                ("num_col",ctypes.c_uint32),("hwctx_id",ctypes.c_uint32),("pid",ctypes.c_int64),
                ("command_submissions",ctypes.c_uint64),("command_completions",ctypes.c_uint64),
                ("migrations",ctypes.c_uint64),("preemptions",ctypes.c_uint64),("errors",ctypes.c_uint64),
                ("priority",ctypes.c_uint64),("heap_usage",ctypes.c_uint64),("suspensions",ctypes.c_uint64),
                ("state",ctypes.c_uint32),("pasid",ctypes.c_uint32),("gops",ctypes.c_uint32),
                ("fps",ctypes.c_uint32),("dma_bandwidth",ctypes.c_uint32),("latency",ctypes.c_uint32),
                ("frame_exec_time",ctypes.c_uint32),("txn_op_idx",ctypes.c_uint32),("ctx_pc",ctypes.c_uint32),
                ("fatal_error_type",ctypes.c_uint32),("fatal_error_exception_type",ctypes.c_uint32),
                ("fatal_error_exception_pc",ctypes.c_uint32),("fatal_error_app_module",ctypes.c_uint32)]
assert ctypes.sizeof(AieTileAccess)==32, ctypes.sizeof(AieTileAccess)
assert ctypes.sizeof(SetState)==16 and ctypes.sizeof(GetArray)==24

_libc = ctypes.CDLL(None, use_errno=True)
_libc.ioctl.restype = ctypes.c_int
_libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
def _ioctl(fd, request, arg):
    ctypes.set_errno(0)
    r = _libc.ioctl(fd, ctypes.c_ulong(request), ctypes.addressof(arg))
    if r < 0:
        e = ctypes.get_errno(); raise OSError(e, os.strerror(e))
    return r

def core_lock_addr(lid): return 0x1F000 + 0x10*lid   # compute/core tile, lid 0..15
def mem_lock_addr(lid):  return 0xC0000 + 0x10*lid    # mem tile (row 1),  lid 0..63
LOCK6 = 0x3F                                          # 6-bit lock value field

class TileIO:
    def __init__(self, node):
        self.fd = os.open(node, os.O_RDWR | os.O_CLOEXEC)
        self.pid = os.getpid()
        self.cid = None            # the id the driver matches (context_id or hwctx_id)
        self.entry = None
    def _get_array(self, param, arr, elem_size, n):
        ga = GetArray(param=param, element_size=elem_size, num_element=n, pad=0,
                      buffer=ctypes.addressof(arr))
        _ioctl(self.fd, _IOWR(GET_ARRAY_NR, ctypes.sizeof(GetArray)), ga)
        return ga.num_element
    def enumerate_ctx(self, maxn=64):
        arr = (HwctxEntry*maxn)()
        n = self._get_array(P_HW_CONTEXT_ALL, arr, ctypes.sizeof(HwctxEntry), maxn)
        return [arr[i] for i in range(min(n, maxn))]
    def read_reg(self, col, row, addr, cid=None):
        cid = self.cid if cid is None else cid
        payload = bytearray(ctypes.sizeof(AieTileAccess) + 4)
        acc = AieTileAccess.from_buffer(payload)
        acc.pid=self.pid; acc.context_id=cid; acc.col=col; acc.row=row
        acc.addr=addr; acc.size=4; acc.type=ACCESS_REG
        buf = (ctypes.c_char*len(payload)).from_buffer(payload)
        self._get_array(P_AIE_TILE_READ, buf, len(payload), 1)   # driver overwrites head with value
        return struct.unpack_from("<I", payload, 0)[0]
    def write_reg(self, col, row, addr, val, cid=None):
        cid = self.cid if cid is None else cid
        payload = bytearray(ctypes.sizeof(AieTileAccess) + 4)
        acc = AieTileAccess.from_buffer(payload)
        acc.pid=self.pid; acc.context_id=cid; acc.col=col; acc.row=row
        acc.addr=addr; acc.size=4; acc.type=ACCESS_REG
        struct.pack_into("<I", payload, ctypes.sizeof(AieTileAccess), val & 0xFFFFFFFF)
        buf = (ctypes.c_char*len(payload)).from_buffer(payload)
        ss = SetState(param=P_AIE_TILE_WRITE, buffer_size=len(payload), buffer=ctypes.addressof(buf))
        _ioctl(self.fd, _IOWR(SET_STATE_NR, ctypes.sizeof(SetState)), ss)
    def bind(self):
        mine = [e for e in self.enumerate_ctx() if e.pid == self.pid]
        if not mine:
            raise RuntimeError("no hw_context for pid %d found (create the XRT context first)" % self.pid)
        mine.sort(key=lambda e: (e.state != 1, -e.command_submissions))
        e = self.entry = mine[0]
        # the write/read handler matches access.context_id against the per-client ctx key;
        # the entry exposes both context_id and hwctx_id -- try context_id, fall back to hwctx_id.
        for name in ("context_id", "hwctx_id"):
            cand = getattr(e, name)
            try:
                self.read_reg(0, 2, core_lock_addr(0), cid=cand)  # benign trial read (col0,row2 core)
                self.cid = cand; self.cid_field = name; return e
            except OSError as ex:
                if ex.errno == errno.EOPNOTSUPP:
                    raise RuntimeError("AIE_TILE_READ -> EOPNOTSUPP: firmware lacks MSG_OP_AIE_RW_ACCESS "
                                       "(0x203). Needs FW >= 6.24 on NPU4-class. Check `xrt-smi examine`.")
                if ex.errno in (errno.EINVAL, errno.EPERM):
                    continue
                raise
        raise RuntimeError("could not bind context id (tried context_id and hwctx_id; last errno EINVAL/EPERM)")

def rows_of_kind(nrows):
    # aie2p/NPU2 tile map: row 0 = shim, row 1 = mem tile, rows 2.. = core tiles.
    core = list(range(2, nrows)); mem = [1] if nrows > 1 else []
    return core, mem
def sweep_locks(io, nrows, max_core_lid, max_mem_lid):
    """Return {(col,row,lid): value6} for every lock reg in the context partition."""
    core_rows, mem_rows = rows_of_kind(nrows)
    out = {}
    for col in range(io.entry.num_col):
        for row in core_rows:
            for lid in range(max_core_lid):
                try: out[(col,row,lid)] = io.read_reg(col,row,core_lock_addr(lid)) & LOCK6
                except OSError: pass
        for row in mem_rows:
            for lid in range(max_mem_lid):
                try: out[(col,row,lid)] = io.read_reg(col,row,mem_lock_addr(lid)) & LOCK6
                except OSError: pass
    return out

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=None, help="accel node (default: autodetect /dev/accel/accel*)")
    ap.add_argument("--mode", choices=["observe","rearm"], default="observe")
    ap.add_argument("--dry-run", action="store_true", help="bind ctx + confirm 0x203 works, then exit")
    ap.add_argument("--rows", type=int, default=6, help="tile rows to sweep (shim=0, mem=1, core>=2)")
    ap.add_argument("--max-core-lid", type=int, default=16)
    ap.add_argument("--max-mem-lid", type=int, default=8)
    ap.add_argument("--locks", default="", help="rearm list: COL:ROW:LID=VAL,... (partition-rel col)")
    args = ap.parse_args()

    node = args.node
    if node is None:
        cands = sorted(glob.glob("/dev/accel/accel*"))
        node = cands[0] if cands else "/dev/accel/accel0"

    # bring up the XRT context + BOs exactly like run_bd_AB
    A = pack([0,1,2,3]); B = pack([10,11,12,13])
    instr = np.fromfile(f"{EX}/insts.bin", dtype=np.uint32)
    xb = pyxrt.xclbin(f"{EX}/final.xclbin"); kn = xb.get_kernels()[0].get_name()
    d = pyxrt.device(0); d.register_xclbin(xb)
    hw = pyxrt.hw_context(d, xb.get_uuid()); kern = pyxrt.kernel(hw, kn)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FR = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bi = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bq = pyxrt.bo(d, A[0].nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bp = pyxrt.bo(d, A[1].nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bk = pyxrt.bo(d, A[2].nbytes, pyxrt.bo.host_only, kern.group_id(5))
    bv = pyxrt.bo(d, A[3].nbytes, pyxrt.bo.host_only, kern.group_id(6))
    bc = pyxrt.bo(d, H*NQ*DK*2, pyxrt.bo.host_only, kern.group_id(7))
    bi.write(instr.tobytes(),0); bi.sync(TO)
    def load(S):
        for bo,a in ((bq,S[0]),(bp,S[1]),(bk,S[2]),(bv,S[3])): bo.write(a.tobytes(),0); bo.sync(TO)
    def run():
        r = kern(3, bi, instr.size, bq, bp, bk, bv, bc); r.wait(); bc.sync(FR)
        return np.frombuffer(bc.read(H*NQ*DK*2,0),dtype=np.uint16).view(bfloat16)\
                 .astype(np.float32).reshape(H*NQ,DK).copy()

    io = TileIO(node)
    e = io.bind()
    print(f"[ctx] node={node} pid={io.pid} bound via {io.cid_field}={io.cid} "
          f"start_col={e.start_col} num_col={e.num_col} state={e.state}")
    if args.dry_run:
        print("[dry-run] 0x203 tile access works on this device/firmware. OK."); return

    # dispatch 1 (A)
    load(A); oA = run()
    relA = rel(oA, A[4])
    locks_A = sweep_locks(io, args.rows, args.max_core_lid, args.max_mem_lid)

    if args.mode == "rearm":
        if not args.locks.strip():
            print("[rearm] --locks empty. Run --mode observe first, then pass the armed values."); return
        targets = []
        for tok in args.locks.split(","):
            cr, val = tok.split("="); col, row, lid = (int(x) for x in cr.split(":"))
            targets.append((col, row, lid, int(val)))
        print("[rearm] pre-write lock values (post-dispatch-A):")
        for col,row,lid,val in targets:
            base = core_lock_addr(lid) if row >= 2 else mem_lock_addr(lid)
            print(f"    ({col},{row}) lid{lid}: now={io.read_reg(col,row,base)&LOCK6} -> writing {val}")
            io.write_reg(col, row, base, val)
        # read-back: did the write LAND on the live resident tile?
        landed = True
        for col,row,lid,val in targets:
            base = core_lock_addr(lid) if row >= 2 else mem_lock_addr(lid)
            got = io.read_reg(col,row,base)&LOCK6
            ok = (got == (val & LOCK6)); landed &= ok
            print(f"    read-back ({col},{row}) lid{lid}: {got} {'OK' if ok else '!= '+str(val)+' (WRITE DROPPED)'}")
        if not landed:
            print("[verdict] WRITE DID NOT LAND on the resident tile -> points to case (4) driver-drop; "
                  "the register poke is not honored on a live context.")

    # dispatch 2 (B)
    load(B); oB = run()
    relBB = rel(oB, B[4]); relBA = rel(oB, A[4])
    print(f"[disp1] A vs goldenA : {relA:.4e}  (expect ~4e-3)")
    print(f"[disp2] B vs goldenB : {relBB:.4e}  (~4e-3 => B re-armed correctly | large => broken)")
    print(f"[disp2] B vs goldenA : {relBA:.4e}  (~small => reused A's stale resident weights)")

    if args.mode == "observe":
        locks_B = sweep_locks(io, args.rows, args.max_core_lid, args.max_mem_lid)
        changed = sorted(k for k in locks_A if locks_A.get(k) != locks_B.get(k))
        nz = sorted(k for k in locks_A if locks_A[k] != 0 or locks_B.get(k,0) != 0)
        print(f"\n[observe] lock regs that CHANGED A->B (candidate objectFIFO locks): {len(changed)}")
        for k in changed:
            col,row,lid = k
            print(f"    ({col},{row}) lid{lid}: postA={locks_A[k]:2d}  postB={locks_B.get(k):2d}")
        print(f"[observe] all non-zero lock regs (postA/postB): {len(nz)}")
        for k in nz:
            col,row,lid = k
            print(f"    ({col},{row}) lid{lid}: postA={locks_A[k]:2d}  postB={locks_B.get(k):2d}")
        print("\nNext: pick the resident weight FIFO's prod/cons locks from the CHANGED set, then\n"
              "  --mode rearm --locks COL:ROW:LID=ARMED,...  (armed = prod:depth, cons:0).")

    # interpretation
    broke = relBB > 5e-2
    stale = relBA < 5e-2
    if args.mode == "observe":
        if broke and stale: print("[verdict] BROKEN build confirmed: dispatch-2 reuses stale resident weights.")
        elif not broke:     print("[verdict] dispatch-2 already correct -> FIXED build loaded (nothing to re-arm).")
    else:
        if not broke:       print("[verdict] RE-ARMED: userspace lock poke fixed dispatch-2 -> Tier-1 ioctl VIABLE.")
        elif stale:         print("[verdict] still stale after lock re-arm -> lock-value alone insufficient "
                                   "(BD-queue residual, case 2/3) OR write did not land (case 4; see read-back).")
        else:               print("[verdict] dispatch-2 garbage (not stale-A): re-arm perturbed state; inspect.")

if __name__ == "__main__":
    main()
