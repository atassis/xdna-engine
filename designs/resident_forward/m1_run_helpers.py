"""M1 host-side helpers (inputs, reference layout, row extraction)."""
import os
import numpy as np
from mlp_ref import h_memtile_layout, a_layout, D
from m1_design import NC, PCAP_T, ABLK, HT

_R = None


def _ref():
    # Lazy: only x_buf/ref_h (device probes, not the design build) need this file; rlayer_design's
    # generator imports this module only for nt_of, which does not touch it.
    global _R
    if _R is None:
        import rf_paths
        _R = np.load(os.environ.get("RF_M1_REF", str(rf_paths.BUILD_ROOT / "scratch/m1/ref.npz")))
    return _R


def nt_of(p):
    return -(-p // 16)


def x_buf(p):
    nt = nt_of(p)
    x = np.zeros((16 * nt, D), np.float32)
    x[:p] = _ref()["x"][:p]
    a = a_layout(x, nt).reshape(NC, nt * ABLK)
    buf = np.zeros((NC, PCAP_T * ABLK), np.uint8)
    buf[:, :nt * ABLK] = a
    return buf.reshape(-1)


def ref_h(p):
    nt = nt_of(p)
    hv = np.zeros((16 * nt, _ref()["hv"].shape[1]), np.float32)
    hv[:p] = _ref()["hv"][:p]
    return h_memtile_layout(hv, nt).reshape(NC, nt * HT)


def rows_of(hb, nt):
    """[NC, nt*HT] bytes -> per row [16*nt, NC*240*9] bytes (block = one row's 8 h columns)."""
    t = hb.reshape(NC, nt, 240, 2, 8, 9)                     # [m][t][kb][r][row][9]
    return t.transpose(1, 3, 4, 0, 2, 5).reshape(16 * nt, -1)


