# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the data-parallel QKV head, composed from the operators it absorbs so it
cannot drift from them independently."""

import torch

from iron.operators.rms_norm.reference import reference as rms_norm_ref
from iron.operators.rope.reference import reference as rope_ref


def reference(cur, n_in, wqkv, n_qn, n_kn, ang, D, HD, Hq, Hkv, eps=1e-6):
    """Args mirror the device buffers exactly (flat, bf16). `wqkv` is (Hq*HD + 2*Hkv*HD, D)
    row-major flat, stacked q then k then v; `ang` is (HD,) interleaved [cos,sin,...] for one
    position. Returns the flat concatenated [q | k | v]."""
    QD, KVD = Hq * HD, Hkv * HD
    cur = torch.as_tensor(cur).reshape(1, D)
    n_in = torch.as_tensor(n_in).reshape(D)
    hn = rms_norm_ref(cur, n_in, weighted=True, eps=eps).reshape(D)

    w = torch.as_tensor(wqkv).reshape(QD + 2 * KVD, D)
    raw = (w.float() @ hn.float()).to(hn.dtype)

    n_qn = torch.as_tensor(n_qn).reshape(HD)
    n_kn = torch.as_tensor(n_kn).reshape(HD)
    ang = torch.as_tensor(ang).reshape(1, HD)

    q = rms_norm_ref(raw[:QD].reshape(Hq, HD), n_qn, weighted=True, eps=eps)
    k = rms_norm_ref(raw[QD:QD + KVD].reshape(Hkv, HD), n_kn, weighted=True, eps=eps)
    q = rope_ref(q.reshape(Hq, 1, HD), ang, method_type=0, rows=1, cols=HD).reshape(QD)
    k = rope_ref(k.reshape(Hkv, 1, HD), ang, method_type=0, rows=1, cols=HD).reshape(KVD)
    return torch.cat([q, k, raw[QD + KVD:]])
