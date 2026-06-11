"""Load encoder weights + reference tensors from artifacts/encoder (produced by
scripts/extract_encoder.py). fp32 on disk; the ops layer bf16-quantizes at use."""
import os, json
import numpy as np

from . import config as C


class WeightStore:
    def __init__(self, artifacts=C.ARTIFACTS):
        self.root = artifacts
        self.man = json.load(open(os.path.join(artifacts, "manifest.json")))
        self._blocks = [self._load_dir(os.path.join(artifacts, f"L{b}"), self.man["blocks"][str(b)])
                        for b in range(self.man["nblocks"])]
        self.pre_encode = self._load_named(os.path.join(artifacts, "pre_encode"), self.man["pre_encode"])
        self.cos = np.load(os.path.join(artifacts, "refs", "pos_cos.npy"))
        self.sin = np.load(os.path.join(artifacts, "refs", "pos_sin.npy"))

    @staticmethod
    def _load_dir(d, keys):
        return {k: np.load(os.path.join(d, f"{k}.npy")) for k in keys}

    @staticmethod
    def _load_named(d, names):
        return {n: np.load(os.path.join(d, f"{n}.npy")) for n in names}

    def block(self, i):
        return self._blocks[i]

    def ref(self, name):
        return np.load(os.path.join(self.root, "refs", f"{name}.npy"))
