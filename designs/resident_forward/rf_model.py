"""Resident rail geometry selected by RF_MODEL; Gemma-4-12B is the default.

The partition is eight columns, four chain rows and 96-row int4 weight elements.
The padded Gemma-3 geometry carries its logical norm width separately.
"""
import os
from typing import NamedTuple


class Model(NamedTuple):
    name: str
    D: int          # d_model as the rail sees it (padded)
    norm_width: int # logical RMSNorm width before rail padding
    norm_eps: float
    FF: int         # MLP width as the rail sees it (padded)
    HQ: int
    HKV: int
    HD: int
    V: int          # vocabulary (LM head rows)
    sliding_window: int

    @property
    def physical_norm_eps(self):
        return self.norm_eps * self.norm_width / self.D


GEMMA4_12B = Model("gemma4-12b", 3840, 3840, 1e-6, 15360, 16, 8, 256, 262144, 1024)
# gemma3-270m (640, 2048, 4 q / 1 kv heads) zero-padded: D / 8 = FF / 32 = 96, one weight element.
GEMMA3_270M_RAIL = Model("gemma3-270m-rail", 768, 640, 1e-6, 3072, 16, 8, 256, 262144, 512)

MODELS = {m.name: m for m in (GEMMA4_12B, GEMMA3_270M_RAIL)}
MODEL = MODELS[os.environ.get("RF_MODEL", GEMMA4_12B.name)]


def norm_defines(width):
    return [f"-DRF_NORM_W={width}", f"-DRF_NORM_EPS={MODEL.physical_norm_eps:.17g}f"]
