"""Static contract for the M256 four-K1024 same-core FC2 schedule.

The device generator consumes this geometry; this module keeps the physical panel
layout and the ordered f32 reduction independently testable without an AIE runtime.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FC2FusedConfig:
    rows: int = 256
    cols: int = 1024
    partial_k: int = 1024
    partials: int = 4
    m: int = 16
    k: int = 32
    n: int = 128
    columns: int = 8

    def __post_init__(self):
        if (self.rows, self.cols, self.partial_k, self.partials, self.m, self.k,
                self.n, self.columns) != (256, 1024, 1024, 4, 16, 32, 128, 8):
            raise ValueError("FC2 fused schedule is qualified only for 256x4096x1024, 16x32x128/8c")
        if self.rows % (4 * self.m) or self.cols % (self.columns * self.n):
            raise ValueError("tile geometry does not cover the physical FC2 output")
        if self.partial_k % self.k:
            raise ValueError("each K1024 partial must contain whole k tiles")


@dataclass(frozen=True)
class TileAccess:
    output_offset: int
    partial: int
    a_panel_offset: int
    w_panel_offset: int
    k_tiles: int


def plan_accesses(config=FC2FusedConfig()):
    """One ordered K1024 group per final C tile and partial, in element offsets."""
    accesses = []
    for row in range(0, config.rows, config.m):
        for col in range(0, config.cols, config.n):
            output_offset = row * config.cols + col
            for partial in range(config.partials):
                accesses.append(TileAccess(
                    output_offset=output_offset,
                    partial=partial,
                    a_panel_offset=partial * config.rows * config.partial_k,
                    w_panel_offset=partial * config.partial_k * config.cols,
                    k_tiles=config.partial_k // config.k,
                ))
    return accesses


def fused_fc2_reference(a, w, config=FC2FusedConfig(), *, partials=None):
    """IEEE-binary32 ordered reference for the device's four materialized partials."""
    if partials is None:
        if a is None or w is None:
            raise ValueError("A and W are required unless materialized partials are supplied")
        if a.shape != (4, config.rows, config.partial_k) or w.shape != (4, config.partial_k, config.cols):
            raise ValueError("A/W must use the physical four-panel FC2 layout")
        partials = np.stack([(a[i] @ w[i]).astype(np.float32) for i in range(4)])
    if partials.shape != (4, config.rows, config.cols) or partials.dtype != np.float32:
        raise ValueError("expected four materialized f32 partials")
    accumulator = np.zeros((config.rows, config.cols), dtype=np.float32)
    for partial in partials:
        accumulator = np.add(accumulator, partial, dtype=np.float32)
    return accumulator
