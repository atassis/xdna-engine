#!/usr/bin/env python3
"""A fused arm's gate must ask what the OPERATOR accepts, not only what the model allows.

Regression for the 2026-09-15 breakage: main opened the MLP-fusion gate for Gemma-4 while
`swiglu_mlp_dp` had no `layout` parameter on any IRON branch, so every quantized planar build died
on a TypeError naming the operator instead of a decline naming the gate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_decode_spec import operator_rejects  # noqa: E402


class _TakesBoth:
    def __init__(self, weight_dtype=None, group_size=None, layout=None):
        pass


class _TakesNeither:
    def __init__(self, weight_dtype=None, group_size=None):
        pass


class _TakesAnything:
    def __init__(self, weight_dtype=None, **kwargs):
        pass


def test_accepted_kwargs_are_not_reported():
    assert operator_rejects(_TakesBoth, {"weight_dtype": "int4", "layout": "row_group_planar"}) == ()


def test_every_unaccepted_kwarg_is_named_not_just_the_first():
    """Python's TypeError names one; the gate must name all of them, which is why this reads the
    signature instead of listing parameters it believes exist."""
    bad = operator_rejects(
        _TakesNeither,
        {"weight_dtype": "int4", "group_size": 32, "layout": "row_group_planar", "scale_dtype": "bf16"},
    )
    assert set(bad) == {"layout", "scale_dtype"}, bad


def test_a_kwargs_operator_accepts_everything():
    assert operator_rejects(_TakesAnything, {"layout": "x", "anything": 1}) == ()


def test_an_unquantized_plan_passes_any_operator():
    """bf16 sites produce no kwargs at all, so the gate must never refuse on their account."""
    assert operator_rejects(_TakesNeither, {}) == ()
