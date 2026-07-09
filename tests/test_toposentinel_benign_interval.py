"""Unit tests for the fixed 5th/95th-percentile accept interval.

DKW calibration has been removed entirely; the fixed-percentile interval
(with additive margin and a small-buffer fallback) is now the only path.
Replaces tests/test_toposentinel_dkw.py, which tested the now-removed
filter_mode="dkw" branch.

Run with:
    pytest tests/test_toposentinel_benign_interval.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.toposentinel import TopoSentinelServer


def _make_server(**kwargs) -> TopoSentinelServer:
    """Construct a minimal CPU server for filter-logic unit tests."""
    model = nn.Linear(8, 4)
    return TopoSentinelServer(model=model, device=torch.device("cpu"), **kwargs)


def _fill(server: TopoSentinelServer, data) -> None:
    server._bias_history.extend(data)


def test_interval_matches_percentile_plus_margin():
    """The interval must equal np.percentile(H, [5, 95]) with
    bias_interval_margin added/subtracted on each end."""
    rng    = np.random.RandomState(99)
    N      = 100   # comfortably under the default deque maxlen (150)
    data   = rng.exponential(scale=1.0, size=N).tolist()
    margin = 0.02

    server = _make_server(bias_interval_margin=margin, min_bias_history_size=50)
    _fill(server, data)

    lo, hi, L = server._get_benign_interval()

    H           = np.array(data)
    expected_lo = max(0.0, float(np.percentile(H, 5.0)) - margin)
    expected_hi = float(np.percentile(H, 95.0)) + margin

    assert L == N
    assert abs(lo - expected_lo) < 1e-10, f"lo={lo} expected={expected_lo}"
    assert abs(hi - expected_hi) < 1e-10, f"hi={hi} expected={expected_hi}"


def test_fallback_triggers_below_min_bias_history_size():
    """Below min_bias_history_size, the fixed bias_fallback_interval is used
    instead of a percentile computed from too little data."""
    server = _make_server(
        bias_fallback_interval=[0.0, 0.3],
        min_bias_history_size=100,
    )
    _fill(server, [0.1, 0.2, 0.15])   # L=3, well below threshold

    lo, hi, L = server._get_benign_interval()

    assert lo == 0.0
    assert hi == 0.3
    assert L  == 3


def test_interval_endpoints_never_negative():
    """theta_min must be clamped to >= 0 even if percentile - margin dips
    below zero (tiny/skewed history)."""
    server = _make_server(bias_interval_margin=1.0, min_bias_history_size=3)
    _fill(server, [0.01, 0.02, 0.015, 0.018, 0.02])

    lo, hi, L = server._get_benign_interval()

    assert lo >= 0.0
    assert hi > lo


def test_empty_history_uses_fallback():
    """L=0 (no history at all) must not crash and must use the fallback."""
    server = _make_server(bias_fallback_interval=[0.0, 0.5], min_bias_history_size=10)

    lo, hi, L = server._get_benign_interval()

    assert L == 0
    assert lo == 0.0
    assert hi == 0.5
