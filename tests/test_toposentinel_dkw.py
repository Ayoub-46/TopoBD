"""Unit tests for the DKW-calibrated accept interval in TopoSentinelServer.

Tests cover:
  1. Large buffer converges to the empirical (delta/2, 1-delta/2) quantiles.
  2. Small buffer yields a strictly wider interval than a large buffer.
  3. L < 2 path accepts everyone (returns [0, inf]).
  4. Probability args are always clamped to [0, 1]; theta_min is always >= 0.
  5. filter_mode='fixed' exactly reproduces the original 5th/95th percentile
     behaviour with bias_interval_margin applied.

Run with:
    pytest tests/test_toposentinel_dkw.py -v
"""

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.toposentinel import TopoSentinelServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(**kwargs) -> TopoSentinelServer:
    """Construct a minimal CPU server for filter-logic unit tests."""
    model = nn.Linear(8, 4)
    return TopoSentinelServer(model=model, device=torch.device("cpu"), **kwargs)


def _fill(server: TopoSentinelServer, data) -> None:
    server._bias_history.extend(data)


# ---------------------------------------------------------------------------
# Test 1: large buffer → interval converges to empirical quantiles
# ---------------------------------------------------------------------------

def test_large_buffer_converges_to_empirical_quantiles():
    """As L → ∞, eps → 0 and the DKW interval approaches the
    empirical (delta/2, 1-delta/2) quantiles of the history."""
    rng = np.random.RandomState(42)
    delta      = 0.10
    dkw_conf   = 0.999   # alpha = 0.001 → eps will be very small

    server = _make_server(target_fpr=delta, dkw_confidence=dkw_conf)

    N    = 200_000
    data = rng.exponential(scale=1.0, size=N).tolist()
    _fill(server, data)

    lo, hi, L, eps = server._get_benign_interval()

    assert L == N

    # eps = sqrt(ln(2/0.001) / (2 * 200000)) ≈ 0.0025
    alpha_expected = 1.0 - dkw_conf
    eps_expected   = math.sqrt(math.log(2.0 / alpha_expected) / (2.0 * N))
    assert abs(eps - eps_expected) < 1e-10

    # Interval must be close to the empirical quantiles
    H            = np.array(data)
    expected_lo  = max(0.0, float(np.quantile(H, delta / 2)))
    expected_hi  = float(np.quantile(H, 1.0 - delta / 2))

    assert abs(lo - expected_lo) < 5e-3, f"lo={lo:.6f}  expected≈{expected_lo:.6f}"
    assert abs(hi - expected_hi) < 5e-3, f"hi={hi:.6f}  expected≈{expected_hi:.6f}"


# ---------------------------------------------------------------------------
# Test 2: small buffer yields strictly wider interval than large buffer
# ---------------------------------------------------------------------------

def test_small_buffer_strictly_wider_than_large():
    """With L_small=100 samples, eps is large enough that p_lo<=0 (so
    theta_min=0) and p_hi>=1 (so theta_max=max(H)).  With L_large=10000,
    the interval is an interior quantile range — strictly narrower."""
    rng  = np.random.RandomState(42)
    data = rng.exponential(scale=1.0, size=10_000).tolist()

    # L=100: eps = sqrt(ln(40)/200) ≈ 0.136 → p_lo = 0.05-0.136 = -0.086 ≤ 0
    #               p_hi = 0.95+0.136 = 1.086 > 1
    server_s = _make_server(target_fpr=0.10, dkw_confidence=0.95)
    _fill(server_s, data[:100])

    # L=10000: eps = sqrt(ln(40)/20000) ≈ 0.014 → 0 < p_lo < p_hi < 1
    server_l = _make_server(target_fpr=0.10, dkw_confidence=0.95)
    _fill(server_l, data)

    lo_s, hi_s, L_s, eps_s = server_s._get_benign_interval()
    lo_l, hi_l, L_l, eps_l = server_l._get_benign_interval()

    assert L_s == 100
    assert L_l == 10_000
    assert eps_s > eps_l, "More samples must give a smaller DKW correction."

    # Small buffer: p_lo <= 0 → theta_min == 0; large buffer: theta_min > 0
    assert lo_s == 0.0, f"Expected lo_s=0.0 (p_lo<=0), got {lo_s}"
    assert lo_l >  0.0, f"Expected lo_l>0.0 (interior quantile), got {lo_l}"

    # Small buffer: p_hi >= 1 → theta_max = max(H_small);
    # large buffer uses interior quantile, which for exp(1) is ≈ 3.3 whereas
    # max(100 exp(1) samples) is typically ≈ 4–6.
    assert hi_s > hi_l, (
        f"Expected hi_s={hi_s:.4f} > hi_l={hi_l:.4f} (small buffer wider)"
    )


# ---------------------------------------------------------------------------
# Test 3: L < 2 accepts everyone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_entries", [0, 1])
def test_l_less_than_2_accepts_everyone(n_entries):
    """With fewer than 2 history entries, DKW calibration is skipped and
    every client is accepted (interval = [0, +inf])."""
    server = _make_server()
    if n_entries == 1:
        _fill(server, [0.42])

    lo, hi, L, eps = server._get_benign_interval()

    assert lo  == 0.0
    assert hi  == float("inf")
    assert L   == n_entries
    assert math.isnan(eps)


# ---------------------------------------------------------------------------
# Test 4: probability args clamped to [0, 1]; theta_min always >= 0
# ---------------------------------------------------------------------------

def test_probability_clamping_and_nonneg_theta_min():
    """With L=5 and alpha=0.05, eps ≈ 0.77 so p_lo < 0 and p_hi > 1.
    The implementation must clamp both before passing to np.quantile and
    return theta_min == 0 and theta_max == max(H)."""
    rng  = np.random.RandomState(0)
    data = rng.exponential(1.0, size=5).tolist()

    server = _make_server(target_fpr=0.05, dkw_confidence=0.95)
    _fill(server, data)

    lo, hi, L, eps = server._get_benign_interval()

    alpha        = 0.05      # 1 - dkw_confidence
    eps_expected = math.sqrt(math.log(2.0 / alpha) / (2.0 * L))
    p_lo         = 0.05 / 2 - eps_expected
    p_hi         = 1.0 - 0.05 / 2 + eps_expected

    # Confirm the preconditions of this test
    assert p_lo < 0,  f"Precondition failed: p_lo={p_lo:.4f} must be < 0"
    assert p_hi > 1,  f"Precondition failed: p_hi={p_hi:.4f} must be > 1"

    # theta_min: p_lo <= 0 → must be 0.0 (not a negative quantile)
    assert lo == 0.0, f"theta_min must be 0.0 when p_lo<=0, got {lo}"

    # theta_max: p_hi clamped to 1.0 → must equal max(H)
    assert abs(hi - max(data)) < 1e-9, (
        f"theta_max must be max(H)={max(data):.6f} when p_hi>=1, got {hi:.6f}"
    )

    # General invariant: theta_min is always non-negative
    assert lo >= 0.0


# ---------------------------------------------------------------------------
# Test 5: filter_mode='fixed' reproduces the original 5th/95th percentile
# ---------------------------------------------------------------------------

def test_fixed_mode_matches_original_percentile_behavior():
    """filter_mode='fixed' must return exactly the same interval as the
    original implementation (np.percentile at 5 and 95, with margin added),
    and eps must be nan since no DKW correction is computed."""
    rng  = np.random.RandomState(99)
    N    = 200
    data = rng.exponential(scale=1.0, size=N).tolist()
    margin = 0.02

    server = _make_server(
        filter_mode="fixed",
        bias_interval_margin=margin,
        min_bias_history_size=100,   # N=200 > 100, so learned interval is used
    )
    _fill(server, data)

    lo, hi, L, eps = server._get_benign_interval()

    H            = np.array(data)
    expected_lo  = max(0.0, float(np.percentile(H, 5.0)) - margin)
    expected_hi  = float(np.percentile(H, 95.0)) + margin

    assert abs(lo - expected_lo) < 1e-10, f"lo={lo} expected={expected_lo}"
    assert abs(hi - expected_hi) < 1e-10, f"hi={hi} expected={expected_hi}"
    assert math.isnan(eps), "fixed mode must return eps=nan"


def test_fixed_mode_uses_fallback_when_history_small():
    """filter_mode='fixed' falls back to bias_fallback_interval when the
    history is below min_bias_history_size."""
    server = _make_server(
        filter_mode="fixed",
        bias_fallback_interval=[0.0, 0.3],
        min_bias_history_size=100,
    )
    _fill(server, [0.1, 0.2, 0.15])   # L=3, well below threshold

    lo, hi, L, eps = server._get_benign_interval()

    assert lo  == 0.0
    assert hi  == 0.3
    assert L   == 3
    assert math.isnan(eps)
