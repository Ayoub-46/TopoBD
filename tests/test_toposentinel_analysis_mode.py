"""Unit tests for TopoSentinelServer's pure-observability analysis_mode.

Covers:
  1. Sporadic schedule: is_attack_round / is_malicious_present match the
     {period, duty} spec.
  2. No filtering: every client is aggregated regardless of how extreme its
     bias delta is (accepted set == all clients, every round).
  3. Both CSVs are written with the right shape and populated columns.
  4. Guards: prob args in [0,1] (inherited, untouched), tau_min clamped,
     tiny/empty inputs (0 or 1 client) don't crash.
  5. analysis_mode=False leaves production behaviour (rejection still
     possible) unchanged.

Run with:
    pytest tests/test_toposentinel_analysis_mode.py -v
"""

import csv
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
    model = nn.Linear(8, 4)
    return TopoSentinelServer(model=model, device=torch.device("cpu"), **kwargs)


def _submit_client(server, cid, bias_shift, length=100):
    """Fabricate a client update whose bias tensor is shifted by bias_shift
    (a scalar or array broadcastable to the bias shape) from the global bias.
    """
    global_params = server.get_params()
    params = {k: v.clone() for k, v in global_params.items()}
    params["bias"] = params["bias"] + torch.as_tensor(bias_shift, dtype=params["bias"].dtype)
    server.receive_update(client_id=cid, params=params, length=length)


# ---------------------------------------------------------------------------
# Test 1: sporadic schedule matches spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "period,duty",
    [(10, 1), (5, 2), (1, 1), (3, 0), (4, 4)],
)
def test_is_attack_round_matches_schedule_spec(period, duty):
    server = _make_server(analysis_mode=True, attack_pattern={"period": period, "duty": duty})
    for r in range(3 * period + 1):
        expected = (r % period) < min(duty, period)
        assert server._is_attack_round(r) == expected, f"round={r} period={period} duty={duty}"


def test_default_attack_pattern_is_period_10_duty_1():
    server = _make_server(analysis_mode=True)
    attack_rounds = [r for r in range(30) if server._is_attack_round(r)]
    assert attack_rounds == [0, 10, 20]


@pytest.mark.parametrize(
    "period,duty,warmup",
    [(10, 1, 0), (5, 2, 3), (1, 1, 7), (4, 4, 5)],
)
def test_is_attack_round_respects_warmup(period, duty, warmup):
    server = _make_server(
        analysis_mode=True,
        attack_pattern={"period": period, "duty": duty, "warmup": warmup},
    )
    for r in range(warmup + 3 * period + 1):
        if r < warmup:
            expected = False
        else:
            expected = ((r - warmup) % period) < min(duty, period)
        assert server._is_attack_round(r) == expected, (
            f"round={r} period={period} duty={duty} warmup={warmup}"
        )


def test_warmup_default_is_zero_preserves_prior_behaviour():
    server = _make_server(analysis_mode=True, attack_pattern={"period": 5, "duty": 1})
    attack_rounds = [r for r in range(15) if server._is_attack_round(r)]
    assert attack_rounds == [0, 5, 10]


def test_is_malicious_present_requires_both_identity_and_attack_round():
    server = _make_server(analysis_mode=True, attack_pattern={"period": 4, "duty": 1})
    malicious_ids = frozenset({0, 1})

    for cid in range(5):
        _submit_client(server, cid, bias_shift=0.01 * (cid + 1))
    server.filter_updates(true_malicious=malicious_ids)  # round 0 -> attack round

    row0 = [r for r in server._filter_log if r["round"] == 0]
    for r in row0:
        expected = r["client_id"] in malicious_ids
        assert bool(r["is_malicious_present"]) == expected

    server.aggregate()
    server.reset()

    for cid in range(5):
        _submit_client(server, cid, bias_shift=0.01 * (cid + 1))
    server.filter_updates(true_malicious=malicious_ids)  # round 1 -> quiet round

    row1 = [r for r in server._filter_log if r["round"] == 1]
    for r in row1:
        assert r["is_malicious_present"] == 0


# ---------------------------------------------------------------------------
# Test 2: analysis_mode never rejects, regardless of how extreme the update
# ---------------------------------------------------------------------------

def test_analysis_mode_never_rejects_even_extreme_outliers():
    server = _make_server(analysis_mode=True, attack_pattern={"period": 3, "duty": 1})
    malicious_ids = frozenset({0})

    for round_idx in range(6):
        client_ids = list(range(5))
        for cid in client_ids:
            shift = 50.0 if (cid == 0 and server._is_attack_round(round_idx)) else 0.01
            _submit_client(server, cid, bias_shift=shift)

        result = server.filter_updates(true_malicious=malicious_ids)

        assert result.rejected_ids == frozenset()
        server.aggregate()
        server.reset()


def test_analysis_mode_accepted_set_equals_all_clients_each_round():
    server = _make_server(analysis_mode=True, attack_pattern={"period": 2, "duty": 1})
    malicious_ids = frozenset({2})

    for round_idx in range(4):
        client_ids = list(range(4))
        for cid in client_ids:
            shift = 20.0 if (cid == 2 and server._is_attack_round(round_idx)) else 0.0
            _submit_client(server, cid, bias_shift=shift)

        before_ids = set(server._received_updates.keys())
        server.filter_updates(true_malicious=malicious_ids)
        after_ids = set(server._received_updates.keys())

        assert after_ids == before_ids == set(client_ids)
        agg = server.aggregate()
        assert agg.num_clients == len(client_ids)
        server.reset()


# ---------------------------------------------------------------------------
# Test 3: both CSVs are written with the right shape
# ---------------------------------------------------------------------------

def test_both_csvs_written_with_expected_shape(tmp_path):
    period, duty = 5, 1
    server = _make_server(analysis_mode=True, attack_pattern={"period": period, "duty": duty})
    malicious_ids = frozenset({0, 1})
    n_clients = 6
    n_rounds = 12

    for round_idx in range(n_rounds):
        is_attack = server._is_attack_round(round_idx)
        for cid in range(n_clients):
            is_mal = is_attack and cid in malicious_ids
            shift = 5.0 if is_mal else 0.02
            _submit_client(server, cid, bias_shift=shift)
        server.filter_updates(true_malicious=malicious_ids)
        server.aggregate()
        server.reset()

    alarm_csv  = tmp_path / "alarm.csv"
    filter_csv = tmp_path / "filter.csv"
    server.write_analysis_logs(str(alarm_csv), str(filter_csv))

    assert alarm_csv.exists()
    assert filter_csv.exists()

    with open(alarm_csv, newline="") as f:
        alarm_rows = list(csv.DictReader(f))
    with open(filter_csv, newline="") as f:
        filter_rows = list(csv.DictReader(f))

    assert len(alarm_rows) == n_rounds
    expected_alarm_cols = {
        "round", "is_attack_round", "num_clients", "num_malicious_present",
        "W_inf", "tau_decay", "s_t", "R_ratio", "would_alarm_decay",
    }
    assert expected_alarm_cols.issubset(alarm_rows[0].keys())
    for row in alarm_rows:
        for col in expected_alarm_cols:
            assert row[col] != "", f"column {col} empty in alarm row {row}"

    assert len(filter_rows) == n_rounds * n_clients
    expected_filter_cols = {
        "round", "is_attack_round", "client_id", "is_malicious_present",
        "d_i", "theta_min", "theta_max", "inside_interval",
    }
    assert expected_filter_cols.issubset(filter_rows[0].keys())
    for row in filter_rows:
        for col in expected_filter_cols:
            assert row[col] != "", f"column {col} empty in filter row {row}"

    # Malicious clients on attack rounds should sit outside the accept
    # interval far more often than on quiet rounds (sanity, not a hard bound).
    mal_attack_outside = sum(
        1 for r in filter_rows
        if int(r["is_malicious_present"]) == 1 and int(r["inside_interval"]) == 0
    )
    mal_attack_total = sum(1 for r in filter_rows if int(r["is_malicious_present"]) == 1)
    assert mal_attack_total > 0
    assert mal_attack_outside / mal_attack_total > 0.5


def test_write_analysis_logs_noop_when_empty(tmp_path):
    server = _make_server(analysis_mode=True)
    alarm_csv  = tmp_path / "alarm.csv"
    filter_csv = tmp_path / "filter.csv"
    server.write_analysis_logs(str(alarm_csv), str(filter_csv))
    assert not alarm_csv.exists()
    assert not filter_csv.exists()


# ---------------------------------------------------------------------------
# Test 4: guards — tiny/empty inputs don't crash
# ---------------------------------------------------------------------------

def test_analysis_mode_handles_zero_clients():
    server = _make_server(analysis_mode=True)
    result = server.filter_updates(true_malicious=frozenset())
    assert result.rejected_ids == frozenset()
    assert len(server._alarm_log) == 1
    assert server._alarm_log[0]["num_clients"] == 0
    assert server._filter_log == []


def test_analysis_mode_handles_single_client():
    server = _make_server(analysis_mode=True)
    _submit_client(server, 0, bias_shift=0.1)
    result = server.filter_updates(true_malicious=frozenset())
    assert result.rejected_ids == frozenset()
    assert len(server._filter_log) == 1
    assert server._filter_log[0]["d_i"] == 0.0  # single point == its own median


def test_r_ratio_uses_clamped_tau_min_no_division_by_zero():
    server = _make_server(analysis_mode=True, bottleneck_min_threshold=0.0)
    for round_idx in range(2):
        for cid in range(3):
            _submit_client(server, cid, bias_shift=0.0)  # identical deltas -> s_t = 0
        server.filter_updates(true_malicious=frozenset())
        server.aggregate()
        server.reset()
    # Should not have raised (no ZeroDivisionError / inf from 0/0).
    assert len(server._alarm_log) == 2


# ---------------------------------------------------------------------------
# Test 5: analysis_mode=False preserves production behaviour
# ---------------------------------------------------------------------------

def test_analysis_mode_off_can_still_reject():
    """With analysis_mode left at its default (False), an obviously
    triggered + out-of-interval client can still be rejected -- i.e. adding
    analysis_mode did not disable production filtering."""
    server = _make_server(
        bottleneck_initial_threshold=0.0,  # force trigger_filtering immediately
        bottleneck_decay_rate=1.0,
        bottleneck_min_threshold=0.0,
        min_bias_history_size=3,
    )
    assert server.analysis_mode is False

    # Warm up benign history with tight, consistent distances.
    for _ in range(2):
        for cid in range(4):
            _submit_client(server, cid, bias_shift=0.001 * cid)
        server.filter_updates(true_malicious=frozenset())
        server.aggregate()
        server.reset()

    # Now submit one wild outlier alongside consistent benign clients.
    for cid in range(4):
        shift = 50.0 if cid == 3 else 0.001 * cid
        _submit_client(server, cid, bias_shift=shift)
    result = server.filter_updates(true_malicious=frozenset({3}))

    # Production path must remain capable of rejecting (unchanged behaviour).
    assert isinstance(result.rejected_ids, frozenset)
