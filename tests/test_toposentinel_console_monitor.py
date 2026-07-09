"""Unit tests for the console_monitor opt-in per-round stdout summary.

Purely observational: it must never change what gets rejected, must print
nothing when off (default) or in analysis_mode, and must report accurate
TPR/FPR/rejected-client data when on.

Run with:
    pytest tests/test_toposentinel_console_monitor.py -v
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.toposentinel import TopoSentinelServer


def _make_server(**kwargs) -> TopoSentinelServer:
    model = nn.Linear(8, 4)
    return TopoSentinelServer(model=model, device=torch.device("cpu"), **kwargs)


def _submit_client(server, cid, bias_shift, length=100):
    global_params = server.get_params()
    params = {k: v.clone() for k, v in global_params.items()}
    params["bias"] = params["bias"] + torch.as_tensor(bias_shift, dtype=params["bias"].dtype)
    server.receive_update(client_id=cid, params=params, length=length)


def test_console_monitor_off_by_default_prints_nothing(capsys):
    server = _make_server(min_clients_for_defense=3)
    for cid in range(4):
        _submit_client(server, cid, bias_shift=0.01 * cid)
    server.filter_updates(true_malicious=frozenset({3}))

    captured = capsys.readouterr()
    assert captured.out == ""


def test_console_monitor_on_prints_summary_with_correct_fields(capsys):
    server = _make_server(console_monitor=True, min_clients_for_defense=3)
    for cid in range(4):
        _submit_client(server, cid, bias_shift=0.01 * cid)
    result = server.filter_updates(true_malicious=frozenset({3}))

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.startswith("[TopoSentinel]")]
    assert len(lines) == 1
    line = lines[0]

    assert "round=   0" in line
    assert "clients=[0, 1, 2, 3]" in line
    assert "malicious=[3]" in line
    assert f"rejected={sorted(result.rejected_ids)}" in line
    assert "threshold=" in line
    assert "bottleneck=" in line
    assert "triggered=" in line
    assert "TPR=" in line
    assert "FPR=" in line


def test_console_monitor_does_not_change_rejection_decision():
    """console_monitor must be pure observability -- identical rejected_ids
    with it on vs off, given the same inputs/seed."""
    def run(monitor: bool):
        server = _make_server(
            console_monitor=monitor,
            bottleneck_initial_threshold=0.0,  # force trigger immediately
            bottleneck_decay_rate=1.0,
            bottleneck_min_threshold=0.0,
            min_bias_history_size=3,
        )
        for _ in range(2):
            for cid in range(4):
                _submit_client(server, cid, bias_shift=0.001 * cid)
            server.filter_updates(true_malicious=frozenset())
            server.aggregate()
            server.reset()

        for cid in range(4):
            shift = 50.0 if cid == 3 else 0.001 * cid
            _submit_client(server, cid, bias_shift=shift)
        return server.filter_updates(true_malicious=frozenset({3})).rejected_ids

    assert run(monitor=False) == run(monitor=True)


def test_console_monitor_silent_in_analysis_mode(capsys):
    """console_monitor is documented as production-path only; analysis_mode
    has its own CSV logging, so no stdout print should occur even if both
    flags are set."""
    server = _make_server(console_monitor=True, analysis_mode=True)
    for cid in range(4):
        _submit_client(server, cid, bias_shift=0.01 * cid)
    server.filter_updates(true_malicious=frozenset({3}))

    captured = capsys.readouterr()
    assert "[TopoSentinel]" not in captured.out


def test_console_monitor_shows_na_when_defense_skipped(capsys):
    """Below min_clients_for_defense, detection is skipped -- bottleneck and
    triggered must show as n/a, not a stale value from a previous round."""
    server = _make_server(console_monitor=True, min_clients_for_defense=3)
    _submit_client(server, 0, bias_shift=0.1)  # only 1 client, below min=3
    server.filter_updates(true_malicious=frozenset())

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.startswith("[TopoSentinel]")]
    assert len(lines) == 1
    assert "bottleneck=n/a" in lines[0]
    assert "triggered=n/a" in lines[0]
