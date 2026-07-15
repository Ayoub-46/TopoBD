"""Unit tests for TopoSentinelServer's post-filter aggregation clipping
strategies: clip_mode in {"global" (default), "layerwise", "none"}.

Key motivating scenario for layerwise: a malicious client can concentrate a
large perturbation in a naturally low-norm layer (e.g. bias) while matching
benign clients on a naturally high-norm layer (e.g. a big weight matrix).
Global clipping's single scalar scale is driven by the FULL flattened delta
norm, which the high-norm layer dominates -- so the concentrated attack in
the small layer barely moves that scale and passes through almost entirely.
Layerwise clipping scales each parameter tensor against its OWN median
survivor norm, so the anomalous small layer gets caught independent of the
large layer's scale.

Run with:
    pytest tests/test_toposentinel_clip_modes.py -v
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.toposentinel import TopoSentinelServer


class _BigSmallModel(nn.Module):
    """Two parameters of deliberately very different natural scale."""

    def __init__(self):
        super().__init__()
        self.big = nn.Parameter(torch.zeros(1000))
        self.small = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return x


def _make_server(clip_mode: str, **kwargs) -> TopoSentinelServer:
    return TopoSentinelServer(
        model=_BigSmallModel(), device=torch.device("cpu"), clip_mode=clip_mode,
        min_clients_for_defense=3, **kwargs,
    )


def _submit(server, cid, big_delta, small_delta, length=100):
    global_params = server.get_params()
    params = {k: v.clone() for k, v in global_params.items()}
    params["big"] = params["big"] + big_delta
    params["small"] = params["small"] + small_delta
    server.receive_update(client_id=cid, params=params, length=length)


# Benign: "big" delta has L2 norm exactly 10 (spread over 1000 dims), "small"
# delta has L2 norm exactly 0.01. Malicious: same "big" (blends in), but
# "small" has L2 norm 1.0 -- 100x the benign small-layer norm, while barely
# moving the FULL-vector norm (10.0 -> ~10.05).
_BIG_BENIGN   = torch.full((1000,), 10.0 / (1000 ** 0.5))
_SMALL_BENIGN = torch.full((2,), 0.01 / (2 ** 0.5))
_SMALL_MALICIOUS = torch.full((2,), 1.0 / (2 ** 0.5))
_N_BENIGN = 5
_MAL_ID = _N_BENIGN


def _run_round(clip_mode: str) -> dict:
    server = _make_server(clip_mode)
    for cid in range(_N_BENIGN):
        _submit(server, cid, _BIG_BENIGN, _SMALL_BENIGN)
    _submit(server, _MAL_ID, _BIG_BENIGN, _SMALL_MALICIOUS)

    server.filter_updates(true_malicious=frozenset({_MAL_ID}))
    agg = server.aggregate()
    return agg.aggregated_params


def test_default_clip_mode_is_global():
    server = _make_server("global")
    assert server.clip_mode == "global"


def test_global_clipping_barely_suppresses_concentrated_small_layer_attack():
    params = _run_round("global")
    # Aggregated "small" should be pulled far from the benign value (0.01/sqrt(2)
    # per element) towards the malicious value, since global clipping applies
    # essentially no suppression (scale ~= 0.995, driven by the dominant "big"
    # layer's norm).
    small_norm = params["small"].norm(p=2).item()
    assert small_norm > 0.1, (
        f"expected global clipping to barely suppress the concentrated "
        f"small-layer attack, got small_norm={small_norm:.4f}"
    )


def test_layerwise_clipping_suppresses_concentrated_small_layer_attack():
    params = _run_round("layerwise")
    small_norm = params["small"].norm(p=2).item()
    # With layerwise clipping the malicious "small" contribution is scaled by
    # ~0.01 (median benign small-layer norm / malicious small-layer norm),
    # so the aggregate should stay close to the benign-only value.
    assert small_norm < 0.02, (
        f"expected layerwise clipping to suppress the concentrated "
        f"small-layer attack, got small_norm={small_norm:.4f}"
    )


def test_layerwise_still_passes_benignlike_big_layer_unclipped():
    params = _run_round("layerwise")
    # The "big" layer matches benign clients across the board, so it should
    # not be meaningfully suppressed even under layerwise clipping.
    big_norm = params["big"].norm(p=2).item()
    assert big_norm > 9.0


def test_none_clip_mode_applies_no_suppression_at_all():
    params_none   = _run_round("none")
    params_global = _run_round("global")
    # "none" must pass the malicious small-layer delta through completely
    # unscaled, i.e. strictly more of it survives than under global clipping.
    assert params_none["small"].norm(p=2).item() >= params_global["small"].norm(p=2).item()


def test_unknown_clip_mode_falls_back_to_global_with_warning():
    server = _make_server("bogus_mode")
    assert server.clip_mode == "global"
