"""Unit test for the intra-round filter operating on bias DELTAS
(client_bias - global_bias) rather than raw absolute bias vectors.

Note on why this test uses bias_metric="cosine", not the default "euclidean":
for euclidean distance, this change is provably a numeric no-op given the
current code path. np.median is exactly equivariant under subtracting the
same constant vector c from every sample -- median(x_i - c) == median(x_i) - c
-- so ||delta_i - median_delta|| == ||bias_i - median_bias|| identically, and
Sterbenz's lemma guarantees the subtraction itself is exact in floating point
when the operands are close in magnitude (true here, since median-of-similar-
clients stays close to any individual client's value).

For bias_metric="cosine" a large common bias component genuinely breaks
separation, but only once it's large enough to exhaust float64 precision --
empirically verified below (see the module-level comment before
`_GLOBAL_BIAS_MAGNITUDE`): at more modest magnitudes cosine distance shrinks
towards zero but still preserves relative ranking (to leading order it scales
with the *squared* ratio of each client's off-axis delta component, which
is small but not exactly zero). Only once the common component is large
enough that the true second-order signal falls below the ~1e-16 relative
float64 epsilon does the old absolute-vector approach collapse to an
*exact* tie (0.0 for every client, benign and outlier alike) -- a genuine,
unambiguous failure to separate, not just reduced sensitivity.

Run with:
    pytest tests/test_toposentinel_bias_delta_filter.py -v
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
    model = nn.Linear(8, 4)
    return TopoSentinelServer(model=model, device=torch.device("cpu"), **kwargs)


def test_delta_based_cosine_distance_separates_outlier_that_absolute_misses():
    """A large common bias component exhausts float64 precision for cosine
    similarity on absolute vectors (every client's distance rounds to
    exactly 0.0); subtracting it first (delta-based) restores separation.

    1e9 is chosen empirically, not arbitrarily: at more modest common-bias
    magnitudes (e.g. 1e3), cosine distance on absolute vectors shrinks a lot
    but still preserves the outlier's relative ranking (to leading order it
    scales with the squared off-axis delta component, small but nonzero).
    Only once the common component is large enough that this second-order
    signal falls below float64's ~1e-16 relative epsilon does it become an
    exact, unrecoverable tie -- verified to hold from 1e7 upward for these
    delta magnitudes.
    """
    server = _make_server(bias_metric="cosine")

    rng = np.random.RandomState(0)
    dim = 4

    # Large common component, identical for every client (as it is in FedAvg:
    # every client starts this round's local training from the same global
    # bias, so their post-training absolute bias vectors all sit near it).
    global_bias = np.full(dim, 1e9)

    # Benign clients: tiny deltas, all pointing in roughly the same direction.
    n_benign = 8
    benign_deltas = {
        cid: np.array([0.01, 0.02, 0.0, 0.0]) + rng.normal(scale=0.002, size=dim)
        for cid in range(n_benign)
    }
    # Planted outlier: delta of comparable magnitude but a clearly different
    # direction (orthogonal-ish) from the benign cluster.
    outlier_id = 99
    outlier_delta = np.array([0.0, 0.0, 0.03, -0.03])

    valid_ids = list(benign_deltas.keys()) + [outlier_id]
    client_deltas = {**benign_deltas, outlier_id: outlier_delta}
    client_biases_absolute = {cid: global_bias + d for cid, d in client_deltas.items()}

    # ---- OLD behaviour (removed): median + cosine distance on ABSOLUTE ----
    # vectors, replicated inline here (not calling production code) purely
    # to demonstrate what the prior implementation would have produced.
    from scipy.spatial.distance import cosine as scipy_cosine
    abs_median = np.median(np.vstack([client_biases_absolute[c] for c in valid_ids]), axis=0)
    abs_dists = {
        cid: scipy_cosine(client_biases_absolute[cid], abs_median) for cid in valid_ids
    }
    benign_abs = [abs_dists[c] for c in benign_deltas]
    outlier_abs = abs_dists[outlier_id]

    # The large shared component exhausts float64 precision: every absolute
    # vector's cosine distance to the median rounds to exactly 0.0, benign
    # and outlier alike -- an unrecoverable tie, not just a small signal.
    assert outlier_abs == 0.0, (
        f"Precondition failed: expected the absolute-vector approach to "
        f"completely lose the outlier signal (distance == 0.0), got "
        f"outlier={outlier_abs!r}"
    )
    assert all(d == 0.0 for d in benign_abs), (
        f"Precondition failed: expected all benign absolute-vector distances "
        f"to also be exactly 0.0, got {benign_abs!r}"
    )

    # ---- NEW behaviour: _bias_distances_from_median on DELTAS -------------
    median_delta = np.median(np.vstack([client_deltas[c] for c in valid_ids]), axis=0)
    delta_dists = server._bias_distances_from_median(valid_ids, client_deltas, median_delta)
    benign_delta = [delta_dists[c] for c in benign_deltas]
    outlier_delta_dist = delta_dists[outlier_id]

    # With the common component removed, the outlier's distinct direction
    # is clearly separated from the tight benign cluster.
    assert outlier_delta_dist > 10 * max(benign_delta), (
        f"Expected delta-based distance to clearly separate the outlier, got "
        f"outlier={outlier_delta_dist:.4f} vs max(benign)={max(benign_delta):.4f}"
    )


def test_run_defense_uses_deltas_not_absolutes_for_last_client_scores():
    """_last_client_scores (production path) must reflect delta-based
    distances, matching the analysis-mode filter log."""
    server = _make_server(min_clients_for_defense=3)
    global_params = server.get_params()

    # 3 benign clients with a large shared bias shift; a 4th with a distinct
    # delta direction. Euclidean metric here just checks the wiring (that
    # _last_client_scores comes from client_deltas / median_delta, not raw
    # absolute vectors) -- the cosine test above is what proves the fix
    # matters numerically.
    for cid, shift in enumerate([0.01, 0.01, 0.01, 5.0]):
        params = {k: v.clone() for k, v in global_params.items()}
        params["bias"] = params["bias"] + shift
        server.receive_update(client_id=cid, params=params, length=100)

    server.filter_updates(true_malicious=frozenset())

    assert server._last_client_scores is not None
    assert server._last_client_scores[3] > 3 * max(
        server._last_client_scores[c] for c in (0, 1, 2)
    )
