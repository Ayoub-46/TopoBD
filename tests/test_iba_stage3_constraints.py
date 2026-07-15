"""Stage-3 unit tests: constrained model poisoning math (paper §3.2).

Tests the pure projection/masking logic of IBAClient._constrain_poison_update
without building a full federated client (constructed via __new__ with only the
attributes the method reads: config + running grad estimate).
"""

import torch

from attacks.iba_client import IBAClient, IBAConfig


def _bare_client(delta, k_percent, grad_running=None):
    c = IBAClient.__new__(IBAClient)
    c.config = IBAConfig(
        trigger=None, target_label=0, normalize_transform=None,
        delta=delta, dimension_k_percent=k_percent,
    )
    c._grad_running = grad_running
    c._grad_rounds = 0 if grad_running is None else 1
    return c


def test_dimension_mask_confines_to_bottom_k():
    """Only the bottom-k% coordinates (smallest running grad) survive."""
    w_global = {"a": torch.zeros(4), "b": torch.zeros(4)}
    w_local = {"a": torch.ones(4), "b": torch.ones(4)}
    # a has the small gradients (1..4), b the large (5..8); bottom-50% = all of a
    grad_running = {"a": torch.tensor([1., 2., 3., 4.]),
                    "b": torch.tensor([5., 6., 7., 8.])}
    c = _bare_client(delta=None, k_percent=0.5, grad_running=grad_running)
    new_w, raw, proj, n = c._constrain_poison_update(w_global, w_local)
    assert torch.allclose(new_w["a"], torch.ones(4))   # kept
    assert torch.allclose(new_w["b"], torch.zeros(4))  # zeroed (top-half)
    assert n == 4
    assert proj == raw  # no space projection (delta None)


def test_space_projection_scales_to_delta():
    """When the update norm exceeds delta it is rescaled onto the L2 ball."""
    w_global = {"a": torch.zeros(2)}
    w_local = {"a": torch.tensor([3.0, 4.0])}          # update norm = 5
    c = _bare_client(delta=1.0, k_percent=1.0)         # k=1.0 disables masking
    new_w, raw, proj, n = c._constrain_poison_update(w_global, w_local)
    assert abs(raw - 5.0) < 1e-5
    assert abs(proj - 1.0) < 1e-5
    assert abs(new_w["a"].norm().item() - 1.0) < 1e-5


def test_no_projection_when_within_delta():
    """Updates inside the ball are untouched."""
    w_global = {"a": torch.zeros(2)}
    w_local = {"a": torch.tensor([0.3, 0.4])}          # norm 0.5 < delta
    c = _bare_client(delta=1.0, k_percent=1.0)
    new_w, raw, proj, n = c._constrain_poison_update(w_global, w_local)
    assert abs(raw - 0.5) < 1e-5 and abs(proj - 0.5) < 1e-5
    assert torch.allclose(new_w["a"], w_local["a"])


def test_identity_when_both_constraints_off():
    """delta=None and k=1.0 -> constrained update equals the raw update."""
    w_global = {"a": torch.zeros(3), "b": torch.tensor([1.0, 2.0, 3.0])}
    w_local = {"a": torch.tensor([1.0, -1.0, 2.0]), "b": torch.tensor([2.0, 2.0, 2.0])}
    c = _bare_client(delta=None, k_percent=1.0)
    new_w, raw, proj, n = c._constrain_poison_update(w_global, w_local)
    assert torch.allclose(new_w["a"], w_local["a"])
    assert torch.allclose(new_w["b"], w_local["b"])
    assert n == 0 and proj == raw


def test_combined_mask_then_project():
    """Mask reduces to bottom-k, then the masked update is projected to delta."""
    w_global = {"a": torch.zeros(4)}
    w_local = {"a": torch.tensor([3.0, 4.0, 10.0, 10.0])}
    grad_running = {"a": torch.tensor([1.0, 2.0, 100.0, 100.0])}  # keep first two
    c = _bare_client(delta=1.0, k_percent=0.5, grad_running=grad_running)
    new_w, raw, proj, n = c._constrain_poison_update(w_global, w_local)
    assert n == 2                                   # bottom-50% of 4 = 2 coords
    assert abs(raw - 5.0) < 1e-5                    # masked update = [3,4,0,0]
    assert abs(new_w["a"].norm().item() - 1.0) < 1e-5
    assert new_w["a"][2] == 0 and new_w["a"][3] == 0
