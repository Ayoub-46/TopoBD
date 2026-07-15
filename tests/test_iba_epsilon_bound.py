"""Stage-1 validity test for the epsilon-bounded IBA trigger.

Verifies the paper Eq. 1 contract: the additive perturbation is L∞-bounded by
epsilon and the poisoned image stays in [0, 1] — regardless of the (possibly
untrained / extreme) generator weights.
"""

import torch

from attacks.triggers import get_trigger
from attacks.triggers.iba import IBATrigger
from attacks.triggers.unet import UNet

_TOL = 1e-6


def test_project_bounds_arbitrary_tensor():
    """The L∞ projection clamps any generator output to [-eps, +eps]."""
    eps = 0.3
    trigger = IBATrigger(unet=UNet(in_channels=3, base_features=16), epsilon=eps)
    huge = torch.randn(4, 3, 8, 8) * 1000.0
    projected = trigger._project(huge)
    assert projected.abs().max().item() <= eps + _TOL


def test_apply_perturbation_linf_bounded_and_in_range():
    """apply(): ||x̃ - x||_∞ ≤ eps and x̃ ∈ [0, 1] for random inputs."""
    torch.manual_seed(0)
    eps = 0.3
    trigger = get_trigger("iba", in_channels=3, base_features=16, epsilon=eps)
    trigger.generator.eval()
    for _ in range(5):
        img = torch.rand(3, 32, 32)
        out = trigger.apply(img)
        assert (out - img).abs().max().item() <= eps + _TOL      # L∞ bound
        assert out.min().item() >= 0.0 - _TOL                    # valid range
        assert out.max().item() <= 1.0 + _TOL


def test_apply_bound_holds_for_extreme_generator():
    """Even with blown-up generator weights, the projection dominates."""
    eps = 0.1
    unet = UNet(in_channels=3, base_features=16)
    with torch.no_grad():
        for p in unet.parameters():
            p.mul_(50.0)
    trigger = IBATrigger(unet=unet, epsilon=eps)
    trigger.generator.eval()
    img = torch.rand(3, 32, 32)
    out = trigger.apply(img)
    assert (out - img).abs().max().item() <= eps + _TOL
    assert out.min().item() >= 0.0 - _TOL
    assert out.max().item() <= 1.0 + _TOL


def test_different_epsilon_changes_bound():
    """A smaller epsilon yields a strictly tighter perturbation bound."""
    torch.manual_seed(1)
    img = torch.rand(3, 32, 32)
    outs = {}
    for eps in (0.05, 0.5):
        t = get_trigger("iba", in_channels=3, base_features=16, epsilon=eps)
        t.generator.eval()
        outs[eps] = (t.apply(img) - img).abs().max().item()
        assert outs[eps] <= eps + _TOL


def test_epsilon_decay_schedule():
    """Eq. 4: eps_t = max(eps_hat, eps_0*(1-lambda)^(t-t_I)), monotone, floored."""
    t = get_trigger("iba", in_channels=3, base_features=16,
                    epsilon=0.05, eps_0=0.3, lambda_decay=0.5)
    t_I = 10
    assert abs(t._epsilon_at(10, t_I) - 0.30) < _TOL     # t-t_I = 0
    assert abs(t._epsilon_at(11, t_I) - 0.15) < _TOL     # 0.3*0.5
    assert abs(t._epsilon_at(12, t_I) - 0.075) < _TOL    # 0.3*0.25
    assert abs(t._epsilon_at(13, t_I) - 0.05) < _TOL     # 0.0375 -> floored
    assert abs(t._epsilon_at(50, t_I) - 0.05) < _TOL     # stays at floor
    # monotone non-increasing
    vals = [t._epsilon_at(r, t_I) for r in range(10, 20)]
    assert all(vals[i] >= vals[i + 1] - _TOL for i in range(len(vals) - 1))
    # before the attack window (t < t_I) the schedule clamps t-t_I to 0
    assert abs(t._epsilon_at(5, t_I) - 0.30) < _TOL


def test_train_trigger_sets_deployed_epsilon_to_floor_after_decay():
    """After enough active rounds the deployed epsilon equals the floor."""
    t = get_trigger("iba", in_channels=3, base_features=16,
                    epsilon=8 / 255, eps_0=0.3, lambda_decay=0.4)
    # simulate the schedule setting the current radius at a late round
    t.epsilon = t._epsilon_at(30, 10)
    assert abs(t.epsilon - 8 / 255) < _TOL
