"""Unit tests for module-type-based bias-parameter selection.

_compute_bias_param_names() walks the model's module tree and selects each
module's `.bias` when it is a learnable nn.Parameter -- not a "bias" in name
substring match. This pins the exact matched-name list for a small synthetic
Conv+BN+Linear model, asserting BatchNorm's weight (gamma) is excluded and
its bias (beta) is included.

Run with:
    pytest tests/test_toposentinel_bias_extraction.py -v
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenses.toposentinel import TopoSentinelServer


class _TinyConvBNLinear(nn.Module):
    """Conv(bias) -> BatchNorm(weight=gamma, bias=beta) -> Linear(bias)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def test_matched_bias_names_include_conv_linear_and_bn_beta_exclude_gamma():
    server = TopoSentinelServer(model=_TinyConvBNLinear(), device=torch.device("cpu"))

    assert server._bias_param_names == ["conv.bias", "bn.bias", "fc.bias"]

    # BatchNorm's weight (gamma, multiplicative scale) must never be selected.
    assert "bn.weight" not in server._bias_param_names
    # BatchNorm's bias (beta, additive) must be selected.
    assert "bn.bias" in server._bias_param_names
    # Conv/Linear weight matrices must never be selected.
    assert "conv.weight" not in server._bias_param_names
    assert "fc.weight" not in server._bias_param_names


def test_exclude_bn_bias_drops_bn_beta_keeps_conv_and_linear():
    server = TopoSentinelServer(
        model=_TinyConvBNLinear(), device=torch.device("cpu"), exclude_bn_bias=True,
    )
    assert server._bias_param_names == ["conv.bias", "fc.bias"]
    assert "bn.bias" not in server._bias_param_names


def test_exclude_bn_bias_default_false_preserves_prior_behaviour():
    server = TopoSentinelServer(model=_TinyConvBNLinear(), device=torch.device("cpu"))
    assert server.exclude_bn_bias is False
    assert server._bias_param_names == ["conv.bias", "bn.bias", "fc.bias"]


def test_no_bias_module_yields_empty_list_and_none_vector():
    class _NoBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=False)

        def forward(self, x):
            return self.conv(x)

    server = TopoSentinelServer(model=_NoBias(), device=torch.device("cpu"))
    assert server._bias_param_names == []
    assert server._extract_bias_vector(server.get_params()) is None


def test_extract_bias_vector_uses_exact_names_not_substring():
    """A parameter whose name merely contains "bias" as a substring, but
    isn't in the module-derived name list, must be excluded."""
    server = TopoSentinelServer(model=_TinyConvBNLinear(), device=torch.device("cpu"))
    params = server.get_params()
    # Inject a decoy key containing "bias" as a substring that is NOT one of
    # the module tree's real bias parameters.
    params = dict(params)
    params["unbiased_estimator"] = torch.zeros(100)

    vec = server._extract_bias_vector(params)
    expected_len = sum(params[name].numel() for name in server._bias_param_names)
    assert vec is not None
    assert vec.shape[0] == expected_len
