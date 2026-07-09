"""Regression tests: outside their attack window, attack clients must train
on normalised data (matching a real BenignClient), not the raw [0,1]
pre-normalisation loader they're constructed with.

Bug this guards against: all five attack client classes were built with
``trainloader=pre_loaders[cid]`` (needed so the trigger/perturbation can be
applied in [0,1] space) and, outside the attack window, called
``return super().local_train(...)`` directly -- which trains on that raw
loader with no normalisation applied, since BenignClient.local_train trusts
its trainloader is already normalised. This produced a large,
attack-independent distributional shift for malicious-labeled clients on
every round they weren't actively poisoning (before attack_start_round,
after attack_end_round, or on quiet rounds of any sporadic schedule).

The fix builds a ``_clean_loader`` at construction time (BackdoorDataset
with poison_fraction=0.0, which still applies normalize_transform to every
sample since BackdoorDataset applies post_trigger_transform unconditionally)
and swaps it in for the benign-fallback branch of local_train.

Run with:
    pytest tests/test_attack_benign_fallback_normalization.py -v
"""
import dataclasses
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.a3fl_client import A3FLClient, A3FLConfig
from attacks.chameleon_client import ChameleonClient, ChameleonConfig
from attacks.iba_client import IBAClient, IBAConfig
from attacks.neurotoxin_client import NeurotoxinClient, NeurotoxinConfig
from attacks.patch_client import PatchClient, PatchConfig

_RAW_VALUE = 0.9   # every raw sample is this constant, in [0, 1] pre-norm space
_DIM = 8


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Distinctive affine map so raw-vs-normalised is trivially distinguishable."""
    return (x - 0.5) * 10.0


class _RawDataset(Dataset):
    """Tiny synthetic pre-normalisation dataset: constant [0,1] vectors."""

    def __init__(self, n: int = 16):
        self.targets = [i % 2 for i in range(n)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.full((_DIM,), _RAW_VALUE), self.targets[idx]


class _AssertNotCalledTrigger:
    """A trigger whose .apply must never be invoked: with poison_fraction=0.0
    (used for the clean/fallback loader) no sample should ever be poisoned."""

    def apply(self, x):
        raise AssertionError(
            "trigger_fn was invoked while building/using the clean "
            "(poison_fraction=0.0) fallback loader -- it should poison nothing."
        )


def _make_client(cls, config_cls, extra_config_kwargs):
    raw_loader = DataLoader(_RawDataset(), batch_size=4)
    model = nn.Linear(_DIM, 2)
    config_kwargs = dict(
        target_label=0,
        normalize_transform=_normalize,
        attack_start_round=100,   # window far in the future --
        attack_end_round=200,     # round_idx=0 below is always a fallback round
        seed=0,
    )
    config_kwargs.update(extra_config_kwargs)
    field_names = {f.name for f in dataclasses.fields(config_cls)}
    if "trigger" in field_names:
        config_kwargs["trigger"] = _AssertNotCalledTrigger()
    config = config_cls(**config_kwargs)
    client = cls(
        config=config,
        id=0,
        trainloader=raw_loader,
        testloader=None,
        model=model,
        lr=0.01,
        weight_decay=0.0,
        epochs=1,
        device=torch.device("cpu"),
    )
    return client, raw_loader


def _assert_clean_loader_is_normalized(client):
    x, _ = client._clean_loader.dataset[0]
    assert torch.allclose(x, _normalize(torch.full((_DIM,), _RAW_VALUE))), (
        "clean fallback loader must apply normalize_transform, not raw [0,1] data"
    )
    assert not torch.allclose(x, torch.full((_DIM,), _RAW_VALUE)), (
        "clean fallback loader is returning raw, un-normalised data"
    )


def _assert_fallback_restores_trainloader(client, raw_loader):
    client.local_train(epochs=1, round_idx=0)  # round 0 is outside [100, 200]
    assert client.trainloader is raw_loader, (
        "local_train must restore the original (raw) trainloader after "
        "temporarily swapping in the clean loader for the fallback branch"
    )


def test_patch_client_benign_fallback_is_normalized():
    client, raw_loader = _make_client(PatchClient, PatchConfig, {})
    _assert_clean_loader_is_normalized(client)
    _assert_fallback_restores_trainloader(client, raw_loader)


def test_neurotoxin_client_benign_fallback_is_normalized():
    client, raw_loader = _make_client(
        NeurotoxinClient, NeurotoxinConfig, {"mask_k_percent": 0.95}
    )
    _assert_clean_loader_is_normalized(client)
    _assert_fallback_restores_trainloader(client, raw_loader)


def test_a3fl_client_benign_fallback_is_normalized():
    client, raw_loader = _make_client(
        A3FLClient, A3FLConfig, {"trigger_sample_size": 4}
    )
    _assert_clean_loader_is_normalized(client)
    _assert_fallback_restores_trainloader(client, raw_loader)


def test_iba_client_benign_fallback_is_normalized():
    client, raw_loader = _make_client(
        IBAClient, IBAConfig, {"trigger_sample_size": 4}
    )
    _assert_clean_loader_is_normalized(client)
    _assert_fallback_restores_trainloader(client, raw_loader)


def test_chameleon_client_benign_fallback_is_normalized():
    client, raw_loader = _make_client(ChameleonClient, ChameleonConfig, {})
    _assert_clean_loader_is_normalized(client)
    _assert_fallback_restores_trainloader(client, raw_loader)


def test_chameleon_client_no_peers_fallback_is_normalized():
    """target_label=0 and every sample in _RawDataset has label 0 or 1, so
    with target_label=1 there ARE peer (target-class) samples -- flip it so
    there are none, exercising the *other* fallback path (no target-class
    samples in local data), which must also use the normalised loader."""
    client, raw_loader = _make_client(
        ChameleonClient, ChameleonConfig, {"target_label": 99}
    )
    # attack_start_round/end_round are set so round 0 IS inside the window,
    # to force the "no peers" branch specifically rather than the window
    # check short-circuiting first.
    client.config.attack_start_round = 0
    client.config.attack_end_round = 10
    result = client.local_train(epochs=1, round_idx=0)
    assert client.trainloader is raw_loader
    assert result.is_malicious is False  # fell back to plain BenignClient training
