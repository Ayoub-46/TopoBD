"""Patch (BadNets) attack client with configurable attack window.

The trigger is a fixed-pattern solid-colour patch applied in [0, 1] pixel
space before normalisation.  The poisoned dataset is built once at
construction time (static trigger, reproducible index selection).  The
attack window is enforced in :meth:`local_train`: outside the window the
client falls back to standard benign training, matching the behaviour of
all other attack clients in this framework.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from fl.client import BenignClient, ClientUpdate
from datasets.backdoor import BackdoorDataset
from attacks.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)


@dataclass
class PatchConfig:
    """Typed configuration for the patch attack client.

    Args:
        trigger:             An instantiated :class:`BaseTrigger`
                             (typically :class:`~attacks.triggers.patch.PatchTrigger`).
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation transform applied after
                             the trigger so the model receives normalised inputs.
        poison_fraction:     Fraction of local training samples to poison
                             each round.
        attack_start_round:  First FL round in which the attack is active.
        attack_end_round:    Last FL round in which the attack is active
                             (inclusive). ``float('inf')`` means no end.
        seed:                Base RNG seed for reproducible poisoning index
                             selection (should be per-client to avoid identical
                             poisoned subsets across clients).
    """
    trigger: BaseTrigger
    target_label: int
    normalize_transform: Callable
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    attack_end_round: float = float("inf")
    seed: int = 42


class PatchClient(BenignClient):
    """FL client that mounts the patch (BadNets) backdoor attack.

    The ``trainloader`` must be built from a **pre-normalisation** dataset
    (i.e. returned by ``DatasetAdapter.get_client_pre_loaders()``), because
    the trigger is applied in ``[0, 1]`` pixel space before normalisation.

    A poisoned :class:`~datasets.backdoor.BackdoorDataset` is constructed
    once in ``__init__`` with fixed poisoned indices (static trigger, seed
    from config).  At each round, :meth:`local_train` consults the attack
    window and either trains on the poisoned dataset (inside window) or
    falls back to standard benign training on the clean pre-norm dataset
    (outside window).

    Args:
        config: :class:`PatchConfig` instance.
        All other args forwarded to :class:`~fl.client.BenignClient`.
    """

    def __init__(self, config: PatchConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config

        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=config.trigger.apply,
            target_label=config.target_label,
            post_trigger_transform=config.normalize_transform,
            poison_fraction=config.poison_fraction,
            seed=config.seed,
            poison_exclude_target=True,
        )
        self._poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=self.trainloader.num_workers,
            pin_memory=self.trainloader.pin_memory,
        )

        logger.info(
            "PatchClient [%d]: %d/%d samples poisoned (target=%d, "
            "window=[%d, %s]).",
            self.id,
            len(poisoned_dataset.poisoned_indices),
            len(poisoned_dataset),
            config.target_label,
            config.attack_start_round,
            config.attack_end_round if config.attack_end_round != float("inf") else "∞",
        )

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Poisoned or benign local training depending on the attack window.

        Outside ``[attack_start_round, attack_end_round]`` the client
        behaves identically to a :class:`~fl.client.BenignClient`.
        """
        cfg = self.config

        if not (cfg.attack_start_round <= round_idx <= cfg.attack_end_round):
            return super().local_train(epochs=epochs, round_idx=round_idx)

        # Temporarily swap in the poisoned loader, run the inherited training
        # loop, then restore the clean loader.
        original_loader = self.trainloader
        self.trainloader = self._poisoned_loader
        result = super().local_train(epochs=epochs, round_idx=round_idx)
        self.trainloader = original_loader

        result.is_malicious = True
        result.metadata.update({
            "attack": "patch",
            "target_label": cfg.target_label,
            "poison_fraction": cfg.poison_fraction,
            "num_poisoned": len(self._poisoned_loader.dataset.poisoned_indices),
        })
        return result
