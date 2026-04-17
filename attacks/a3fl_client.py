"""A3FL attack client.

Reference
---------
Zhang et al., "A3FL: Adversarially Adaptive Backdoor Attacks to
Federated Learning", NeurIPS 2023.

The client performs two stages every active round:

1. **Trigger optimisation** — adapts the learnable trigger to the current
   global model using a small local subset.
2. **Poisoned local training** — trains on a mix of clean and triggered
   samples, with the normalisation transform applied correctly after the
   trigger so the model receives properly normalised inputs.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fl.client import BenignClient, ClientUpdate
from datasets.backdoor import BackdoorDataset
from attacks.triggers.base import LearnableTrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class A3FLConfig:
    """Typed configuration for the A3FL attack client.

    Args:
        trigger:              An instantiated :class:`LearnableTrigger`.
                              Must be provided — there is no safe default
                              because the trigger shape depends on the dataset.
        target_label:         Backdoor target class index.
        normalize_transform:  The dataset's normalisation transform (e.g.
                              ``Normalize(mean, std)``).  Applied after the
                              trigger so the model receives normalised inputs.
        poison_fraction:      Fraction of local training samples to poison
                              each round.
        attack_start_round:   First FL round in which the attack is active.
        attack_end_round:     Last FL round in which the attack is active
                              (inclusive).  Defaults to ``float('inf')``
                              (always active once started).
        trigger_sample_size:  Maximum number of local samples used to
                              optimise the trigger in Stage 1.
        seed:                 Base RNG seed for reproducible poisoning.
                              The actual seed used each round is
                              ``seed + round_idx`` so that the poisoned
                              subset varies across rounds.
    """

    trigger: LearnableTrigger
    target_label: int
    normalize_transform: Callable
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    attack_end_round: float = float("inf")
    trigger_sample_size: int = 512
    seed: int = 42


# ---------------------------------------------------------------------------
# Attack client
# ---------------------------------------------------------------------------

class A3FLClient(BenignClient):
    """Federated learning client that mounts the A3FL backdoor attack.

    Expects ``trainloader`` to be built from a **pre-normalisation** dataset
    (i.e. returned by ``DatasetAdapter.get_client_pre_loaders()``).  The
    normalisation transform is applied inside ``BackdoorDataset`` via
    ``config.normalize_transform``.

    Args:
        config:      :class:`A3FLConfig` instance.
        *args:       Forwarded to :class:`BenignClient`.
        **kwargs:    Forwarded to :class:`BenignClient`.
    """

    def __init__(self, config: A3FLConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    # ------------------------------------------------------------------
    # BenignClient interface
    # ------------------------------------------------------------------

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Two-stage A3FL attack, falling back to benign training outside
        the configured attack window.

        Args:
            epochs:    Local epochs.  ``None`` uses the constructor default.
            round_idx: Current FL round index.

        Returns:
            :class:`ClientUpdate` with ``is_malicious=True`` during attack
            rounds, ``False`` otherwise.
        """
        cfg = self.config

        # ---- Benign fallback ------------------------------------------------
        if not (cfg.attack_start_round <= round_idx <= cfg.attack_end_round):
            return super().local_train(epochs=epochs, round_idx=round_idx)

        # ---- Stage 1: optimise trigger against the current global model -----
        logger.info(
            "A3FL client [%d] — round %d: optimising trigger.", self.id, round_idx
        )
        trigger_loader = self._build_trigger_dataloader()
        if trigger_loader is not None:
            cfg.trigger.train_trigger(
                model=self._model,
                dataloader=trigger_loader,
                target_class=cfg.target_label,
            )
        else:
            logger.warning(
                "A3FL client [%d] — trigger optimisation skipped: no local data.",
                self.id,
            )

        # ---- Stage 2: train on poisoned + clean data -----------------------
        # FIX: seed varies per round so the poisoned subset changes each round
        round_seed = cfg.seed + round_idx

        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,  # [0, 1] pre-norm Subset
            trigger_fn=cfg.trigger.apply,
            target_label=cfg.target_label,
            post_trigger_transform=cfg.normalize_transform,  # FIX: normalise after trigger
            poison_fraction=cfg.poison_fraction,
            seed=round_seed,
            poison_exclude_target=True,
        )

        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=self.trainloader.num_workers
            if hasattr(self.trainloader, "num_workers")
            else 0,
        )

        # Swap trainloader, run benign training loop, restore
        original_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(epochs=epochs, round_idx=round_idx)
        finally:
            self.trainloader = original_loader  # always restored, even on exception

        # FIX: mark the update as malicious for server-side evaluation
        result.is_malicious = True

        # Attach attack metadata for logging and defense evaluation
        result.metadata.update(
            {
                "attack": "a3fl",
                "target_label": cfg.target_label,
                "poison_fraction": cfg.poison_fraction,
                "num_poisoned": len(poisoned_dataset.poisoned_indices),
                "round_seed": round_seed,
            }
        )

        logger.info(
            "A3FL client [%d] — round %d: poisoned %d / %d samples (target=%d).",
            self.id,
            round_idx,
            len(poisoned_dataset.poisoned_indices),
            len(poisoned_dataset),
            cfg.target_label,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trigger_dataloader(self) -> Optional[DataLoader]:
        """Sample a small subset of local (pre-norm) data for trigger
        optimisation.

        Returns:
            A ``DataLoader`` over at most ``config.trigger_sample_size``
            local samples, or ``None`` if the local dataset is empty.
        """
        base_dataset = self.trainloader.dataset
        num_samples = len(base_dataset)

        if num_samples == 0:
            return None

        k = min(self.config.trigger_sample_size, num_samples)
        # Use a fixed seed for the trigger subset so that trigger optimisation
        # is reproducible independently of the poisoning seed.
        rng = np.random.RandomState(self.config.seed)
        indices = rng.choice(num_samples, size=k, replace=False).tolist()

        batch_size = min(
            getattr(self.trainloader, "batch_size", 32), k
        )
        return DataLoader(
            Subset(base_dataset, indices),
            batch_size=batch_size,
            shuffle=True,
        )