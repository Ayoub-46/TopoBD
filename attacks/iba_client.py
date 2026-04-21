"""IBA backdoor attack client.

Reference
---------
Nguyen et al., "IBA: Towards Irreversible Backdoor Attacks in Federated
Learning." NeurIPS 2023.

Each active round the client performs two stages:

1. **Generator training** — adapts the shared U-Net trigger against the
   current global model using a small local subset.
2. **Poisoned local training** — trains on a mix of clean and triggered
   samples; normalisation is applied correctly after trigger injection via
   ``BackdoorDataset.post_trigger_transform``.

All malicious clients receive the same ``IBATrigger`` instance (shared
reference).  Training is sequential within a round, so each client fine-
tunes the same generator in turn.  This is consistent with the assumption
in the paper that the adversary controls all malicious clients and
coordinates the attack.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fl.client import BenignClient, ClientUpdate
from datasets.backdoor import BackdoorDataset
from attacks.triggers.iba import IBATrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class IBAConfig:
    """Typed configuration for the IBA attack client.

    Args:
        trigger:             Shared :class:`IBATrigger` instance.  All
                             malicious clients in the same federation must
                             reference the *same* object so that generator
                             training is cumulative and the ASR evaluator
                             in the runner reflects the trained state.
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation applied after the trigger
                             inside ``BackdoorDataset``.
        poison_fraction:     Fraction of local samples to poison per round.
        attack_start_round:  First FL round in which the attack is active.
        attack_end_round:    Last FL round (inclusive).  ``float('inf')``
                             means no end.
        trigger_sample_size: Local samples used for generator fine-tuning.
                             Smaller values are faster; larger values give a
                             better gradient estimate.
        seed:                Base RNG seed.  Effective seed per round is
                             ``seed + round_idx``.
    """
    trigger: IBATrigger
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

class IBAClient(BenignClient):
    """FL client that mounts the IBA backdoor attack.

    The ``trainloader`` must be built from a **pre-normalisation** dataset
    (returned by ``DatasetAdapter.get_client_pre_loaders()``), because the
    trigger is applied in raw ``[0, 1]`` pixel space before normalisation.

    Args:
        config: :class:`IBAConfig` instance.
        All other args forwarded to :class:`~fl.client.BenignClient`.
    """

    def __init__(self, config: IBAConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    # ------------------------------------------------------------------
    # local_train override
    # ------------------------------------------------------------------

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Two-stage IBA attack, falling back to benign training outside
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

        # ---- Stage 1: fine-tune the shared U-Net generator ------------------
        logger.info(
            "IBA client [%d] — round %d: training generator.", self.id, round_idx
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
                "IBA client [%d] — trigger training skipped: no local data.", self.id
            )

        # ---- Stage 2: train on poisoned + clean data -----------------------
        round_seed = cfg.seed + round_idx

        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=cfg.trigger.apply,
            target_label=cfg.target_label,
            post_trigger_transform=cfg.normalize_transform,
            poison_fraction=cfg.poison_fraction,
            seed=round_seed,
            poison_exclude_target=True,
        )
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=getattr(self.trainloader, "num_workers", 0),
        )

        original_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(epochs=epochs, round_idx=round_idx)
        finally:
            self.trainloader = original_loader

        result.is_malicious = True
        result.metadata.update({
            "attack": "iba",
            "target_label": cfg.target_label,
            "poison_fraction": cfg.poison_fraction,
            "num_poisoned": len(poisoned_dataset.poisoned_indices),
            "round_seed": round_seed,
        })

        logger.info(
            "IBA client [%d] — round %d: poisoned %d / %d samples (target=%d).",
            self.id, round_idx,
            len(poisoned_dataset.poisoned_indices), len(poisoned_dataset),
            cfg.target_label,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trigger_dataloader(self) -> Optional[DataLoader]:
        """Sample a small subset for generator fine-tuning.

        Returns:
            DataLoader over at most ``config.trigger_sample_size`` local
            (pre-norm) samples, or ``None`` if the local dataset is empty.
        """
        base_dataset = self.trainloader.dataset
        num_samples  = len(base_dataset)
        if num_samples == 0:
            return None

        k   = min(self.config.trigger_sample_size, num_samples)
        rng = np.random.RandomState(self.config.seed)
        idx = rng.choice(num_samples, size=k, replace=False).tolist()

        batch_size = min(getattr(self.trainloader, "batch_size", 32), k)
        return DataLoader(Subset(base_dataset, idx), batch_size=batch_size, shuffle=True)
