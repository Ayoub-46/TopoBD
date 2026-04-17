"""Experiment utility functions.

Covers: reproducibility seeding, device resolution, dataset adapter
construction, client / server factory, and defense detection scoring.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all relevant RNG seeds for full reproducibility.

    Covers Python ``random``, NumPy, PyTorch CPU, and all CUDA devices.
    Also forces deterministic cuDNN operations (small performance cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device_str: ``"auto"`` selects GPU if available, else CPU.
                    ``"cpu"`` or ``"cuda"`` / ``"cuda:N"`` force a device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Dataset adapter factory
# ---------------------------------------------------------------------------

def build_adapter(config):
    """Construct and set up the dataset adapter specified in *config*.

    Args:
        config: :class:`~experiment.config.ExperimentConfig`.

    Returns:
        A fully set-up :class:`~datasets.adapter.DatasetAdapter`.

    Raises:
        ValueError: if ``config.dataset`` is not registered.
    """
    from datasets.cifar10 import CIFAR10Dataset

    registry = {
        "cifar10": lambda: CIFAR10Dataset(root=config.data_root, download=True),
    }

    key = config.dataset.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. "
            f"Available: {sorted(registry)}"
        )
    adapter = registry[key]()
    adapter.setup()
    return adapter


# ---------------------------------------------------------------------------
# Malicious ID assignment
# ---------------------------------------------------------------------------

def assign_malicious_ids(
    num_clients: int,
    num_malicious: int,
    seed: int,
) -> FrozenSet[int]:
    """Deterministically assign a fixed set of malicious client IDs.

    Stable across runs given the same ``(num_clients, seed)`` pair,
    regardless of how many rounds are run or which clients are sampled.

    Args:
        num_clients:   Total federation size.
        num_malicious: Number of malicious clients (clamped to num_clients).
        seed:          Master RNG seed.

    Returns:
        Immutable set of malicious client IDs.
    """
    rng = random.Random(seed)
    ids = rng.sample(range(num_clients), min(num_malicious, num_clients))
    return frozenset(ids)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_clients(
    config,
    adapter,
    model_cfg,          # pre-built ModelConfig passed from runner
    device: torch.device,
) -> Tuple[FrozenSet[int], Dict[int, object]]:
    """Construct all client objects for the federation.

    Benign clients receive fully-normalised DataLoaders.
    Attack clients receive pre-normalisation DataLoaders so the trigger can
    be applied in ``[0, 1]`` pixel space before ``Normalize``.

    Args:
        config:    :class:`~experiment.config.ExperimentConfig`.
        adapter:   Set-up dataset adapter.
        model_cfg: :class:`~models.base.ModelConfig` — computed once by the
                   runner and shared here to avoid duplicate construction.
        device:    Torch device for all client models.

    Returns:
        ``(malicious_ids, clients)`` — clients is ``{client_id: BenignClient}``.
    """
    # Deferred imports keep the module importable without installing every dep
    from fl.client import BenignClient
    from models import get_model
    from attacks.triggers import get_trigger
    # FIX: correct path — attacks/a3fl_client.py, no clients/ subdirectory
    from attacks.a3fl_client import A3FLClient, A3FLConfig
    from datasets.backdoor import BackdoorDataset

    cfg = config
    atk = config.attack

    # ---- Malicious client assignment ----------------------------------------
    malicious_ids: FrozenSet[int] = (
        assign_malicious_ids(cfg.num_clients, atk.num_malicious, cfg.seed)
        if atk.attack_type != "none" and atk.num_malicious > 0
        else frozenset()
    )

    # ---- Partition strategy kwargs ------------------------------------------
    partition_kw: dict = {}
    if cfg.partition == "dirichlet":
        partition_kw["alpha"] = cfg.dirichlet_alpha

    loader_kw = dict(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        strategy=cfg.partition,
        seed=cfg.seed,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        **partition_kw,
    )

    # ---- DataLoaders --------------------------------------------------------
    full_loaders = adapter.get_client_loaders(**loader_kw)
    pre_loaders  = adapter.get_client_pre_loaders(**loader_kw) if malicious_ids else {}

    # ---- Trigger kwargs: dataset-shape defaults merged with user overrides --
    # These are injected into every get_trigger() call.  Each trigger factory
    # in triggers/__init__.py is tolerant of unknown kwargs (only forwards
    # what its concrete class understands), so static triggers like PatchTrigger
    # safely ignore in_channels / image_size.
    C, H, W = adapter.input_shape
    trigger_kw = {"in_channels": C, "image_size": (H, W)}
    trigger_kw.update(atk.trigger_kwargs)

    # ---- Build clients ------------------------------------------------------
    clients: Dict[int, object] = {}

    for cid in range(cfg.num_clients):
        model = get_model(model_cfg).to(device)

        # ---- Malicious clients ----------------------------------------------
        if cid in malicious_ids:
            if cid not in pre_loaders:
                logger.warning(
                    "Malicious client %d has no data after partitioning — skipped.",
                    cid,
                )
                continue

            if atk.attack_type == "a3fl":
                trigger = get_trigger("a3fl", **trigger_kw)
                a3fl_cfg = A3FLConfig(
                    trigger=trigger,
                    target_label=atk.target_label,
                    normalize_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    attack_start_round=atk.attack_start_round,
                    trigger_sample_size=atk.trigger_sample_size,
                    seed=cfg.seed + cid,
                )
                clients[cid] = A3FLClient(
                    config=a3fl_cfg,
                    id=cid,
                    trainloader=pre_loaders[cid],
                    testloader=None,
                    model=model,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    epochs=cfg.local_epochs,
                    device=device,
                )

            elif atk.attack_type == "patch":
                trigger = get_trigger("patch", **trigger_kw)
                bd_ds = BackdoorDataset(
                    original_dataset=pre_loaders[cid].dataset,
                    trigger_fn=trigger.apply,
                    target_label=atk.target_label,
                    post_trigger_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    seed=cfg.seed + cid,
                )
                poisoned_loader = DataLoader(
                    bd_ds,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=(device.type == "cuda"),
                )
                # Static-trigger attack clients are plain BenignClients whose
                # trainloader has already been poisoned at construction time.
                clients[cid] = BenignClient(
                    id=cid,
                    trainloader=poisoned_loader,
                    testloader=None,
                    model=model,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    epochs=cfg.local_epochs,
                    device=device,
                )

            else:
                raise ValueError(
                    f"Unknown attack type '{atk.attack_type}'. "
                    "Valid options: 'none', 'patch', 'a3fl'."
                )

        # ---- Benign clients -------------------------------------------------
        else:
            if cid not in full_loaders:
                logger.warning(
                    "Benign client %d has no data after partitioning — skipped.",
                    cid,
                )
                continue
            clients[cid] = BenignClient(
                id=cid,
                trainloader=full_loaders[cid],
                testloader=None,
                model=model,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                epochs=cfg.local_epochs,
                device=device,
            )

    logger.info(
        "Built %d clients — %d malicious, %d benign.",
        len(clients),
        len([c for c in clients if c in malicious_ids]),
        len([c for c in clients if c not in malicious_ids]),
    )
    return malicious_ids, clients


# ---------------------------------------------------------------------------
# Defense detection scoring
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Outcome of one round of defense filtering.

    Populated by the defense server's ``filter_updates()`` method and
    consumed by the runner to compute per-round TPR / FPR.

    Args:
        rejected_ids:   Client IDs whose updates were rejected this round.
        true_malicious: Ground-truth set of malicious IDs that were selected
                        this round (passed in by the runner).
    """
    rejected_ids: FrozenSet[int]
    true_malicious: FrozenSet[int]

    @property
    def tpr(self) -> float:
        """True-positive rate: correctly rejected malicious / total malicious
        selected this round.  ``NaN`` when no malicious clients were selected.
        """
        if not self.true_malicious:
            return math.nan
        tp = len(self.rejected_ids & self.true_malicious)
        return tp / len(self.true_malicious)

    def compute_fpr(self, n_selected_benign: int) -> float:
        """False-positive rate: incorrectly rejected benign / total benign
        selected this round.

        Args:
            n_selected_benign: Number of benign clients selected this round.
                               Supplied by the runner, which has the full
                               client roster.

        Returns:
            FPR in ``[0, 1]``, or ``NaN`` when no benign clients were selected.
        """
        if n_selected_benign == 0:
            return math.nan
        fp = len(self.rejected_ids - self.true_malicious)
        return fp / n_selected_benign