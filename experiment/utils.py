"""Experiment utility functions.

Covers: reproducibility seeding, device resolution, dataset adapter
construction, client / server factory, and defense detection scoring.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
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
    from datasets.femnist import FEMNISTDataset
    from datasets.gtsrb import GTSRBDataset
    from datasets.mnist import MNISTDataset
    from datasets.tiny_imagenet import TinyImageNetDataset

    from datasets.leaf_femnist import LEAFFEMNISTDataset

    registry = {
        "cifar10":        lambda: CIFAR10Dataset(root=config.data_root, download=True),
        "femnist":        lambda: FEMNISTDataset(root=config.data_root, download=True),
        "femnist_leaf":   lambda: LEAFFEMNISTDataset(root=config.data_root),
        "gtsrb":          lambda: GTSRBDataset(root=config.data_root, download=True),
        "mnist":          lambda: MNISTDataset(root=config.data_root, download=True),
        "tiny_imagenet":  lambda: TinyImageNetDataset(root=config.data_root, download=True),
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

    # ---- IBA / A3FL: build ONE shared trigger for all malicious clients ----
    # For IBA: all clients fine-tune the same U-Net generator sequentially each
    # round.  For A3FL: all clients optimise the same pattern so ASR evaluation
    # always reflects the trigger that was actually trained this round.
    # The runner's _build_backdoor_loader obtains the trigger from the first
    # malicious client (BackdoorDataset calls trigger.apply on-the-fly).
    iba_shared_trigger = None
    a3fl_shared_trigger = None
    if atk.attack_type == "iba":
        from attacks.iba_client import IBAClient, IBAConfig
        iba_trigger_kw = dict(trigger_kw)
        iba_trigger_kw["normalize_transform"] = adapter.normalize_transform
        iba_shared_trigger = get_trigger("iba", **iba_trigger_kw)
    elif atk.attack_type == "a3fl":
        a3fl_trigger_kw = dict(trigger_kw)
        a3fl_trigger_kw["normalize_transform"] = adapter.normalize_transform
        a3fl_shared_trigger = get_trigger("a3fl", **a3fl_trigger_kw)

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
                trigger = a3fl_shared_trigger
                a3fl_cfg = A3FLConfig(
                    trigger=trigger,
                    target_label=atk.target_label,
                    normalize_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    attack_start_round=atk.attack_start_round,
                    attack_end_round=(
                        float("inf") if atk.attack_end_round is None
                        else atk.attack_end_round
                    ),
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

            elif atk.attack_type == "neurotoxin":
                from attacks.neurotoxin_client import NeurotoxinClient, NeurotoxinConfig
                trigger = get_trigger("patch", **trigger_kw)
                nt_cfg = NeurotoxinConfig(
                    trigger=trigger,
                    target_label=atk.target_label,
                    normalize_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    attack_start_round=atk.attack_start_round,
                    attack_end_round=(
                        float("inf") if atk.attack_end_round is None
                        else atk.attack_end_round
                    ),
                    mask_k_percent=atk.trigger_kwargs.get("mask_k_percent", 0.95),
                    seed=cfg.seed + cid,
                )
                clients[cid] = NeurotoxinClient(
                    config=nt_cfg,
                    id=cid,
                    trainloader=pre_loaders[cid],
                    testloader=None,
                    model=model,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    epochs=cfg.local_epochs,
                    device=device,
                )

            elif atk.attack_type == "iba":
                iba_cfg = IBAConfig(
                    trigger=iba_shared_trigger,
                    target_label=atk.target_label,
                    normalize_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    attack_start_round=atk.attack_start_round,
                    attack_end_round=(
                        float("inf") if atk.attack_end_round is None
                        else atk.attack_end_round
                    ),
                    trigger_sample_size=atk.trigger_kwargs.get("trigger_sample_size", 512),
                    seed=cfg.seed + cid,
                )
                clients[cid] = IBAClient(
                    config=iba_cfg,
                    id=cid,
                    trainloader=pre_loaders[cid],
                    testloader=None,
                    model=model,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    epochs=cfg.local_epochs,
                    device=device,
                )

            elif atk.attack_type == "chameleon":
                from attacks.chameleon_client import ChameleonClient, ChameleonConfig
                ch_cfg = ChameleonConfig(
                    target_label=atk.target_label,
                    normalize_transform=adapter.normalize_transform,
                    poison_fraction=atk.poison_fraction,
                    attack_start_round=atk.attack_start_round,
                    attack_end_round=(
                        float("inf") if atk.attack_end_round is None
                        else atk.attack_end_round
                    ),
                    epsilon=atk.trigger_kwargs.get("epsilon", 0.3),
                    lambda_sim=atk.trigger_kwargs.get("lambda_sim", 1.0),
                    num_pgd_steps=atk.trigger_kwargs.get("num_pgd_steps", 100),
                    pgd_lr=atk.trigger_kwargs.get("pgd_lr", 0.01),
                    peer_pool_size=atk.trigger_kwargs.get("peer_pool_size", 128),
                    seed=cfg.seed + cid,
                )
                clients[cid] = ChameleonClient(
                    config=ch_cfg,
                    id=cid,
                    trainloader=pre_loaders[cid],
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
                    "Valid options: 'none', 'patch', 'a3fl', 'neurotoxin', 'chameleon', 'iba'."
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
        client_scores:  Optional per-client anomaly score (the actual decision
                        variable for score-based defenses, e.g. Krum score or
                        bias-distance).  Used for AUPRC / PR curves.  ``None``
                        for hard-decision defenses (FLAME, DeepSight) or when
                        the defense does not expose a score.
    """
    rejected_ids: FrozenSet[int]
    true_malicious: FrozenSet[int]
    client_scores: Optional[Dict[int, float]] = field(default=None)

    def raw_counts(self, n_selected_benign: int) -> Tuple[int, int, int, int]:
        """Return (TP, FP, TN, FN) for this round.

        TP/FN are -1 when no malicious clients were selected (undefined).
        FP/TN are always defined when a detection defense is active.

        Args:
            n_selected_benign: Benign clients sampled this round.

        Returns:
            ``(tp, fp, tn, fn)`` with -1 as "not applicable" sentinel.
        """
        fp = len(self.rejected_ids - self.true_malicious)
        tn = n_selected_benign - fp
        if self.true_malicious:
            tp = len(self.rejected_ids & self.true_malicious)
            fn = len(self.true_malicious) - tp
        else:
            tp = fn = -1
        return tp, fp, tn, fn

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

        Returns:
            FPR in ``[0, 1]``, or ``NaN`` when no benign clients were selected.
        """
        if n_selected_benign == 0:
            return math.nan
        fp = len(self.rejected_ids - self.true_malicious)
        return fp / n_selected_benign


# ---------------------------------------------------------------------------
# Client atypicality
# ---------------------------------------------------------------------------

def compute_client_atypicality(
    adapter,
    num_clients: int,
    strategy: str,
    seed: int,
    alpha: float = 0.5,
) -> Dict[int, float]:
    """Return per-client Total-Variation distance from the global label distribution.

    Atypicality = 0 for IID clients; rises toward 1 as the client's label
    distribution diverges from the global empirical distribution.

    Under IID partitioning every client gets a near-uniform slice, so TV
    distance ≈ 0 for all.  Under Dirichlet, minority clients can reach ≈ 0.9.

    Args:
        adapter:     A set-up :class:`~datasets.adapter.DatasetAdapter`.
        num_clients: Total number of clients.
        strategy:    Partition strategy (``"iid"`` or ``"dirichlet"``).
        seed:        Partition seed (must match the seed used for the run).
        alpha:       Dirichlet concentration (ignored for IID).

    Returns:
        ``{client_id: tv_distance}`` for every client 0 … num_clients-1.
        Returns an empty dict if the partition cannot be computed.
    """
    import numpy as np
    from datasets.utils import extract_labels

    try:
        kw: dict = {"alpha": alpha} if strategy == "dirichlet" else {}
        partitions = adapter._make_partitions(num_clients, strategy, seed, **kw)
        all_labels = extract_labels(adapter.train_dataset)

        num_classes = int(np.max(all_labels)) + 1
        global_counts = np.bincount(all_labels, minlength=num_classes).astype(float)
        global_dist   = global_counts / global_counts.sum()

        atypicality: Dict[int, float] = {}
        for cid, indices in partitions.items():
            if not indices:
                atypicality[cid] = float("nan")
                continue
            client_labels  = all_labels[indices]
            client_counts  = np.bincount(client_labels, minlength=num_classes).astype(float)
            client_dist    = client_counts / client_counts.sum()
            tv_dist        = 0.5 * float(np.abs(client_dist - global_dist).sum())
            atypicality[cid] = tv_dist
        return atypicality
    except Exception as exc:
        logger.warning("compute_client_atypicality failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Defense server factory
# ---------------------------------------------------------------------------

def build_server(config, model, device, adapter=None):
    """Construct the defense server specified in *config*.

    Args:
        config:  :class:`~experiment.config.ExperimentConfig`.
        model:   Global model (already on *device*).
        device:  Torch device.
        adapter: Optional dataset adapter — used to inject ``input_shape``
                 for defenses that need it (e.g. DeepSight).

    Returns:
        A server instance compatible with :class:`~fl.server.FedAvgAggregator`.

    Raises:
        ValueError: if ``config.defense.defense_type`` is not registered.
    """
    from fl.server import FedAvgAggregator

    defense_type = config.defense.defense_type.lower()
    kwargs = dict(config.defense.defense_kwargs)  # copy so we can mutate safely

    if defense_type == "none":
        return FedAvgAggregator(model=model, device=device)

    if defense_type == "mkrum":
        from defenses.mkrum import MKrumServer
        return MKrumServer(model=model, device=device, **kwargs)

    if defense_type == "deepsight":
        from defenses.deepsight import DeepSightServer
        if "input_shape" not in kwargs and adapter is not None:
            C, H, W = adapter.input_shape
            kwargs["input_shape"] = (C, H, W)
        return DeepSightServer(model=model, device=device, **kwargs)

    if defense_type == "flame":
        from defenses.flame import FlameServer
        return FlameServer(model=model, device=device, **kwargs)

    if defense_type == "nnm":
        from defenses.nnm import NNMServer
        return NNMServer(model=model, device=device, **kwargs)

    if defense_type == "toposentinel":
        from defenses.toposentinel import TopoSentinelServer
        return TopoSentinelServer(model=model, device=device, **kwargs)

    raise ValueError(
        f"Unknown defense '{defense_type}'. "
        f"Available: 'none', 'mkrum', 'deepsight', 'flame', 'nnm', 'toposentinel'."
    )