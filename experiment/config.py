"""Experiment configuration dataclasses.

Every aspect of an FL run — dataset, federation, model, attack, defense,
logging — is captured in a single :class:`ExperimentConfig` object so that
experiments are fully reproducible and serialisable to JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import json
import os


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    """Backdoor attack parameters.

    Args:
        attack_type:         ``"none"``, ``"patch"``, or ``"a3fl"``.
        num_malicious:       Number of permanently malicious clients.
        target_label:        Backdoor target class index.
        poison_fraction:     Fraction of each malicious client's local data
                             to poison per round.
        attack_start_round:  First round in which attack clients activate.
        trigger_kwargs:      Forwarded verbatim to
                             :func:`~attacks.triggers.get_trigger`.
        trigger_sample_size: (A3FL) local samples used for trigger optimisation.
    """
    attack_type: str = "none"
    num_malicious: int = 0
    target_label: int = 0
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    trigger_kwargs: Dict[str, Any] = field(default_factory=dict)
    trigger_sample_size: int = 512


@dataclass
class DefenseConfig:
    """Defense parameters.

    Populated when a defense server is used.  The runner uses
    ``defense_type`` to construct the appropriate server subclass from the
    defenses registry.  ``defense_kwargs`` are forwarded to its constructor.

    Defense types currently planned:
        ``"none"``         — plain FedAvg (no defense).
        ``"flame"``        — FLAME.
        ``"foolsgold"``    — FoolsGold.
        ``"krum"``         — Krum / Multi-Krum.
        ``"median"``       — coordinate-wise median.
        ``"trimmed_mean"`` — trimmed mean.
    """
    defense_type: str = "none"
    defense_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Complete specification for one FL experiment.

    Args:
        name:              Human-readable run identifier (used as the output
                           sub-directory name).
        dataset:           Dataset key — ``"cifar10"`` or ``"synthetic"``.
        data_root:         Root path for dataset downloads / caching.
        partition:         Data partition strategy — ``"iid"`` or
                           ``"dirichlet"``.
        dirichlet_alpha:   Concentration for Dirichlet partition (lower →
                           more heterogeneous).
        batch_size:        Mini-batch size for all DataLoaders.
        num_clients:       Total number of federation clients.
        num_rounds:        Total FL communication rounds.
        clients_per_round: Clients sampled uniformly each round.
        local_epochs:      Local SGD epochs per round.
        model:             Model registry key — ``"resnet18"``,
                           ``"simple_cnn"``, etc.
        lr:                SGD learning rate.
        weight_decay:      SGD weight decay (L2 regularisation).
        attack:            :class:`AttackConfig` instance.
        defense:           :class:`DefenseConfig` instance.
        eval_every:        Evaluate the global model every N rounds.
                           The final round is always evaluated.
        output_dir:        Root directory for results.  Experiment outputs
                           go into ``<output_dir>/<name>/``.
        device:            ``"auto"`` selects GPU if available, else CPU.
                           Pass ``"cpu"`` or ``"cuda"`` to force.
        seed:              Master RNG seed (controls data split, client
                           selection, and attack-client assignment).
    """
    name: str

    # Dataset
    dataset: str = "cifar10"
    data_root: str = "data"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    batch_size: int = 64

    # Federation
    num_clients: int = 100
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5

    # Model
    model: str = "resnet18"

    # Optimiser
    lr: float = 0.01
    weight_decay: float = 1e-4

    # Attack / Defense
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)

    # Evaluation & output
    eval_every: int = 1
    output_dir: str = "results"
    device: str = "auto"
    seed: int = 42

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write this config to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Restore a config from a JSON file produced by :meth:`save`."""
        with open(path) as f:
            data = json.load(f)
        data["attack"] = AttackConfig(**data.get("attack", {}))
        data["defense"] = DefenseConfig(**data.get("defense", {}))
        return cls(**data)