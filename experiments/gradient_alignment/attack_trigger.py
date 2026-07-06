"""Minimal trigger application for each attack type.

Does NOT reimplement trigger logic — imports existing trigger classes
and replicates only the minimal apply() step needed for a standalone batch.

Trigger function contract:
    trigger_fn(image: Tensor[C,H,W] in [0,1]) -> Tensor[C,H,W] in [0,1]
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from experiment.config import ExperimentConfig
from models import ModelConfig, get_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_cfg: ModelConfig,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> nn.Module:
    """Load a model from a checkpoint file (state dict).

    Falls back to a randomly-initialised model when the checkpoint is absent.
    """
    model = get_model(model_cfg).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  WARNING: checkpoint not found ({checkpoint_path}); using random init.")
    model.eval()
    return model


def find_checkpoint(results_dir: str, run_name: str) -> Optional[str]:
    """Return path to final_model.pt for a given run, or None."""
    path = os.path.join(results_dir, run_name, "final_model.pt")
    return path if os.path.exists(path) else None


def find_trigger_state(results_dir: str, run_name: str) -> Optional[str]:
    """Return path to trigger.pt for a given run, or None."""
    path = os.path.join(results_dir, run_name, "trigger.pt")
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Trigger factories — one per attack type
# ---------------------------------------------------------------------------

def build_neurotoxin_trigger(config: ExperimentConfig) -> Callable:
    """Neurotoxin uses a static PatchTrigger — no training required."""
    from attacks.triggers import get_trigger

    C, H, W = config.attack.trigger_kwargs.get("in_channels", 3), None, None
    C_real, H_real, W_real = _infer_shape(config)
    trigger_kw = {"in_channels": C_real, "image_size": (H_real, W_real)}
    trigger_kw.update(config.attack.trigger_kwargs)
    trigger = get_trigger("patch", **trigger_kw)
    return trigger.apply


def build_a3fl_trigger(config: ExperimentConfig, results_dir: str = "results") -> Callable:
    """A3FL — loads the trained pattern from trigger.pt when available."""
    from attacks.triggers import get_trigger

    C, H, W = _infer_shape(config)
    trigger_kw = {"in_channels": C, "image_size": (H, W)}
    trigger_kw.update(config.attack.trigger_kwargs)
    trigger = get_trigger("a3fl", **trigger_kw)

    ckpt_map = _ATTACK_CHECKPOINT_MAPS.get(config.dataset.lower(), ATTACK_CHECKPOINT_MAP)
    run_name = ckpt_map.get("a3fl")
    trigger_path = find_trigger_state(results_dir, run_name) if run_name else None
    if trigger_path:
        state = torch.load(trigger_path, map_location="cpu", weights_only=True)
        trigger.pattern = state["pattern"]
        print(f"  Loaded trained A3FL pattern from {trigger_path}")
    else:
        print("  WARNING: no trained A3FL trigger found; using default 0.5 pattern.")

    return trigger.apply


def build_iba_trigger(config: ExperimentConfig, results_dir: str = "results") -> Callable:
    """IBA — loads the trained U-Net generator from trigger.pt when available."""
    from attacks.triggers import get_trigger

    C, H, W = _infer_shape(config)
    trigger_kw = {
        "in_channels": C,
        "image_size": (H, W),
        "normalize_transform": None,  # apply() works without normalisation
    }
    trigger_kw.update(config.attack.trigger_kwargs)
    trigger = get_trigger("iba", **trigger_kw)

    ckpt_map = _ATTACK_CHECKPOINT_MAPS.get(config.dataset.lower(), ATTACK_CHECKPOINT_MAP)
    run_name = ckpt_map.get("iba")
    trigger_path = find_trigger_state(results_dir, run_name) if run_name else None
    if trigger_path:
        state = torch.load(trigger_path, map_location="cpu", weights_only=True)
        trigger.generator.load_state_dict(state["unet_state_dict"])
        print(f"  Loaded trained IBA generator from {trigger_path}")
    else:
        print("  WARNING: no trained IBA generator found; using random init.")

    trigger.generator.eval()
    return trigger.apply


def build_chameleon_trigger(config: ExperimentConfig, epsilon: float = 0.3) -> Callable:
    """Chameleon has no fixed trigger; we use a fixed checkerboard perturbation
    of magnitude epsilon as a structural proxy.

    The pattern is deterministic (seed=0) so results are reproducible.
    """
    C, H, W = _infer_shape(config)
    torch.manual_seed(0)
    # Checkerboard pattern: +ε on even pixels, −ε on odd pixels
    delta = torch.zeros(C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                delta[c, i, j] = epsilon if (i + j) % 2 == 0 else -epsilon

    def apply(image: torch.Tensor) -> torch.Tensor:
        return (image + delta.to(image.device)).clamp(0.0, 1.0)

    return apply


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Per-dataset maps from attack name → FL run name (directory under results/).
# Extend this dict when adding a new dataset.
_ATTACK_CHECKPOINT_MAPS: dict = {
    "gtsrb": {
        "neurotoxin": "gtsrb_fedavg_neurotoxin",
        "a3fl":       "gtsrb_fedavg_a3fl",
        "iba":        "gtsrb_fedavg_iba",
        "chameleon":  "gtsrb_fedavg_chameleon",
    },
    "cifar10": {
        "neurotoxin": "cifar10_fedavg_neurotoxin",
        "a3fl":       "cifar10_fedavg_a3fl",
        "iba":        "cifar10_fedavg_iba",
        "chameleon":  "cifar10_fedavg_chameleon",
    },
    "femnist": {
        "neurotoxin": "femnist_fedavg_neurotoxin",
        "a3fl":       "femnist_fedavg_a3fl",
        "iba":        "femnist_fedavg_iba",
        "chameleon":  "femnist_fedavg_chameleon",
    },
}

_BENIGN_RUNS: dict = {
    "gtsrb":   "gtsrb_benign_iid",
    "cifar10": "cifar10_benign_iid",
    "femnist": "femnist_benign_iid",
}

# Kept for backward compatibility (existing code that imports these names).
ATTACK_CHECKPOINT_MAP = _ATTACK_CHECKPOINT_MAPS["gtsrb"]
BENIGN_RUN            = _BENIGN_RUNS["gtsrb"]


def get_trigger_fn(
    attack_name: str,
    config: ExperimentConfig,
    results_dir: str = "results",
) -> Callable:
    """Return the trigger_fn callable for ``attack_name``.

    Args:
        attack_name:  One of "neurotoxin", "a3fl", "iba", "chameleon".
        config:       Experiment config (used for shape/hyperparams).
        results_dir:  Root results directory; used to locate trigger.pt for
                      learnable triggers (A3FL, IBA).

    Returns:
        Callable ``(image: Tensor[C,H,W]) -> Tensor[C,H,W]``, both in [0,1].
    """
    if attack_name not in ("neurotoxin", "a3fl", "iba", "chameleon"):
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            "Choose from ['neurotoxin', 'a3fl', 'iba', 'chameleon']."
        )
    if attack_name == "neurotoxin":
        return build_neurotoxin_trigger(config)
    if attack_name == "chameleon":
        return build_chameleon_trigger(config, epsilon=config.attack.trigger_kwargs.get("epsilon", 0.3))
    if attack_name == "a3fl":
        return build_a3fl_trigger(config, results_dir=results_dir)
    if attack_name == "iba":
        return build_iba_trigger(config, results_dir=results_dir)


def get_attack_checkpoint(
    attack_name: str,
    results_dir: str = "results",
    dataset: str = "gtsrb",
) -> Optional[str]:
    """Return the path to the attack's final_model.pt, or the benign
    checkpoint when the attack-specific one is missing.

    Args:
        attack_name:  One of "neurotoxin", "a3fl", "iba", "chameleon".
        results_dir:  Root results directory.
        dataset:      Dataset key ("gtsrb", "cifar10", …).  Controls which
                      run-name map is used for checkpoint lookup.
    """
    ckpt_map  = _ATTACK_CHECKPOINT_MAPS.get(dataset.lower(), ATTACK_CHECKPOINT_MAP)
    benign_run = _BENIGN_RUNS.get(dataset.lower(), BENIGN_RUN)

    run_name = ckpt_map.get(attack_name)
    if run_name:
        path = find_checkpoint(results_dir, run_name)
        if path:
            return path
    # Fall back to benign checkpoint for this dataset
    fallback = find_checkpoint(results_dir, benign_run)
    if fallback:
        print(f"  Falling back to benign checkpoint for attack '{attack_name}'.")
    return fallback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_shape(config: ExperimentConfig) -> Tuple[int, int, int]:
    """Read the dataset input shape from the config without building an adapter."""
    shape_map = {
        "gtsrb":        (3, 32, 32),
        "cifar10":      (3, 32, 32),
        "mnist":        (1, 28, 28),
        "femnist":      (1, 28, 28),
        "femnist_leaf": (1, 28, 28),
        "tiny_imagenet": (3, 64, 64),
    }
    key = config.dataset.lower()
    if key not in shape_map:
        print(f"  WARNING: unknown dataset '{key}' in _infer_shape; defaulting to (3,32,32).")
    return shape_map.get(key, (3, 32, 32))
