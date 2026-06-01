"""Experiment 2 — γ Per Layer (Clean Model).

Measures how orthogonal the class-conditional mean bias gradient directions
are for a clean (benign) model. Low γ validates Lemma 2.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiment.config import ExperimentConfig
from experiment.utils import build_adapter
from models import ModelConfig

from .attack_trigger import load_model
from .per_sample_grads import get_bias_bearing_layers, per_sample_bias_gradients


# ---------------------------------------------------------------------------
# Class-stratified data collection
# ---------------------------------------------------------------------------

def collect_class_data(
    adapter,
    n_per_class: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Return a dict mapping class_id → list of dataset indices.

    Uses the normalised test dataset. Caps each class at ``n_per_class``.
    """
    from datasets.utils import extract_labels

    labels = extract_labels(adapter.test_dataset)
    rng = np.random.RandomState(seed)

    class_indices: Dict[int, List[int]] = {}
    num_classes = adapter.num_classes

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            class_indices[c] = []
            continue
        if len(idx) > n_per_class:
            idx = rng.choice(idx, size=n_per_class, replace=False)
        class_indices[c] = idx.tolist()

    return class_indices


# ---------------------------------------------------------------------------
# Mean bias gradient direction per class
# ---------------------------------------------------------------------------

def compute_class_mean_bias_grad(
    model: nn.Module,
    class_indices: List[int],
    adapter,
    batch_size: int,
    class_label: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute mean bias gradient direction for one class.

    Args:
        model:         Eval-mode model on ``device``.
        class_indices: Dataset indices for this class.
        adapter:       Set-up dataset adapter.
        batch_size:    Batch size for gradient computation.
        class_label:   The class label (integer).
        device:        Torch device.

    Returns:
        Dict {layer_name: mean_grad (bias_dim,)} — NOT yet normalised.
    """
    if not class_indices:
        return {}

    subset = Subset(adapter.test_dataset, class_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    accumulator: Dict[str, torch.Tensor] = {}
    n_samples = 0

    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        n = x_batch.shape[0]

        with torch.enable_grad():
            g_bias = per_sample_bias_gradients(model, x_batch, class_label, device)

        for name, grads in g_bias.items():
            # grads: (n, bias_dim)
            batch_sum = grads.sum(dim=0).to(torch.float64)
            if name in accumulator:
                accumulator[name] = accumulator[name] + batch_sum
            else:
                accumulator[name] = batch_sum

        n_samples += n

    # Normalise to mean
    return {name: (g / n_samples).to(torch.float32) for name, g in accumulator.items()}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_gamma_experiment(
    config: ExperimentConfig,
    benign_checkpoint: Optional[str],
    n_per_class: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Experiment 2 — γ per layer for the benign model.

    Args:
        config:             Experiment config.
        benign_checkpoint:  Path to the benign model checkpoint.
        n_per_class:        Number of samples per class (min 100 recommended).
        batch_size:         Batch size for gradient computation.
        device:             Torch device.
        output_dir:         Directory for output CSV files.

    Returns:
        (df_raw, df_summary) DataFrames (also written to disk).
    """
    print(f"\n{'='*60}")
    print("Experiment 2 (γ) — clean model gradient diversity")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset & model ----------------------------------------------------
    adapter = build_adapter(config)
    model_cfg = ModelConfig.from_adapter(config.model, adapter)
    model = load_model(model_cfg, benign_checkpoint, device)
    model.eval()

    num_classes = adapter.num_classes
    bias_layers = get_bias_bearing_layers(model)
    print(f"  Classes: {num_classes}")
    print(f"  Bias layers: {[name for _, name, _ in bias_layers]}")

    # ---- Collect class indices -----------------------------------------------
    print(f"  Collecting up to {n_per_class} samples per class...")
    class_indices = collect_class_data(adapter, n_per_class=n_per_class, seed=42)

    actual_counts = {c: len(idx) for c, idx in class_indices.items()}
    print(f"  Actual counts: min={min(actual_counts.values())}, "
          f"max={max(actual_counts.values())}")

    # ---- Compute class fractions (for zero-mean residual) --------------------
    total_samples = sum(actual_counts.values())
    pi = {c: cnt / total_samples for c, cnt in actual_counts.items()}

    # ---- Compute mean bias gradient direction per class ----------------------
    print("  Computing class mean bias gradient directions...")
    # u[class][layer_name] = unit vector (bias_dim,)
    class_mean_grads: Dict[int, Dict[str, torch.Tensor]] = {}

    for c in range(num_classes):
        if len(class_indices[c]) == 0:
            class_mean_grads[c] = {}
            continue
        mean_grads = compute_class_mean_bias_grad(
            model, class_indices[c], adapter, batch_size, c, device
        )
        class_mean_grads[c] = mean_grads
        if (c + 1) % 10 == 0 or c == num_classes - 1:
            print(f"  Class {c + 1}/{num_classes} done")

    # ---- Compute zero-mean residual per layer --------------------------------
    residuals: Dict[str, float] = {}
    for _, layer_name, _ in bias_layers:
        weighted_sum = None
        for c in range(num_classes):
            if layer_name not in class_mean_grads[c]:
                continue
            g = class_mean_grads[c][layer_name]
            term = pi[c] * g
            if weighted_sum is None:
                weighted_sum = term
            else:
                weighted_sum = weighted_sum + term

        if weighted_sum is not None:
            residuals[layer_name] = weighted_sum.norm(p=2).item()
        else:
            residuals[layer_name] = float("nan")

    print("\n  Zero-mean residuals per layer:")
    for layer_name, res in residuals.items():
        flag = " (large — model may not be converged)" if res > 0.1 else ""
        print(f"    {layer_name:40s}: {res:.4f}{flag}")

    # ---- Compute unit vectors -----------------------------------------------
    unit_vecs: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
    for c in range(num_classes):
        unit_vecs[c] = {}
        for layer_name, mean_g in class_mean_grads[c].items():
            norm = mean_g.norm(p=2)
            if norm < 1e-12:
                unit_vecs[c][layer_name] = None
            else:
                unit_vecs[c][layer_name] = mean_g / norm

    # ---- Compute pairwise cosine similarity matrix per layer ----------------
    print("\n  Computing pairwise cosine similarities...")

    raw_records: List[dict] = []
    summary_records: List[dict] = []

    for layer_idx, layer_name, _ in bias_layers:
        # Build matrix U: (C_valid, bias_dim)
        valid_classes = [
            c for c in range(num_classes)
            if unit_vecs[c].get(layer_name) is not None
        ]
        if len(valid_classes) < 2:
            continue

        U = torch.stack([unit_vecs[c][layer_name] for c in valid_classes])  # (C_v, dim)

        # Full similarity matrix: (C_v, C_v)
        S = (U @ U.T).abs()   # absolute cosine similarity

        # Off-diagonal elements only
        C_v = len(valid_classes)
        off_diag_vals = []
        for i in range(C_v):
            for j in range(C_v):
                if i == j:
                    continue
                ci = valid_classes[i]
                cj = valid_classes[j]
                sim_val = S[i, j].item()
                raw_records.append({
                    "layer_name": layer_name,
                    "layer_idx": layer_idx,
                    "class_i": ci,
                    "class_j": cj,
                    "cosine_similarity": sim_val,
                })
                off_diag_vals.append(sim_val)

        off_diag_t = torch.tensor(off_diag_vals)
        gamma = off_diag_t.max().item()
        mean_sim = off_diag_t.mean().item()
        C_eff = len(valid_classes)
        eff_rank_lb = (
            (C_eff - 1) / (1.0 + (C_eff - 2) * gamma ** 2)
            if C_eff > 2 and gamma < 1.0 else float("nan")
        )

        summary_records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "gamma": gamma,
            "mean_sim": mean_sim,
            "eff_rank_lb": eff_rank_lb,
            "residual_norm": residuals.get(layer_name, float("nan")),
        })

        print(f"    {layer_name:40s}  γ={gamma:.4f}  mean_sim={mean_sim:.4f}  "
              f"eff_rank_lb={eff_rank_lb:.2f}  residual={residuals.get(layer_name, float('nan')):.4f}")

    # ---- Save ---------------------------------------------------------------
    df_raw = pd.DataFrame(raw_records)
    df_summary = pd.DataFrame(summary_records)

    raw_path = os.path.join(output_dir, "gamma_raw.csv")
    summary_path = os.path.join(output_dir, "gamma_summary.csv")
    df_raw.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  Saved: {raw_path}")
    print(f"  Saved: {summary_path}")

    return df_raw, df_summary
