"""Experiment 1 — α Distribution Per Attack.

Measures per-sample cosine alignment of triggered inputs' bias gradients
with the mean gradient direction, for each attack type and each layer.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.backdoor import BackdoorDataset
from experiment.config import ExperimentConfig
from experiment.utils import build_adapter
from models import ModelConfig, get_model

from .attack_trigger import get_attack_checkpoint, get_trigger_fn, load_model
from .per_sample_grads import (
    compute_alpha,
    get_bias_bearing_layers,
    get_weight_bearing_layers,
    per_sample_bias_gradients,
    per_sample_weight_gradients,
    sanity_check_gradients,
)


# ---------------------------------------------------------------------------
# Per-batch α computation
# ---------------------------------------------------------------------------

def compute_alpha_for_batch(
    model: nn.Module,
    triggered_norm: torch.Tensor,     # (N, C, H, W) normalised triggered inputs
    target_label: int,
    device: torch.device,
    bias_layer_names: List[str],
    weight_layer_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute bias α and weight α for one batch of triggered samples.

    Returns:
        (alpha_bias, alpha_weight) — each a dict {layer_name: (N,) cosine sims}.
    """
    model.eval()
    with torch.enable_grad():
        g_bias = per_sample_bias_gradients(model, triggered_norm, target_label, device)
        g_weight = per_sample_weight_gradients(model, triggered_norm, target_label, device)

    alpha_bias = compute_alpha(g_bias)
    alpha_weight = compute_alpha(g_weight)
    return alpha_bias, alpha_weight


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_alpha_experiment(
    attack_name: str,
    config: ExperimentConfig,
    checkpoint_path: Optional[str],
    n_batches: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    results_dir: str = "results",
    dataset: str = "gtsrb",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Experiment 1 for one attack type.

    Loads model + trigger, applies trigger to clean test images, computes
    per-sample α for bias and weight gradients at every bias-bearing layer.

    Args:
        attack_name:     One of "neurotoxin", "a3fl", "iba", "chameleon".
        config:          Experiment config (dataset / model / attack params).
        checkpoint_path: Path to the model .pt file; None → auto-detect.
        n_batches:       Number of batches to process.
        batch_size:      Samples per batch.
        device:          Torch device.
        output_dir:      Directory where CSV and PNG files are written.
        results_dir:     Root results directory (for fallback checkpoint lookup).

    Returns:
        (df_raw, df_summary) DataFrames (also written to disk).
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1 (α) — attack: {attack_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset & adapter --------------------------------------------------
    adapter = build_adapter(config)

    # ---- Model --------------------------------------------------------------
    model_cfg = ModelConfig.from_adapter(config.model, adapter)

    if checkpoint_path is None:
        checkpoint_path = get_attack_checkpoint(attack_name, results_dir, dataset=dataset)

    model = load_model(model_cfg, checkpoint_path, device)
    model.eval()

    # ---- Trigger ------------------------------------------------------------
    trigger_fn = get_trigger_fn(attack_name, config, results_dir=results_dir)

    # ---- Layer info ---------------------------------------------------------
    bias_layers = get_bias_bearing_layers(model)
    weight_layers = get_weight_bearing_layers(model)
    bias_layer_map = {name: idx for idx, name, _ in bias_layers}
    weight_layer_map = {name: idx for idx, name, _ in weight_layers}

    print(f"  Bias-bearing layers ({len(bias_layers)}): "
          f"{[name for _, name, _ in bias_layers]}")

    # ---- Sanity check (on one batch) ----------------------------------------
    print("\n  Running sanity check...")
    pre_loader_one = DataLoader(
        adapter.test_pre_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    raw_batch, _ = next(iter(pre_loader_one))
    triggered_batch = torch.stack([trigger_fn(img) for img in raw_batch]).to(device)
    norm_fn = adapter.normalize_transform
    C_chan = triggered_batch.shape[1]
    mean_t = torch.tensor(norm_fn.mean, dtype=torch.float32).view(1, C_chan, 1, 1).to(device)
    std_t = torch.tensor(norm_fn.std, dtype=torch.float32).view(1, C_chan, 1, 1).to(device)
    triggered_norm_check = (triggered_batch - mean_t) / std_t
    passed = sanity_check_gradients(
        model, triggered_norm_check, config.attack.target_label, device
    )
    print(f"  Sanity check: {'ALL PASS' if passed else 'SOME FAIL'}")

    # ---- Build triggered test loader ----------------------------------------
    target_label = config.attack.target_label
    triggered_dataset = BackdoorDataset(
        original_dataset=adapter.test_pre_dataset,
        trigger_fn=trigger_fn,
        target_label=target_label,
        post_trigger_transform=adapter.normalize_transform,
        poison_fraction=1.0,        # trigger every sample
        seed=42,
        poison_exclude_target=True,  # skip target-class samples (correct ASR denom)
    )
    triggered_loader = DataLoader(
        triggered_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ---- Iterate over batches -----------------------------------------------
    raw_records: List[dict] = []
    bias_accumulator: Dict[str, List[torch.Tensor]] = {
        name: [] for _, name, _ in bias_layers
    }
    weight_accumulator: Dict[str, List[torch.Tensor]] = {
        name: [] for _, name, _ in bias_layers  # same layers for comparison
    }

    loader_iter = iter(triggered_loader)
    for batch_idx in range(n_batches):
        try:
            x_norm, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(triggered_loader)
            x_norm, _ = next(loader_iter)

        x_norm = x_norm.to(device)
        N = x_norm.shape[0]

        with torch.enable_grad():
            g_bias_raw = per_sample_bias_gradients(
                model, x_norm, target_label, device
            )
            g_weight_raw = per_sample_weight_gradients(
                model, x_norm, target_label, device
            )

        alpha_bias = compute_alpha(g_bias_raw)
        alpha_weight = compute_alpha(g_weight_raw)

        for layer_name in bias_layer_map:
            a_b = alpha_bias.get(layer_name)
            a_w = alpha_weight.get(layer_name)
            if a_b is None:
                continue

            # Accumulate for summary stats
            bias_accumulator[layer_name].append(a_b)
            if a_w is not None:
                weight_accumulator[layer_name].append(a_w)

            # Raw records
            for sample_idx in range(N):
                raw_records.append({
                    "layer_name": layer_name,
                    "layer_idx": bias_layer_map[layer_name],
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "alpha_bias": a_b[sample_idx].item(),
                    "alpha_weight": a_w[sample_idx].item() if a_w is not None else float("nan"),
                })

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{n_batches} batches")

    # ---- Build DataFrames ---------------------------------------------------
    df_raw = pd.DataFrame(raw_records)

    summary_records = []
    for idx, layer_name, _ in bias_layers:
        ab_all = bias_accumulator[layer_name]
        aw_all = weight_accumulator[layer_name]

        if not ab_all:
            continue

        ab = torch.cat(ab_all)          # (N_total,)
        N_total = len(ab)

        mean_ab = ab.mean().item()
        std_ab = ab.std().item()
        mean_sq_ab = (ab ** 2).mean().item()
        eff_rank_b = (
            (N_total - 1) / (1.0 + (N_total - 2) * mean_sq_ab)
            if N_total > 2 and mean_sq_ab > 1e-12 else float("nan")
        )

        mean_aw = float("nan")
        eff_rank_w = float("nan")
        if aw_all:
            aw = torch.cat(aw_all)
            mean_aw = (aw ** 2).mean().item()
            eff_rank_w = (
                (N_total - 1) / (1.0 + (N_total - 2) * mean_aw)
                if N_total > 2 and mean_aw > 1e-12 else float("nan")
            )

        summary_records.append({
            "layer_name": layer_name,
            "layer_idx": idx,
            "mean_alpha_bias": mean_ab,
            "std_alpha_bias": std_ab,
            "mean_alpha_sq_bias": mean_sq_ab,
            "eff_rank_bound_bias": eff_rank_b,
            "mean_alpha_weight": mean_aw,
            "eff_rank_bound_weight": eff_rank_w,
        })

    df_summary = pd.DataFrame(summary_records)

    # ---- Save ---------------------------------------------------------------
    raw_path = os.path.join(output_dir, f"alpha_raw_{attack_name}.csv")
    summary_path = os.path.join(output_dir, f"alpha_summary_{attack_name}.csv")
    df_raw.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"  Saved: {raw_path}")
    print(f"  Saved: {summary_path}")

    return df_raw, df_summary
