"""Extended Lemma 2 experiment — Effective Rank Comparison.

Computes and contrasts four quantities per bias-bearing layer:

  γ_max                  max off-diagonal |cosine sim| of class-mean directions
                         (already computed by exp_gamma; loaded from gamma_raw.csv)

  γ_mean                 mean off-diagonal |cosine sim|
                         (derived from gamma_raw.csv off-diagonal entries)

  eff_rank_clean_direct  empirical effective rank of Σ_clean:
                           Σ_clean = (1/N) Σ_i  g_i g_i^T
                         where g_i are per-sample bias gradients on CLEAN data
                         with each sample's own true label. Computed via
                         eigendecomposition of Σ or (for n_bias > 512) the
                         Gram matrix trick.

  eff_rank_bd_upper_bound  1 / mean_alpha_sq_bias per attack (from
                            alpha_summary_{attack}.csv). Reported as a
                            [min, max] range across attacks.

Outputs
-------
  rank_comparison_table.csv   — one row per layer
  rank_comparison_table.png   — formatted table (highlighted where rank_gap > 0)
  eigenvalue_spectra.png      — sorted eigenvalue curves per layer

Sanity checks (aborts with a clear message if violated)
--------------------------------------------------------
  1. Σ_clean is positive semi-definite (all eigenvalues ≥ −1e-6)
  2. eff_rank_clean ∈ [1, n_bias]
  3. eff_rank_clean >> 1 for FC layers (warns if not)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment.config import ExperimentConfig
from experiment.utils import build_adapter
from models import ModelConfig

from .attack_trigger import get_attack_checkpoint, get_trigger_fn, load_model
from .per_sample_grads import (
    get_bias_bearing_layers,
    per_sample_bias_gradients,
    per_sample_bias_gradients_from_labels,
)


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def _eff_rank(eigenvalues: torch.Tensor) -> float:
    """Participation-ratio effective rank: (Σλ)² / Σλ²."""
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) == 0:
        return 0.0
    s  = pos.sum()
    s2 = (pos ** 2).sum()
    return (s * s / s2).item()


# ---------------------------------------------------------------------------
# Covariance eigenvalues
# ---------------------------------------------------------------------------

def _covariance_eigenvalues(
    grad_batches: List[torch.Tensor],   # list of (n_i, d) CPU float32 tensors
) -> torch.Tensor:
    """Eigenvalues of the empirical gradient covariance matrix.

    Uses the direct (d×d) covariance for n_bias ≤ 512 and the Gram matrix
    (N×N) trick for n_bias > 512, as specified.

    Returns eigenvalues sorted ascending (torch.float64).
    """
    if not grad_batches:
        return torch.zeros(0, dtype=torch.float64)

    N = sum(g.shape[0] for g in grad_batches)
    d = grad_batches[0].shape[1]

    if d <= 512:
        # Incremental (d, d) covariance — O(d²) memory, any N
        cov = torch.zeros(d, d, dtype=torch.float64)
        for g in grad_batches:
            g64 = g.to(torch.float64)
            cov.add_(g64.T @ g64)
        cov /= N
        return torch.linalg.eigvalsh(cov)
    else:
        # Gram matrix trick: store all gradients → (N, N)
        # Equivalent non-zero eigenvalues to (d, d) covariance.
        G = torch.cat(grad_batches, dim=0).to(torch.float64)  # (N, d)
        gram = (G @ G.T) / N                                   # (N, N)
        eigvals = torch.linalg.eigvalsh(gram)
        return eigvals


# ---------------------------------------------------------------------------
# Clean gradient covariance accumulation
# ---------------------------------------------------------------------------

def _accumulate_clean_gradients(
    model: nn.Module,
    loader: DataLoader,
    n_batches: int,
    device: torch.device,
    bias_layer_names: List[str],
) -> Tuple[Dict[str, List[torch.Tensor]], int]:
    """Accumulate per-sample clean bias gradients using true labels.

    Returns:
        (accum, total_n): accum maps layer_name → list of (n_i, d) tensors.
    """
    model.eval()
    accum: Dict[str, List[torch.Tensor]] = {name: [] for name in bias_layer_names}
    total_n = 0
    loader_iter = iter(loader)

    for batch_idx in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x = x.to(device)
        n = x.shape[0]

        with torch.enable_grad():
            g = per_sample_bias_gradients_from_labels(model, x, y, device)

        for name in bias_layer_names:
            if name in g:
                accum[name].append(g[name].cpu())

        total_n += n
        if (batch_idx + 1) % 20 == 0:
            print(f"    clean gradients: {batch_idx + 1}/{n_batches} batches")

    return accum, total_n


# ---------------------------------------------------------------------------
# Backdoor gradient covariance accumulation
# ---------------------------------------------------------------------------

def _accumulate_bd_gradients(
    model: nn.Module,
    trigger_fn,
    adapter,
    target_label: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    bias_layer_names: List[str],
) -> Tuple[Dict[str, List[torch.Tensor]], int]:
    """Accumulate per-sample backdoor bias gradients (triggered inputs, target label)."""
    from datasets.backdoor import BackdoorDataset

    triggered_ds = BackdoorDataset(
        original_dataset=adapter.test_pre_dataset,
        trigger_fn=trigger_fn,
        target_label=target_label,
        post_trigger_transform=adapter.normalize_transform,
        poison_fraction=1.0,
        seed=42,
        poison_exclude_target=True,
    )
    loader = DataLoader(triggered_ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model.eval()
    accum: Dict[str, List[torch.Tensor]] = {name: [] for name in bias_layer_names}
    total_n = 0
    loader_iter = iter(loader)

    for batch_idx in range(n_batches):
        try:
            x_norm, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x_norm, _ = next(loader_iter)

        x_norm = x_norm.to(device)
        n = x_norm.shape[0]

        with torch.enable_grad():
            g = per_sample_bias_gradients(model, x_norm, target_label, device)

        for name in bias_layer_names:
            if name in g:
                accum[name].append(g[name].cpu())

        total_n += n

    return accum, total_n


# ---------------------------------------------------------------------------
# γ_mean from gamma_raw DataFrame
# ---------------------------------------------------------------------------

def _compute_gamma_mean(gamma_raw: pd.DataFrame) -> pd.DataFrame:
    """Return per-layer γ_max and γ_mean from the gamma_raw CSV.

    gamma_raw has columns: layer_name, layer_idx, class_i, class_j,
    cosine_similarity.  Off-diagonal pairs are already stored (i ≠ j).
    """
    rows = []
    for (layer_name, layer_idx), grp in gamma_raw.groupby(
        ["layer_name", "layer_idx"], sort=False
    ):
        sims = grp["cosine_similarity"].values
        rows.append({
            "layer_name": layer_name,
            "layer_idx":  int(layer_idx),
            "gamma_max":  float(sims.max()),
            "gamma_mean": float(sims.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Eigenvalue spectrum plots
# ---------------------------------------------------------------------------

def _plot_eigenvalue_spectra(
    clean_eigvals: Dict[str, torch.Tensor],
    bd_eigvals: Dict[str, Dict[str, torch.Tensor]],   # attack → layer → eigvals
    output_dir: str,
) -> None:
    """One subplot per layer; sorted eigenvalue spectrum on log y-scale."""
    bias_layers = list(clean_eigvals.keys())
    n_layers = len(bias_layers)
    if n_layers == 0:
        return

    cols = min(n_layers, 3)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if n_layers > 1 else [axes]

    attack_colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax_idx, layer_name in enumerate(bias_layers):
        ax = axes_flat[ax_idx]
        c_eig = clean_eigvals[layer_name].numpy()
        c_eig_sorted = np.sort(c_eig[c_eig > 0])[::-1]

        if len(c_eig_sorted):
            ax.semilogy(np.arange(1, len(c_eig_sorted) + 1), c_eig_sorted,
                        color="black", linewidth=2, label="clean")

        for atk_idx, (attack, layer_dict) in enumerate(bd_eigvals.items()):
            if layer_name not in layer_dict:
                continue
            b_eig = layer_dict[layer_name].numpy()
            b_eig_sorted = np.sort(b_eig[b_eig > 0])[::-1]
            if len(b_eig_sorted):
                ax.semilogy(np.arange(1, len(b_eig_sorted) + 1), b_eig_sorted,
                            color=attack_colors[atk_idx % len(attack_colors)],
                            linestyle="--", linewidth=1.5, label=attack)

        ax.set_title(layer_name, fontsize=9)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("eigenvalue (log scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n_layers, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle("Gradient Covariance Eigenvalue Spectra", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "eigenvalue_spectra.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Table figure
# ---------------------------------------------------------------------------

def _plot_rank_table(df: pd.DataFrame, output_dir: str) -> None:
    """Render rank_comparison_table as a matplotlib table figure."""
    display_cols = [
        "layer_name", "gamma_max", "gamma_mean",
        "eff_rank_clean_direct", "eff_rank_bd_min", "eff_rank_bd_max",
        "rank_gap",
    ]
    df_display = df[[c for c in display_cols if c in df.columns]].copy()

    float_cols = [c for c in display_cols if c != "layer_name"]
    for col in float_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )

    col_labels = [c.replace("_", "\n") for c in df_display.columns]
    n_rows, n_cols = df_display.shape

    fig_w = max(10, n_cols * 1.6)
    fig_h = max(2,  n_rows * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_display.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # Highlight rows where rank_gap > 0 (column may not exist if bd data absent)
    if "rank_gap" in df.columns:
        for row_idx, gap in enumerate(df["rank_gap"]):
            if pd.notna(gap) and gap > 0:
                for col_idx in range(n_cols):
                    tbl[row_idx + 1, col_idx].set_facecolor("#d4edda")  # light green

    ax.set_title("Effective Rank Comparison (Extended Lemma 2)", fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "rank_comparison_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_lemma2_experiment(
    config: ExperimentConfig,
    benign_checkpoint: Optional[str],
    attacks: List[str],
    n_batches: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    results_dir: str = "results",
    gamma_raw_df: Optional[pd.DataFrame] = None,
    alpha_summary_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Compute and compare effective ranks across clean and backdoor settings.

    Args:
        config:             Experiment config (dataset / model).
        benign_checkpoint:  Path to the clean model checkpoint.
        attacks:            Attack names to include in the BD upper-bound.
        n_batches:          Batches used for covariance estimation.
        batch_size:         Samples per batch.
        device:             Torch device.
        output_dir:         Where to write outputs.
        results_dir:        Root results directory.
        gamma_raw_df:       Pre-loaded gamma_raw DataFrame; loaded from CSV if None.
        alpha_summary_dfs:  Pre-loaded alpha_summary DataFrames per attack; loaded
                            from CSV if None.

    Returns:
        rank_comparison DataFrame (also saved to disk).
    """
    print(f"\n{'='*60}")
    print("Extended Lemma 2 — Effective Rank Comparison")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load / compute γ statistics ------------------------------------
    if gamma_raw_df is None:
        p = os.path.join(output_dir, "gamma_raw.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"gamma_raw.csv not found at {p}. "
                "Run exp_gamma first (or omit --skip-exp2)."
            )
        gamma_raw_df = pd.read_csv(p)
        print(f"  Loaded gamma_raw from {p}")

    gamma_stats = _compute_gamma_mean(gamma_raw_df)
    print(f"  γ stats computed for {len(gamma_stats)} layers")

    # ---- 2. Load α summaries for BD upper-bound ----------------------------
    if alpha_summary_dfs is None:
        alpha_summary_dfs = {}
    for attack in attacks:
        if attack in alpha_summary_dfs:
            continue
        p = os.path.join(output_dir, f"alpha_summary_{attack}.csv")
        if os.path.exists(p):
            alpha_summary_dfs[attack] = pd.read_csv(p)
            print(f"  Loaded alpha_summary_{attack} from {p}")
        else:
            print(f"  WARNING: alpha_summary_{attack}.csv not found; skipping.")

    # ---- 3. Build model and adapter ----------------------------------------
    adapter = build_adapter(config)
    model_cfg = ModelConfig.from_adapter(config.model, adapter)
    model = load_model(model_cfg, benign_checkpoint, device)
    model.eval()

    bias_layers = get_bias_bearing_layers(model)
    bias_layer_names = [name for _, name, _ in bias_layers]
    layer_idx_map = {name: idx for idx, name, _ in bias_layers}
    print(f"  Bias-bearing layers: {bias_layer_names}")

    # ---- 4. Compute Σ_clean eigenvalues ------------------------------------
    print(f"\n  Computing Σ_clean ({n_batches} batches × {batch_size} samples)...")
    clean_loader = DataLoader(
        adapter.test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    clean_accum, clean_n = _accumulate_clean_gradients(
        model, clean_loader, n_batches, device, bias_layer_names
    )
    print(f"  Σ_clean accumulated ({clean_n} total samples)")

    clean_eigvals: Dict[str, torch.Tensor] = {}
    clean_eff_ranks: Dict[str, float] = {}

    for layer_name in bias_layer_names:
        batches = clean_accum.get(layer_name, [])
        if not batches:
            continue
        eigvals = _covariance_eigenvalues(batches)
        clean_eigvals[layer_name] = eigvals

        # Sanity 1: PSD check
        min_eig = eigvals.min().item()
        assert min_eig >= -1e-6, (
            f"Σ_clean not PSD at layer '{layer_name}': min eigenvalue={min_eig:.2e}"
        )
        pos_eigvals = eigvals[eigvals > 0]

        er = _eff_rank(eigvals)
        n_bias = batches[0].shape[1]

        # Sanity 2: plausible range
        assert 1.0 <= er <= n_bias + 1e-3, (
            f"eff_rank out of bounds at layer '{layer_name}': "
            f"eff_rank={er:.2f}, n_bias={n_bias}"
        )
        clean_eff_ranks[layer_name] = er

        # Sanity 4: warn if FC layer has eff_rank ≈ 1
        is_fc = ("fc" in layer_name.lower() or "linear" in layer_name.lower()
                  or "classifier" in layer_name.lower())
        if is_fc and er < 5.0:
            print(f"  WARNING: FC layer '{layer_name}' has eff_rank={er:.2f} — "
                  f"close to 1. Model may not be fully converged.")

        print(f"    {layer_name:42s}  n_bias={n_bias:4d}  "
              f"eff_rank_clean={er:.2f}  (n_pos_eigvals={len(pos_eigvals)})")

    # Also reprint zero-mean diagnostic from exp_gamma summary
    gamma_summary_path = os.path.join(output_dir, "gamma_summary.csv")
    if os.path.exists(gamma_summary_path):
        gs = pd.read_csv(gamma_summary_path)
        print("\n  Zero-mean residuals (from exp_gamma):")
        for _, row in gs.iterrows():
            print(f"    {row['layer_name']:42s}  residual={row['residual_norm']:.4f}")

    # ---- 5. Compute Σ_bd eigenvalues per attack (for spectrum plot) --------
    bd_eigvals: Dict[str, Dict[str, torch.Tensor]] = {}  # attack → layer → eigvals
    target_label = config.attack.target_label

    for attack in attacks:
        print(f"\n  Computing Σ_bd for '{attack}' ...")
        try:
            trigger_fn = get_trigger_fn(attack, config, results_dir=results_dir)
            bd_accum, bd_n = _accumulate_bd_gradients(
                model, trigger_fn, adapter, target_label,
                batch_size, n_batches, device, bias_layer_names,
            )
            bd_eigvals[attack] = {}
            for layer_name in bias_layer_names:
                batches = bd_accum.get(layer_name, [])
                if not batches:
                    continue
                bd_eigvals[attack][layer_name] = _covariance_eigenvalues(batches)
            print(f"    done ({bd_n} samples)")
        except Exception as exc:
            print(f"  WARNING: Σ_bd for '{attack}' failed: {exc}")

    # ---- 6. BD upper bounds from alpha summaries ---------------------------
    # eff_rank_bd = 1 / mean_alpha_sq_bias per layer per attack
    bd_upper_bounds: Dict[str, Dict[str, float]] = {}  # layer → attack → bound
    for attack, df_sum in alpha_summary_dfs.items():
        for _, row in df_sum.iterrows():
            lname = row["layer_name"]
            msq = row.get("mean_alpha_sq_bias", float("nan"))
            if pd.isna(msq) or msq < 1e-12:
                bound = float("nan")
            else:
                bound = 1.0 / float(msq)
            bd_upper_bounds.setdefault(lname, {})[attack] = bound

    # ---- 7. Assemble rank comparison table ---------------------------------
    gamma_lkp = gamma_stats.set_index("layer_name")
    table_rows = []

    for _, layer_name, _ in bias_layers:
        row: dict = {
            "layer_name":  layer_name,
            "layer_idx":   layer_idx_map[layer_name],
            "gamma_max":   gamma_lkp.loc[layer_name, "gamma_max"] if layer_name in gamma_lkp.index else float("nan"),
            "gamma_mean":  gamma_lkp.loc[layer_name, "gamma_mean"] if layer_name in gamma_lkp.index else float("nan"),
            "eff_rank_clean_direct": clean_eff_ranks.get(layer_name, float("nan")),
        }

        bd_bounds = bd_upper_bounds.get(layer_name, {})
        finite_bounds = [v for v in bd_bounds.values() if not np.isnan(v)]
        row["eff_rank_bd_min"] = min(finite_bounds) if finite_bounds else float("nan")
        row["eff_rank_bd_max"] = max(finite_bounds) if finite_bounds else float("nan")
        row["rank_gap"] = (
            row["eff_rank_clean_direct"] - row["eff_rank_bd_max"]
            if not np.isnan(row["eff_rank_clean_direct"])
               and not np.isnan(row["eff_rank_bd_max"])
            else float("nan")
        )
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)

    # ---- 8. Print summary --------------------------------------------------
    print(f"\n  {'Layer':<42s} {'γ_max':>7} {'γ_mean':>7} "
          f"{'rank_clean':>11} {'rank_bd_max':>11} {'rank_gap':>10}")
    print("  " + "-" * 95)
    for _, r in df_table.iterrows():
        def _f(v): return f"{v:.3f}" if pd.notna(v) else "  —  "
        print(f"  {r['layer_name']:<42s} {_f(r['gamma_max']):>7} {_f(r['gamma_mean']):>7} "
              f"{_f(r['eff_rank_clean_direct']):>11} {_f(r['eff_rank_bd_max']):>11} "
              f"{_f(r['rank_gap']):>10}")

    # ---- 9. Save outputs ---------------------------------------------------
    csv_path = os.path.join(output_dir, "rank_comparison_table.csv")
    df_table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")

    _plot_rank_table(df_table, output_dir)
    _plot_eigenvalue_spectra(clean_eigvals, bd_eigvals, output_dir)

    return df_table
