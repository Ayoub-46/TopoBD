"""Plotting functions for the gradient alignment diagnostic experiments."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ATTACK_COLORS = {
    "neurotoxin": "#e74c3c",
    "a3fl":       "#3498db",
    "iba":        "#2ecc71",
    "chameleon":  "#9b59b6",
}
WEIGHT_COLOR = "#95a5a6"
BIAS_LINE_STYLE = "-"
WEIGHT_LINE_STYLE = "--"


# ---------------------------------------------------------------------------
# Helper: KDE plot on an axis
# ---------------------------------------------------------------------------

def _kde_on_ax(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    bw_method: Optional[float] = None,
) -> None:
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return
    kde = gaussian_kde(values, bw_method=bw_method or "scott")
    xs = np.linspace(max(-1.05, values.min() - 0.05), min(1.05, values.max() + 0.05), 300)
    ax.plot(xs, kde(xs), color=color, linestyle=linestyle, label=label, linewidth=1.5)
    ax.fill_between(xs, kde(xs), alpha=0.12, color=color)


# ---------------------------------------------------------------------------
# Figure 1 — α distributions
# ---------------------------------------------------------------------------

def plot_alpha_distributions(
    alpha_raw_per_attack: Dict[str, pd.DataFrame],
    alpha_summary_per_attack: Dict[str, pd.DataFrame],
    output_dir: str,
    every_n_layers: int = 2,
) -> str:
    """Grid of KDE plots: rows=layers (every other), cols=attacks.

    Each cell: overlapping KDE of α_bias (attack colour) and α_weight (grey).
    Annotation: mean α² and effective rank bound.
    """
    # Collect all layer names (use first available attack)
    first_df = next(iter(alpha_raw_per_attack.values()))
    all_layers = (
        first_df[["layer_idx", "layer_name"]]
        .drop_duplicates()
        .sort_values("layer_idx")
    )
    selected = all_layers.iloc[::every_n_layers]
    n_layers = len(selected)

    attacks = list(alpha_raw_per_attack.keys())
    n_attacks = len(attacks)

    fig, axes = plt.subplots(
        n_layers, n_attacks,
        figsize=(4 * n_attacks, 3 * n_layers),
        squeeze=False,
    )
    fig.suptitle("Lemma 1: Per-Sample Bias Gradient Alignment (α) by Attack",
                 fontsize=14, y=1.01)

    for row_idx, (_, layer_row) in enumerate(selected.iterrows()):
        layer_name = layer_row["layer_name"]
        layer_idx = int(layer_row["layer_idx"])

        for col_idx, attack_name in enumerate(attacks):
            ax = axes[row_idx][col_idx]
            df_raw = alpha_raw_per_attack.get(attack_name, pd.DataFrame())
            df_sum = alpha_summary_per_attack.get(attack_name, pd.DataFrame())

            if df_raw.empty:
                ax.set_visible(False)
                continue

            layer_mask = df_raw["layer_name"] == layer_name

            # --- bias KDE ---
            ab = df_raw.loc[layer_mask, "alpha_bias"].dropna().values
            _kde_on_ax(ax, ab, ATTACK_COLORS.get(attack_name, "steelblue"),
                       "bias", linestyle=BIAS_LINE_STYLE)

            # --- weight KDE ---
            aw = df_raw.loc[layer_mask, "alpha_weight"].dropna().values
            _kde_on_ax(ax, aw, WEIGHT_COLOR, "weight", linestyle=WEIGHT_LINE_STYLE)

            # --- annotation ---
            row_sum = df_sum[df_sum["layer_name"] == layer_name]
            if not row_sum.empty:
                msq_b = row_sum["mean_alpha_sq_bias"].values[0]
                erb = row_sum["eff_rank_bound_bias"].values[0]
                ax.text(
                    0.03, 0.97,
                    f"E[α²]={msq_b:.3f}\nRank≤{erb:.1f}",
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )

            ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel("α (cosine alignment)", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)

            if row_idx == 0:
                ax.set_title(attack_name, fontsize=10, fontweight="bold")
            if col_idx == 0:
                short = layer_name.split(".")[-1] if "." in layer_name else layer_name
                ax.set_ylabel(f"L{layer_idx}: {short}\nDensity", fontsize=7)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "alpha_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — γ heatmaps
# ---------------------------------------------------------------------------

def plot_gamma_heatmaps(
    gamma_raw: pd.DataFrame,
    gamma_summary: pd.DataFrame,
    output_dir: str,
    every_n_layers: int = 2,
) -> str:
    """One C×C cosine similarity heatmap per layer (every other layer)."""
    layers_to_plot = (
        gamma_summary[["layer_idx", "layer_name"]]
        .sort_values("layer_idx")
        .iloc[::every_n_layers]
    )
    n_plots = len(layers_to_plot)
    if n_plots == 0:
        return ""

    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Lemma 2: Class-Conditional Bias Gradient Cosine Similarity (|S[c,c']|)",
                 fontsize=13)

    for plot_idx, (_, row) in enumerate(layers_to_plot.iterrows()):
        layer_name = row["layer_name"]
        layer_idx = int(row["layer_idx"])
        ax = axes[plot_idx // ncols][plot_idx % ncols]

        # Pivot to get symmetric C×C matrix
        df_layer = gamma_raw[gamma_raw["layer_name"] == layer_name].copy()
        if df_layer.empty:
            ax.set_visible(False)
            continue

        classes = sorted(set(df_layer["class_i"].tolist() + df_layer["class_j"].tolist()))
        C = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        mat = np.eye(C)  # diagonal = 1 (self-similarity)
        for _, r in df_layer.iterrows():
            i = class_to_idx[r["class_i"]]
            j = class_to_idx[r["class_j"]]
            mat[i, j] = r["cosine_similarity"]

        gamma_val = gamma_summary.loc[
            gamma_summary["layer_name"] == layer_name, "gamma"
        ].values
        gamma_str = f" γ={gamma_val[0]:.3f}" if len(gamma_val) > 0 else ""

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"L{layer_idx}: {layer_name}{gamma_str}", fontsize=9)
        ax.set_xlabel("class j", fontsize=8)
        ax.set_ylabel("class i", fontsize=8)
        if C <= 20:
            ax.set_xticks(range(C))
            ax.set_yticks(range(C))
            ax.set_xticklabels(classes, fontsize=6, rotation=90)
            ax.set_yticklabels(classes, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for k in range(n_plots, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gamma_heatmaps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — Theory validation (combined)
# ---------------------------------------------------------------------------

def plot_theory_validation(
    alpha_summaries: Dict[str, pd.DataFrame],
    gamma_summary: pd.DataFrame,
    output_dir: str,
) -> str:
    """Two-panel figure: Lemma 1 (α) and Lemma 2 (γ) vs layer index."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left panel: mean α² per layer per attack ---------------------------
    ax_left.set_title("Lemma 1: Bias Gradient Concentration by Attack", fontsize=11)
    ax_left.set_xlabel("Layer index (input → output)", fontsize=10)
    ax_left.set_ylabel("Mean α² (higher = more concentrated)", fontsize=10)

    for attack_name, df_sum in alpha_summaries.items():
        if df_sum.empty:
            continue
        df_sorted = df_sum.sort_values("layer_idx")
        color = ATTACK_COLORS.get(attack_name, "black")
        ax_left.plot(
            df_sorted["layer_idx"], df_sorted["mean_alpha_sq_bias"],
            marker="o", markersize=5, linewidth=2,
            color=color, label=f"{attack_name} (bias)",
        )

    # Weight line for first attack (dashed, grey)
    first_key = next(iter(alpha_summaries), None)
    if first_key:
        df_sum = alpha_summaries[first_key]
        if not df_sum.empty:
            df_sorted = df_sum.sort_values("layer_idx")
            ax_left.plot(
                df_sorted["layer_idx"], df_sorted["mean_alpha_weight"],
                marker="^", markersize=4, linewidth=1.5,
                color=WEIGHT_COLOR, linestyle="--", label="weight (ref.)",
            )

    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)
    ax_left.set_ylim(0, 1.05)

    # ---- Right panel: γ per layer -------------------------------------------
    ax_right.set_title("Lemma 2: Clean Gradient Diversity (γ)", fontsize=11)
    ax_right.set_xlabel("Layer index (input → output)", fontsize=10)
    ax_right.set_ylabel("γ = max pairwise |cos similarity| (lower = more diverse)", fontsize=10)

    if not gamma_summary.empty:
        df_sorted = gamma_summary.sort_values("layer_idx")
        ax_right.plot(
            df_sorted["layer_idx"], df_sorted["gamma"],
            marker="s", markersize=6, linewidth=2,
            color="#2c3e50", label="clean model",
        )

    threshold = 0.3
    ax_right.axhline(threshold, color="green", linewidth=1.5, linestyle="--",
                     label=f"threshold γ={threshold}")

    if not gamma_summary.empty:
        df_sorted = gamma_summary.sort_values("layer_idx")
        xs = df_sorted["layer_idx"].values
        ax_right.fill_between(
            xs,
            0, threshold,
            where=[True] * len(xs),
            alpha=0.15, color="green", label="Lemma 2 holds",
        )

    ax_right.set_ylim(0, 1.05)
    ax_right.legend(fontsize=8)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "theory_validation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
