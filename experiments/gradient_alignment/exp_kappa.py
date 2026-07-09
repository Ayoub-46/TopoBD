"""kappa comparison experiment — bias vs weight effective-rank contrast.

STEP 0 — Convention (detected from exp_lemma2._covariance_eigenvalues / _eff_rank,
replicated exactly so kappa_bias / kappa_weight comparison is valid):

  UNCENTERED second moment:  Σ = (1/N) Σ_i  g_i g_i^T  (no mean subtracted)
  Gram trick:                eigvals( (1/N) G @ G.T ),  N×N Gram matrix;
                             nonzero eigenvalues equal those of (1/N) G.T @ G.
  Positive eigenvalues only: pos = eigvals[eigvals > 0]  (no tolerance, same as bias)
  eff_rank formula:          (Σ pos_λ)² / (Σ pos_λ²)   [participation ratio]

Weight-specific design choices
-------------------------------
Coordinate subsampling (D > COORD_SUBSAMPLE_THRESHOLD):
  d' = COORD_SUBSAMPLE_DIM random coordinates, seeded with COORD_SEED + layer_idx.
  CRITICAL: applied DURING accumulation (not after), so we never allocate
  N_total × D_full in RAM.  Same seed → same coordinates for clean and BD of the
  same layer, so kappa_weight = clean_rank / bd_rank is unbiased under this
  projection.  Justification: coordinate subsampling preserves effective rank
  in expectation for any distribution whose support is low-rank or uniformly
  spread; the ratio between clean and BD eff_rank is preserved as long as
  both are projected identically.

Early-stop accumulation:
  We accumulate at most n_gram_samples rows per layer, then stop.  With
  n_gram_samples=1000 and batch_size=64 this means ≈16 batches, not 200.
  Peak RAM per layer ≈ n_gram_samples × d' × 4 bytes = 1000×5000×4 = 20 MB.
  (Without this, 200 batches × 64 × D_full for fc_layers.0 in GTSRBNet
   would be 200 × 64 × 2 097 152 × 4 ≈ 102 GB — the original crash cause.)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    per_sample_weight_gradients,
    per_sample_weight_gradients_from_labels,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COORD_SUBSAMPLE_THRESHOLD = 20_000   # use coord subsampling when D > this
COORD_SUBSAMPLE_DIM       = 5_000    # d' after coordinate subsampling
COORD_SEED                = 42       # base seed; actual seed = COORD_SEED + layer_idx


# ---------------------------------------------------------------------------
# Effective rank (identical to exp_lemma2._eff_rank)
# ---------------------------------------------------------------------------

def _eff_rank(eigenvalues: torch.Tensor) -> float:
    """Participation-ratio effective rank: (Σλ)² / Σλ² over pos eigenvalues."""
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) == 0:
        return 0.0
    s  = pos.sum()
    s2 = (pos ** 2).sum()
    return (s * s / s2).item()


# ---------------------------------------------------------------------------
# Coordinate subsampling
# ---------------------------------------------------------------------------

def _get_coord_subset(D: int, d_prime: int, seed: int) -> torch.Tensor:
    """Fixed permutation-based subset of d_prime indices out of D.

    Deterministic: the same seed always yields the same subset regardless
    of the global torch RNG state.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(D, generator=g)
    return perm[:d_prime]


# ---------------------------------------------------------------------------
# Compact weight gradient accumulation (THE FIX)
# ---------------------------------------------------------------------------

def _accumulate_weight_gradients(
    model: nn.Module,
    loader,                         # iterable of (x, y) batches
    device: torch.device,
    active_layers: List[str],
    layer_D: Dict[str, int],
    layer_idx_map: Dict[str, int],
    n_gram_samples: int,
    coord_subsample_dim: int,
    coord_seed: int,
    target_label: Optional[int],    # None → use true labels (clean); int → BD
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Accumulate per-sample weight gradients with early-stop and in-loop
    coordinate subsampling.

    Memory contract: at most n_gram_samples × d' rows are ever held in RAM
    per layer (d' = D if D ≤ threshold, else coord_subsample_dim).
    Stops iterating once every active layer has n_gram_samples rows.

    Returns:
        G_dict   : {layer_name: tensor (N_actual, D_eff)} ready for Gram trick
        coord_idx: {layer_name: coord_index tensor} for large-D layers
                   (same tensor must be used for clean and BD of same layer)
    """
    model.eval()

    # Pre-compute coordinate subsets for large-D layers.
    # These are FIXED per layer so clean and BD use the same projection.
    coord_idx_map: Dict[str, torch.Tensor] = {}
    for name in active_layers:
        D = layer_D[name]
        if D > COORD_SUBSAMPLE_THRESHOLD:
            seed = coord_seed + layer_idx_map[name]
            coord_idx_map[name] = _get_coord_subset(D, coord_subsample_dim, seed)

    # Running accumulation: list of (n_i, D_eff) tensors per layer
    accum: Dict[str, List[torch.Tensor]] = {name: [] for name in active_layers}
    counts: Dict[str, int] = {name: 0 for name in active_layers}

    loader_iter = iter(loader)
    batch_idx = 0

    while True:
        # Check if every layer is saturated
        if all(counts[n] >= n_gram_samples for n in active_layers):
            break

        try:
            x, y = next(loader_iter)
        except StopIteration:
            break   # exhausted dataset before reaching n_gram_samples

        x = x.to(device)
        with torch.enable_grad():
            if target_label is not None:
                g_dict = per_sample_weight_gradients(
                    model, x, target_label, device
                )
            else:
                g_dict = per_sample_weight_gradients_from_labels(
                    model, x, y, device
                )

        for name in active_layers:
            if counts[name] >= n_gram_samples:
                continue
            g = g_dict.get(name)
            if g is None:
                continue

            g = g.cpu()  # (batch, D_full)

            # Apply coordinate subsampling immediately — never hold D_full rows
            if name in coord_idx_map:
                g = g[:, coord_idx_map[name]]   # (batch, d')

            # Take only the rows we still need
            need = n_gram_samples - counts[name]
            g = g[:need]

            accum[name].append(g)
            counts[name] += g.shape[0]

        batch_idx += 1
        if batch_idx % 10 == 0:
            sat = sum(1 for n in active_layers if counts[n] >= n_gram_samples)
            print(f"    batch {batch_idx}: {sat}/{len(active_layers)} layers saturated "
                  f"(counts: { {n: counts[n] for n in active_layers} })")

    # Concatenate into single tensors
    G_dict: Dict[str, torch.Tensor] = {}
    for name in active_layers:
        if accum[name]:
            G_dict[name] = torch.cat(accum[name], dim=0)   # (N_actual, D_eff)
            D_eff = G_dict[name].shape[1]
            note  = f" [coord d'={D_eff}]" if name in coord_idx_map else ""
            print(f"    {name:42s}  N={G_dict[name].shape[0]:4d}  D_eff={D_eff:8d}{note}")
        else:
            print(f"    WARNING: no data for layer '{name}'")

    return G_dict, coord_idx_map


# ---------------------------------------------------------------------------
# Gram eigenvalues (operates on already-compact G matrix)
# ---------------------------------------------------------------------------

def _gram_eigenvalues(G: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of (1/N) G G^T — identical convention to exp_lemma2.

    G is (N, D_eff) float32; computation in float64 for numerical stability.
    Returns eigvalsh result (ascending) as float64 tensor.
    """
    N = G.shape[0]
    G64  = G.to(torch.float64)
    gram = (G64 @ G64.T) / N          # (N, N)
    return torch.linalg.eigvalsh(gram)


# ---------------------------------------------------------------------------
# Step 3 — Gram trick validation
# ---------------------------------------------------------------------------

def _validate_gram_trick(
    G: torch.Tensor,      # (N, D) float32 CPU, for a SMALL-D layer only
    layer_name: str,
    tol: float = 1e-3,
) -> bool:
    """Assert eff_rank(D×D cov) ≈ eff_rank(N×N Gram) to relative tol.

    Only called on the smallest-D layer where D×D covariance is cheap.
    """
    N, D = G.shape
    G64  = G.to(torch.float64)

    # (a) D×D covariance (ground truth)
    cov_dd   = (G64.T @ G64) / N
    eig_dd   = torch.linalg.eigvalsh(cov_dd)
    er_dd    = _eff_rank(eig_dd)

    # (b) N×N Gram
    gram_nn  = (G64 @ G64.T) / N
    eig_nn   = torch.linalg.eigvalsh(gram_nn)
    er_nn    = _eff_rank(eig_nn)

    if er_dd < 1e-9 and er_nn < 1e-9:
        print(f"  [Gram validation] {layer_name}: both ranks ≈ 0, trivially PASS")
        return True

    denom   = max(er_dd, er_nn, 1e-9)
    rel_err = abs(er_dd - er_nn) / denom
    status  = "PASS" if rel_err < tol else "FAIL"
    print(f"  [Gram validation] {layer_name}: "
          f"eff_rank(D×D)={er_dd:.4f}  eff_rank(N×N Gram)={er_nn:.4f}  "
          f"rel_err={rel_err:.2e}  → {status}")

    if status == "FAIL":
        raise AssertionError(
            f"Gram trick validation FAILED for '{layer_name}': "
            f"eff_rank(D×D)={er_dd:.4f} vs eff_rank(N×N Gram)={er_nn:.4f}, "
            f"rel_err={rel_err:.2e} > tol={tol}"
        )
    return True


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_kappa_table(df: pd.DataFrame, output_dir: str) -> None:
    """Formatted table figure; green rows where kappa_bias > kappa_weight."""
    display_cols = [
        "layer_name",
        "eff_rank_clean_bias", "eff_rank_bd_max_bias", "kappa_bias",
        "eff_rank_clean_weight", "eff_rank_bd_max_weight", "kappa_weight",
        "kappa_ratio",
    ]
    df_d = df[[c for c in display_cols if c in df.columns]].copy()
    for col in display_cols:
        if col != "layer_name" and col in df_d.columns:
            df_d[col] = df_d[col].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "—"
            )

    col_labels = [c.replace("_", "\n") for c in df_d.columns]
    n_rows, n_cols = df_d.shape

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 1.8), max(2, n_rows * 0.55 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=df_d.values.tolist(), colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    if "kappa_bias" in df.columns and "kappa_weight" in df.columns:
        for i in range(n_rows):
            kb = df["kappa_bias"].iloc[i]
            kw = df["kappa_weight"].iloc[i]
            if pd.notna(kb) and pd.notna(kw) and kb > kw:
                for j in range(n_cols):
                    tbl[i + 1, j].set_facecolor("#d4edda")

    ax.set_title("κ Comparison: Bias vs Weight Subspace (Theorem 1)",
                 fontsize=11, pad=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "kappa_comparison_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_kappa_barplot(df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bars kappa_bias vs kappa_weight per layer — headline figure."""
    if df.empty:
        return

    labels = [f"{r['layer_name']}\n(idx {r['layer_idx']})"
              for _, r in df.iterrows()]
    kb = df["kappa_bias"].values.astype(float)
    kw = df["kappa_weight"].values.astype(float)

    n = len(df)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    ax.bar(x - w / 2, kb, w, label="κ_bias",   color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, kw, w, label="κ_weight", color="#ff7f0e", alpha=0.85)

    ylo, yhi = ax.get_ylim()
    for i, (b, ww) in enumerate(zip(kb, kw)):
        if np.isfinite(b) and np.isfinite(ww) and b > ww:
            ax.text(x[i], max(b, ww) + 0.02 * (yhi - ylo),
                    "★", ha="center", va="bottom", fontsize=10, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("κ  (eff_rank_clean / eff_rank_bd_max)")
    ax.set_title("Theorem 1: κ_bias vs κ_weight per layer\n"
                 "(★ = κ_bias > κ_weight; Theorem 1 prediction holds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "kappa_barplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_kappa_experiment(
    config: ExperimentConfig,
    benign_checkpoint: Optional[str],
    attacks: List[str],
    n_batches: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
    results_dir: str = "results",
    n_gram_samples: int = 1000,
    coord_subsample_dim: int = COORD_SUBSAMPLE_DIM,
    coord_seed: int = COORD_SEED,
) -> pd.DataFrame:
    """Compute kappa_weight and compare with kappa_bias from existing CSV.

    n_batches is ignored here — accumulation stops as soon as every layer
    has n_gram_samples rows (typically ≈ ceil(n_gram_samples / batch_size)
    batches, far fewer than n_batches).
    """
    print(f"\n{'='*60}")
    print("kappa experiment — bias vs weight rank contrast (Theorem 1)")
    print(f"{'='*60}")

    print("\n[CONVENTION — identical to exp_lemma2, replicated exactly]")
    print("  Σ_W = (1/N) Σ_i g_i g_i^T   (UNCENTERED second moment)")
    print("  Gram: eigvals((1/N) G @ G.T), N×N")
    print("  eff_rank = (Σλ)²/Σλ² over λ>0 only (no tolerance)")
    print(f"  Coord subsampling: D>{COORD_SUBSAMPLE_THRESHOLD} "
          f"→ d'={coord_subsample_dim}, seed={coord_seed}+layer_idx, "
          f"applied DURING accumulation")
    print(f"  Early-stop: accumulate at most {n_gram_samples} rows per layer\n")

    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load bias side ---------------------------------------------------
    bias_csv = os.path.join(output_dir, "rank_comparison_table.csv")
    if not os.path.exists(bias_csv):
        raise FileNotFoundError(
            f"rank_comparison_table.csv not found at {bias_csv}. "
            "Run --extended-lemma2 first."
        )
    df_bias = pd.read_csv(bias_csv)
    print(f"  Loaded bias ranks from {bias_csv} ({len(df_bias)} layers)")

    bias_layer_names = df_bias["layer_name"].tolist()
    layer_idx_map    = dict(zip(df_bias["layer_name"], df_bias["layer_idx"]))

    # ---- 2. Build model and adapter -----------------------------------------
    adapter    = build_adapter(config)
    model_cfg  = ModelConfig.from_adapter(config.model, adapter)
    benign_model = load_model(model_cfg, benign_checkpoint, device)
    benign_model.eval()

    target_label = config.attack.target_label

    # Identify which bias-table layers also have a weight param
    weight_bearing = {
        name
        for name, module in benign_model.named_modules()
        if name and isinstance(getattr(module, "weight", None), nn.Parameter)
    }
    missing = [n for n in bias_layer_names if n not in weight_bearing]
    if missing:
        print(f"  WARNING: no weight param for: {missing}")
    active_layers = [n for n in bias_layer_names if n in weight_bearing]

    # Weight dims (full D before any subsampling)
    layer_D: Dict[str, int] = {}
    for name, module in benign_model.named_modules():
        if name in active_layers:
            layer_D[name] = module.weight.numel()

    print(f"  Active layers: {active_layers}")
    print(f"  Weight dims:   { {n: layer_D[n] for n in active_layers} }")

    # Peak RAM estimate (after coord subsampling)
    for name in active_layers:
        D_eff = min(layer_D[name], coord_subsample_dim) \
                if layer_D[name] > COORD_SUBSAMPLE_THRESHOLD else layer_D[name]
        mb = n_gram_samples * D_eff * 4 / 1e6
        print(f"    {name}: peak G matrix = {n_gram_samples}×{D_eff} = {mb:.1f} MB")

    # ---- 3. Accumulate CLEAN weight gradients --------------------------------
    print(f"\n  [STEP 1] Accumulating clean weight gradients "
          f"(stop at {n_gram_samples} rows per layer)...")
    clean_loader = DataLoader(
        adapter.test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    G_clean, coord_idx_map = _accumulate_weight_gradients(
        model=benign_model,
        loader=clean_loader,
        device=device,
        active_layers=active_layers,
        layer_D=layer_D,
        layer_idx_map=layer_idx_map,
        n_gram_samples=n_gram_samples,
        coord_subsample_dim=coord_subsample_dim,
        coord_seed=coord_seed,
        target_label=None,   # clean: use true labels
    )

    # ---- 4. Gram trick validation (STEP 3) — smallest-D layer ---------------
    smallest_layer = min(active_layers, key=lambda n: layer_D.get(n, 10**9))
    smallest_D     = layer_D.get(smallest_layer, 0)
    print(f"\n  [STEP 3] Gram validation on '{smallest_layer}' (D={smallest_D})")

    if smallest_layer in G_clean and smallest_D <= 50_000:
        G_val = G_clean[smallest_layer]   # (N, D) — no coord subsampling here
        _validate_gram_trick(G_val, smallest_layer)
    else:
        print(f"  Skipping Gram validation (D={smallest_D} or no data)")

    # ---- 5. Compute clean eff_ranks ------------------------------------------
    clean_eff_ranks: Dict[str, float] = {}
    for name in active_layers:
        if name not in G_clean:
            continue
        eigvals = _gram_eigenvalues(G_clean[name])
        er      = _eff_rank(eigvals)
        clean_eff_ranks[name] = er
        D_eff   = G_clean[name].shape[1]
        note    = f" [coord d'={D_eff}]" if name in coord_idx_map else ""
        print(f"    {name:42s}  eff_rank_clean_W={er:.3f}{note}")

    # Free clean G matrices — no longer needed
    del G_clean
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- 6. BD weight gradients per attack (STEP 2) -------------------------
    bd_eff_ranks: Dict[str, Dict[str, float]] = {}  # attack → layer → eff_rank

    for attack in attacks:
        print(f"\n  [STEP 2] BD weight grads for '{attack}'...")
        try:
            atk_ckpt = get_attack_checkpoint(attack, results_dir,
                                              dataset=config.dataset)
            attack_model = load_model(model_cfg, atk_ckpt, device)
            attack_model.eval()

            trigger_fn = get_trigger_fn(attack, config, results_dir=results_dir)

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
            bd_loader = DataLoader(triggered_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)

            G_bd, _ = _accumulate_weight_gradients(
                model=attack_model,
                loader=bd_loader,
                device=device,
                active_layers=active_layers,
                layer_D=layer_D,
                layer_idx_map=layer_idx_map,
                n_gram_samples=n_gram_samples,
                coord_subsample_dim=coord_subsample_dim,
                coord_seed=coord_seed,   # SAME seed → SAME coord subset as clean
                target_label=target_label,
            )

            bd_eff_ranks[attack] = {}
            for name in active_layers:
                if name not in G_bd:
                    continue
                eigvals = _gram_eigenvalues(G_bd[name])
                er = _eff_rank(eigvals)
                bd_eff_ranks[attack][name] = er
                print(f"    {name:42s}  eff_rank_bd_W[{attack}]={er:.3f}")

            del G_bd, attack_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            print(f"  WARNING: BD for '{attack}' failed: {exc}")
            import traceback
            traceback.print_exc()

    # ---- 7. Assemble kappa comparison table (STEP 4) -------------------------
    bias_lkp   = df_bias.set_index("layer_name")
    table_rows = []

    for layer_name in bias_layer_names:
        row: dict = {"layer_name": layer_name,
                     "layer_idx":  layer_idx_map[layer_name]}

        # Bias columns — loaded from CSV, NOT recomputed
        erb_clean = float(bias_lkp.loc[layer_name, "eff_rank_clean_direct"]) \
                    if layer_name in bias_lkp.index else float("nan")
        erb_bd    = float(bias_lkp.loc[layer_name, "eff_rank_bd_max"]) \
                    if layer_name in bias_lkp.index else float("nan")
        kappa_b   = (erb_clean / erb_bd
                     if (not np.isnan(erb_clean) and not np.isnan(erb_bd)
                         and erb_bd > 1e-12) else float("nan"))

        # Weight columns
        erw_clean  = clean_eff_ranks.get(layer_name, float("nan"))
        bd_list    = [bd_eff_ranks[a][layer_name]
                      for a in bd_eff_ranks if layer_name in bd_eff_ranks[a]]
        erw_bd_max = max(bd_list) if bd_list else float("nan")
        kappa_w    = (erw_clean / erw_bd_max
                      if (not np.isnan(erw_clean) and not np.isnan(erw_bd_max)
                          and erw_bd_max > 1e-12) else float("nan"))

        row.update({
            "eff_rank_clean_bias":   erb_clean,
            "eff_rank_bd_max_bias":  erb_bd,
            "kappa_bias":            kappa_b,
            "eff_rank_clean_weight": erw_clean,
            "eff_rank_bd_max_weight":erw_bd_max,
            "kappa_weight":          kappa_w,
            "kappa_ratio":           (kappa_b / kappa_w
                                      if (not np.isnan(kappa_b)
                                          and not np.isnan(kappa_w)
                                          and kappa_w > 1e-12)
                                      else float("nan")),
        })
        table_rows.append(row)

    df_kappa = pd.DataFrame(table_rows)

    # ---- 8. Print summary + Theorem 1 verdict (STEP 6) ----------------------
    print(f"\n  {'Layer':<42s}  {'κ_bias':>8}  {'κ_weight':>9}  "
          f"{'κ_ratio':>8}  {'Thm1?':>8}")
    print("  " + "-" * 86)
    for _, r in df_kappa.iterrows():
        def _f(v): return f"{v:.3f}" if pd.notna(v) else "   —   "
        holds = ("HOLDS"
                 if (pd.notna(r.get("kappa_bias")) and pd.notna(r.get("kappa_weight"))
                     and r["kappa_bias"] > r["kappa_weight"])
                 else "VIOLATED")
        print(f"  {r['layer_name']:<42s}  {_f(r.get('kappa_bias')):>8}  "
              f"{_f(r.get('kappa_weight')):>9}  {_f(r.get('kappa_ratio')):>8}  "
              f"{holds:>8}")

    # FC/classifier layers — headline verdict
    fc_kw  = ("fc", "linear", "classifier", "head", "out")
    fc_rows = df_kappa[df_kappa["layer_name"].str.lower()
                       .str.contains("|".join(fc_kw), regex=True)]
    print(f"\n  === STEP 6 — FC-layer Theorem 1 check ({config.dataset}) ===")
    if fc_rows.empty:
        print("  No FC/classifier layers found in table.")
    else:
        for _, r in fc_rows.iterrows():
            kb = r.get("kappa_bias",   float("nan"))
            kw = r.get("kappa_weight", float("nan"))
            vd = "HOLDS" if (pd.notna(kb) and pd.notna(kw) and kb > kw) \
                 else "VIOLATED"
            print(f"  FC-layer Theorem 1 check ({r['layer_name']}): "
                  f"kappa_bias={kb:.4f} vs kappa_weight={kw:.4f} -> {vd}")

    # ---- 9. Save (STEP 4) ----------------------------------------------------
    csv_path = os.path.join(output_dir, "kappa_comparison_table.csv")
    df_kappa.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Saved: {csv_path}")
    _plot_kappa_table(df_kappa, output_dir)
    _plot_kappa_barplot(df_kappa, output_dir)

    return df_kappa
