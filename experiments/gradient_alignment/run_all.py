"""CLI entry point for gradient alignment diagnostic experiments.

Usage
-----
    python -m experiments.gradient_alignment.run_all \\
        --config configs/gtsrb_fedavg_iba.yaml \\
        --benign-checkpoint results/gtsrb_benign_iid/final_model.pt \\
        --attacks neurotoxin a3fl iba chameleon \\
        --n-batches 200 \\
        --batch-size 64 \\
        --n-per-class 200 \\
        --output-dir experiments/gradient_alignment/outputs \\
        --device cuda

    # Also run the extended Lemma 2 experiment (effective rank comparison):
    python -m experiments.gradient_alignment.run_all \\
        --config configs/gtsrb_fedavg_iba.yaml \\
        --benign-checkpoint results/gtsrb_benign_iid/final_model.pt \\
        --extended-lemma2 \\
        --output-dir experiments/gradient_alignment/outputs \\
        --device cuda

Skip logic
----------
If alpha_raw_{attack}.csv / gamma_raw.csv / rank_comparison_table.csv already
exist in the output directory, the corresponding experiment is skipped.
Pass --force to re-run everything from scratch.
"""

from __future__ import annotations

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch

from experiment.config import ExperimentConfig
from experiment.utils import resolve_device

from .exp_alpha import run_alpha_experiment
from .exp_gamma import run_gamma_experiment
from .exp_lemma2 import run_lemma2_experiment
from .exp_kappa import run_kappa_experiment
from .visualize import (
    plot_alpha_distributions,
    plot_gamma_heatmaps,
    plot_theory_validation,
)


# ---------------------------------------------------------------------------
# Skip-if-exists helpers
# ---------------------------------------------------------------------------

def _alpha_exists(output_dir: str, attack: str) -> bool:
    return (
        os.path.exists(os.path.join(output_dir, f"alpha_raw_{attack}.csv"))
        and os.path.exists(os.path.join(output_dir, f"alpha_summary_{attack}.csv"))
    )


def _gamma_exists(output_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(output_dir, "gamma_raw.csv"))
        and os.path.exists(os.path.join(output_dir, "gamma_summary.csv"))
    )


def _lemma2_exists(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "rank_comparison_table.csv"))


def _kappa_exists(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "kappa_comparison_table.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gradient alignment diagnostics (Experiments 1, 2, and Lemma 2)."
    )
    p.add_argument("--config", required=True,
                   help="Path to experiment YAML config file.")
    p.add_argument("--checkpoint", default=None,
                   help="Attack model checkpoint (.pt state dict). "
                        "Auto-detected from results/ when omitted.")
    p.add_argument("--benign-checkpoint", default=None, dest="benign_checkpoint",
                   help="Benign model checkpoint for Experiment 2 and Lemma 2. "
                        "Auto-detected from results/gtsrb_benign_iid/ when omitted.")
    p.add_argument("--attacks", nargs="+",
                   default=["neurotoxin", "a3fl", "iba", "chameleon"],
                   help="List of attack names to run Experiment 1 for.")
    p.add_argument("--n-batches", type=int, default=50,
                   help="Number of triggered batches per attack (Exp 1 and Lemma 2).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Samples per batch.")
    p.add_argument("--n-per-class", type=int, default=100,
                   help="Samples per class for Experiment 2.")
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(__file__), "outputs"),
                   help="Directory for all output files.")
    p.add_argument("--device", default="auto",
                   help="Torch device: 'auto', 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--results-dir", default="results",
                   help="Root results directory for checkpoint auto-detection.")
    p.add_argument("--skip-exp1", action="store_true",
                   help="Skip Experiment 1 (α distribution).")
    p.add_argument("--skip-exp2", action="store_true",
                   help="Skip Experiment 2 (γ clean model).")
    p.add_argument("--extended-lemma2", action="store_true",
                   help="Run the extended Lemma 2 experiment (effective rank "
                        "comparison: eff_rank_clean_direct vs eff_rank_bd_upper_bound). "
                        "Requires Exp 2 (gamma_raw.csv) and Exp 1 alpha summaries.")
    p.add_argument("--force", action="store_true",
                   help="Re-run experiments even if output CSVs already exist.")
    p.add_argument("--kappa-weight", action="store_true", dest="kappa_weight",
                   help="Run the kappa comparison experiment (bias vs weight "
                        "effective-rank contrast, Theorem 1). Loads bias columns "
                        "from the existing rank_comparison_table.csv and computes "
                        "weight-side effective ranks. Requires --benign-checkpoint "
                        "and rank_comparison_table.csv in --output-dir.")
    p.add_argument("--n-gram-samples", type=int, default=1000,
                   dest="n_gram_samples",
                   help="Max N rows in the Gram matrix for weight eff_rank "
                        "(default 1000). Effective rank converges well below this.")
    p.add_argument("--coord-subsample", type=int, default=5000,
                   dest="coord_subsample",
                   help="d' for coordinate subsampling when weight D > 20000 "
                        "(default 5000). Same coords used for clean and BD.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load config --------------------------------------------------------
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {args.config} (dataset={config.dataset}, model={config.model})")

    # ---- Benign checkpoint auto-detection -----------------------------------
    from .attack_trigger import _BENIGN_RUNS
    benign_ckpt = args.benign_checkpoint
    if benign_ckpt is None:
        benign_run = _BENIGN_RUNS.get(config.dataset.lower(), "gtsrb_benign_iid")
        candidate = os.path.join(args.results_dir, benign_run, "final_model.pt")
        if os.path.exists(candidate):
            benign_ckpt = candidate
            print(f"Auto-detected benign checkpoint: {benign_ckpt}")
        else:
            print("WARNING: no benign checkpoint found; using random init for Exp 2.")

    import pandas as pd

    # ---- Experiment 1: α distribution ---------------------------------------
    alpha_raws = {}
    alpha_summaries = {}

    if not args.skip_exp1:
        for attack in args.attacks:
            if not args.force and _alpha_exists(args.output_dir, attack):
                print(f"  [skip] Exp 1 '{attack}' — CSVs already exist (--force to rerun)")
                alpha_raws[attack] = pd.read_csv(
                    os.path.join(args.output_dir, f"alpha_raw_{attack}.csv")
                )
                alpha_summaries[attack] = pd.read_csv(
                    os.path.join(args.output_dir, f"alpha_summary_{attack}.csv")
                )
                continue

            ckpt = args.checkpoint
            if ckpt is None:
                from .attack_trigger import get_attack_checkpoint
                ckpt = get_attack_checkpoint(attack, args.results_dir,
                                              dataset=config.dataset)

            try:
                df_raw, df_sum = run_alpha_experiment(
                    attack_name=attack,
                    config=config,
                    checkpoint_path=ckpt,
                    n_batches=args.n_batches,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=args.output_dir,
                    results_dir=args.results_dir,
                    dataset=config.dataset,
                )
                alpha_raws[attack] = df_raw
                alpha_summaries[attack] = df_sum
            except Exception as exc:
                print(f"ERROR in Exp 1 for '{attack}': {exc}")
                import traceback
                traceback.print_exc()

    # ---- Experiment 2: γ per layer ------------------------------------------
    gamma_raw = None
    gamma_summary = None

    if not args.skip_exp2:
        if not args.force and _gamma_exists(args.output_dir):
            print("  [skip] Exp 2 — CSVs already exist (--force to rerun)")
            gamma_raw     = pd.read_csv(os.path.join(args.output_dir, "gamma_raw.csv"))
            gamma_summary = pd.read_csv(os.path.join(args.output_dir, "gamma_summary.csv"))
        else:
            try:
                gamma_raw, gamma_summary = run_gamma_experiment(
                    config=config,
                    benign_checkpoint=benign_ckpt,
                    n_per_class=args.n_per_class,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=args.output_dir,
                )
            except Exception as exc:
                print(f"ERROR in Exp 2: {exc}")
                import traceback
                traceback.print_exc()

    # ---- Extended Lemma 2 ---------------------------------------------------
    if args.extended_lemma2:
        if not args.force and _lemma2_exists(args.output_dir):
            print("  [skip] Extended Lemma 2 — rank_comparison_table.csv exists "
                  "(--force to rerun)")
        else:
            # Load from disk if in-memory DFs are absent (e.g. --skip-exp2)
            _gamma_df = gamma_raw
            if _gamma_df is None:
                p = os.path.join(args.output_dir, "gamma_raw.csv")
                if os.path.exists(p):
                    _gamma_df = pd.read_csv(p)

            _alpha_dfs = dict(alpha_summaries)
            for attack in args.attacks:
                if attack not in _alpha_dfs:
                    p = os.path.join(args.output_dir, f"alpha_summary_{attack}.csv")
                    if os.path.exists(p):
                        _alpha_dfs[attack] = pd.read_csv(p)

            try:
                run_lemma2_experiment(
                    config=config,
                    benign_checkpoint=benign_ckpt,
                    attacks=args.attacks,
                    n_batches=args.n_batches,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=args.output_dir,
                    results_dir=args.results_dir,
                    gamma_raw_df=_gamma_df,
                    alpha_summary_dfs=_alpha_dfs if _alpha_dfs else None,
                )
            except Exception as exc:
                print(f"ERROR in Extended Lemma 2: {exc}")
                import traceback
                traceback.print_exc()

    # ---- kappa weight experiment (Step 4–6) ---------------------------------
    if args.kappa_weight:
        if not args.force and _kappa_exists(args.output_dir):
            print("  [skip] kappa experiment — kappa_comparison_table.csv exists "
                  "(--force to rerun)")
        else:
            try:
                run_kappa_experiment(
                    config=config,
                    benign_checkpoint=benign_ckpt,
                    attacks=args.attacks,
                    n_batches=args.n_batches,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=args.output_dir,
                    results_dir=args.results_dir,
                    n_gram_samples=args.n_gram_samples,
                    coord_subsample_dim=args.coord_subsample,
                )
            except Exception as exc:
                print(f"ERROR in kappa experiment: {exc}")
                import traceback
                traceback.print_exc()

    # ---- Visualisations -----------------------------------------------------
    print("\nGenerating figures...")

    # Load from disk if we skipped an experiment
    if args.skip_exp1:
        for attack in args.attacks:
            raw_path = os.path.join(args.output_dir, f"alpha_raw_{attack}.csv")
            sum_path = os.path.join(args.output_dir, f"alpha_summary_{attack}.csv")
            if os.path.exists(raw_path):
                alpha_raws[attack] = pd.read_csv(raw_path)
            if os.path.exists(sum_path):
                alpha_summaries[attack] = pd.read_csv(sum_path)

    if args.skip_exp2:
        rp = os.path.join(args.output_dir, "gamma_raw.csv")
        sp = os.path.join(args.output_dir, "gamma_summary.csv")
        if os.path.exists(rp):
            gamma_raw = pd.read_csv(rp)
        if os.path.exists(sp):
            gamma_summary = pd.read_csv(sp)

    if alpha_raws and alpha_summaries:
        try:
            plot_alpha_distributions(alpha_raws, alpha_summaries, args.output_dir)
        except Exception as exc:
            print(f"WARNING: alpha_distributions plot failed: {exc}")

    if gamma_raw is not None and gamma_summary is not None:
        try:
            plot_gamma_heatmaps(gamma_raw, gamma_summary, args.output_dir)
        except Exception as exc:
            print(f"WARNING: gamma_heatmaps plot failed: {exc}")

    if alpha_summaries and gamma_summary is not None:
        try:
            plot_theory_validation(alpha_summaries, gamma_summary, args.output_dir)
        except Exception as exc:
            print(f"WARNING: theory_validation plot failed: {exc}")

    print("\nDone. All outputs written to:", args.output_dir)


if __name__ == "__main__":
    main()
