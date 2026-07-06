"""TopoSentinel ablation: DKW vs. fixed-quantile filter modes.

Compares two TopoSentinel variants across all attack × dataset × seed combos:
  dkw   — DKW-calibrated interval (filter_mode="dkw",  target_fpr=0.05, dkw_confidence=0.95)
  fixed — original 5th/95th percentile  (filter_mode="fixed")

Run names encode the variant:
  {dataset}_{attack}_topo_dkw_seed{s}
  {dataset}_{attack}_topo_fixed_seed{s}

Results are written to <results-dir>/toposentinel_ablation/.
Summary CSV: <results-dir>/toposentinel_ablation_summary.csv.

Usage
-----
    python -m experiments.benchmark.run_toposentinel_ablation \\
        --results-dir results --device cuda

    # Single variant
    python -m experiments.benchmark.run_toposentinel_ablation --variants dkw

    # Dry-run
    python -m experiments.benchmark.run_toposentinel_ablation --dry-run

    # Summary only
    python -m experiments.benchmark.run_toposentinel_ablation --summarize-only
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from typing import Dict, List

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd

from experiment.config import DefenseConfig, ExperimentConfig
from experiments.benchmark.matrix import (
    DATASET_CFG,
    ESTIMATED_RUN_SECS,
    EVAL_EVERY,
    N_SEEDS,
    _make_attack_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASETS = ["femnist", "gtsrb", "cifar10", "cifar100"]
_ATTACKS  = ["a3fl", "iba", "neurotoxin", "chameleon"]

# Each entry maps variant name → kwargs forwarded to TopoSentinelServer
_VARIANT_CFG: Dict[str, dict] = {
    "dkw":   {"filter_mode": "dkw",   "target_fpr": 0.05, "dkw_confidence": 0.95},
    "fixed": {"filter_mode": "fixed"},
}

_SUBDIR = "toposentinel_ablation"

_SAFETY_MARGIN_SECS = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("topo_ablation")


# ---------------------------------------------------------------------------
# Graceful stop
# ---------------------------------------------------------------------------

class _GracefulStop(Exception):
    pass


_stop_requested = False


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):
        global _stop_requested
        _stop_requested = True
        logger.warning("SIGTERM received — will stop after the current run.")
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _run_name(dataset: str, attack: str, variant: str, seed: int) -> str:
    return f"{dataset}_{attack}_topo_{variant}_seed{seed}"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _make_ablation_config(
    dataset: str,
    attack: str,
    variant: str,
    seed: int,
    results_dir: str,
    device: str,
) -> ExperimentConfig:
    dc   = DATASET_CFG[dataset]
    name = _run_name(dataset, attack, variant, seed)
    vkw  = _VARIANT_CFG[variant]

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        data_root="data",
        partition="iid",
        batch_size=dc["batch_size"],
        num_clients=dc["num_clients"],
        num_rounds=dc["num_rounds"],
        clients_per_round=dc["clients_per_round"],
        local_epochs=dc["local_epochs"],
        model=dc["model"],
        lr=dc["lr"],
        weight_decay=dc["weight_decay"],
        attack=_make_attack_config(attack, dc),
        defense=DefenseConfig(
            defense_type="toposentinel",
            defense_kwargs=dict(vkw),
        ),
        eval_every=EVAL_EVERY,
        output_dir=os.path.join(results_dir, _SUBDIR),
        device=device,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _run_dir(run_name: str, results_dir: str) -> str:
    return os.path.join(results_dir, _SUBDIR, run_name)


def is_complete(run_name: str, results_dir: str, num_rounds: int) -> bool:
    d = _run_dir(run_name, results_dir)
    if not os.path.isfile(os.path.join(d, "final_model.pt")):
        return False
    metrics_path = os.path.join(d, "metrics.csv")
    if not os.path.isfile(metrics_path):
        return False
    try:
        df = pd.read_csv(metrics_path)
        return len(df) >= num_rounds // EVAL_EVERY
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def run_one(
    dataset: str,
    attack: str,
    variant: str,
    seed: int,
    results_dir: str,
    device: str,
) -> bool:
    from experiment.runner import FLRunner

    rn = _run_name(dataset, attack, variant, seed)
    logger.info("▶  %s", rn)
    t0 = time.time()
    try:
        config = _make_ablation_config(dataset, attack, variant, seed, results_dir, device)
        FLRunner(config).run()
        logger.info("✓  %s  (%.1f min)", rn, (time.time() - t0) / 60)
        return True
    except Exception as exc:
        logger.error("✗  %s — %s", rn, exc)
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(
    results_dir: str,
    datasets: List[str],
    attacks: List[str],
    variants: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    METRICS = ["clean_acc", "asr", "defense_tpr", "defense_fpr"]
    records = []

    for dataset in datasets:
        for attack in attacks:
            for variant in variants:
                rows_per_seed = []
                seeds_found: List[int] = []

                for seed in seeds:
                    rn   = _run_name(dataset, attack, variant, seed)
                    path = os.path.join(results_dir, _SUBDIR, rn, "metrics.csv")
                    if not os.path.isfile(path):
                        continue
                    try:
                        df = pd.read_csv(path)
                        if len(df) == 0:
                            continue
                        rows_per_seed.append(df.iloc[-1])
                        seeds_found.append(seed)
                    except Exception:
                        continue

                if not rows_per_seed:
                    continue

                frame = pd.DataFrame(rows_per_seed)
                rec: dict = {
                    "dataset":    dataset,
                    "attack":     attack,
                    "variant":    variant,
                    "n_seeds":    len(seeds_found),
                    "seeds_done": str(seeds_found),
                }
                for col in METRICS:
                    if col not in frame.columns:
                        rec[f"{col}_mean"] = float("nan")
                        rec[f"{col}_std"]  = float("nan")
                        continue
                    vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                    rec[f"{col}_mean"] = vals.mean() if len(vals) else float("nan")
                    rec[f"{col}_std"]  = vals.std()  if len(vals) > 1 else float("nan")

                records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TopoSentinel ablation: DKW vs. fixed-quantile filter mode."
    )
    p.add_argument("--results-dir", default="results",
                   help="Root results directory.")
    p.add_argument("--time-limit-hours", type=float, default=47.0,
                   help="Wall-clock budget in hours.")
    p.add_argument("--device",    default="auto")
    p.add_argument("--datasets",  nargs="+", default=None,
                   help="Subset of datasets (default: all).")
    p.add_argument("--attacks",   nargs="+", default=None,
                   help="Subset of attacks (default: all).")
    p.add_argument("--variants",  nargs="+", default=None,
                   choices=list(_VARIANT_CFG),
                   help=f"Filter-mode variants to run. "
                        f"Choices: {list(_VARIANT_CFG)}. Default: all.")
    p.add_argument("--seeds",     nargs="+", type=int, default=None,
                   help="Subset of seeds (default: 0–9).")
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip training; regenerate summary CSV only.")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print the run plan without executing anything.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _install_sigterm_handler()

    datasets = args.datasets or _DATASETS
    attacks  = args.attacks  or _ATTACKS
    variants = args.variants or list(_VARIANT_CFG)
    seeds    = args.seeds if args.seeds is not None else list(range(N_SEEDS))

    budget_secs  = args.time_limit_hours * 3600
    script_start = time.time()

    # Sort datasets shortest-first; interleave variants so both run on every
    # dataset before moving to the next, giving early comparable results.
    matrix = [
        (d, a, v, s)
        for d in sorted(datasets, key=lambda x: ESTIMATED_RUN_SECS.get(x, 0))
        for a in attacks
        for v in variants
        for s in seeds
    ]
    n_total = len(matrix)

    counts = {"complete": 0, "partial": 0, "missing": 0}
    for dataset, attack, variant, seed in matrix:
        rn = _run_name(dataset, attack, variant, seed)
        nr = DATASET_CFG[dataset]["num_rounds"]
        if is_complete(rn, args.results_dir, nr):
            counts["complete"] += 1
        elif os.path.isdir(_run_dir(rn, args.results_dir)):
            counts["partial"] += 1
        else:
            counts["missing"] += 1

    logger.info(
        "Matrix: %d datasets × %d attacks × %d variants × %d seeds = %d runs",
        len(datasets), len(attacks), len(variants), len(seeds), n_total,
    )
    logger.info("Variants: %s", variants)
    logger.info(
        "Status — complete: %d  partial: %d  missing: %d",
        counts["complete"], counts["partial"], counts["missing"],
    )
    logger.info("Time budget: %.1f h", args.time_limit_hours)

    if args.dry_run:
        print("\nDry-run — runs that WOULD execute (not yet complete):")
        for dataset, attack, variant, seed in matrix:
            rn = _run_name(dataset, attack, variant, seed)
            nr = DATASET_CFG[dataset]["num_rounds"]
            if not is_complete(rn, args.results_dir, nr):
                est_min = ESTIMATED_RUN_SECS.get(dataset, 0) / 60
                print(f"  {rn}  (~{est_min:.0f} min)")
        return

    # ---- Training phase -------------------------------------------------------
    if not args.summarize_only:
        n_run = n_failed = 0

        try:
            for dataset, attack, variant, seed in matrix:
                if _stop_requested:
                    raise _GracefulStop("stop flag set (SIGTERM).")

                rn = _run_name(dataset, attack, variant, seed)
                nr = DATASET_CFG[dataset]["num_rounds"]

                if is_complete(rn, args.results_dir, nr):
                    continue

                elapsed   = time.time() - script_start
                remaining = budget_secs - elapsed - _SAFETY_MARGIN_SECS
                est_secs  = ESTIMATED_RUN_SECS.get(dataset, 0)

                if remaining < est_secs:
                    logger.warning(
                        "Time budget: %.1f min remaining < %.1f min estimated "
                        "for '%s'. Stopping to write summary.",
                        remaining / 60, est_secs / 60, rn,
                    )
                    break

                ok = run_one(dataset, attack, variant, seed, args.results_dir, args.device)
                n_run += 1
                if not ok:
                    n_failed += 1

        except _GracefulStop as exc:
            logger.warning("Graceful stop: %s", exc)

        logger.info("Training done — %d new runs  (%d failed).", n_run, n_failed)

    # ---- Summary --------------------------------------------------------------
    logger.info("Generating ablation summary CSV …")
    summary = summarize(args.results_dir, datasets, attacks, variants, seeds)
    out_path = os.path.join(args.results_dir, "toposentinel_ablation_summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.4f")
    logger.info("Summary → %s  (%d rows)", out_path, len(summary))

    if len(summary):
        display_cols = [
            "dataset", "attack", "variant", "n_seeds",
            "clean_acc_mean", "clean_acc_std",
            "asr_mean", "asr_std",
            "defense_tpr_mean", "defense_fpr_mean",
        ]
        present = [c for c in display_cols if c in summary.columns]
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.float_format", "{:.3f}".format)
        print("\n" + summary[present].to_string(index=False))


if __name__ == "__main__":
    main()
