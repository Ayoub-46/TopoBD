"""TopoSentinel benchmark runner — all attack × dataset × seed combinations.

Runs TopoSentinel (fixed 5th/95th-percentile intra-round filter; DKW
calibration was removed) against every combination of:
  datasets : cifar10, cifar100, gtsrb, femnist
  attacks  : a3fl, iba, neurotoxin, chameleon
  seeds    : 0 … N_SEEDS-1 (default 10)

Results are written to <results-dir>/toposentinel_benchmark/.

Features
--------
* Skips already-complete runs (``final_model.pt`` present + metrics.csv full).
* Graceful stopping: checks the remaining time budget before each run and
  stops before the SLURM wall-clock limit is reached.
* SIGTERM handler: catches SLURM's preemption signal and exits cleanly.
* Summary CSV written on every exit.

Usage
-----
    python -m experiments.benchmark.run_toposentinel \\
        --results-dir results \\
        --time-limit-hours 47 \\
        --device cuda

    # Subset
    python -m experiments.benchmark.run_toposentinel \\
        --datasets cifar10 gtsrb --attacks a3fl --seeds 0 1

    # Summary only
    python -m experiments.benchmark.run_toposentinel --summarize-only
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from typing import List, Optional

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
_DEFENSE  = "toposentinel"
_SUBDIR   = "toposentinel_benchmark"

_SAFETY_MARGIN_SECS = 600

# ---------------------------------------------------------------------------
# Per-dataset bottleneck-alarm thresholds
# ---------------------------------------------------------------------------
# The bottleneck distance W_inf of the client bias persistence diagram is
# model/dataset dependent — from the IBA analysis_mode sweep
# (results/analysis/iba_fixed_detection_summary.csv) the ATTACK-round W_inf is
# roughly gtsrb≈0.024, femnist≈0.054, cifar≈0.099, with QUIET-round W_inf about
# half of that (0.012 / 0.026 / 0.033). A single min_threshold therefore cannot
# separate attack from quiet across datasets: the previous uniform 0.05 sat
# ABOVE gtsrb's attack signal (0.024), so the alarm could never fire there
# (measured recall 0.0), while barely clearing femnist's (recall 0.2).
#
# Each dataset's min_threshold is set ~0.7×(attack W_inf) — comfortably below
# the attack signal yet above that dataset's quiet W_inf (near-zero FPR) — and
# the initial threshold decays to it before the attack window opens
# (attack_start: femnist 37, gtsrb 50, cifar 80).
_DEFAULT_BOTTLENECK = {
    "bottleneck_initial_threshold": 0.15,
    "bottleneck_decay_rate": 0.96,
    "bottleneck_min_threshold": 0.05,
}
_BOTTLENECK_THRESHOLDS = {
    "gtsrb":    {"bottleneck_initial_threshold": 0.05,
                 "bottleneck_decay_rate": 0.95, "bottleneck_min_threshold": 0.017},
    "femnist":  {"bottleneck_initial_threshold": 0.08,
                 "bottleneck_decay_rate": 0.95, "bottleneck_min_threshold": 0.038},
    "cifar10":  {"bottleneck_initial_threshold": 0.15,
                 "bottleneck_decay_rate": 0.96, "bottleneck_min_threshold": 0.060},
    "cifar100": {"bottleneck_initial_threshold": 0.15,
                 "bottleneck_decay_rate": 0.96, "bottleneck_min_threshold": 0.060},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("topo_benchmark")


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
# Config builder
# ---------------------------------------------------------------------------

def _make_topo_config(
    dataset: str,
    attack: str,
    seed: int,
    results_dir: str,
    device: str,
) -> ExperimentConfig:
    dc   = DATASET_CFG[dataset]
    name = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
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
            # Per-dataset bottleneck-alarm thresholds calibrated to that
            # dataset's W_inf scale (see _BOTTLENECK_THRESHOLDS above). The
            # intra-round bias-distance filter is unchanged.
            defense_kwargs=dict(_BOTTLENECK_THRESHOLDS.get(dataset, _DEFAULT_BOTTLENECK)),
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
    seed: int,
    results_dir: str,
    device: str,
) -> bool:
    from experiment.runner import FLRunner

    run_name = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
    logger.info("▶  %s", run_name)
    t0 = time.time()
    try:
        config = _make_topo_config(dataset, attack, seed, results_dir, device)
        FLRunner(config).run()
        logger.info("✓  %s  (%.1f min)", run_name, (time.time() - t0) / 60)
        return True
    except Exception as exc:
        logger.error("✗  %s — %s", run_name, exc)
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(
    results_dir: str,
    datasets: List[str],
    attacks: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    METRICS = ["clean_acc", "asr", "defense_tpr", "defense_fpr"]
    records = []

    for dataset in datasets:
        for attack in attacks:
            rows_per_seed = []
            seeds_found: List[int] = []

            for seed in seeds:
                run_name = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
                path = os.path.join(results_dir, _SUBDIR, run_name, "metrics.csv")
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
                "defense":    _DEFENSE,
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
        description="TopoSentinel benchmark: all dataset × attack × seed combinations."
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
    seeds    = args.seeds if args.seeds is not None else list(range(N_SEEDS))

    budget_secs  = args.time_limit_hours * 3600
    script_start = time.time()

    # Sort datasets shortest-first for better time-budget utilisation
    matrix = [
        (d, a, s)
        for d in sorted(datasets, key=lambda x: ESTIMATED_RUN_SECS.get(x, 0))
        for a in attacks
        for s in seeds
    ]
    n_total = len(matrix)

    counts = {"complete": 0, "partial": 0, "missing": 0}
    for dataset, attack, seed in matrix:
        rn = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
        nr = DATASET_CFG[dataset]["num_rounds"]
        if is_complete(rn, args.results_dir, nr):
            counts["complete"] += 1
        elif os.path.isdir(_run_dir(rn, args.results_dir)):
            counts["partial"] += 1
        else:
            counts["missing"] += 1

    logger.info(
        "Matrix: %d datasets × %d attacks × %d seeds = %d runs",
        len(datasets), len(attacks), len(seeds), n_total,
    )
    logger.info(
        "Status — complete: %d  partial: %d  missing: %d",
        counts["complete"], counts["partial"], counts["missing"],
    )
    logger.info("Time budget: %.1f h", args.time_limit_hours)

    if args.dry_run:
        print("\nDry-run — runs that WOULD execute (not yet complete):")
        for dataset, attack, seed in matrix:
            rn = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
            nr = DATASET_CFG[dataset]["num_rounds"]
            if not is_complete(rn, args.results_dir, nr):
                est_min = ESTIMATED_RUN_SECS.get(dataset, 0) / 60
                print(f"  {rn}  (~{est_min:.0f} min)")
        return

    # ---- Training phase -------------------------------------------------------
    if not args.summarize_only:
        n_run = n_failed = 0

        try:
            for dataset, attack, seed in matrix:
                if _stop_requested:
                    raise _GracefulStop("stop flag set (SIGTERM).")

                rn = f"{dataset}_{attack}_{_DEFENSE}_seed{seed}"
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

                ok = run_one(dataset, attack, seed, args.results_dir, args.device)
                n_run += 1
                if not ok:
                    n_failed += 1

        except _GracefulStop as exc:
            logger.warning("Graceful stop: %s", exc)

        logger.info("Training done — %d new runs  (%d failed).", n_run, n_failed)

    # ---- Summary --------------------------------------------------------------
    logger.info("Generating summary CSV …")
    summary = summarize(args.results_dir, datasets, attacks, seeds)
    out_path = os.path.join(args.results_dir, "toposentinel_summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.4f")
    logger.info("Summary → %s  (%d rows)", out_path, len(summary))

    if len(summary):
        display_cols = [
            "dataset", "attack", "defense", "n_seeds",
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
