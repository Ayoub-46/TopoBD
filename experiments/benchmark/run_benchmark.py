"""FL benchmark runner — full attack × defense × dataset × seed matrix.

Features
--------
* Skips already-complete runs (``final_model.pt`` present + metrics.csv full).
* Graceful stopping: checks the remaining time budget before each run and
  stops before the SLURM wall-clock limit is reached.
* SIGTERM handler: catches SLURM's preemption signal and exits cleanly after
  the current run finishes.
* Summary CSV written on every exit (normal or graceful stop).

Usage
-----
    python -m experiments.benchmark.run_benchmark \\
        --results-dir results \\
        --time-limit-hours 47 \\
        --device cuda

    # Partial runs (resume by resubmitting — already-complete runs are skipped)
    sbatch experiments/benchmark/slurm_run.sh

    # Re-generate summary only (no training)
    python -m experiments.benchmark.run_benchmark --summarize-only
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
import traceback
from typing import List, Optional, Tuple

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd

from experiments.benchmark.matrix import (
    ATTACKS,
    DATASET_CFG,
    DATASETS,
    DEFENSES,
    ESTIMATED_RUN_SECS,
    EVAL_EVERY,
    N_SEEDS,
    get_run_matrix,
    make_config,
    make_run_name,
)

# 10-minute safety margin — time reserved for writing the summary CSV and
# cleaning up before SLURM terminates the job.
_SAFETY_MARGIN_SECS = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Graceful stop
# ---------------------------------------------------------------------------

class _GracefulStop(Exception):
    """Raised on SIGTERM or exhausted time budget."""


_stop_requested = False


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):
        global _stop_requested
        _stop_requested = True
        logger.warning("SIGTERM received — will stop after the current run.")
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------

def _run_dir(run_name: str, results_dir: str) -> str:
    return os.path.join(results_dir, "benchmark", run_name)


def is_complete(run_name: str, results_dir: str, num_rounds: int) -> bool:
    """True when final_model.pt exists and metrics.csv has enough rows."""
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


def _classify_run(run_name: str, results_dir: str, num_rounds: int) -> str:
    """Return 'complete', 'partial', or 'missing'."""
    d = _run_dir(run_name, results_dir)
    if is_complete(run_name, results_dir, num_rounds):
        return "complete"
    if os.path.isdir(d):
        return "partial"
    return "missing"


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def run_one(
    dataset: str,
    attack: str,
    defense: str,
    seed: int,
    results_dir: str,
    device: str,
) -> bool:
    """Execute one (dataset, attack, defense, seed) cell. Returns True on success."""
    from experiment.runner import FLRunner

    run_name = make_run_name(dataset, attack, defense, seed)
    logger.info("▶  %s", run_name)
    t0 = time.time()
    try:
        config = make_config(dataset, attack, defense, seed, results_dir, device)
        FLRunner(config).run()
        logger.info("✓  %s  (%.1f min)", run_name, (time.time() - t0) / 60)
        return True
    except Exception as exc:
        logger.error("✗  %s — %s", run_name, exc)
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def summarize(
    results_dir: str,
    datasets: List[str],
    attacks: List[str],
    defenses: List[str],
    seeds: List[int],
) -> pd.DataFrame:
    """Aggregate final-round metrics across seeds → mean ± std per cell."""
    METRICS = ["clean_acc", "asr", "defense_tpr", "defense_fpr"]
    records = []

    for dataset in datasets:
        for attack in attacks:
            for defense in defenses:
                rows_per_seed = []
                seeds_found   = []

                for seed in seeds:
                    run_name = make_run_name(dataset, attack, defense, seed)
                    path = os.path.join(
                        results_dir, "benchmark", run_name, "metrics.csv"
                    )
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
                    "defense":    defense,
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
        description="FL benchmark: full attack × defense × dataset × seed matrix."
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Root results directory (runs go into <results-dir>/benchmark/).",
    )
    p.add_argument(
        "--time-limit-hours", type=float, default=47.0,
        help="Wall-clock budget in hours. Set slightly below the SBATCH --time "
             "value to allow a clean summary write at the end.",
    )
    p.add_argument("--device",   default="auto")
    p.add_argument("--datasets",  nargs="+", default=None,
                   help="Subset of datasets to run (default: all).")
    p.add_argument("--attacks",   nargs="+", default=None,
                   help="Subset of attacks to run (default: all).")
    p.add_argument("--defenses",  nargs="+", default=None,
                   help="Subset of defenses to run (default: all).")
    p.add_argument("--seeds",     nargs="+", type=int, default=None,
                   help="Subset of seeds (default: 0–9).")
    p.add_argument(
        "--summarize-only", action="store_true",
        help="Skip training and only (re-)generate the summary CSV.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without executing anything.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _install_sigterm_handler()

    datasets = args.datasets or DATASETS
    attacks  = args.attacks  or ATTACKS
    defenses = args.defenses or DEFENSES
    seeds    = args.seeds if args.seeds is not None else list(range(N_SEEDS))

    budget_secs  = args.time_limit_hours * 3600
    script_start = time.time()

    matrix = get_run_matrix(datasets, attacks, defenses, seeds)
    n_total = len(matrix)

    # ---- Status report -------------------------------------------------------
    counts = {"complete": 0, "partial": 0, "missing": 0}
    for dataset, attack, defense, seed in matrix:
        run_name = make_run_name(dataset, attack, defense, seed)
        status = _classify_run(run_name, args.results_dir, DATASET_CFG[dataset]["num_rounds"])
        counts[status] += 1

    logger.info(
        "Matrix: %d datasets × %d attacks × %d defenses × %d seeds = %d runs",
        len(datasets), len(attacks), len(defenses), len(seeds), n_total,
    )
    logger.info(
        "Status — complete: %d  partial/incomplete: %d  not started: %d",
        counts["complete"], counts["partial"], counts["missing"],
    )
    logger.info("Time budget: %.1f h", args.time_limit_hours)

    if args.dry_run:
        print("\nDry-run — runs that WOULD execute (not yet complete):")
        for dataset, attack, defense, seed in matrix:
            run_name = make_run_name(dataset, attack, defense, seed)
            nr = DATASET_CFG[dataset]["num_rounds"]
            if not is_complete(run_name, args.results_dir, nr):
                est_min = ESTIMATED_RUN_SECS[dataset] / 60
                print(f"  {run_name}  (~{est_min:.0f} min)")
        return

    # ---- Training phase ------------------------------------------------------
    if not args.summarize_only:
        n_run = n_failed = 0

        try:
            for dataset, attack, defense, seed in matrix:
                if _stop_requested:
                    raise _GracefulStop("stop flag set (SIGTERM).")

                run_name = make_run_name(dataset, attack, defense, seed)
                nr = DATASET_CFG[dataset]["num_rounds"]

                if is_complete(run_name, args.results_dir, nr):
                    continue    # already done — skip

                # Time budget gate
                elapsed   = time.time() - script_start
                remaining = budget_secs - elapsed - _SAFETY_MARGIN_SECS
                est_secs  = ESTIMATED_RUN_SECS[dataset]

                if remaining < est_secs:
                    logger.warning(
                        "Time budget: %.1f min remaining < %.1f min estimated "
                        "for '%s'. Stopping to write summary.",
                        remaining / 60, est_secs / 60, run_name,
                    )
                    break

                ok = run_one(dataset, attack, defense, seed, args.results_dir, args.device)
                n_run += 1
                if not ok:
                    n_failed += 1

        except _GracefulStop as exc:
            logger.warning("Graceful stop: %s", exc)

        logger.info(
            "Training done — %d new runs  (%d failed).", n_run, n_failed
        )

    # ---- Summary -------------------------------------------------------------
    logger.info("Generating summary CSV …")
    summary = summarize(args.results_dir, datasets, attacks, defenses, seeds)
    out_path = os.path.join(args.results_dir, "benchmark_summary.csv")
    os.makedirs(args.results_dir, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.4f")
    logger.info("Summary → %s  (%d rows)", out_path, len(summary))

    if len(summary):
        # Pretty-print a compact version to stdout
        display_cols = [
            "dataset", "attack", "defense", "n_seeds",
            "clean_acc_mean", "clean_acc_std",
            "asr_mean", "asr_std",
        ]
        present = [c for c in display_cols if c in summary.columns]
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.float_format", "{:.3f}".format)
        print("\n" + summary[present].to_string(index=False))


if __name__ == "__main__":
    main()
