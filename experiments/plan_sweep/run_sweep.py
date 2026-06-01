"""Plan-sweep runner — executes the §8 matrix with graceful SLURM stopping.

Features
--------
* Skips complete runs (final_model.pt + sufficient metrics.csv rows).
* Time-budget gate: stops before SLURM wall-clock limit, writes summary.
* SIGTERM handler: finishes current run then exits cleanly.
* Tier selection: --tier 1|2|all.
* Dry-run mode: prints what would run without executing.

Usage
-----
    python -m experiments.plan_sweep.run_sweep \\
        --tier 1 \\
        --results-dir results \\
        --time-limit-hours 99 \\
        --device cuda

    # Smoke check (fast, subset):
    python -m experiments.plan_sweep.run_sweep \\
        --config configs/smoke.yaml \\
        --tier 1 --datasets cifar10 --alphas iid \\
        --attacks neurotoxin --defenses none toposentinel \\
        --seeds 0 --time-limit-hours 1 --device cpu
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

from experiments.plan_sweep.matrix import (
    ATTACKS, DEFENSES, ESTIMATED_RUN_SECS, EVAL_EVERY, N_SEEDS,
    TIER1_DATASETS, TIER2_DATASETS,
    get_full_matrix, get_tier1_matrix, get_tier2_matrix,
    make_config, make_run_name,
)

_SAFETY_MARGIN_SECS = 900   # 15-min buffer for summary + cleanup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("plan_sweep")

_stop_requested = False


def _install_sigterm() -> None:
    def _handler(sig, frame):
        global _stop_requested
        _stop_requested = True
        logger.warning("SIGTERM received — will stop after the current run.")
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _run_dir(run_name: str, results_dir: str) -> str:
    return os.path.join(results_dir, "plan_sweep", run_name)


def is_complete(run_name: str, results_dir: str, num_rounds: int) -> bool:
    d = _run_dir(run_name, results_dir)
    if not os.path.isfile(os.path.join(d, "final_model.pt")):
        return False
    mp = os.path.join(d, "metrics.csv")
    if not os.path.isfile(mp):
        return False
    try:
        return len(pd.read_csv(mp)) >= num_rounds // EVAL_EVERY
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def run_one(
    dataset: str, alpha, attack: str, defense: str, seed: int,
    results_dir: str, device: str,
) -> bool:
    from experiment.runner import FLRunner
    from experiments.plan_sweep.matrix import DATASET_CFG

    run_name = make_run_name(dataset, alpha, attack, defense, seed)
    logger.info("▶  %s", run_name)
    t0 = time.time()
    try:
        cfg = make_config(dataset, alpha, attack, defense, seed, results_dir, device)
        FLRunner(cfg).run()
        logger.info("✓  %s  (%.1f min)", run_name, (time.time() - t0) / 60)
        return True
    except Exception as exc:
        logger.error("✗  %s — %s", run_name, exc)
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary(results_dir: str, matrix: List[Tuple]) -> None:
    """Aggregate final-round metrics across seeds."""
    METRIC_COLS = ["clean_acc", "asr", "macro_f1", "defense_tpr", "defense_fpr"]
    records = []
    seen: set = set()

    for (dataset, alpha, attack, defense, seed) in matrix:
        key = (dataset, alpha, attack, defense)
        if key in seen:
            continue

        rows = []
        for s in range(N_SEEDS):
            rname = make_run_name(dataset, alpha, attack, defense, s)
            mp = os.path.join(results_dir, "plan_sweep", rname, "metrics.csv")
            if not os.path.isfile(mp):
                continue
            try:
                df = pd.read_csv(mp)
                if len(df) > 0:
                    rows.append(df.iloc[-1])
            except Exception:
                pass

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        alpha_str = "iid" if alpha == "iid" else str(alpha)
        rec: dict = {
            "dataset": dataset, "alpha": alpha_str,
            "attack": attack, "defense": defense,
            "n_seeds": len(rows),
        }
        for col in METRIC_COLS:
            if col not in frame.columns:
                rec[f"{col}_mean"] = rec[f"{col}_std"] = float("nan")
                continue
            vals = pd.to_numeric(frame[col], errors="coerce").dropna()
            rec[f"{col}_mean"] = vals.mean() if len(vals) else float("nan")
            rec[f"{col}_std"]  = vals.std()  if len(vals) > 1 else float("nan")
        records.append(rec)
        seen.add(key)

    if not records:
        logger.info("No completed runs to summarise yet.")
        return

    out = pd.DataFrame(records)
    path = os.path.join(results_dir, "plan_sweep_summary.csv")
    out.to_csv(path, index=False, float_format="%.4f")
    logger.info("Summary → %s  (%d rows)", path, len(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FL plan sweep runner (§8 matrix).")
    p.add_argument("--results-dir",      default="results")
    p.add_argument("--time-limit-hours", type=float, default=99.0)
    p.add_argument("--device",           default="auto")
    p.add_argument("--tier",             choices=["1", "2", "all"], default="all")
    p.add_argument("--datasets",  nargs="+", default=None)
    p.add_argument("--alphas",    nargs="+", default=None,
                   help="Subset of alpha levels: iid 1.0 0.5 0.3 0.1 natural")
    p.add_argument("--attacks",   nargs="+", default=None)
    p.add_argument("--defenses",  nargs="+", default=None)
    p.add_argument("--seeds",     nargs="+", type=int, default=None)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()
    _install_sigterm()

    seeds    = args.seeds   if args.seeds   is not None else list(range(N_SEEDS))
    attacks  = args.attacks  or ATTACKS
    defenses = args.defenses or DEFENSES

    # Parse alphas (mix of strings and floats)
    def _parse_alpha(s: str):
        if s in ("iid", "natural"):
            return s
        try:
            return float(s)
        except ValueError:
            return s

    alphas = [_parse_alpha(a) for a in args.alphas] if args.alphas else None

    kw = dict(attacks=attacks, defenses=defenses, seeds=seeds)
    if args.tier == "1":
        kw["datasets"] = args.datasets or TIER1_DATASETS
        if alphas:
            kw["alphas"] = alphas
        matrix = get_tier1_matrix(**kw)
    elif args.tier == "2":
        kw["datasets"] = args.datasets or TIER2_DATASETS
        matrix = get_tier2_matrix(**kw)
    else:
        t1kw = dict(kw)
        t2kw = dict(kw)
        if args.datasets:
            t1kw["datasets"] = [d for d in args.datasets if d in TIER1_DATASETS]
            t2kw["datasets"] = [d for d in args.datasets if d in TIER2_DATASETS]
        if alphas:
            t1kw["alphas"] = alphas
        matrix = get_tier1_matrix(**t1kw) + get_tier2_matrix(**t2kw)

    from experiments.plan_sweep.matrix import DATASET_CFG

    n_total = len(matrix)
    n_complete = sum(
        1 for (d, a, atk, de, s) in matrix
        if is_complete(make_run_name(d, a, atk, de, s), args.results_dir,
                       DATASET_CFG[d]["num_rounds"])
    )
    logger.info(
        "Matrix: %d runs  |  complete: %d  |  remaining: %d",
        n_total, n_complete, n_total - n_complete,
    )
    logger.info("Time budget: %.1f h", args.time_limit_hours)

    if args.dry_run:
        print("\nDry-run — pending runs:")
        for (d, a, atk, de, s) in matrix:
            rn = make_run_name(d, a, atk, de, s)
            nr = DATASET_CFG[d]["num_rounds"]
            if not is_complete(rn, args.results_dir, nr):
                est = ESTIMATED_RUN_SECS.get(d, 0) / 60
                print(f"  {rn}  (~{est:.0f} min)")
        return

    budget_secs  = args.time_limit_hours * 3600
    script_start = time.time()
    n_run = n_failed = 0

    if not args.summarize_only:
        try:
            for (dataset, alpha, attack, defense, seed) in matrix:
                if _stop_requested:
                    raise RuntimeError("stop flag")

                rn = make_run_name(dataset, alpha, attack, defense, seed)
                nr = DATASET_CFG[dataset]["num_rounds"]
                if is_complete(rn, args.results_dir, nr):
                    continue

                elapsed   = time.time() - script_start
                remaining = budget_secs - elapsed - _SAFETY_MARGIN_SECS
                est       = ESTIMATED_RUN_SECS.get(dataset, 0)
                if remaining < est:
                    logger.warning(
                        "Budget: %.1f min left < %.1f min est for '%s'. Stopping.",
                        remaining / 60, est / 60, rn,
                    )
                    break

                ok = run_one(dataset, alpha, attack, defense, seed,
                             args.results_dir, args.device)
                n_run += 1
                if not ok:
                    n_failed += 1

        except Exception as exc:
            logger.warning("Early stop: %s", exc)

        logger.info("Training phase: %d runs, %d failed.", n_run, n_failed)

    write_summary(args.results_dir, matrix)


if __name__ == "__main__":
    main()
