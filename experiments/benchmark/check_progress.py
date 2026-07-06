"""Progress tracker for all FL benchmark experiment suites.

Scans run directories and reports completion status for three suites:
  • Main benchmark   — dataset × attack × defense × seed
  • TopoSentinel     — dataset × attack × seed  (DKW filter)
  • Ablation         — dataset × attack × variant × seed

Usage
-----
    python experiments/benchmark/check_progress.py
    python experiments/benchmark/check_progress.py --results-dir /path/to/results
    python experiments/benchmark/check_progress.py --suite toposentinel
    python experiments/benchmark/check_progress.py --verbose        # per-cell grids
    python experiments/benchmark/check_progress.py --missing-only   # list incomplete runs
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Tuple

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
    make_run_name,
)

_TOPO_VARIANTS = ["dkw", "fixed"]
_BLOCK = {0: "·", 1: "▁", 2: "▂", 3: "▃", 4: "▄", 5: "▅", 6: "▆", 7: "▇", 8: "█", 9: "█", 10: "█"}


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def _classify(run_dir: str, num_rounds: int) -> str:
    """Return 'done', 'partial', or 'missing'."""
    if not os.path.isdir(run_dir):
        return "missing"
    has_model   = os.path.isfile(os.path.join(run_dir, "final_model.pt"))
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    if has_model and os.path.isfile(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            if len(df) >= num_rounds // EVAL_EVERY:
                return "done"
        except Exception:
            pass
    return "partial"


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _eta_str(n_remaining: int, dataset: str) -> str:
    if n_remaining == 0:
        return "—"
    secs = n_remaining * ESTIMATED_RUN_SECS.get(dataset, 0)
    if secs == 0:
        return "?"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    return f"~{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def _bar(n_done: int, n_total: int, width: int = 30) -> str:
    filled = round(width * n_done / n_total) if n_total else 0
    return "[" + "█" * filled + "·" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Seed completion grid  (one cell = one seed, filled = done)
# ---------------------------------------------------------------------------

def _seed_grid(
    seed_statuses: Dict[str, Dict[int, str]],  # {(a,d_or_v): {seed: status}}
    row_labels: List[str],
    col_labels: List[str],
    seeds: List[int],
    col_width: int = 4,
) -> str:
    """ASCII grid: rows = row_labels, cols = col_labels, each cell = seed bar."""
    n_seeds  = len(seeds)
    rl_width = max(len(r) for r in row_labels) + 1

    lines = []
    # Header
    header = " " * rl_width + "  " + "  ".join(f"{c[:col_width]:<{col_width}}" for c in col_labels)
    lines.append(header)
    lines.append(" " * rl_width + "  " + ("─" * col_width + "  ") * len(col_labels))

    for row in row_labels:
        cells = []
        for col in col_labels:
            key = (row, col)
            counts = seed_statuses.get(key, {})
            n_done = sum(1 for s in seeds if counts.get(s) == "done")
            # Spark-bar: N chars where each char represents n_seeds / N seeds
            # Simple: show n_done/n_seeds as 4 chars
            filled = round(col_width * n_done / n_seeds) if n_seeds else 0
            cell   = "█" * filled + "·" * (col_width - filled)
            if n_done == n_seeds:
                cells.append(cell)
            elif n_done == 0:
                n_part = sum(1 for s in seeds if counts.get(s) == "partial")
                cells.append("░" * min(n_part, col_width) + "·" * (col_width - min(n_part, col_width)))
            else:
                cells.append(cell)
        lines.append(f"  {row:<{rl_width}}" + "  ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suite: main benchmark
# ---------------------------------------------------------------------------

def check_main_benchmark(
    datasets: List[str],
    attacks:  List[str],
    defenses: List[str],
    seeds:    List[int],
    results_dir: str,
    verbose: bool,
    missing_only: bool,
) -> None:
    print(f"\n{'═' * 70}")
    n_exp = len(datasets) * len(attacks) * len(defenses) * len(seeds)
    print(f"  MAIN BENCHMARK   {len(datasets)} datasets × {len(attacks)} attacks × "
          f"{len(defenses)} defenses × {len(seeds)} seeds = {n_exp} runs")
    print(f"{'═' * 70}")

    total: Counter = Counter()
    # dataset → (attack, defense) → seed → status
    by_dataset: Dict[str, Dict[Tuple[str,str], Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))

    for d in datasets:
        nr = DATASET_CFG[d]["num_rounds"]
        for a in attacks:
            for de in defenses:
                for s in seeds:
                    rn  = make_run_name(d, a, de, s)
                    rd  = os.path.join(results_dir, "benchmark", rn)
                    st  = _classify(rd, nr)
                    total[st] += 1
                    by_dataset[d][(a, de)][s] = st

    n_total   = sum(total.values())
    n_done    = total["done"]
    n_partial = total["partial"]
    n_missing = total["missing"]
    pct       = 100 * n_done / n_total if n_total else 0

    print(f"  ✓ done: {n_done:>5}  ({pct:.0f}%)   "
          f"▷ partial: {n_partial:>4}   · missing: {n_missing:>4}")
    print(f"  {_bar(n_done, n_total, 40)}")

    # Per-dataset summary table
    print()
    col_w = max(len(d) for d in datasets)
    print(f"  {'Dataset':<{col_w}}   Done   Part   Miss  Total   ETA (single GPU)")
    print("  " + "─" * 60)

    for d in datasets:
        c     = Counter(st for cell in by_dataset[d].values() for st in cell.values())
        nd    = c["done"]; np_ = c["partial"]; nm = c["missing"]
        nt    = nd + np_ + nm
        eta   = _eta_str(nm + np_, d)
        print(f"  {d:<{col_w}}   {nd:>4}   {np_:>4}   {nm:>4}   {nt:>4}   {eta}")

    # Verbose: per-dataset grid (attacks × defenses, each cell = seed bar)
    if verbose:
        for d in datasets:
            nr    = DATASET_CFG[d]["num_rounds"]
            c     = Counter(st for cell in by_dataset[d].values() for st in cell.values())
            nd    = c["done"]; nt = sum(c.values())
            print(f"\n  ── {d}  ({nd}/{nt}) ──────────────────────────────")
            grid_data = {(a, de): by_dataset[d].get((a, de), {})
                         for a in attacks for de in defenses}
            print(_seed_grid(grid_data, attacks, defenses, seeds, col_width=4))

    # Missing-only: list incomplete runs
    if missing_only:
        print(f"\n  Incomplete runs:")
        for d in datasets:
            nr = DATASET_CFG[d]["num_rounds"]
            for a in attacks:
                for de in defenses:
                    for s in seeds:
                        st = by_dataset[d].get((a, de), {}).get(s, "missing")
                        if st != "done":
                            rn = make_run_name(d, a, de, s)
                            sym = "▷" if st == "partial" else "·"
                            print(f"    {sym}  {rn}")


# ---------------------------------------------------------------------------
# Suite: TopoSentinel benchmark
# ---------------------------------------------------------------------------

def check_topo_benchmark(
    datasets: List[str],
    attacks:  List[str],
    seeds:    List[int],
    results_dir: str,
    verbose: bool,
    missing_only: bool,
) -> None:
    print(f"\n{'═' * 70}")
    n_exp = len(datasets) * len(attacks) * len(seeds)
    print(f"  TOPOSENTINEL BENCHMARK   {len(datasets)} datasets × {len(attacks)} attacks × "
          f"{len(seeds)} seeds = {n_exp} runs")
    print(f"{'═' * 70}")

    total: Counter = Counter()
    by_dataset: Dict[str, Dict[Tuple[str, str], Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))

    for d in datasets:
        nr = DATASET_CFG[d]["num_rounds"]
        for a in attacks:
            for s in seeds:
                rn = f"{d}_{a}_toposentinel_seed{s}"
                rd = os.path.join(results_dir, "toposentinel_benchmark", rn)
                st = _classify(rd, nr)
                total[st] += 1
                by_dataset[d][(a, "topo")][s] = st

    n_total   = sum(total.values())
    n_done    = total["done"]
    n_partial = total["partial"]
    n_missing = total["missing"]
    pct       = 100 * n_done / n_total if n_total else 0

    print(f"  ✓ done: {n_done:>5}  ({pct:.0f}%)   "
          f"▷ partial: {n_partial:>4}   · missing: {n_missing:>4}")
    print(f"  {_bar(n_done, n_total, 40)}")

    print()
    col_w = max(len(d) for d in datasets)
    print(f"  {'Dataset':<{col_w}}   Done   Part   Miss  Total   ETA (single GPU)")
    print("  " + "─" * 60)

    for d in datasets:
        c   = Counter(st for cell in by_dataset[d].values() for st in cell.values())
        nd  = c["done"]; np_ = c["partial"]; nm = c["missing"]
        nt  = nd + np_ + nm
        eta = _eta_str(nm + np_, d)
        print(f"  {d:<{col_w}}   {nd:>4}   {np_:>4}   {nm:>4}   {nt:>4}   {eta}")

    if verbose:
        for d in datasets:
            c  = Counter(st for cell in by_dataset[d].values() for st in cell.values())
            nd = c["done"]; nt = sum(c.values())
            print(f"\n  ── {d}  ({nd}/{nt}) ──────────────────────────────")
            grid_data = {(a, "topo"): by_dataset[d].get((a, "topo"), {}) for a in attacks}
            print(_seed_grid(grid_data, attacks, ["topo"], seeds, col_width=len(seeds)))

    if missing_only:
        print(f"\n  Incomplete runs:")
        for d in datasets:
            nr = DATASET_CFG[d]["num_rounds"]
            for a in attacks:
                for s in seeds:
                    st = by_dataset[d].get((a, "topo"), {}).get(s, "missing")
                    if st != "done":
                        rn  = f"{d}_{a}_toposentinel_seed{s}"
                        sym = "▷" if st == "partial" else "·"
                        print(f"    {sym}  {rn}")


# ---------------------------------------------------------------------------
# Suite: Ablation
# ---------------------------------------------------------------------------

def check_ablation(
    datasets: List[str],
    attacks:  List[str],
    variants: List[str],
    seeds:    List[int],
    results_dir: str,
    verbose: bool,
    missing_only: bool,
) -> None:
    print(f"\n{'═' * 70}")
    n_exp = len(datasets) * len(attacks) * len(variants) * len(seeds)
    print(f"  ABLATION   {len(datasets)} datasets × {len(attacks)} attacks × "
          f"{len(variants)} variants × {len(seeds)} seeds = {n_exp} runs")
    print(f"{'═' * 70}")

    total: Counter = Counter()
    # dataset → (attack, variant) → seed → status
    by_dataset: Dict[str, Dict[Tuple[str, str], Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))

    for d in datasets:
        nr = DATASET_CFG[d]["num_rounds"]
        for a in attacks:
            for v in variants:
                for s in seeds:
                    rn = f"{d}_{a}_topo_{v}_seed{s}"
                    rd = os.path.join(results_dir, "toposentinel_ablation", rn)
                    st = _classify(rd, nr)
                    total[st] += 1
                    by_dataset[d][(a, v)][s] = st

    n_total   = sum(total.values())
    n_done    = total["done"]
    n_partial = total["partial"]
    n_missing = total["missing"]
    pct       = 100 * n_done / n_total if n_total else 0

    print(f"  ✓ done: {n_done:>5}  ({pct:.0f}%)   "
          f"▷ partial: {n_partial:>4}   · missing: {n_missing:>4}")
    print(f"  {_bar(n_done, n_total, 40)}")

    print()
    col_w = max(len(d) for d in datasets)
    print(f"  {'Dataset/Variant':<{col_w + 6}}   Done   Part   Miss  Total   ETA (single GPU)")
    print("  " + "─" * 65)

    for d in datasets:
        for v in variants:
            c   = Counter(st for s in seeds
                          for a in attacks
                          for st in [by_dataset[d].get((a, v), {}).get(s, "missing")])
            nd  = c["done"]; np_ = c["partial"]; nm = c["missing"]
            nt  = nd + np_ + nm
            eta = _eta_str(nm + np_, d)
            label = f"{d}/{v}"
            print(f"  {label:<{col_w + 6}}   {nd:>4}   {np_:>4}   {nm:>4}   {nt:>4}   {eta}")

    if verbose:
        for d in datasets:
            for v in variants:
                c  = Counter(by_dataset[d].get((a, v), {}).get(s, "missing")
                             for a in attacks for s in seeds)
                nd = c["done"]; nt = sum(c.values())
                print(f"\n  ── {d} / {v}  ({nd}/{nt}) ──────────────────────────────")
                grid_data = {(a, v): by_dataset[d].get((a, v), {}) for a in attacks}
                print(_seed_grid(grid_data, attacks, [v], seeds, col_width=len(seeds)))

    if missing_only:
        print(f"\n  Incomplete runs:")
        for d in datasets:
            nr = DATASET_CFG[d]["num_rounds"]
            for a in attacks:
                for v in variants:
                    for s in seeds:
                        st  = by_dataset[d].get((a, v), {}).get(s, "missing")
                        if st != "done":
                            rn  = f"{d}_{a}_topo_{v}_seed{s}"
                            sym = "▷" if st == "partial" else "·"
                            print(f"    {sym}  {rn}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track completion progress across all FL benchmark suites."
    )
    p.add_argument("--results-dir", default="results",
                   help="Root results directory (default: results/).")
    p.add_argument(
        "--suite",
        choices=["benchmark", "toposentinel", "ablation", "all"],
        default="all",
        help="Which suite(s) to report (default: all).",
    )
    p.add_argument("--datasets",  nargs="+", default=None,
                   help="Subset of datasets to check.")
    p.add_argument("--attacks",   nargs="+", default=None,
                   help="Subset of attacks to check.")
    p.add_argument("--defenses",  nargs="+", default=None,
                   help="Subset of defenses to check (main benchmark only).")
    p.add_argument("--seeds",     nargs="+", type=int, default=None,
                   help="Subset of seeds to check.")
    p.add_argument(
        "--verbose", action="store_true",
        help="Show per-dataset seed-completion grids (attacks × defenses/variants).",
    )
    p.add_argument(
        "--missing-only", action="store_true",
        help="List every incomplete run name (useful for manual re-queuing).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = args.datasets  or DATASETS
    attacks  = args.attacks   or ATTACKS
    defenses = args.defenses  or DEFENSES
    seeds    = args.seeds if args.seeds is not None else list(range(N_SEEDS))
    suite    = args.suite
    rdir     = args.results_dir

    if suite in ("benchmark", "all"):
        check_main_benchmark(
            datasets, attacks, defenses, seeds, rdir,
            verbose=args.verbose, missing_only=args.missing_only,
        )

    if suite in ("toposentinel", "all"):
        check_topo_benchmark(
            datasets, attacks, seeds, rdir,
            verbose=args.verbose, missing_only=args.missing_only,
        )

    if suite in ("ablation", "all"):
        check_ablation(
            datasets, attacks, _TOPO_VARIANTS, seeds, rdir,
            verbose=args.verbose, missing_only=args.missing_only,
        )

    print()


if __name__ == "__main__":
    main()
