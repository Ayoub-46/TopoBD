"""Train all FL experiments whose checkpoints are required by run_all.py.

For each required run, checks whether results/<name>/final_model.pt already
exists and skips it if so.  Runs everything that is missing sequentially.

Checkpoints required by run_all.py (via attack_trigger.ATTACK_CHECKPOINT_MAP):
    results/gtsrb_benign_iid/final_model.pt       → Experiment 2 (γ)
    results/gtsrb_fedavg_neurotoxin/final_model.pt → Experiment 1, neurotoxin
    results/gtsrb_fedavg_a3fl/final_model.pt       → Experiment 1, a3fl
    results/gtsrb_fedavg_iba/final_model.pt        → Experiment 1, iba
    results/gtsrb_fedavg_chameleon/final_model.pt  → Experiment 1, chameleon

Usage
-----
    # From the repo root:
    python experiments/gradient_alignment/train_prerequisites.py

    # Skip runs whose checkpoints already exist (default behaviour):
    python experiments/gradient_alignment/train_prerequisites.py --skip-existing

    # Force re-run everything even if checkpoints exist:
    python experiments/gradient_alignment/train_prerequisites.py --force

    # Run only a subset:
    python experiments/gradient_alignment/train_prerequisites.py \\
        --runs iba chameleon

    # Run training then immediately launch the diagnostics:
    python experiments/gradient_alignment/train_prerequisites.py --then-run-diagnostics
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_repo_root, "results", "experiment.log"), mode="a"),
    ],
)

os.makedirs(os.path.join(_repo_root, "results"), exist_ok=True)

from experiment import ExperimentConfig, FLRunner


# ---------------------------------------------------------------------------
# Run registry — (run_name, config_path)
# ---------------------------------------------------------------------------

_REPO = _repo_root

GTSRB_RUNS = [
    {
        "name":    "gtsrb_benign_iid",
        "config":  os.path.join(_REPO, "configs", "gtsrb_benign_iid.yaml"),
        "purpose": "benign baseline (required by Experiment 2 / γ)",
    },
    {
        "name":    "gtsrb_fedavg_neurotoxin",
        "config":  os.path.join(_REPO, "configs", "gtsrb_fedavg_neurotoxin.yml"),
        "purpose": "Neurotoxin attack checkpoint (Experiment 1)",
    },
    {
        "name":    "gtsrb_fedavg_a3fl",
        "config":  os.path.join(_REPO, "configs", "gtsrb_fedavg_a3fl.yaml"),
        "purpose": "A3FL attack checkpoint (Experiment 1)",
    },
    {
        "name":    "gtsrb_fedavg_iba",
        "config":  os.path.join(_REPO, "configs", "gtsrb_fedavg_iba.yaml"),
        "purpose": "IBA attack checkpoint (Experiment 1)",
    },
    {
        "name":    "gtsrb_fedavg_chameleon",
        "config":  os.path.join(_REPO, "configs", "gtsrb_fedavg_chameleon.yaml"),
        "purpose": "Chameleon attack checkpoint (Experiment 1)",
    },
]

FEMNIST_RUNS = [
    {
        "name":    "femnist_benign_iid",
        "config":  os.path.join(_REPO, "configs", "femnist_benign_iid.yaml"),
        "purpose": "benign baseline (required by Experiment 2 / γ)",
    },
    {
        "name":    "femnist_fedavg_neurotoxin",
        "config":  os.path.join(_REPO, "configs", "femnist_fedavg_neurotoxin.yaml"),
        "purpose": "Neurotoxin attack checkpoint (Experiment 1)",
    },
    {
        "name":    "femnist_fedavg_a3fl",
        "config":  os.path.join(_REPO, "configs", "femnist_fedavg_a3fl.yaml"),
        "purpose": "A3FL attack checkpoint (Experiment 1)",
    },
    {
        "name":    "femnist_fedavg_iba",
        "config":  os.path.join(_REPO, "configs", "femnist_fedavg_iba.yaml"),
        "purpose": "IBA attack checkpoint (Experiment 1)",
    },
    {
        "name":    "femnist_fedavg_chameleon",
        "config":  os.path.join(_REPO, "configs", "femnist_fedavg_chameleon.yaml"),
        "purpose": "Chameleon attack checkpoint (Experiment 1)",
    },
]

CIFAR10_RUNS = [
    {
        "name":    "cifar10_benign_iid",
        "config":  os.path.join(_REPO, "configs", "cifar10_benign_iid.yaml"),
        "purpose": "benign baseline (required by Experiment 2 / γ)",
    },
    {
        "name":    "cifar10_fedavg_neurotoxin",
        "config":  os.path.join(_REPO, "configs", "cifar10_fedavg_neurotoxin.yaml"),
        "purpose": "Neurotoxin attack checkpoint (Experiment 1)",
    },
    {
        "name":    "cifar10_fedavg_a3fl",
        "config":  os.path.join(_REPO, "configs", "cifar10_fedavg_a3fl.yaml"),
        "purpose": "A3FL attack checkpoint (Experiment 1)",
    },
    {
        "name":    "cifar10_fedavg_iba",
        "config":  os.path.join(_REPO, "configs", "cifar10_fedavg_iba.yaml"),
        "purpose": "IBA attack checkpoint (Experiment 1)",
    },
    {
        "name":    "cifar10_fedavg_chameleon",
        "config":  os.path.join(_REPO, "configs", "cifar10_fedavg_chameleon.yaml"),
        "purpose": "Chameleon attack checkpoint (Experiment 1)",
    },
]

# Default to GTSRB for backward compatibility; overridden by --dataset.
RUNS = GTSRB_RUNS

_ALL_RUNS = {r["name"]: r for r in GTSRB_RUNS + CIFAR10_RUNS + FEMNIST_RUNS}
RUN_BY_NAME = {r["name"]: r for r in RUNS}   # updated in main() after arg parsing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def checkpoint_exists(run_name: str, results_dir: str) -> bool:
    return os.path.exists(os.path.join(results_dir, run_name, "final_model.pt"))


def _fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_one(run: dict, results_dir: str) -> None:
    """Run a single FL experiment and print a summary."""
    name = run["name"]
    cfg_path = run["config"]

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"Create it before running this script."
        )

    cfg = ExperimentConfig.from_yaml(cfg_path)

    print(f"\n{'=' * 66}")
    print(f"  Run      : {name}")
    print(f"  Purpose  : {run['purpose']}")
    print(f"  Config   : {cfg_path}")
    print(f"  Attack   : {cfg.attack.attack_type}  |  Defense: {cfg.defense.defense_type}")
    print(f"  Rounds   : {cfg.num_rounds}  |  Clients: {cfg.num_clients}/{cfg.clients_per_round}")
    print(f"{'=' * 66}\n")

    t0 = time.time()
    runner = FLRunner(cfg)
    tracker = runner.run()
    elapsed = time.time() - t0

    asr_str = (
        f"{tracker.final_asr * 100:.2f}%"
        if not math.isnan(tracker.final_asr)
        else "—"
    )
    print(f"\n{'=' * 66}")
    print(f"  Done: {name}  ({_fmt_time(elapsed)})")
    print(f"  Clean accuracy : {tracker.final_clean_acc * 100:.2f}%")
    print(f"  ASR            : {asr_str}")
    ckpt_path = os.path.join(results_dir, name, "final_model.pt")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"{'=' * 66}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FL experiments whose checkpoints are needed by run_all.py."
    )
    p.add_argument(
        "--dataset", choices=["gtsrb", "cifar10", "femnist"], default="gtsrb",
        help="Dataset to train prerequisite runs for (default: gtsrb).",
    )
    p.add_argument(
        "--runs", nargs="+", default=None,
        help="Subset of run names to consider (default: all five for the dataset).",
    )
    p.add_argument(
        "--results-dir", default=os.path.join(_REPO, "results"),
        help="Root results directory.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-train even if final_model.pt already exists.",
    )
    p.add_argument(
        "--then-run-diagnostics", action="store_true",
        help="After training, immediately launch run_all.py with default settings.",
    )
    p.add_argument(
        "--diagnostic-n-batches", type=int, default=50,
        help="--n-batches passed to run_all.py when --then-run-diagnostics is set.",
    )
    p.add_argument(
        "--diagnostic-batch-size", type=int, default=64,
        help="--batch-size passed to run_all.py when --then-run-diagnostics is set.",
    )
    p.add_argument(
        "--diagnostic-n-per-class", type=int, default=100,
        help="--n-per-class passed to run_all.py when --then-run-diagnostics is set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Resolve run list for the requested dataset
    _dataset_map = {"cifar10": CIFAR10_RUNS, "femnist": FEMNIST_RUNS, "gtsrb": GTSRB_RUNS}
    dataset_runs = _dataset_map[args.dataset]
    run_by_name  = {r["name"]: r for r in dataset_runs}
    run_names    = args.runs if args.runs else list(run_by_name.keys())

    # Validate any explicitly requested run names
    unknown = [n for n in run_names if n not in run_by_name]
    if unknown:
        # Allow cross-dataset run names in _ALL_RUNS
        for n in unknown:
            if n in _ALL_RUNS:
                run_by_name[n] = _ALL_RUNS[n]
            else:
                print(f"ERROR: unknown run '{n}'. "
                      f"Available for --dataset {args.dataset}: {list(run_by_name)}")
                sys.exit(1)

    # ---- Status report ------------------------------------------------------
    print(f"\nDataset: {args.dataset}")
    print("\nCheckpoint status:")
    for run_name in run_names:
        exists = checkpoint_exists(run_name, results_dir)
        status = "PRESENT (will skip)" if exists and not args.force else (
            "PRESENT (will re-run)" if exists and args.force else "MISSING (will train)"
        )
        print(f"  {run_name:<40s}  {status}")
    print()

    # ---- Train missing runs -------------------------------------------------
    trained: list[str] = []
    skipped: list[str] = []
    failed:  list[str] = []
    t_total = time.time()

    for run_name in run_names:
        run = run_by_name[run_name]

        if checkpoint_exists(run_name, results_dir) and not args.force:
            print(f"SKIP  {run_name}  (final_model.pt exists)")
            skipped.append(run_name)
            continue

        try:
            run_one(run, results_dir)
            trained.append(run_name)
        except Exception as exc:
            print(f"\nERROR training '{run_name}': {exc}")
            import traceback
            traceback.print_exc()
            failed.append(run_name)

    # ---- Final summary ------------------------------------------------------
    print(f"\n{'=' * 66}")
    print(f"  Training complete  ({_fmt_time(time.time() - t_total)} total)")
    print(f"  Trained : {trained or '—'}")
    print(f"  Skipped : {skipped or '—'}")
    if failed:
        print(f"  FAILED  : {failed}")
    print(f"{'=' * 66}")

    # ---- Optional: launch diagnostics ---------------------------------------
    if args.then_run_diagnostics:
        if failed:
            print("\nWARNING: some runs failed; diagnostics may use fallback checkpoints.")

        _benign_run  = f"{args.dataset}_benign_iid"
        _iba_config  = os.path.join(_REPO, "configs", f"{args.dataset}_fedavg_iba.yaml")
        benign_ckpt  = os.path.join(results_dir, _benign_run, "final_model.pt")
        diag_output  = os.path.join(
            _REPO, "experiments", "gradient_alignment",
            f"outputs_{args.dataset}",
        )

        print(f"\nLaunching diagnostic experiments (run_all.py) for {args.dataset}...")
        from experiments.gradient_alignment.run_all import main as run_diag_main

        sys.argv = [
            "run_all.py",
            "--config",               _iba_config,
            "--benign-checkpoint",    benign_ckpt,
            "--attacks",              "neurotoxin", "a3fl", "iba", "chameleon",
            "--n-batches",            str(args.diagnostic_n_batches),
            "--batch-size",           str(args.diagnostic_batch_size),
            "--n-per-class",          str(args.diagnostic_n_per_class),
            "--output-dir",           diag_output,
            "--results-dir",          results_dir,
            "--device",               "auto",
        ]
        run_diag_main()


if __name__ == "__main__":
    main()
