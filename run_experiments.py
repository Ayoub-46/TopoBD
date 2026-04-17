"""Validation experiment script.

Run from the project root::

    python run_experiments.py

Results are written to ``results/<experiment_name>/``:
    metrics.csv   — per-round tabular log
    metrics.json  — same data in JSON
    config.json   — full experiment config for reproducibility

Experiments
-----------
1. Benign FedAvg baseline
2. Patch attack  (15% malicious clients, bottom-right red trigger)
3. A3FL attack   (15% malicious clients, adaptive learnable trigger)
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment.log", mode="w"),
    ],
)
os.makedirs("results", exist_ok=True)

from experiment import ExperimentConfig, AttackConfig, FLRunner

# ---------------------------------------------------------------------------
# Shared FL hyper-parameters
# ---------------------------------------------------------------------------
SHARED = dict(
    dataset="cifar10",          # swap to "synthetic" if download is blocked
    data_root="data",
    partition="dirichlet",
    dirichlet_alpha=0.5,
    model="resnet18",
    num_clients=100,
    clients_per_round=10,
    local_epochs=5,
    num_rounds=200,
    batch_size=64,
    lr=0.01,
    weight_decay=1e-4,
    eval_every=5,
    output_dir="results",
    seed=42,
)

# ---------------------------------------------------------------------------
# Experiment 1 — Benign baseline
# ---------------------------------------------------------------------------
cfg_baseline = ExperimentConfig(
    name="01_baseline",
    attack=AttackConfig(attack_type="none"),
    **SHARED,
)

# ---------------------------------------------------------------------------
# Experiment 2 — Patch attack
# ---------------------------------------------------------------------------
cfg_patch = ExperimentConfig(
    name="02_patch_attack",
    attack=AttackConfig(
        attack_type="patch",
        num_malicious=15,           # 15% of 100 clients
        target_label=0,             # "airplane" in CIFAR-10
        poison_fraction=0.5,
        attack_start_round=0,
        trigger_kwargs={
            "position": (29, 29),   # bottom-right 3×3 patch
            "size": (3, 3),
            "color": (1.0, 0.0, 0.0),
        },
    ),
    **SHARED,
)

# ---------------------------------------------------------------------------
# Experiment 3 — A3FL attack
# ---------------------------------------------------------------------------
cfg_a3fl = ExperimentConfig(
    name="03_a3fl_attack",
    attack=AttackConfig(
        attack_type="a3fl",
        num_malicious=15,
        target_label=0,
        poison_fraction=0.5,
        attack_start_round=10,      # warm up the global model first
        trigger_sample_size=256,
        trigger_kwargs={
            "position": (2, 2),
            "size": (5, 5),
            "trigger_epochs": 10,
            "trigger_lr": 0.01,
            "lambda_balance": 0.1,
            "adv_epochs": 50,
            "adv_lr": 0.01,
        },
    ),
    **SHARED,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    experiments = [
        ("Benign baseline",  cfg_baseline),
        ("Patch attack",     cfg_patch),
        ("A3FL attack",      cfg_a3fl),
    ]

    all_metrics = {}
    for label, cfg in experiments:
        print(f"\n{'=' * 62}")
        print(f"  Running: {label}")
        print(f"{'=' * 62}")
        runner = FLRunner(cfg)
        all_metrics[label] = runner.run()

    # ---- Side-by-side final-round comparison --------------------------------
    print(f"\n{'=' * 62}")
    print("  Final-round comparison")
    print(f"{'=' * 62}")
    print(f"  {'Experiment':<24}  {'CleanAcc':>9}  {'ASR':>9}")
    print(f"  {'-'*24}  {'-'*9}  {'-'*9}")
    for label, m in all_metrics.items():
        asr_str = f"{m.final_asr * 100:>8.2f}%" if not __import__('math').isnan(m.final_asr) else "       —"
        print(f"  {label:<24}  {m.final_clean_acc * 100:>8.2f}%  {asr_str}")
    print(f"{'=' * 62}\n")