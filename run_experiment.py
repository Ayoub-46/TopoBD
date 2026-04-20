"""Run a single FL experiment from a YAML config file.

Usage::

    python run_experiment.py configs/gtsrb_mkrum_a3fl.yaml
"""

import argparse
import logging
import os
import sys

os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment.log", mode="a"),
    ],
)

from experiment import ExperimentConfig, FLRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FL experiment from a YAML config.")
    parser.add_argument("--config", help="Path to the YAML experiment config file.")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    print(f"\n{'=' * 62}")
    print(f"  Experiment : {cfg.name}")
    print(f"  Dataset    : {cfg.dataset}  |  Model: {cfg.model}")
    print(f"  Attack     : {cfg.attack.attack_type}  |  Defense: {cfg.defense.defense_type}")
    print(f"  Rounds     : {cfg.num_rounds}  |  Clients: {cfg.num_clients}")
    print(f"{'=' * 62}\n")

    runner = FLRunner(cfg)
    metrics = runner.run()

    print(f"\n{'=' * 62}")
    print(f"  Results — {cfg.name}")
    print(f"{'=' * 62}")

    import math
    asr_str = (
        f"{metrics.final_asr * 100:.2f}%"
        if not math.isnan(metrics.final_asr)
        else "—"
    )
    print(f"  Clean accuracy : {metrics.final_clean_acc * 100:.2f}%")
    print(f"  ASR            : {asr_str}")
    print(f"  Results saved  : results/{cfg.name}/")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
