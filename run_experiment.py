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
    parser.add_argument("--config", required=True, help="Path to the YAML experiment config file.")
    parser.add_argument("--device",     default=None, help="Override config device (auto/cpu/cuda/cuda:N).")
    parser.add_argument("--seed",       type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-dir", default=None, dest="output_dir", help="Override config output_dir.")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

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
