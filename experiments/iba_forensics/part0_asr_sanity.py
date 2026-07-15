"""Part 0 -- ASR sanity check for the IBA forensics diagnostic.

Verifies that the ASR metric measures an IMPLANTED backdoor rather than a
pre-existing decision-boundary artifact, on CIFAR-10 / VGG-13-noBN.

For each attack (iba, neurotoxin, a3fl) it computes, reusing existing code only:

  round0_asr             ASR of the INITIAL (random-init) global model with the
                         FINAL trained trigger applied. Probes whether the
                         trigger pattern alone biases an un-backdoored model.
  clean_model_asr        ASR of a model trained with NO malicious clients
                         (benign run) with the FINAL trained trigger. The key
                         artifact probe: does a competent clean model already
                         map triggered inputs to the target class?
  benign_global_asr_by_round
                         The real training trajectory's ASR on QUIET
                         (is_attack_round == 0) rounds, read straight from the
                         attack run's metrics.csv (the runner's own per-round
                         ASR, using the round-current trigger). Tests whether
                         the "backdoor" exists/persists when no attacker is
                         active -- pre-attack (0..9) and post-attack (19..29).

Reused, unmodified:
  * experiment.runner.FLRunner                     -- FL training + metrics.csv
  * experiments.gradient_alignment.attack_trigger  -- get_trigger_fn / get_attack_checkpoint
  * datasets.adapter.get_backdoor_test_loader      -- triggered ASR loader
  * fl.server.FedAvgAggregator.evaluate            -- the ASR metric itself

Interpretation flag: if round0_asr or clean_model_asr > 0.2 for an attack, the
trigger exploits a pre-existing artifact and that attack's ASR metric is
suspect -- reported loudly.

Usage:
    python -m experiments.iba_forensics.part0_asr_sanity           # train missing + sanity
    python -m experiments.iba_forensics.part0_asr_sanity --sanity-only
    python -m experiments.iba_forensics.part0_asr_sanity --train-only
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

import experiment  # noqa: F401 -- breaks fl.server <-> experiment import cycle
from experiment.utils import build_adapter, resolve_device
from experiment.runner import FLRunner
from fl.server import FedAvgAggregator
from models import ModelConfig, get_model

from experiments.gradient_alignment.attack_trigger import (
    find_checkpoint,
    get_trigger_fn,
)
from experiments.iba_forensics.configs import (
    ATTACKS,
    DATASET,
    MODEL,
    RESULTS_ROOT,
    RUN_NAMES,
    SEED,
    TARGET_LABEL,
    make_config,
)

ARTIFACT_THRESHOLD = 0.2   # round0/clean ASR above this => suspected artifact


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Step 1 -- train any missing FL runs (benign + 3 attacks)
# ---------------------------------------------------------------------------

def train_runs(force: bool = False) -> None:
    for kind in ["benign", *ATTACKS]:
        run_name = RUN_NAMES[kind]
        ckpt = os.path.join(RESULTS_ROOT, run_name, "final_model.pt")
        if os.path.exists(ckpt) and not force:
            print(f"[train] SKIP {run_name} (final_model.pt exists)")
            continue
        print(f"\n[train] === {run_name} (kind={kind}) ===", flush=True)
        _seed_everything(SEED)
        cfg = make_config(kind)
        FLRunner(cfg).run()


# ---------------------------------------------------------------------------
# Step 2 -- ASR sanity measurements (reuse server.evaluate on triggered loader)
# ---------------------------------------------------------------------------

def _measure_asr(model_cfg, state_dict, trigger_fn, adapter, device) -> float:
    """ASR = fraction of non-target triggered inputs predicted as target,
    computed by the same FedAvg server evaluate() used in training."""
    server = FedAvgAggregator(model=get_model(model_cfg), device=device)
    server.set_params(state_dict)
    loader = adapter.get_backdoor_test_loader(
        trigger_fn=trigger_fn,
        target_label=TARGET_LABEL,
        batch_size=256,
        num_workers=2,
    )
    return float(server.evaluate(loader).metrics["main_accuracy"])


def run_sanity() -> pd.DataFrame:
    device = resolve_device("auto")
    print(f"[sanity] device={device}  seed={SEED}  model={MODEL}  dataset={DATASET}")

    adapter = build_adapter(make_config("benign"))
    model_cfg = ModelConfig.from_adapter(MODEL, adapter)

    # Random-init (round-0) global model -- deterministic.
    _seed_everything(SEED)
    round0_state = {k: v.detach().cpu().clone()
                    for k, v in get_model(model_cfg).state_dict().items()}

    # Clean (benign-trained) global model.
    clean_ckpt = find_checkpoint(RESULTS_ROOT, RUN_NAMES["benign"])
    if clean_ckpt is None:
        raise FileNotFoundError(
            f"benign checkpoint missing under {RESULTS_ROOT}/{RUN_NAMES['benign']}; "
            "run training first."
        )
    clean_state = torch.load(clean_ckpt, map_location="cpu", weights_only=True)

    long_rows = []
    scalar_rows = []
    for attack in ATTACKS:
        print(f"\n[sanity] --- {attack} ---")
        cfg = make_config(attack)
        # Trained trigger loaded from OUR run dir (get_trigger_fn looks up
        # results_dir/<run_name>/trigger.pt for iba/a3fl; patch for neurotoxin).
        trigger_fn = get_trigger_fn(attack, cfg, results_dir=RESULTS_ROOT)

        round0_asr = _measure_asr(model_cfg, round0_state, trigger_fn, adapter, device)
        clean_asr = _measure_asr(model_cfg, clean_state, trigger_fn, adapter, device)
        print(f"[sanity] {attack}: round0_asr={round0_asr:.4f}  "
              f"clean_model_asr={clean_asr:.4f}")

        # Per-round quiet-round ASR straight from the run's metrics.csv.
        metrics_path = os.path.join(RESULTS_ROOT, RUN_NAMES[attack], "metrics.csv")
        mdf = pd.read_csv(metrics_path)
        for _, r in mdf.iterrows():
            long_rows.append({
                "attack": attack,
                "round0_asr": round0_asr,
                "clean_model_asr": clean_asr,
                "round": int(r["round"]),
                "is_attack_round": int(r["is_attack_round"]),
                "benign_global_asr": float(r["asr"]),
            })
        scalar_rows.append({
            "attack": attack,
            "round0_asr": round0_asr,
            "clean_model_asr": clean_asr,
            "quiet_asr_mean": float(mdf.loc[mdf["is_attack_round"] == 0, "asr"].mean()),
            "attack_asr_max": float(mdf.loc[mdf["is_attack_round"] == 1, "asr"].max())
            if (mdf["is_attack_round"] == 1).any() else float("nan"),
        })

    long_df = pd.DataFrame(long_rows)
    out_path = os.path.join(RESULTS_ROOT, "asr_sanity.csv")
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    long_df.to_csv(out_path, index=False)
    print(f"\n[sanity] wrote {out_path}  ({len(long_df)} rows)")

    _print_verdict(pd.DataFrame(scalar_rows))
    return long_df


def _print_verdict(sdf: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART 0 -- ASR SANITY VERDICT")
    print("=" * 70)
    print(f"{'attack':<12} {'round0_asr':>11} {'clean_asr':>10} "
          f"{'quiet_mean':>11} {'attack_max':>11}  flag")
    for _, r in sdf.iterrows():
        artifact = (r["round0_asr"] > ARTIFACT_THRESHOLD
                    or r["clean_model_asr"] > ARTIFACT_THRESHOLD)
        flag = "*** ARTIFACT SUSPECTED ***" if artifact else "ok"
        print(f"{r['attack']:<12} {r['round0_asr']:>11.4f} "
              f"{r['clean_model_asr']:>10.4f} {r['quiet_asr_mean']:>11.4f} "
              f"{r['attack_asr_max']:>11.4f}  {flag}")
    print("=" * 70)
    print(f"Threshold for 'artifact suspected': round0_asr or clean_model_asr "
          f"> {ARTIFACT_THRESHOLD}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--sanity-only", action="store_true")
    p.add_argument("--force-train", action="store_true",
                   help="Retrain even if final_model.pt exists.")
    args = p.parse_args()

    if not args.sanity_only:
        train_runs(force=args.force_train)
    if not args.train_only:
        run_sanity()


if __name__ == "__main__":
    main()
