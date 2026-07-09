# Running the FL Backdoor Defense Experiment Suite

## Quick reference

| Goal | Command |
|------|---------|
| Single experiment | `python run_experiment.py --config configs/gtsrb_fedavg_a3fl.yaml` |
| Smoke check | `python run_experiment.py --config configs/smoke.yaml --device cpu` |
| Reproduction sanity | `python run_experiment.py --config experiments/plan_sweep/reproduction_sanity/neurotoxin_cifar10.yaml` |
| Full sweep (local) | `python -m experiments.plan_sweep.run_sweep --tier all --device cuda` |
| Full sweep (SLURM) | `sbatch experiments/plan_sweep/slurm_run.sh` |
| Resume sweep | resubmit same sbatch command — already-complete runs are skipped |
| Summary only | `python -m experiments.plan_sweep.run_sweep --summarize-only` |
| Dry-run (what would run) | `python -m experiments.plan_sweep.run_sweep --dry-run` |

---

## Expected run counts

| Tier | Description | Runs |
|------|-------------|------|
| Tier 1 | CIFAR-10 + GTSRB × 5 α × 5 attacks × 7 defenses × 10 seeds | 3,500 |
| Tier 2 | Tiny-ImageNet (IID+0.3) + FEMNIST-natural × 5 attacks × 7 defenses × 10 seeds | 1,050 |
| **Total** | | **~4,550** |

Tier 1 contains the clean baselines (attack=none) within its count.

---

## Output directory structure

```
results/
  plan_sweep/
    {dataset}_{alpha}_{attack}_{defense}_seed{seed}/
      config.json              — full ExperimentConfig
      metrics.csv              — one row per eval round
      detection_raw.csv        — one row per FL round, raw TP/FP/TN/FN
      per_class_metrics.csv    — one row per (round, class)
      convergence.json         — pooled TPR/FPR, AUC, rounds-to-target
      client_atypicality.json  — TV-distance from global label dist per client
      client_inclusion.csv     — cumulative benign-client inclusion rates
      anomaly_scores.csv       — per-round per-client anomaly score (AUPRC)
      final_model.pt           — global model at final round
      trigger.pt               — trained trigger state (A3FL, IBA only)
  plan_sweep_summary.csv       — mean ± std across seeds per (dataset, α, attack, defense)
```

---

## Prerequisite: LEAF FEMNIST data

Natural FEMNIST (Tier 2) requires offline preprocessing. See
[`data/femnist_leaf/README.md`](data/femnist_leaf/README.md) for instructions.

---

## Reproduction-sanity workflow

Before running the full sweep, confirm each attack reaches its published ASR:

```bash
for attack in neurotoxin a3fl iba chameleon; do
    python run_experiment.py \
        --config experiments/plan_sweep/reproduction_sanity/${attack}_cifar10.yaml \
        --device cuda
done
```

Check `results/reproduction_sanity/repro_*/convergence.json` for final ASR.
Published targets (no-defense, sparse participation ~1/round):
- Neurotoxin: ASR ≥ 80%
- A3FL: ASR ≥ 70%
- IBA: ASR ≥ 70%
- Chameleon: ASR ≥ 60% (highly round-dependent)

Adjust attack hyperparameters in the YAML if ASR falls short.

---

## SLURM: GPU assignment

```
GPU 0  cifar10  (all Tier 1 conditions)
GPU 1  gtsrb    (all Tier 1 conditions)
GPU 2  Tier 2   (tiny_imagenet + femnist_leaf)
```

Wall-clock limit: 100 h (`--time=100:00:00`).
Python soft limit: 98 h (`TIME_LIMIT_HOURS=98`), leaving 2 h for summary + cleanup.

Override subsets via `--export`:
```bash
# Run only CIFAR-10, IID, neurotoxin, our defense + no-defense baseline
sbatch --export=ALL,DATASETS="cifar10",ALPHAS="iid",\
ATTACKS="none neurotoxin",DEFENSES="none toposentinel" \
    experiments/plan_sweep/slurm_run.sh
```

---

## Adding a new attack or defense

- **New attack**: add a branch to `experiment/utils.py::build_clients` and register
  the trigger in `attacks/triggers/__init__.py`.  Add a reproduction-sanity config.
  Do NOT modify existing attack clients.

- **New defense**: add a class in `defenses/` inheriting `FedAvgAggregator`.
  Register it in `experiment/utils.py::build_server`.
  Add the key to `experiments/plan_sweep/matrix.py::DEFENSES`.

---

## Key metric locations

| Metric (§6) | File | Column/field |
|-------------|------|--------------|
| TPR (per round) | `metrics.csv` | `defense_tpr` |
| FPR (per round) | `metrics.csv` | `defense_fpr` |
| Raw counts per round | `detection_raw.csv` | `tp, fp, tn, fn` |
| **Pooled TPR (attack period)** | `convergence.json` | `pooled_tpr` |
| **Pooled FPR (all rounds)** | `convergence.json` | `pooled_fpr` |
| ASR (final) | `metrics.csv` | `asr` (last row) |
| ASR (mean / max attack window) | `convergence.json` | `mean_asr_attack_window` |
| Macro F1 | `metrics.csv` | `macro_f1` |
| Per-class F1 | `per_class_metrics.csv` | `f1` grouped by `class` |
| Convergence speed | `convergence.json` | `rounds_to_70pct`, `acc_auc_normalised` |
| Client atypicality | `client_atypicality.json` | `atypicality` |
| Inclusion rate | `client_inclusion.csv` | `inclusion_rate` |
| FPR vs atypicality | `client_inclusion.csv` | `empirical_fpr` + `atypicality` |
| AUPRC scores | `anomaly_scores.csv` | `score`, `is_malicious` |

python -m experiments.gradient_alignment.run_all \
  --config configs/femnist_fedavg_iba.yaml \
  --benign-checkpoint results/femnist_benign_iid/final_model.pt \
  --kappa-weight \
  --skip-exp1 --skip-exp2 \
  --attacks neurotoxin a3fl iba chameleon \
  --n-batches 200 --batch-size 64 \
  --n-gram-samples 1000 --coord-subsample 5000 \
  --output-dir experiments/gradient_alignment/outputs/outputs_femnist \
  --device cuda