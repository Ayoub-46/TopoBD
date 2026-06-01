# Gap Analysis — FL Backdoor Defense Framework
*Audit of code vs. `fl_backdoor_run_plan.md` (the plan). Written before any Phase 2 implementation.*

---

## 0. How to read this document

Each requirement from the plan is mapped to one of three statuses:

| Symbol | Meaning |
|--------|---------|
| ✅ **Present & matches** | Code does what the plan requires |
| ⚠️ **Present but diverges** | Feature exists; behaviour differs from plan in a described way |
| ❌ **Missing** | Not implemented at all |

Ambiguities and plan-vs-code conflicts are called out explicitly in §5.
The proposed Phase 2 implementation plan is in §6.

---

## 1. Code map (entry points and key files)

| Component | File | Key class / function |
|-----------|------|---------------------|
| CLI entry point | `run_experiments.py` | **DELETED** (git status `D`) |
| Config dataclasses | `experiment/config.py` | `ExperimentConfig`, `AttackConfig`, `DefenseConfig` |
| Main training loop | `experiment/runner.py` | `FLRunner.__init__`, `FLRunner.run` |
| Per-round metrics | `experiment/metrics.py` | `RoundMetrics`, `MetricsTracker` |
| Dataset adapters | `datasets/*.py` | `DatasetAdapter` ABC + per-dataset subclasses |
| Partitioning | `datasets/adapter.py:292-345` | `partition_iid`, `partition_dirichlet` |
| Client factory | `experiment/utils.py:127-401` | `build_clients` |
| Server / aggregation | `fl/server.py` | `FedAvgAggregator`, `AggregationResult` |
| Defense servers | `defenses/*.py` | `FlameServer`, `DeepSightServer`, `MKrumServer`, `NNMServer`, `TopoSentinelServer` |
| Detection accounting | `experiment/utils.py:400-448` | `DetectionResult` |
| Sweep launcher | `experiments/benchmark/` | `matrix.py`, `run_benchmark.py`, `slurm_run.sh` |
| Gradient diagnostics | `experiments/gradient_alignment/` | `run_all.py` (separate; not part of main sweep) |

---

## 2. Federated configuration — §2 of plan

| Plan requirement | Status | Notes |
|-----------------|--------|-------|
| N=100 clients, 10/round | ✅ | `num_clients=100`, `clients_per_round=10` defaults; enforced in runner |
| Uniform random client sampling | ✅ | `random.sample(list(self.clients.keys()), cfg.clients_per_round)` — runner.py:177 |
| Baseline aggregation: FedAvg | ✅ | `FedAvgAggregator` in `fl/server.py`; overridden by defense |
| SGD, momentum 0.9 | ✅ | `fl/client.py`: `torch.optim.SGD(..., momentum=0.9)` |
| Weight decay **5e-4** | ⚠️ | `ExperimentConfig.weight_decay` defaults to **1e-4**, not 5e-4. Benchmark matrix also uses 1e-4. |
| Local epochs: FEMNIST 2, others 5 | ✅ | `matrix.py` sets `local_epochs=2` for mnist/femnist, `local_epochs=5` for gtsrb/cifar10/tiny_imagenet |
| Batch size 64, LR 0.01 | ✅ | Both are benchmark matrix defaults |
| 10 seeds (mean ± std) | ✅ | `N_SEEDS=10` in `matrix.py`; seeds 0–9 |
| 10 malicious (10%, fixed identities) | ✅ | `assign_malicious_ids(num_clients, num_malicious, seed)` in utils.py; deterministic per seed |
| ~1 malicious sampled/round | ✅ | 10/100 × 10/round → 1 in expectation; correctly noted in plan |

**Action required:** change `weight_decay` default to `5e-4` or set it explicitly in sweep configs.

---

## 3. Datasets & models — §1 of plan

| Plan requirement | Status | Notes |
|-----------------|--------|-------|
| CIFAR-10 / ResNet-18 | ✅ | Implemented; benchmark uses ResNet-18 |
| GTSRB / CNN (3-conv) | ✅ | `gtsrb_cnn` / `GTSRBNet` in models/ |
| Tiny-ImageNet / ResNet-18 | ✅ | Dataset adapter implemented (`datasets/tiny_imagenet.py`) |
| FEMNIST / 2-conv CNN | ⚠️ | FEMNIST dataset uses EMNIST-byclass. **No 2-conv CNN registered** — benchmark uses `lenet5`. Plan calls for a "2-conv CNN" (may differ from LeNet-5). |
| FEMNIST: **natural** writer-based heterogeneity | ⚠️ | **Major divergence.** `datasets/femnist.py` explicitly replaces writer-based partitioning with IID/Dirichlet. There is no `partition="natural_femnist"` strategy. Plan §1 says "Natural (+ optional synthetic α=0.3)". |
| Partitions regenerated per seed | ✅ | Both IID and Dirichlet create a seeded `np.random.RandomState(seed)`; same seed → same partition. |
| α sweep on CIFAR-10 and GTSRB: IID + {1.0, 0.5, 0.3, 0.1} | ❌ | **Benchmark matrix uses only IID for all datasets** (`partition="iid"` hardcoded in `matrix.py:159`). No α sweep is implemented. |
| Tiny-ImageNet: IID + α=0.3 only | ❌ | Same; benchmark uses IID only. |

---

## 4. Attacks — §4 of plan

| Plan requirement | Status | Notes |
|-----------------|--------|-------|
| A3FL (adversarially optimized trigger) | ✅ | `attacks/a3fl_client.py`, `A3FLTrigger` |
| Neurotoxin (static pattern + bottom-k) | ✅ | `attacks/neurotoxin_client.py` |
| IBA (UNet-generated, imperceptible) | ✅ | `attacks/iba_client.py`, `IBATrigger` |
| Chameleon (per-round PGD, peer adaptation) | ✅ | `attacks/chameleon_client.py` |
| **Adaptive attack** (white-box, minimize proposed detector score) | ❌ | **Not implemented.** Chameleon uses PGD but is NOT aware of the detector score; it minimizes target-class loss + peer similarity. A detector-evasion variant starting from IBA must be built as a new attack client. Algorithm not in repo — **must ask researcher for the optimization objective.** |
| Attack continuous from end-of-warmup to final round | ⚠️ | `attack_end_round=None` in `AttackConfig` does mean "run to end." But the **benchmark matrix** sets explicit `attack_end=100` for most datasets (out of 200 rounds) — attack covers only half the run, not the full second half. Plan explicitly says "no durability window/tail." Fix: set `attack_end_round=None` in all sweep configs. |
| Same target label across datasets | ✅ | Benchmark matrix sets `target_label=0` everywhere |
| Same poison fraction per batch | ✅ | `poison_fraction=0.5` across all attacks |
| Trigger-perturbation norm (L2/LPIPS) per attack | ❌ | Not logged anywhere. IBA computes L2 internally (`iba.py:142`) but doesn't expose it. Patch/Neurotoxin have static triggers (norm can be computed offline). A3FL/IBA/Chameleon need logging. |
| "patch" attack | ⚠️ | Implemented in benchmark matrix but **not in the plan's attack list**. Plan has 4 attacks: A3FL, Neurotoxin, IBA, Chameleon. "patch" is a plain static-patch baseline not in §4 — either treat as a sub-case of Neurotoxin or exclude from plan sweep. |
| Reproduction sanity check (original participation model) | ❌ | No sanity-check harness or published-number target specified in code. |
| Fair-comparison controls (trigger budgets) | ❌ | Trigger budget caps (A3FL epochs, IBA epochs) are in configs but no systematic enforcement or reporting of matched budgets exists. |

---

## 5. Defenses — §5 of plan

| Plan defense | Code status | Defense key | filter_updates()? | Anomaly score exposed? |
|-------------|-------------|-------------|-------------------|----------------------|
| **Proposed (TopoSentinel)** | ✅ Implemented | `"toposentinel"` | ✅ Yes | ❌ Binary only |
| FLAME | ✅ Implemented | `"flame"` | ✅ Yes | ❌ Binary only |
| DeepSight | ✅ Implemented | `"deepsight"` | ✅ Yes | ❌ Binary only |
| Multi-Krum | ✅ Implemented | `"mkrum"` (with `num_to_select>1`) | ✅ Yes | ❌ Binary only |
| **Krum (single)** | ⚠️ | `"mkrum"` with `num_to_select=1` | ✅ Yes | ❌ Binary only |
| **NNM + Krum** | ⚠️ | `"nnm"` with `base_rule="krum"` (default is `"cwtm"`) | ❌ NNM has no filter_updates | ❌ N/A |
| FedAvg (no defense) | ✅ | `"none"` | ❌ N/A | ❌ N/A |

**Gaps in benchmark matrix (`experiments/benchmark/matrix.py`):**
- `"toposentinel"` is absent from `DEFENSES = ["none", "mkrum", "flame", "nnm", "deepsight"]` — the proposed defense is excluded from its own benchmark. ❌
- No separate `"krum"` entry (single Krum, `num_to_select=1`). ❌
- `"nnm"` uses `base_rule="cwtm"` by default; plan calls for NNM+Krum specifically. ❌

**AUPRC/PR curves:**
- Plan §6a lists "AUPRC / PR curve" for score-based defenses.
- **None of the five detection defenses expose a per-client continuous anomaly score.** They all return binary `rejected_ids` sets.
- FLAME uses HDBSCAN cluster assignment; DeepSight uses combined distance matrix; MKrum uses Krum scores; TopoSentinel uses bottleneck distance + bias distance.
- To support AUPRC, each defense's `filter_updates()` would need to return per-client scores alongside the binary decision. This requires a thin wrapper (not modifying the core algorithm).

---

## 6. Metrics — §6 of plan

### 6a. Detection metrics

| Plan requirement | Status | Notes |
|----------------|--------|-------|
| TPR = poisoned updates correctly flagged (per participation event) | ✅ | `DetectionResult.tpr` in `utils.py:424-431`. Correct formula: `|rejected ∩ malicious| / |malicious_selected|` |
| FPR = benign updates incorrectly discarded (per participation event) | ✅ | `DetectionResult.compute_fpr` in `utils.py:433-448`. Correct: `|rejected − malicious| / n_selected_benign` |
| Counting unit: (client, round) participation event | ✅ | Each call to `filter_updates()` operates on one round's sampled clients. |
| Pool events over the **attack period** | ⚠️ | Per-round TPR/FPR are logged (NaN outside attack window). **No pooled-period aggregate is computed.** The raw per-round data is in the CSV so it can be recomputed, but the plan's stated metric (pooled) should be derived and reported. |
| Raw TP/FP/TN/FN counts per round | ❌ | Only scalar TPR and FPR written. DetectionResult has the sets needed to derive raw counts, but they are not persisted. Plan §6a says "log raw confusion counts per round so anything can be re-derived." |
| MCC / balanced accuracy (optional) | ❌ | Not computed. Can be derived from raw counts once those are added. |
| ASR (mean & max over run) | ⚠️ | Per-round ASR is in the CSV; mean and max can be computed post-hoc but are not pre-computed. `MetricsTracker.final_asr` returns only the last-round value. |
| AUPRC / PR curve (score-based only) | ❌ | No defense exposes a continuous score (see §5). |
| Selection-FPR for Krum/Multi-Krum | ⚠️ | MKrumServer has `filter_updates()` so the standard FPR is computed. This is correct for selection-based Krum — "selection-FPR" is equivalent here. |

### 6b. Learning-task metrics

| Plan requirement | Status | Notes |
|----------------|--------|-------|
| Global accuracy (overall) | ✅ | `clean_acc` in `RoundMetrics`; `server.evaluate()` returns `main_accuracy` |
| **Macro-averaged precision / recall / F1** | ❌ | `server.evaluate()` returns only `loss` and `main_accuracy`. No per-class computation exists. |
| **Per-class F1 (tail classes)** | ❌ | Completely absent. Requires storing per-class TP/FP/FN counts from the test set. |
| **Convergence speed** (rounds-to-target-acc, AUC of acc-vs-round) | ❌ | Not computed. The full acc-vs-round trajectory is in the CSV (computable post-hoc), but no target-acc threshold or AUC is derived online. |
| Per-client accuracy distribution (min, worst-10%, variance) | ❌ | No per-client evaluation exists. `server.evaluate()` evaluates only the global model on the global test set. A separate eval pass over each client's local test set (or a held-out per-client split) is needed. |
| Cumulative benign-client inclusion rate | ❌ | No cumulative tracking. Would require a running counter per client in the runner. |
| FPR stratified by client atypicality | ❌ | No atypicality score computed, and no per-client FP tracking. |
| Attribution baselines (clean FedAvg, FedAvg-under-attack) | ⚠️ | These cells exist implicitly in the matrix (attack="none"/defense="none" and attack=X/defense="none"). They are not explicitly labeled or isolated as attribution references. |

---

## 7. Sweep launcher — §8 of plan

| Plan §8 requirement | Status | Notes |
|--------------------|--------|-------|
| Tier 1: CIFAR-10, GTSRB × 5 α-levels × 4 attacks × 7 defenses × 10 seeds | ❌ | Benchmark is 5 datasets × 6 attacks × **5 defenses** × 10 seeds, all IID. No α sweep. |
| Tier 2: Tiny-ImageNet (IID+α=0.3) + FEMNIST natural | ❌ | No Dirichlet sweep, no natural FEMNIST partitioning. |
| Tier 3: adaptive attack × top-3 defenses | ❌ | Adaptive attack not implemented. |
| Clean baselines (no attack) × all conditions | ⚠️ | attack="none" exists in matrix. Not isolated or labeled as "clean baseline" group. |
| Structured per-run output path | ✅ | `results/benchmark/{run_name}/` with `metrics.csv`, `final_model.pt`, `config.json`. |
| Graceful stop + resume on SLURM | ✅ | `run_benchmark.py` checks `is_complete()` before each run; SIGTERM handler present. |
| Parallel execution (3 GPUs) | ✅ | `slurm_run.sh` launches 3 background workers with `CUDA_VISIBLE_DEVICES`. |
| Determinism per run | ⚠️ | Global seed is set once; per-round client sampling advances a shared RNG state (not independently seeded per round). Runs are reproducible given the same seed and execution order, but per-round sampling is not independently re-seeded if the run is interrupted and resumed partway. |

---

## 8. Per-seed determinism — invariant verification

| Invariant | Status | Notes |
|-----------|--------|-------|
| Seed data partition | ✅ | `np.random.RandomState(seed)` in both partition functions |
| Seed client sampling (each round) | ⚠️ | Global `random.seed(seed)` set once at run start. Per-round sampling is deterministic *within an uninterrupted run*, but not independently re-seeded per round (i.e., stopping mid-run and resuming won't reproduce the original sampling sequence for remaining rounds). |
| Seed model init | ✅ | `torch.manual_seed(seed)` before model construction |
| Seed attack-client assignment | ✅ | `assign_malicious_ids(..., seed)` uses `random.Random(seed)` |
| Seed malicious client's local state | ✅ | Attack client created with `seed=cfg.seed + cid` |
| Full cuDNN determinism | ✅ | `deterministic=True`, `benchmark=False` |

---

## 9. Missing infrastructure

| Item | Status |
|------|--------|
| CLI entry point (`run_experiments.py`) | ❌ DELETED (git `D run_experiments.py`). No way to launch a single YAML-configured run from the command line. Needs to be re-created or replaced. |
| Smoke-run config (10 rounds, 20 clients, 5/round, 2 malicious, 2 seeds) | ❌ Not present. |
| `RUN.md` (how to launch the full matrix, expected run count, output paths) | ❌ Not present. |
| `GAP_ANALYSIS.md` | → This file. |

---

## 10. Explicit ambiguities and plan-vs-code conflicts

**A1 — TopoSentinel is already implemented.**
Plan §5 says "if it isn't already in the repo, add a clearly-marked stub and ASK me for the algorithm." TopoSentinel IS in the repo (`defenses/toposentinel.py`) with a full TDA-based implementation using `persistent_homology/`. Treating it as the proposed defense going forward. No stub needed; no algorithm question needed. Do NOT modify it.

**A2 — FEMNIST model: plan says "2-conv CNN," benchmark uses LeNet-5.**
Are these the same architecture? LeNet-5 is a 2-conv CNN for grayscale images, so they may be equivalent. But if the plan intends a specific (different) architecture, this needs clarification. **Flagging for researcher decision.**

**A3 — FEMNIST natural partitioning.**
The plan calls for natural heterogeneity. True LEAF/natural-FEMNIST partitioning requires the EMNIST dataset split by original writer identity (each client = one writer). The current dataset uses EMNIST-byclass with synthetic splits. Implementing true natural FEMNIST requires either (a) downloading the LEAF benchmark data, or (b) using the official EMNIST writer split. This is a significant implementation effort. **Flagging for researcher decision: acceptable to use Dirichlet α=0.3 as a proxy for natural FEMNIST, or is the writer-based split required?**

**A4 — Adaptive attack algorithm.**
Plan §4 says "Start from IBA." The optimization objective is not specified in the plan. To minimize the TopoSentinel anomaly score, the attacker would need white-box access to the topological analysis pipeline and an end-to-end differentiable evasion objective. **Must ask researcher for the specific optimization objective and whether gradient-based or black-box optimization is intended.**

**A5 — Krum (single) vs Multi-Krum.**
Plan §5 lists them as separate baselines. Code has one class (`MKrumServer`) parameterized by `num_to_select`. Single Krum = `num_to_select=1`. This is handled by separate defense config entries, not separate classes — acceptable. The benchmark matrix needs two entries: `krum` (num_to_select=1) and `mkrum` (num_to_select=n-f).

**A6 — "Pooling" TPR/FPR over the attack period.**
Plan §6a: "Pool events over the attack period." The current code correctly computes TPR/FPR per round (with NaN outside the attack window). The pooled period metric = `sum(TP) / sum(TP+FN)` over all attack-window rounds. This requires raw counts (TP, FP, TN, FN) per round — the current code only stores derived scalars. **Raw count logging must be added to RoundMetrics.**

**A7 — "Detection metrics during attack period only."**
Current code: `defense_tpr` is NaN when `true_malicious_this_round` is empty (outside attack window or no malicious selected). This means TPR is undefined (not just ignored) in benign rounds. FPR, however, is meaningful even in benign rounds (any false positives are real). The plan says "detection metrics during attack period only." Clarification: does FPR during benign rounds count? Under continuous attack (attack_end_round=None), this distinction disappears after warmup.

**A8 — AUPRC: no continuous scores.**
The plan's AUPRC requirement ("AUPRC / PR curve — score-based defenses only") assumes defenses expose per-client anomaly scores. None currently do. Adding score-return to `filter_updates()` would require a new return type or side-channel. **Recommend: add an optional `client_scores: Dict[int, float]` field to `DetectionResult` and have each detection defense populate it (thin wrapper, algorithm unchanged).**

**A9 — "Patch" attack not in plan.**
Benchmark matrix has `"patch"` as an attack. Plan §4 lists only A3FL, Neurotoxin, IBA, Chameleon. Patch is redundant (Neurotoxin uses a patch trigger but adds gradient masking). **Recommend: keep `"patch"` in benchmark as a static-trigger sanity check but exclude from the plan sweep matrix.**

**A10 — Per-round client sampling reproducibility.**
Current: `random.seed(seed)` once at run start. Per-round `random.sample()` advances the global state. If two seeds produce the same first K rounds but diverge at round K+1 (e.g., due to different attack branch code paths taken), their sampling sequences will diverge. For strict per-round reproducibility, consider `random.Random(seed + round_idx).sample(...)`. **Low priority but worth fixing for exact replication.**

---

## 11. Proposed Phase 2 implementation plan

> Ordered by priority as specified in the instructions. No code written until approved.

### P1 — Metrics layer (highest priority, blocks everything else)

**P1.1 — Raw detection counts per round**
- Extend `RoundMetrics` with `defense_tp: int`, `defense_fp: int`, `defense_tn: int`, `defense_fn: int` (all default 0 / -1 for N/A).
- Compute from `DetectionResult.rejected_ids`, `true_malicious`, `n_selected` in runner.py. Log alongside TPR/FPR.
- Add `attack_period_tpr` and `attack_period_fpr` to `MetricsTracker` summary (pooled over attack window).

**P1.2 — Per-class F1 and macro metrics**
- Add `evaluate_per_class(loader)` to `FedAvgAggregator` (thin wrapper over the model's forward pass, no change to existing `evaluate()`). Returns `{class_id: {"tp": int, "fp": int, "fn": int}}`.
- Add to `RoundMetrics`: `macro_f1: float`, `macro_precision: float`, `macro_recall: float`. Write a separate `per_class_f1.csv` (one row per round per class; same output dir).
- Only call this on eval rounds (same as current `evaluate()`). For num_classes ≥ 200 (Tiny-ImageNet), amortize cost.

**P1.3 — Convergence speed**
- Derive post-hoc in `MetricsTracker.save()`: compute `rounds_to_target_acc` (round index when `clean_acc` first exceeds a configurable threshold, default 0.70) and `acc_auc` (trapezoidal integral of acc-vs-round over full run). Append to a `convergence.json` in the run directory.

**P1.4 — Client atypicality score**
- At setup time, compute per-client label distribution from partition indices (available in `build_clients` via the adapter). Store as a dict `{client_id: label_counts}`.
- Compute atypicality = Total Variation distance (or KL) of client's label distribution from the global empirical distribution.
- Write `client_atypicality.json` to the run output directory.

**P1.5 — Cumulative benign-client inclusion rate**
- In runner.py, maintain `cumulative_benign_included: Dict[int, int]` — count of rounds each benign client's update survived filtering.
- Write to `client_inclusion.csv` (client_id, rounds_selected, rounds_included, inclusion_rate) at run end.

**P1.6 — Per-client FPR stratified by atypicality**
- Combine P1.1 per-client FP events (track per-client FP flags in runner) with P1.4 atypicality.
- Write `client_fp_atypicality.csv` (client_id, n_sampled, n_fp, fpr_empirical, atypicality).
- Correlation coefficient/slope computable from this CSV.

**P1.7 — Anomaly scores (thin wrappers for AUPRC)**
- Add optional `client_scores: Optional[Dict[int, float]] = None` to `DetectionResult`.
- For FLAME: expose the per-client distance-to-cluster-centroid as score.
- For DeepSight: expose the combined distance-matrix row mean as score.
- For MKrum: expose the Krum score as score.
- For TopoSentinel: expose the per-client bias-distance as score.
- **Constraint: no change to the detection algorithm itself** — only add score extraction and recording.

**P1.8 — ASR summary statistics**
- Add `asr_mean` and `asr_max` (over attack window) to `MetricsTracker.summary`.

### P2 — Config & partitioning

**P2.1 — Weight decay correction**
- Update `ExperimentConfig.weight_decay` default to `5e-4` (or set explicitly in sweep).

**P2.2 — Continuous attack schedule**
- In sweep configs, set `attack_end_round=None` (or omit, as None is already the default). Update benchmark matrix to remove explicit `attack_end` and use `attack_end_round=None`.

**P2.3 — α sweep support**
- `dirichlet_alpha` is already in `ExperimentConfig`. The sweep launcher only needs to enumerate it.
- IID mode = `partition="iid"` (ignore `dirichlet_alpha`). Already supported.

**P2.4 — FEMNIST natural partitioning** *(pending researcher decision on A3)*
- If required: add `partition="natural_femnist"` strategy in `adapter._make_partitions`. Reads the EMNIST writer-split file to build `{writer_id: [sample_indices]}`, then assigns one writer per client (truncate/pad to num_clients).
- If not required: use Dirichlet α=0.3 as proxy and note it in the appendix.

**P2.5 — Re-create CLI entry point**
- A minimal `run_experiment.py` (accepts `--config path/to/yaml`, loads `ExperimentConfig.from_yaml`, runs `FLRunner`). Deleted file must be restored.

### P3 — Sweep launcher (§8 matrix)

**P3.1 — Plan-aligned matrix definition**
New `experiments/plan_sweep/matrix.py` (separate from the existing benchmark module):
- Tier 1: {cifar10, gtsrb} × {iid, α=1.0, α=0.5, α=0.3, α=0.1} × {none, neurotoxin, a3fl, iba, chameleon} × {none, toposentinel, flame, deepsight, mkrum, krum, nnm_krum} × seeds 0-9 = 2 × 5 × 5 × 7 × 10 = **3,500 runs** *(plan says 2,800 using 4 attacks; add "none"/clean-baseline outside this count → see below)*
- Clean baselines: all dataset-conditions × 7 defenses × 10 seeds (attack=none)
- Tier 2: {tiny_imagenet IID, tiny_imagenet α=0.3, femnist natural/α=0.3} × same attack/defense/seed grid
- Tier 3: adaptive attack (pending P4)

**P3.2 — Defense key mapping**
| Defense key | Config | filter_updates? |
|------------|--------|----------------|
| `"none"` | FedAvg | No |
| `"toposentinel"` | TopoSentinelServer | Yes |
| `"flame"` | FlameServer | Yes |
| `"deepsight"` | DeepSightServer | Yes |
| `"mkrum"` | MKrumServer(num_to_select=n-f) | Yes |
| `"krum"` | MKrumServer(num_to_select=1) | Yes |
| `"nnm_krum"` | NNMServer(base_rule="krum") | No |

**P3.3 — SLURM job-array launcher**
- Generate one config file per run (YAML), written to `experiments/plan_sweep/configs/`.
- `sbatch --array=0-N` launcher that indexes into the config list.
- Graceful stop and skip-if-complete logic (reuse from existing benchmark runner).

### P4 — Adaptive attack *(blocked on A4 — must ask researcher)*

**STOP:** cannot implement this without the specific evasion objective from the researcher.
Placeholder:
- New file `attacks/adaptive_client.py` with `class AdaptiveIBAClient(IBAClient)`.
- Override `local_train()` to add a penalty term `λ * detector_score(update)`.
- The `detector_score` function signature will be determined by the researcher's answer to A4.

### P5 — Attribution baselines

**P5.1** — Already implicit in the sweep matrix (attack="none" + defense="none" = clean FedAvg; attack=X + defense="none" = FedAvg-under-attack). Expose them as named group labels in the summary CSV.

**P5.2** — In the per-class F1 analysis (P1.2), compute the gap: `F1_defense - F1_clean_fedavg` and `F1_defense - F1_fedavg_under_attack` per class. Write to analysis CSV.

### P6 — Smoke-run validation (before any full-matrix launch)

Config: 1 dataset (cifar10), 10 rounds, 20 clients, 5/round, 2 malicious, 2 seeds, α=0.3, all 7 defenses, 1 attack (neurotoxin).
Expected: all metrics populate; no NaN where values are expected; per-class F1 has 10 rows; client_inclusion.csv has 20 rows; client_fp_atypicality.csv populated.
**STOP after smoke run passes. Do not launch full matrix.**

---

## 12. Summary table of all gaps

| # | Requirement | Status | Priority |
|---|-------------|--------|----------|
| 1 | Weight decay 5e-4 | ⚠️ default is 1e-4 | P2.1 |
| 2 | FEMNIST natural partitioning | ⚠️/❌ synthetic only | P2.4 (decision pending) |
| 3 | α sweep (IID + 1.0/0.5/0.3/0.1) | ❌ IID only | P3.1 |
| 4 | Continuous attack (end_round=None) | ⚠️ benchmark sets explicit end | P2.2 |
| 5 | TopoSentinel in benchmark defenses | ❌ excluded from matrix | P3.2 |
| 6 | Krum (single) as separate defense | ❌ not in matrix | P3.2 |
| 7 | NNM+Krum (base_rule="krum") | ❌ NNM uses CWTM | P3.2 |
| 8 | Raw TP/FP/TN/FN counts per round | ❌ not logged | P1.1 |
| 9 | Pooled TPR/FPR over attack period | ❌ not aggregated | P1.1 |
| 10 | MCC / balanced accuracy | ❌ not computed | P1.1 (derives from #8) |
| 11 | Per-class F1 (tail classes) | ❌ not computed | P1.2 |
| 12 | Macro precision/recall/F1 | ❌ not computed | P1.2 |
| 13 | Convergence speed (rounds-to-target, AUC) | ❌ not computed | P1.3 |
| 14 | Client atypicality score | ❌ not computed | P1.4 |
| 15 | Cumulative benign-client inclusion rate | ❌ not tracked | P1.5 |
| 16 | Per-client FPR stratified by atypicality | ❌ not tracked | P1.6 |
| 17 | AUPRC / PR curve (anomaly scores) | ❌ no scores exposed | P1.7 |
| 18 | ASR mean & max over run | ⚠️ only last-round ASR | P1.8 |
| 19 | Adaptive attack | ❌ not implemented | P4 (blocked on A4) |
| 20 | Trigger perturbation norm (L2/LPIPS) | ❌ not logged | P3 / analysis pass |
| 21 | CLI entry point (`run_experiment.py`) | ❌ deleted | P2.5 |
| 22 | Smoke-run config | ❌ not present | P6 |
| 23 | `RUN.md` | ❌ not present | P3 |
| 24 | Per-round client sampling (per-round seed) | ⚠️ global seed only | P2 (low priority) |
| 25 | Per-client accuracy distribution | ❌ no per-client eval | P1 (defer; expensive) |
| 26 | Attribution baselines explicitly labeled | ⚠️ implicit in matrix | P5 |

---

## 13. Questions requiring researcher decisions before Phase 2

| ID | Question |
|----|---------|
| A2 | Is the FEMNIST model "2-conv CNN" specifically different from LeNet-5, or is LeNet-5 acceptable? |
| A3 | Is true writer-based FEMNIST natural partitioning required, or is Dirichlet α=0.3 acceptable as a proxy? |
| A4 | What is the optimization objective for the adaptive attack? Specifically: how should the evasion penalty be defined given TopoSentinel's bottleneck-distance and bias-distance computations? Is gradient-based or black-box optimization intended? |
| A7 | Should FPR be reported for benign rounds (rounds with no malicious sampled) under the continuous-attack protocol? Under attack_end_round=None, there are no purely-benign rounds after warmup — this distinction may be moot. Confirm. |

**STOP — awaiting approval before implementing anything.**
