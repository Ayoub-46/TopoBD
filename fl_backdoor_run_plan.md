# FL Backdoor Defense — Experimental Run Plan
*(Detect-and-filter defense; evaluated on robustness, utility, and fairness)*

## 0. Contribution & objective

We propose a **detect-and-filter** defense against backdoor attacks in cross-device FL. Its claimed advantage is a **low false-positive rate (FPR)** — i.e., it rarely excludes benign clients — **especially under non-IID heterogeneity**, where baseline filters over-flag benign-but-atypical clients.

The thesis to demonstrate, on three axes:
- **Robustness** — the defense still catches attacks (high TPR, low resulting ASR), i.e., low FPR is *not* bought by under-detection.
- **Utility** — by discarding fewer honest updates, it preserves more (especially tail-class) data, yielding higher main-task accuracy than over-aggressive baselines under heterogeneity.
- **Fairness** — its false positives do not concentrate on minority/atypical clients, so distinctive clients are not silenced.

Mechanism under test: under non-IID, honest clients with skewed local data produce anomalous-looking updates; baselines confuse "atypical but honest" with "malicious," which is simultaneously a utility loss and a fairness harm, worst at low α. **The non-IID regime is the headline, not a stress test.**

---

## 1. Datasets & models

| Dataset | Model | Classes | Role | Conditions |
|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | 10 | **Headline** — full α-sweep | IID, α∈{1.0,0.5,0.3,0.1} |
| GTSRB | CNN (3-conv) | 43 | Trend confirmation | IID, α∈{1.0,0.5,0.3,0.1} |
| Tiny-ImageNet | ResNet-18 | 200 | Scale/generality | IID, α=0.3 |
| FEMNIST | 2-conv CNN | 62 | External validity (natural het.) | Natural (+ optional synthetic α=0.3) |

Partitions regenerated per seed. FEMNIST is the natural-heterogeneity check, not a point on the Dirichlet axis (its skew is feature/quantity, not label).

---

## 2. Federated configuration (shared)

| Parameter | Value |
|---|---|
| Clients N / sampled per round | 100 / 10 (uniform random) |
| Baseline aggregation | FedAvg (overridden by defense) |
| Client optimizer | SGD, momentum 0.9, weight decay 5e-4 |
| Local epochs | FEMNIST 2 · others 5 |
| Batch size / client LR | 64 / 0.01 (tune per dataset) |
| Seeds | 10 (mean ± std) |
| Malicious clients | 10 (10% of N, fixed identities, ~1 sampled/round) |

---

## 3. Heterogeneity sweep (the central axis)

The headline figure is **FPR (and precision) vs α**: baselines rise as α drops while the proposed method stays flat. Run the full sweep IID → 1.0 → 0.5 → 0.3 → 0.1 on CIFAR-10 and GTSRB; pin Tiny-ImageNet and FEMNIST at their single representative conditions for breadth. Two points cannot distinguish a cliff from a slope — the curve is the evidence.

---

## 4. Attacks

Threat model: 10/100 malicious, random sampling. **Attack is continuous** from end of warmup to the final round (maximizes detection events; enables steady-state utility/fairness measurement). No durability window/tail — durability is not the goal.

| Attack | Trigger | Mechanism | Role for a detector |
|---|---|---|---|
| A3FL | Adversarially optimized | Trigger adapted to global dynamics | Optimized-trigger case |
| Neurotoxin | Static pattern | Bottom-k coordinate projection | Easiest trigger to flag (baseline) |
| IBA | UNet-generated (imperceptible) | Bottom-k params + update near global | **Hardest non-adaptive case** |
| Chameleon | Per-round optimized | Contrastive peer adaptation | Dynamic-trigger case |

**Adaptive attack (near-mandatory):** a white-box attacker that optimizes its update to minimize the *proposed detector's* anomaly score while preserving the backdoor. Start from IBA. Surviving this at low FPR + high TPR is likely the strongest result; omitting it invites the first reviewer objection.

**Fair-comparison controls:** identical target label per dataset, identical per-batch poison fraction, matched trigger-optimization budgets (A3FL/IBA/Chameleon), no naive scaling, and report a trigger-perturbation norm (L2/LPIPS) per attack so detection differences can be read against trigger stealth.

**Sanity check:** reproduce each attack's published numbers under its original participation model before switching to the ~1/round model used here.

---

## 5. Defenses (baselines to beat)

| Defense | Per-client label? | Compared on |
|---|---|---|
| **Proposed (ours)** | Yes | Detection + utility + fairness + robustness |
| FLAME | Yes (cluster filter) | Detection (head-to-head) + all |
| DeepSight | Yes (cluster filter) | Detection (head-to-head) + all |
| Multi-Krum | Selection-based | Robustness + utility (selection-FPR only) |
| Krum | Selection-based | Robustness baseline + NNM ablation |
| NNM + Krum | No (mixes) | Robustness/utility (no detection metric) |
| FedAvg (no defense) | — | ASR/utility reference |

The **detection-quality head-to-head (FPR / precision / F1 / AUPRC) is vs FLAME and DeepSight** — the other explicit filters. Krum/Multi-Krum/NNM+Krum are robustness/utility baselines; report ASR + utility for them and a selection-based FPR only where definable. Keep Krum present so Krum vs NNM+Krum isolates NNM.

---

## 6. Metrics

Two distinct families. Do not conflate them.

### 6a. Detection metrics — of the malicious-client filter
- **TPR** = fraction of *poisoned* updates correctly flagged/discarded during the attack period.
- **FPR** = fraction of *benign* updates incorrectly flagged and discarded from aggregation.

Each is computed *within* its own class → both are **prevalence-independent** (unaffected by the 10% base rate); they are the detection headline. **Counting unit: the (client, round) participation event** — discarding is a per-round action, so each evaluation of a sampled client is one accept/reject decision. Pool events over the attack period. With the continuous attack (§4), a malicious-identity update during the attack period *is* a poisoned update, so identity ≈ ground truth. Optional prevalence-robust single-number summary: MCC or balanced accuracy. Precision/F1 do **not** belong here.

| Detection metric | Applies to |
|---|---|
| TPR, FPR (operating point + confusion matrix) | Ours, FLAME, DeepSight |
| MCC / balanced accuracy (optional summary) | Ours, FLAME, DeepSight |
| ASR (mean & max over run) | All defenses |
| AUPRC / PR curve | Score-based defenses only (hard filters give one operating point) |
| Selection-FPR | Krum, Multi-Krum (where definable); not NNM |

### 6b. Learning-task metrics — of the global model (utility + fairness)
This is where **precision and F1 live**, as per-class classification metrics on the *actual task* — the instrument that exposes the cost of over-filtering. The causal chain: high detection FPR → benign (often minority) contributions discarded → tail-class signal lost → tail-class F1 and convergence both drop. Overall accuracy hides this; macro/per-class does not.

| Learning-task metric | Why |
|---|---|
| Global accuracy (overall) | Top-line utility (head-dominated; not sufficient alone) |
| **Macro-averaged precision / recall / F1** | Each class weighted equally → surfaces tail-class collapse |
| **Per-class F1 for tail classes** | The headline utility-fairness instrument (strongest on GTSRB, FEMNIST, Tiny-ImageNet) |
| **Convergence speed** (rounds-to-target-acc, or AUC of acc-vs-round) | High FPR discards usable signal → slower convergence |
| Per-client accuracy distribution (min, worst-10%, variance) | Client-level fairness |
| Cumulative benign-client inclusion rate | How often each benign client actually reaches the aggregate |
| **FPR stratified by client atypicality** | Links detection FP to which clients are silenced (see §7) |

**Attribution caution:** tail classes degrade under non-IID *even with no defense*, so a raw tail-class F1 drop does not prove over-filtering caused it. Anchor every defense against two references — clean FedAvg (no attack, no filtering = ceiling) and FedAvg-under-attack (backdoor present, no benign data discarded) — and report each defense's tail-class F1 *gap* from them. Decisive version: show the per-class F1 drop is largest for exactly the classes whose data sat in the clients the baselines false-flagged most (ties 6b to §7).

Do **not** force Krum/Multi-Krum/NNM+Krum into detection FPR; after NNM mixing there is no client to label. They are evaluated on ASR + the learning-task metrics.

---

## 7. Key fairness analysis (the differentiating result)

For each benign client, compute (a) its empirical false-positive rate under each defense and (b) an atypicality score = distance of its local label distribution from the global distribution (or its Dirichlet skew). Plot/correlate (a) against (b) per defense.

Expected story: baseline FPs rise steeply with client atypicality (minority clients silenced); the proposed method's FP stays flat. This is the evidence that converts the fairness claim from assertion to result. Report the correlation coefficient/slope per defense per α.

---

## 8. Run matrix (tiered)

**Tier 1 — headline FPR-vs-α curve**
2 datasets (CIFAR-10, GTSRB) × 5 α-levels × 4 attacks × 7 defenses × 10 seeds = **2,800 runs**

**Tier 2 — breadth**
(Tiny-ImageNet IID+α=0.3) + (FEMNIST natural): 3 dataset-conditions × 4 attacks × 7 defenses × 10 seeds = **840 runs**

**Tier 3 — adaptive attack**
2 datasets × {IID, 0.3} × 1 adaptive attack × top-3 defenses (ours, FLAME, DeepSight) × 10 seeds = **120 runs**

**Clean baselines (no attack)**
all dataset-conditions × 7 defenses × 10 seeds for utility/fairness reference ≈ **~900 runs**

**Core total ≈ 4,700 runs.** Optional: synthetic-α=0.3 FEMNIST (skew-type isolation), malicious-fraction sweep {1,5,10,20,40}% on CIFAR-10/α=0.3.

If compute-bound, protect Tier 1 + Tier 3 + the §7 analysis (they carry the contribution); trim Tier-2 breadth first.

---

## 9. Convergence & reporting

- Convergence gate: confirm the clean model has plateaued by the warmup/injection round in **every** condition (α=0.1 is slowest); extend rounds (CIFAR-10 ~240, Tiny-ImageNet ~300, FEMNIST ~180, GTSRB ~200 as starting points) if not.
- Report mean ± std over 10 seeds everywhere.
- Support every "lower FPR under heterogeneity" claim with paired per-dataset deltas across α, not pooled averages.

---

## 10. Figures/tables for the paper

- **Fig 1 (headline):** FPR vs α, one line per defense — the gap opening up under heterogeneity.
- **Fig 2 (fairness):** per-client FPR vs client atypicality, per defense (§7).
- **Fig 3 (utility):** tail-class F1 and macro-F1 vs α; convergence curves (accuracy vs round) per defense; per-client accuracy distribution (box/violin) at α=0.3.
- **Fig 4 (robustness):** ASR and TPR vs α for ours vs FLAME/DeepSight (PR curve/AUPRC only if a defense exposes a tunable score).
- **Table 1:** ours vs FLAME/DeepSight — TPR, FPR, MCC, balanced accuracy at IID and α=0.3 (precision/F1 in appendix with base-rate caveat).
- **Table 2:** adaptive-attack results.
- **Appendix:** trigger-norm table, reproduction sanity, Krum vs NNM+Krum ablation, optional sweeps.

---

## 11. Execution order

1. Clean baselines + convergence gating.
2. Reproduction sanity (original participation).
3. Tier 1 curve: no-defense first, then ours + FLAME + DeepSight, then Krum-family.
4. §7 fairness analysis on Tier-1 outputs.
5. Tier 3 adaptive attack.
6. Tier 2 breadth, then optional sweeps.