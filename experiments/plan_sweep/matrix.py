"""Plan-aligned sweep matrix (§8 of fl_backdoor_run_plan.md).

Tier 1 — Headline FPR-vs-α curve
---------------------------------
CIFAR-10 and GTSRB × 5 α-levels × 5 attacks × 7 defenses × 10 seeds.

  datasets  : cifar10, gtsrb
  α-levels  : iid, 1.0, 0.5, 0.3, 0.1
  attacks   : none (clean baseline), neurotoxin, a3fl, iba, chameleon
  defenses  : none, toposentinel, flame, deepsight, mkrum, krum, nnm_krum
  seeds     : 0–9

Tier 2 — Breadth
-----------------
  Tiny-ImageNet: IID + α=0.3
  FEMNIST (natural): femnist_leaf (natural writer partition)

Tier 3 — Adaptive attack
--------------------------
DEFERRED (out of scope per researcher decision).

Run matrix
----------
Tier 1:  2 × 5 × 5 × 7 × 10  =  3,500 runs
Tier 2:  3 × 5 × 7 × 10       =  1,050 runs  (3 dataset-conditions)
Clean baselines (attack=none) are included in the counts above.

Total ≈ 4,550 runs.

If compute-bound, protect Tier 1 + the §7 fairness analysis first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from experiment.config import AttackConfig, DefenseConfig, ExperimentConfig

# ---------------------------------------------------------------------------
# Per-dataset hyperparameters
# ---------------------------------------------------------------------------

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "cifar10": dict(
        model="resnet18",
        num_rounds=240,             # extended for convergence (§9)
        num_clients=100,
        clients_per_round=10,
        local_epochs=5,
        lr=0.01,
        batch_size=64,
        attack_start=60,            # ~25 % of 240, model partly converged
        num_malicious=10,
        img_h=32, img_w=32,
        in_channels=3,
        patch_pos=(29, 29),
        patch_size=(3, 3),
        patch_color=[1.0, 1.0, 1.0],
    ),
    "gtsrb": dict(
        model="gtsrb_cnn",
        num_rounds=200,
        num_clients=100,
        clients_per_round=10,
        local_epochs=5,
        lr=0.01,
        batch_size=64,
        attack_start=50,
        num_malicious=10,
        img_h=32, img_w=32,
        in_channels=3,
        patch_pos=(29, 29),
        patch_size=(3, 3),
        patch_color=[1.0, 1.0, 1.0],
    ),
    "tiny_imagenet": dict(
        model="resnet18",
        num_rounds=300,
        num_clients=100,
        clients_per_round=10,
        local_epochs=5,
        lr=0.01,
        batch_size=64,
        attack_start=75,
        num_malicious=10,
        img_h=64, img_w=64,
        in_channels=3,
        patch_pos=(61, 61),
        patch_size=(3, 3),
        patch_color=[1.0, 1.0, 1.0],
    ),
    "femnist_leaf": dict(
        model="lenet5",
        num_rounds=180,
        num_clients=100,
        clients_per_round=10,
        local_epochs=2,
        lr=0.01,
        batch_size=64,
        attack_start=45,
        num_malicious=10,
        img_h=28, img_w=28,
        in_channels=1,
        patch_pos=(25, 25),
        patch_size=(3, 3),
        patch_color=[1.0],
    ),
}

# Estimated per-run wall-clock (seconds on A6000 GPU) for time-budget ordering
ESTIMATED_RUN_SECS: Dict[str, int] = {
    "cifar10":      240 * 60,
    "gtsrb":        120 * 60,
    "tiny_imagenet": 360 * 60,
    "femnist_leaf": 100 * 60,
}

# §8 matrix axes
TIER1_DATASETS    = ["cifar10", "gtsrb"]
TIER2_DATASETS    = ["tiny_imagenet", "femnist_leaf"]

# α-levels: "iid" means partition="iid"; floats mean partition="dirichlet"
TIER1_ALPHAS      = ["iid", 1.0, 0.5, 0.3, 0.1]
TIER2_ALPHAS: Dict[str, List] = {
    "tiny_imagenet": ["iid", 0.3],
    "femnist_leaf":  ["natural"],   # natural writer partition
}

ATTACKS   = ["none", "neurotoxin", "a3fl", "iba", "chameleon"]
DEFENSES  = ["none", "toposentinel", "flame", "deepsight", "mkrum", "krum", "nnm_krum"]
N_SEEDS   = 10
EVAL_EVERY = 5          # more frequent than benchmark (every 5 rounds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_run_name(
    dataset: str, alpha, attack: str, defense: str, seed: int
) -> str:
    alpha_str = "iid" if alpha == "iid" else (
        "natural" if alpha == "natural" else f"a{str(alpha).replace('.', '')}"
    )
    return f"{dataset}_{alpha_str}_{attack}_{defense}_seed{seed}"


def make_config(
    dataset: str,
    alpha,           # "iid" | "natural" | float
    attack: str,
    defense: str,
    seed: int,
    results_dir: str = "results",
    device: str = "auto",
) -> ExperimentConfig:
    dc = DATASET_CFG[dataset]

    # Partition strategy and alpha
    if alpha == "iid":
        partition    = "iid"
        d_alpha      = 0.5          # unused for iid
    elif alpha == "natural":
        partition    = "natural_femnist"   # LEAFFEMNISTDataset ignores strategy string
        d_alpha      = 0.5
    else:
        partition    = "dirichlet"
        d_alpha      = float(alpha)

    return ExperimentConfig(
        name=make_run_name(dataset, alpha, attack, defense, seed),
        dataset=dataset,
        data_root="data",
        partition=partition,
        dirichlet_alpha=d_alpha,
        batch_size=dc["batch_size"],
        num_clients=dc["num_clients"],
        num_rounds=dc["num_rounds"],
        clients_per_round=dc["clients_per_round"],
        local_epochs=dc["local_epochs"],
        model=dc["model"],
        lr=dc["lr"],
        weight_decay=5e-4,
        attack=_make_attack_config(attack, dc),
        defense=_make_defense_config(defense, dc),
        eval_every=EVAL_EVERY,
        output_dir=f"{results_dir}/plan_sweep",
        device=device,
        seed=seed,
    )


def get_tier1_matrix(
    datasets: Optional[List[str]] = None,
    alphas: Optional[List] = None,
    attacks: Optional[List[str]] = None,
    defenses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Tuple]:
    """All (dataset, alpha, attack, defense, seed) for Tier 1."""
    ds   = datasets  or TIER1_DATASETS
    alph = alphas    or TIER1_ALPHAS
    atks = attacks   or ATTACKS
    defs = defenses  or DEFENSES
    sds  = seeds if seeds is not None else list(range(N_SEEDS))
    runs = [
        (d, a, atk, de, s)
        for d   in ds
        for a   in alph
        for atk in atks
        for de  in defs
        for s   in sds
    ]
    runs.sort(key=lambda t: ESTIMATED_RUN_SECS.get(t[0], 9999))
    return runs


def get_tier2_matrix(
    datasets: Optional[List[str]] = None,
    attacks: Optional[List[str]] = None,
    defenses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Tuple]:
    """All (dataset, alpha, attack, defense, seed) for Tier 2."""
    ds   = datasets  or TIER2_DATASETS
    atks = attacks   or ATTACKS
    defs = defenses  or DEFENSES
    sds  = seeds if seeds is not None else list(range(N_SEEDS))
    runs = []
    for d in ds:
        for a in TIER2_ALPHAS.get(d, ["iid"]):
            for atk in atks:
                for de in defs:
                    for s in sds:
                        runs.append((d, a, atk, de, s))
    runs.sort(key=lambda t: ESTIMATED_RUN_SECS.get(t[0], 9999))
    return runs


def get_full_matrix(**kwargs) -> List[Tuple]:
    return get_tier1_matrix(**kwargs) + get_tier2_matrix(**kwargs)


# ---------------------------------------------------------------------------
# Internal config builders
# ---------------------------------------------------------------------------

def _make_attack_config(attack: str, dc: Dict[str, Any]) -> AttackConfig:
    if attack == "none":
        return AttackConfig(attack_type="none")

    common = dict(
        num_malicious=dc["num_malicious"],
        target_label=0,
        poison_fraction=0.5,
        attack_start_round=dc["attack_start"],
        attack_end_round=None,          # continuous to final round (§4)
    )

    if attack == "neurotoxin":
        return AttackConfig(
            attack_type="neurotoxin",
            trigger_kwargs={
                "position":       list(dc["patch_pos"]),
                "size":           list(dc["patch_size"]),
                "color":          dc["patch_color"],
                "mask_k_percent": 0.95,
            },
            **common,
        )

    if attack == "a3fl":
        return AttackConfig(
            attack_type="a3fl",
            trigger_sample_size=512,
            trigger_kwargs={
                "position":       [2, 2],
                "size":           [5, 5],
                "trigger_epochs": 5,
                "adv_epochs":     20,
                "trigger_lr":     0.01,
                "lambda_balance": 0.1,
            },
            **common,
        )

    if attack == "iba":
        return AttackConfig(
            attack_type="iba",
            trigger_kwargs={
                "base_features":       32,
                "generator_epochs":    5,
                "generator_lr":        0.001,
                "lambda_noise":        0.01,
                "trigger_sample_size": 512,
            },
            **common,
        )

    if attack == "chameleon":
        return AttackConfig(
            attack_type="chameleon",
            trigger_type="none",
            trigger_kwargs={
                "epsilon":        0.3,
                "num_pgd_steps":  100,
                "pgd_lr":         0.01,
                "peer_pool_size": 128,
                "lambda_sim":     1.0,
            },
            **common,
        )

    raise ValueError(f"Unknown attack: {attack!r}")


def _make_defense_config(defense: str, dc: Dict[str, Any]) -> DefenseConfig:
    if defense == "none":
        return DefenseConfig(defense_type="none")

    c_per_r = dc["clients_per_round"]
    f = max(1, round(0.1 * c_per_r))

    if defense == "toposentinel":
        return DefenseConfig(defense_type="toposentinel")

    if defense == "flame":
        return DefenseConfig(
            defense_type="flame",
            defense_kwargs={"lamda": 0.001, "eta": 1.0},
        )

    if defense == "deepsight":
        return DefenseConfig(defense_type="deepsight")

    if defense == "mkrum":
        return DefenseConfig(
            defense_type="mkrum",
            defense_kwargs={"num_byzantine": f, "num_to_select": c_per_r - f},
        )

    if defense == "krum":
        return DefenseConfig(
            defense_type="mkrum",
            defense_kwargs={"num_byzantine": f, "num_to_select": 1},
        )

    if defense == "nnm_krum":
        return DefenseConfig(
            defense_type="nnm",
            defense_kwargs={"num_byzantine": f, "base_rule": "krum"},
        )

    raise ValueError(f"Unknown defense: {defense!r}")
