"""Full benchmark configuration matrix.

One `ExperimentConfig` is built programmatically for every
(dataset, attack, defense, seed) cell — no YAML files required.

Dataset-specific convergence parameters
----------------------------------------
* mnist   — 100 rounds, 50 clients, 5/round, 2 local epochs (fast convergence)
* femnist — 150 rounds, 100 clients, 10/round, 2 local epochs (62 classes, moderate)
* gtsrb   — 200 rounds, 100 clients, 10/round, 5 local epochs (43 classes, CNN)
* cifar10 — 200 rounds, 100 clients, 10/round, 5 local epochs (ResNet, slower)

Attack window
-------------
Always 50 rounds, placed at 25 %–25 %+50 of total rounds so the model
is partially converged before the attack and has recovery time after.

Malicious ratio
---------------
Fixed at 10 % of total clients (not clients-per-round).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from experiment.config import AttackConfig, DefenseConfig, ExperimentConfig


# ---------------------------------------------------------------------------
# Per-dataset hyperparameters
# ---------------------------------------------------------------------------

DATASET_CFG: Dict[str, Dict[str, Any]] = {
    "mnist": dict(
        model="lenet5",
        num_rounds=100,
        num_clients=50,
        clients_per_round=5,
        local_epochs=2,
        lr=0.01,
        batch_size=64,
        attack_start=25,
        attack_end=75,
        num_malicious=5,        # 10 % of 50
        img_h=28, img_w=28,
        in_channels=1,
        patch_pos=(25, 25),
        patch_size=(3, 3),
        patch_color=[1.0],
    ),
    "femnist": dict(
        model="lenet5",
        num_rounds=150,
        num_clients=100,
        clients_per_round=10,
        local_epochs=2,
        lr=0.01,
        batch_size=64,
        attack_start=37,        # ~25 % of 150
        attack_end=87,
        num_malicious=10,       # 10 % of 100
        img_h=28, img_w=28,
        in_channels=1,
        patch_pos=(25, 25),
        patch_size=(3, 3),
        patch_color=[1.0],
    ),
    "gtsrb": dict(
        model="gtsrb_cnn",
        num_rounds=200,
        num_clients=100,
        clients_per_round=10,
        local_epochs=5,
        lr=0.01,
        batch_size=64,
        attack_start=50,        # 25 % of 200
        attack_end=100,
        num_malicious=10,
        img_h=32, img_w=32,
        in_channels=3,
        patch_pos=(29, 29),
        patch_size=(3, 3),
        patch_color=[1.0, 1.0, 1.0],
    ),
    "cifar10": dict(
        model="resnet18",
        num_rounds=200,
        num_clients=100,
        clients_per_round=10,
        local_epochs=5,
        lr=0.01,
        batch_size=64,
        attack_start=50,
        attack_end=100,
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
        attack_start=75,        # 25 % of 300
        attack_end=125,
        num_malicious=10,
        img_h=64, img_w=64,
        in_channels=3,
        patch_pos=(61, 61),     # bottom-right corner of 64×64
        patch_size=(3, 3),
        patch_color=[1.0, 1.0, 1.0],
    ),
}

# Conservative GPU-hour estimates per run (seconds); used for time-budget ordering.
ESTIMATED_RUN_SECS: Dict[str, int] = {
    "mnist":         20 * 60,
    "femnist":       50 * 60,
    "gtsrb":         90 * 60,
    "cifar10":      180 * 60,
    "tiny_imagenet": 300 * 60,
}

DATASETS  = ["mnist", "femnist", "gtsrb", "cifar10", "tiny_imagenet"]
ATTACKS   = ["none", "patch", "neurotoxin", "a3fl", "iba", "chameleon"]
# "krum" = single Krum (MKrumServer with num_to_select=1)
# "nnm_krum" = NNM + Krum base rule
DEFENSES  = ["none", "toposentinel", "flame", "deepsight", "mkrum", "krum", "nnm_krum"]
N_SEEDS   = 10
EVAL_EVERY = 10   # evaluate every N rounds; final round always evaluated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_run_name(dataset: str, attack: str, defense: str, seed: int) -> str:
    return f"{dataset}_{attack}_{defense}_seed{seed}"


def make_config(
    dataset: str,
    attack: str,
    defense: str,
    seed: int,
    results_dir: str = "results",
    device: str = "auto",
) -> ExperimentConfig:
    dc = DATASET_CFG[dataset]
    return ExperimentConfig(
        name=make_run_name(dataset, attack, defense, seed),
        dataset=dataset,
        data_root="data",
        partition="iid",
        batch_size=dc["batch_size"],
        num_clients=dc["num_clients"],
        num_rounds=dc["num_rounds"],
        clients_per_round=dc["clients_per_round"],
        local_epochs=dc["local_epochs"],
        model=dc["model"],
        lr=dc["lr"],
        weight_decay=1e-4,
        attack=_make_attack_config(attack, dc),
        defense=_make_defense_config(defense, dc),
        eval_every=EVAL_EVERY,
        output_dir=f"{results_dir}/benchmark",
        device=device,
        seed=seed,
    )


def get_run_matrix(
    datasets: Optional[List[str]] = None,
    attacks: Optional[List[str]] = None,
    defenses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[str, str, str, int]]:
    """All (dataset, attack, defense, seed) tuples, shortest-first for SLURM."""
    ds   = datasets  or DATASETS
    atks = attacks   or ATTACKS
    defs = defenses  or DEFENSES
    sds  = seeds if seeds is not None else list(range(N_SEEDS))

    tuples = [
        (d, a, de, s)
        for d  in ds
        for a  in atks
        for de in defs
        for s  in sds
    ]
    tuples.sort(key=lambda t: ESTIMATED_RUN_SECS[t[0]])
    return tuples


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
        attack_end_round=None,      # continuous to final round (§4)
    )

    if attack == "patch":
        return AttackConfig(
            attack_type="patch",
            trigger_kwargs={
                "position": list(dc["patch_pos"]),
                "size":     list(dc["patch_size"]),
                "color":    dc["patch_color"],
            },
            **common,
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
                "base_features":    32,
                "generator_epochs": 5,
                "generator_lr":     0.001,
                "lambda_noise":     0.01,
                "trigger_sample_size": 512,
            },
            **common,
        )

    if attack == "chameleon":
        return AttackConfig(
            attack_type="chameleon",
            trigger_type="none",        # sample-specific, no fixed trigger for ASR
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

    # Assumed Byzantine clients per round: 10 % of clients_per_round, min 1.
    c_per_r = dc["clients_per_round"]
    f = max(1, round(0.1 * c_per_r))

    if defense == "toposentinel":
        return DefenseConfig(defense_type="toposentinel")

    if defense == "mkrum":
        return DefenseConfig(
            defense_type="mkrum",
            defense_kwargs={
                "num_byzantine": f,
                "num_to_select": c_per_r - f,
            },
        )

    if defense == "krum":
        # Single Krum: select exactly 1 update (lowest Krum score)
        return DefenseConfig(
            defense_type="mkrum",
            defense_kwargs={
                "num_byzantine": f,
                "num_to_select": 1,
            },
        )

    if defense == "flame":
        return DefenseConfig(
            defense_type="flame",
            defense_kwargs={"lamda": 0.001, "eta": 1.0},
        )

    if defense == "nnm":
        return DefenseConfig(
            defense_type="nnm",
            defense_kwargs={"num_byzantine": f, "base_rule": "cwtm"},
        )

    if defense == "nnm_krum":
        return DefenseConfig(
            defense_type="nnm",
            defense_kwargs={"num_byzantine": f, "base_rule": "krum"},
        )

    if defense == "deepsight":
        return DefenseConfig(defense_type="deepsight")

    raise ValueError(f"Unknown defense: {defense!r}")
