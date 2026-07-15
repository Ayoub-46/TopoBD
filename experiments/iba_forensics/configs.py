"""Shared configuration for the IBA-forensics diagnostic runs.

Defines the modest, deterministic CIFAR-10 / VGG-13-noBN federation used by
Part 0 (ASR sanity) and Parts 1-2. The run *names* deliberately match the keys
in ``experiments.gradient_alignment.attack_trigger._ATTACK_CHECKPOINT_MAPS`` and
``_BENIGN_RUNS`` for ``cifar10`` so that ``get_trigger_fn`` / ``get_attack_checkpoint``
resolve our trained triggers/checkpoints when called with
``results_dir=RESULTS_ROOT`` -- i.e. zero reimplementation of trigger loading.

Trigger hyper-parameters are copied verbatim from the existing (resnet18)
``results/cifar10_fedavg_{iba,neurotoxin,a3fl}/config.json`` so the attacks are
exactly as validated; only the model, federation scale, and attack window differ.
"""

from __future__ import annotations

from experiment.config import AttackConfig, DefenseConfig, ExperimentConfig

# ---------------------------------------------------------------------------
# Where diagnostic runs live. Kept separate from the production resnet18 runs.
# ---------------------------------------------------------------------------
RESULTS_ROOT = "results/iba_forensics"

# ---------------------------------------------------------------------------
# Fixed, logged experiment invariants
# ---------------------------------------------------------------------------
SEED = 42
DATASET = "cifar10"
MODEL = "vgg13_nobn"        # no BatchNorm, bias=True on every conv (per request)
TARGET_LABEL = 0

# Modest federation / schedule (stated in the deliverable).
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 10
NUM_MALICIOUS = 4
LOCAL_EPOCHS = 2
NUM_ROUNDS = 30
ATTACK_START = 10          # pre-attack quiet rounds: 0..9
ATTACK_END = 15            # post-attack quiet rounds: 16..29 (persistence test)
EVAL_EVERY = 1             # dense per-round ASR trajectory in metrics.csv
BATCH_SIZE = 64
LR = 0.01
WEIGHT_DECAY = 5e-4

ATTACKS = ["iba", "neurotoxin", "a3fl"]

# Run-name map: MUST match attack_trigger._ATTACK_CHECKPOINT_MAPS["cifar10"].
RUN_NAMES = {
    "benign":     "cifar10_benign_iid",
    "iba":        "cifar10_fedavg_iba",
    "neurotoxin": "cifar10_fedavg_neurotoxin",
    "a3fl":       "cifar10_fedavg_a3fl",
}

# Trigger kwargs verbatim from the existing resnet18 configs.
_TRIGGER_KWARGS = {
    "iba": {
        "alpha": 0.2, "lambda_noise": 0.01, "generator_epochs": 5,
        "generator_lr": 0.001, "base_features": 32, "trigger_sample_size": 512,
    },
    "neurotoxin": {
        "position": [29, 29], "size": [3, 3], "color": [1.0, 1.0, 1.0],
        "mask_k_percent": 0.95,
    },
    "a3fl": {
        "position": [2, 2], "size": [5, 5], "trigger_epochs": 5,
        "trigger_lr": 0.01, "lambda_balance": 0.1, "adv_epochs": 20, "adv_lr": 0.01,
    },
}
_TRIGGER_TYPE = {"iba": "iba", "neurotoxin": "patch", "a3fl": None}
_TRIGGER_SAMPLE_SIZE = {"iba": 512, "neurotoxin": 512, "a3fl": 256}


def make_config(kind: str) -> ExperimentConfig:
    """Build the ExperimentConfig for ``kind`` in {"benign","iba","neurotoxin","a3fl"}."""
    if kind == "benign":
        attack = AttackConfig(attack_type="none", num_malicious=0)
    elif kind in ATTACKS:
        attack = AttackConfig(
            attack_type=kind,
            num_malicious=NUM_MALICIOUS,
            target_label=TARGET_LABEL,
            poison_fraction=0.5,
            attack_start_round=ATTACK_START,
            attack_end_round=ATTACK_END,
            trigger_type=_TRIGGER_TYPE[kind],
            trigger_kwargs=dict(_TRIGGER_KWARGS[kind]),
            trigger_sample_size=_TRIGGER_SAMPLE_SIZE[kind],
        )
    else:
        raise ValueError(f"unknown run kind '{kind}'")

    return ExperimentConfig(
        name=RUN_NAMES[kind],
        dataset=DATASET,
        data_root="data",
        partition="iid",
        dirichlet_alpha=0.5,
        batch_size=BATCH_SIZE,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        clients_per_round=CLIENTS_PER_ROUND,
        local_epochs=LOCAL_EPOCHS,
        model=MODEL,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        attack=attack,
        defense=DefenseConfig(defense_type="none"),
        eval_every=EVAL_EVERY,
        output_dir=RESULTS_ROOT,
        device="auto",
        seed=SEED,
    )
