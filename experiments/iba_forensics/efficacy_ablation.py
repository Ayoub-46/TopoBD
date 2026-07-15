"""IBA efficacy ablation: is femnist's failure about 1-channel, #classes, or ε?

Runs (undefended FedAvg, so we measure the attack's ceiling):
  * MNIST   (1-channel, 10-class, lenet5) at the default ε regime — the control
    that separates "1-channel" from "62-class": same channels+model as femnist,
    but cifar's class count.
  * femnist (1-channel, 62-class, lenet5) with a LARGER floor ε — tests whether
    a bigger trigger budget rescues it (trigger-strength hypothesis).

Reports max/final attacked ASR (from the run's own per-round eval) and, where a
benign checkpoint exists, clean_model_asr (artifact check).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import experiment  # noqa: F401
from experiment.config import AttackConfig, DefenseConfig, ExperimentConfig
from experiment.runner import FLRunner


def _seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _force_epsilon(runner, eps):
    """Pin the shared IBA trigger to a constant floor ε (disable the decay)."""
    for cid in runner.malicious_ids:
        if cid in runner.clients:
            t = runner.clients[cid].config.trigger
            t.eps_0 = eps; t.eps_hat = eps; t.epsilon = eps


def _asr_summary(name):
    m = pd.read_csv(os.path.join("results", "analysis", name, "metrics.csv"))
    atk = m[m.is_attack_round == 1]
    return {
        "run": name,
        "clean_acc": float(m.clean_acc.iloc[-1]),
        "asr_final": float(m.asr.iloc[-1]),
        "asr_max": float(m.asr.max()),
        "asr_attackwin_mean": float(atk.asr.mean()) if len(atk) else float("nan"),
    }


def _iba_config(name, dataset, model, num_clients, num_malicious, num_rounds,
                attack_start, local_epochs, lr, wd, batch=64) -> ExperimentConfig:
    return ExperimentConfig(
        name=name, dataset=dataset, data_root="data", partition="iid",
        batch_size=batch, num_clients=num_clients, num_rounds=num_rounds,
        clients_per_round=min(10, num_clients), local_epochs=local_epochs,
        model=model, lr=lr, weight_decay=wd,
        attack=AttackConfig(
            attack_type="iba", num_malicious=num_malicious, target_label=0,
            poison_fraction=0.5, attack_start_round=attack_start,
            attack_end_round=num_rounds, trigger_type="iba",
            trigger_kwargs={"base_features": 32, "generator_epochs": 5,
                            "generator_lr": 0.001},
        ),
        defense=DefenseConfig(defense_type="none"),
        eval_every=2, output_dir=os.path.join("results", "analysis"),
        device="auto", seed=0,
    )


def run_mnist():
    """MNIST 1-channel 10-class control (default ε regime = eps_0=0.3 decay→8/255)."""
    _seed(0)
    cfg = _iba_config("mnist_iba_efficacy", "mnist", "lenet5",
                      num_clients=10, num_malicious=2, num_rounds=30,
                      attack_start=8, local_epochs=2, lr=0.01, wd=1e-4)
    print(f"\n=== MNIST/iba efficacy (1-ch, 10-class, lenet5, default ε regime) ===",
          flush=True)
    FLRunner(cfg).run()
    return _asr_summary("mnist_iba_efficacy")


def run_femnist_eps(eps):
    """femnist with a larger constant floor ε (trigger-strength test)."""
    from experiments.benchmark.run_toposentinel import _make_topo_config
    _seed(0)
    cfg = _make_topo_config("femnist", "iba", 0, "results", "auto")
    cfg.defense = DefenseConfig(defense_type="none")
    cfg.output_dir = os.path.join("results", "analysis")
    cfg.name = f"femnist_iba_eps{eps:.2f}_efficacy"
    print(f"\n=== femnist/iba efficacy at ε={eps} (undefended) ===", flush=True)
    runner = FLRunner(cfg)
    _force_epsilon(runner, eps)
    runner.run()
    s = _asr_summary(cfg.name)
    # clean-model artifact check with the benign femnist checkpoint + trained trigger
    try:
        s["clean_model_asr"] = _clean_asr_femnist(cfg.name)
    except Exception as exc:
        s["clean_model_asr"] = f"n/a ({exc})"
    return s


def _clean_asr_femnist(run_name, n=5000):
    from experiment.utils import build_adapter, resolve_device
    from experiments.benchmark.run_toposentinel import _make_topo_config
    from models import ModelConfig, get_model
    from attacks.triggers import get_trigger
    from datasets.utils import extract_labels
    dev = resolve_device("auto")
    cfg = _make_topo_config("femnist", "iba", 0, "results", "auto")
    ad = build_adapter(cfg); mc = ModelConfig.from_adapter("lenet5", ad); tl = 0
    trig = get_trigger("iba", in_channels=1, image_size=(28, 28), normalize_transform=None)
    st = torch.load(os.path.join("results", "analysis", run_name, "trigger.pt"),
                    map_location="cpu", weights_only=True)
    trig.generator.load_state_dict(st["unet_state_dict"]); trig.generator.eval()
    benign = torch.load("results/femnist_benign_iid/final_model.pt",
                        map_location="cpu", weights_only=True)
    m = get_model(mc).to(dev); m.load_state_dict({k: v.to(dev) for k, v in benign.items()}); m.eval()
    labels = extract_labels(ad.test_pre_dataset)
    nt = [i for i, l in enumerate(labels) if l != tl][:n]
    mean = torch.tensor(ad.normalize_transform.mean).view(1, 1, 1, 1).to(dev)
    std = torch.tensor(ad.normalize_transform.std).view(1, 1, 1, 1).to(dev)
    ct = tot = 0
    with torch.no_grad():
        for s in range(0, len(nt), 256):
            idx = nt[s:s + 256]
            xb = torch.stack([trig.apply(ad.test_pre_dataset[i][0]) for i in idx]).to(dev)
            xb = (xb - mean) / std
            ct += (m(xb).argmax(1) == tl).sum().item(); tot += len(idx)
    return round(ct / tot, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+", default=["mnist", "femnist"])
    p.add_argument("--femnist-eps", type=float, default=0.1)
    args = p.parse_args()
    results = []
    if "mnist" in args.which:
        results.append(run_mnist())
    if "femnist" in args.which:
        results.append(run_femnist_eps(args.femnist_eps))
    print("\n" + "=" * 80)
    print("IBA EFFICACY ABLATION")
    print("=" * 80)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
