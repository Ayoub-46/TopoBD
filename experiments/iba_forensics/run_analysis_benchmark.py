"""Analysis-mode logging of W_inf (alarm) + d_i (filter) at the BENCHMARK setup.

Same configuration as experiments.benchmark.run_toposentinel (production scale,
continuous attack window, the REAL IBA — ε_0=0.3 decay + δ=2.3 constrained
poisoning — the dataset's model, and the new per-dataset bottleneck thresholds),
but with TopoSentinel in analysis_mode=True so it LOGS every round's bottleneck
distance and per-client bias distance without ever rejecting.  This exposes why
production detection collapses after the backdoor consolidates.

Only difference vs the benchmark: no rejection (that is what "analysis" means),
so the federation evolves undefended and we see the raw detector signal on every
attack round, including the consolidation phase.

Usage:
    python -m experiments.iba_forensics.run_analysis_benchmark               # gtsrb/iba
    python -m experiments.iba_forensics.run_analysis_benchmark --dataset femnist
    python -m experiments.iba_forensics.run_analysis_benchmark --analyze-only
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import experiment  # noqa: F401
from experiments.benchmark.run_toposentinel import _make_topo_config


def _run_dir(dataset: str, attack: str, model_override=None, epsilon=None) -> str:
    suffix = (f"_{model_override}" if model_override else "") \
             + (f"_eps{epsilon:.2f}" if epsilon else "")
    return os.path.join("results", "analysis",
                        f"{dataset}_{attack}_benchmark_analysis{suffix}")


def run(dataset: str, attack: str, seed: int = 0, model_override=None,
        num_clients=None, num_malicious=None, epsilon=None) -> str:
    from experiment.runner import FLRunner

    config = _make_topo_config(dataset, attack, seed, "results", "auto")
    if model_override:
        config.model = model_override                 # e.g. cifar10 -> vgg13_nobn
    # Federation shrink for laptop GPUs: build_clients keeps one resident model+
    # optimizer per client, so 100 vgg13_nobn copies OOM a 7.6GB GPU. Shrinking
    # num_clients (keeping the malicious ratio) is what demo_analysis_mode does.
    if num_clients is not None:
        config.num_clients = num_clients
        config.clients_per_round = min(config.clients_per_round, num_clients)
    if num_malicious is not None:
        config.attack.num_malicious = num_malicious
    t_start = config.attack.attack_start_round        # continuous window origin

    # analysis_mode: log, never reject. attack_pattern labels rounds >= t_start
    # as attack rounds (period=duty=1 => every post-warmup round), matching the
    # continuous benchmark window.
    dk = dict(config.defense.defense_kwargs)
    dk["analysis_mode"] = True
    dk["attack_pattern"] = {"warmup": t_start, "period": 1, "duty": 1}
    config.defense.defense_kwargs = dk

    config.output_dir = os.path.join("results", "analysis")
    config.name = os.path.basename(_run_dir(dataset, attack, model_override, epsilon))

    print(f"=== analysis@benchmark — {dataset}/{attack} "
          f"(model={config.model}, rounds={config.num_rounds}, "
          f"attack_start={t_start}, thresholds={dk['bottleneck_min_threshold']}min"
          f"{f', ε≡{epsilon}' if epsilon else ''}) ===", flush=True)
    runner = FLRunner(config)
    # Force a constant floor ε (disable the decay) when requested — e.g. femnist
    # needs ε=0.1 to implant a real backdoor (8/255 is too weak; see
    # efficacy_ablation.py). Set on each malicious client's shared trigger.
    if epsilon is not None:
        for cid in runner.malicious_ids:
            if cid in runner.clients:
                trig = runner.clients[cid].config.trigger
                trig.eps_0 = epsilon; trig.eps_hat = epsilon; trig.epsilon = epsilon
    runner.run()

    out = _run_dir(dataset, attack, model_override, epsilon)
    os.makedirs(out, exist_ok=True)
    runner.server.write_analysis_logs(
        os.path.join(out, "alarm_log.csv"), os.path.join(out, "filter_log.csv"))
    print(f"  wrote {out}/alarm_log.csv & filter_log.csv", flush=True)
    return out


def analyze(dataset: str, attack: str, model_override=None, epsilon=None) -> None:
    out = _run_dir(dataset, attack, model_override, epsilon)
    alarm = pd.read_csv(os.path.join(out, "alarm_log.csv"))
    filt = pd.read_csv(os.path.join(out, "filter_log.csv"))

    # attack window origin
    t_start = int(alarm.loc[alarm.is_attack_round == 1, "round"].min())
    atk_rounds = alarm[alarm.is_attack_round == 1]["round"]
    mid = int(t_start + 0.35 * (atk_rounds.max() - t_start))   # implant / consolidate split

    def phase(r):
        if r < t_start:
            return "pre"
        return "implant" if r <= mid else "consolidate"

    alarm = alarm.copy(); alarm["phase"] = alarm["round"].map(phase)
    filt = filt.copy(); filt["phase"] = filt["round"].map(phase)

    print(f"\n{'='*88}\nW_inf (alarm) by phase — {dataset}/{attack} "
          f"(attack_start={t_start}, implant≤{mid}<consolidate)\n{'='*88}")
    ag = alarm.groupby("phase").agg(
        n=("round", "count"),
        W_inf_mean=("W_inf", "mean"),
        tau_mean=("tau_decay", "mean"),
        alarm_fire_rate=("would_alarm_decay", "mean"),
    ).reindex(["pre", "implant", "consolidate"])
    print(ag.round(4).to_string())

    print(f"\n{'='*88}\nd_i (filter) by phase — malicious vs benign\n{'='*88}")
    rows = []
    for ph in ["pre", "implant", "consolidate"]:
        fp = filt[filt.phase == ph]
        mal = fp[fp.is_malicious_present == 1]
        ben = fp[fp.is_malicious_present == 0]
        rows.append({
            "phase": ph,
            "d_i_malicious": mal.d_i.mean() if len(mal) else float("nan"),
            "d_i_benign": ben.d_i.mean() if len(ben) else float("nan"),
            "mal_flagged_rate": (mal.inside_interval == 0).mean() if len(mal) else float("nan"),
            "ben_flagged_rate": (ben.inside_interval == 0).mean() if len(ben) else float("nan"),
            "n_mal": len(mal), "n_ben": len(ben),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="gtsrb")
    p.add_argument("--attack", default="iba")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None, help="Override the dataset's default model.")
    p.add_argument("--num-clients", type=int, default=None,
                   help="Shrink the federation (laptop GPU memory).")
    p.add_argument("--num-malicious", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=None,
                   help="Force a constant floor ε (e.g. 0.1 for femnist).")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    if not args.analyze_only:
        run(args.dataset, args.attack, args.seed, model_override=args.model,
            num_clients=args.num_clients, num_malicious=args.num_malicious,
            epsilon=args.epsilon)
    analyze(args.dataset, args.attack, model_override=args.model, epsilon=args.epsilon)


if __name__ == "__main__":
    main()
