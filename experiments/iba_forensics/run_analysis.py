"""Production-like TopoSentinel analysis of the FIXED IBA across 3 datasets.

Reuses results/analysis/demo_analysis_mode.build_config (real FL stack,
analysis_mode=True so TopoSentinel logs but never rejects, sporadic
period/duty attack schedule with a warmup, and the calibrated bottleneck
thresholds) and writes the same alarm_log.csv / filter_log.csv.

Two deliberate overrides vs the vanilla demo, applied per malicious client at
runtime (no edits to shared harness/attack/defense code):

  * Faithful floor ε = 8/255.  The sporadic schedule sets each malicious
    client's attack_start_round = round_idx every active round, so the Eq. 4
    decay would see t - t_I = 0 and pin ε at eps_0 = 0.3 — the input-artifact
    budget we fixed.  At ε = 0.3 the backdoor lives in the INPUT (a clean model
    already fires), the poison training barely moves the weights, and the bias
    update is tiny — IBA would look "undetectable" for the wrong reason.  We
    force ε ≡ eps_hat (= 8/255) so the analysis sees the MODEL-resident
    backdoor whose weight/bias update is real.

  * Space constraint disabled (delta=None) for this pass.  It is a separate
    evasion mechanism against *norm*-based defenses and would need per-dataset
    calibration; TopoSentinel's alarm is topological and its filter is a bias
    DISTANCE, so we first measure detectability with only the (dataset-agnostic)
    bottom-k% dimension constraint active, then can layer the norm cap on.

Datasets/models: cifar10→vgg13_nobn (best-separation arch), gtsrb→gtsrb_cnn,
femnist→lenet5 (dataset defaults from benchmark matrix).

Usage:
    python -m experiments.iba_forensics.run_analysis            # run + analyze
    python -m experiments.iba_forensics.run_analysis --analyze-only
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_ANALYSIS_DIR = os.path.join(_REPO_ROOT, "results", "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

import experiment  # noqa: F401 -- breaks fl.server <-> experiment import cycle
import demo_analysis_mode as dm

# (dataset, model_override) — cifar10 uses the VGG-13-noBN best-separation arch.
RUNS = [("cifar10", "vgg13_nobn"), ("gtsrb", None), ("femnist", None)]


def run_one_iba(dataset: str, model_override) -> str:
    """Run one IBA analysis_mode federation; return the output directory."""
    config = dm.build_config(dataset, "iba", model_override=model_override)
    # Tag the dir so this faithful-ε analysis never clobbers the vanilla demo.
    config.output_dir = config.output_dir + "_fixediba"
    os.makedirs(config.output_dir, exist_ok=True)

    from experiment.runner import FLRunner
    print(f"\n=== IBA analysis — {dataset} (model={config.model}) ===", flush=True)
    runner = FLRunner(config)
    server, clients, malicious_ids = runner.server, runner.clients, runner.malicious_ids

    # --- faithful floor ε + disable the L2 space cap for this pass ---
    forced = False
    for cid in malicious_ids:
        if cid in clients:
            trig = clients[cid].config.trigger
            trig.eps_0 = trig.eps_hat            # ε ≡ 8/255 (no decay in sporadic mode)
            clients[cid].config.delta = None     # dimension constraint only
            forced = True
    print(f"malicious ids: {sorted(malicious_ids)}  | forced ε≡{trig.eps_hat:.4f}={forced}",
          flush=True)

    # --- sporadic-schedule round loop (mirrors demo_analysis_mode.run_one) ---
    global_params = server.get_params()
    rng = random.Random(config.seed + 1_000_003)
    for round_idx in range(config.num_rounds):
        is_attack = server._is_attack_round(round_idx)
        selected_ids = rng.sample(list(clients.keys()), config.clients_per_round)
        malicious_selected = frozenset(c for c in selected_ids if c in malicious_ids)

        for cid in selected_ids:
            client = clients[cid]
            if cid in malicious_ids:
                if is_attack:
                    client.config.attack_start_round = round_idx
                    client.config.attack_end_round = round_idx
                else:
                    client.config.attack_start_round = round_idx + 1
                    client.config.attack_end_round = round_idx - 1
            client.set_params(global_params)
            update = client.local_train(epochs=config.local_epochs, round_idx=round_idx)
            server.receive_update(client_id=update.client_id,
                                  params=update.weights, length=update.num_samples)

        server.filter_updates(true_malicious=malicious_selected)
        agg = server.aggregate()
        server.reset()
        global_params = agg.aggregated_params
        if round_idx % 5 == 0 or is_attack:
            print(f"  round {round_idx:2d} attack={int(is_attack)} "
                  f"malicious_selected={sorted(malicious_selected)}", flush=True)

    acc = server.evaluate(runner.test_loader).metrics["main_accuracy"]
    alarm_path = os.path.join(config.output_dir, "alarm_log.csv")
    filter_path = os.path.join(config.output_dir, "filter_log.csv")
    server.write_analysis_logs(alarm_path, filter_path)
    print(f"  final clean acc={acc:.4f}  | wrote {alarm_path}", flush=True)
    return config.output_dir


# ---------------------------------------------------------------------------
# Analysis of alarm + filter performance
# ---------------------------------------------------------------------------

def analyze(out_dirs: dict) -> pd.DataFrame:
    rows = []
    for dataset, out_dir in out_dirs.items():
        alarm = pd.read_csv(os.path.join(out_dir, "alarm_log.csv"))
        filt = pd.read_csv(os.path.join(out_dir, "filter_log.csv"))

        # ---- ALARM: bottleneck W_inf vs decaying threshold ----
        atk = alarm[alarm.is_attack_round == 1]
        quiet = alarm[alarm.is_attack_round == 0]
        alarm_recall = float((atk.would_alarm_decay == 1).mean()) if len(atk) else float("nan")
        alarm_fpr = float((quiet.would_alarm_decay == 1).mean()) if len(quiet) else float("nan")
        w_atk = float(atk.W_inf.mean(skipna=True))
        w_quiet = float(quiet.W_inf.mean(skipna=True))

        # ---- FILTER: per-client bias distance d_i vs interval ----
        # inside_interval == 0 => flagged. Ground truth malicious = is_malicious_present.
        fa = filt[filt.is_attack_round == 1]
        mal = fa[fa.is_malicious_present == 1]
        ben = fa[fa.is_malicious_present == 0]
        filt_tpr = float((mal.inside_interval == 0).mean()) if len(mal) else float("nan")
        filt_fpr = float((ben.inside_interval == 0).mean()) if len(ben) else float("nan")
        di_mal = float(mal.d_i.mean()) if len(mal) else float("nan")
        di_ben = float(ben.d_i.mean()) if len(ben) else float("nan")

        rows.append({
            "dataset": dataset,
            "attack_rounds": int(len(atk)),
            "alarm_recall": alarm_recall, "alarm_fpr": alarm_fpr,
            "W_inf_attack": w_atk, "W_inf_quiet": w_quiet,
            "filter_TPR": filt_tpr, "filter_FPR": filt_fpr,
            "d_i_malicious": di_mal, "d_i_benign": di_ben,
            "d_i_sep_ratio": (di_mal / di_ben) if di_ben else float("nan"),
            "n_mal_obs": int(len(mal)), "n_ben_obs": int(len(ben)),
        })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(_ANALYSIS_DIR, "iba_fixed_detection_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n" + "=" * 92)
    print("FIXED-IBA vs TopoSentinel — alarm (topological) + filter (bias-distance) performance")
    print("=" * 92)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.round(4).to_string(index=False))
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()

    out_dirs = {}
    for dataset, model in RUNS:
        cfg = dm.build_config(dataset, "iba", model_override=model)
        out_dirs[dataset] = cfg.output_dir + "_fixediba"
        if not args.analyze_only:
            run_one_iba(dataset, model)
    analyze(out_dirs)


if __name__ == "__main__":
    main()
