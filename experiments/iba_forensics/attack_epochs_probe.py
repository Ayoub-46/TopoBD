"""Fast single-round probe: at FIXED low ε (0.1) and δ disabled, does increasing
the malicious backdoor-training epochs raise the bias-space footprint (W_inf +
d_i), i.e. make an effective backdoor MORE detectable?

Tests the "use training effort, not trigger budget" hypothesis: more attacker
epochs at low ε should push the backdoor deeper into the MODEL (larger weight/
bias delta) rather than the input, growing both effectiveness and detectability.

Mirrors winf_epsilon_probe.py but sweeps attack_epochs instead of ε.
Caveat: single-round implantation into a converged model (ASR not measured here;
this isolates the footprint-vs-epochs trend). Same converged-model regime caveat.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import persim
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import experiment  # noqa: F401
from experiment.utils import build_adapter, build_clients, resolve_device
from experiments.benchmark.run_toposentinel import _BOTTLENECK_THRESHOLDS, _make_topo_config
from models import ModelConfig, get_model
from defenses.toposentinel import TopoSentinelServer

EPSILON = 0.1
EPOCHS_SWEEP = [5, 10, 20, 40]


def _seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _h0(analyser, deltas):
    d = analyser.compute_diagram(deltas)
    h0 = d[d[:, 2] == 0]
    return h0[np.isfinite(h0[:, 1])][:, :2]


def main():
    _seed(0)
    dev = resolve_device("auto")
    cfg = _make_topo_config("femnist", "iba", 0, "results", "auto")
    cfg.num_clients = 20
    cfg.attack.num_malicious = 8
    ad = build_adapter(cfg)
    mc = ModelConfig.from_adapter("lenet5", ad)
    mal_ids, clients = build_clients(cfg, ad, mc, dev)
    g = torch.load("results/femnist_benign_iid/final_model.pt",
                   map_location="cpu", weights_only=True)

    srv = TopoSentinelServer(model=get_model(mc), device=dev, analysis_mode=True,
                             min_clients_for_defense=3,
                             **_BOTTLENECK_THRESHOLDS["femnist"])
    gbias = srv._extract_bias_vector(g)

    def delta(state):
        return srv._extract_bias_vector(state) - gbias

    # benign reference cohort — trained at the normal local_epochs (once)
    ben_cohort = [c for c in clients if c not in mal_ids][:8]
    ben_deltas = []
    for c in ben_cohort:
        clients[c].set_params(g)
        r = clients[c].local_train(epochs=cfg.local_epochs, round_idx=50)
        ben_deltas.append(delta(r.weights))
    ben_deltas = np.vstack(ben_deltas)
    ref_h0 = _h0(srv._analyser, ben_deltas)

    mal_cohort = sorted(mal_ids)[:4]
    tau = _BOTTLENECK_THRESHOLDS["femnist"]["bottleneck_min_threshold"]

    print(f"\nFixed ε={EPSILON}, δ=None, benign_epochs={cfg.local_epochs}")
    print(f"{'atk_epochs':>10} {'mal_upd_norm':>13} {'W_inf':>8} {'>tau?':>6} "
          f"{'d_i_mal':>8} {'d_i_ben':>8} {'|sep|':>8}")
    for E in EPOCHS_SWEEP:
        mal_deltas, norms = [], []
        for c in mal_cohort:
            t = clients[c].config.trigger
            t.eps_0 = EPSILON; t.eps_hat = EPSILON; t.epsilon = EPSILON
            clients[c].config.delta = None                # no L2 cap
            clients[c].config.attack_epochs = E           # <-- the swept knob
            clients[c].set_params(g)
            r = clients[c].local_train(epochs=cfg.local_epochs, round_idx=50)
            mal_deltas.append(delta(r.weights))
            norms.append(r.metadata.get("update_norm_projected", float("nan")))
        mal_deltas = np.vstack(mal_deltas)

        alld = np.vstack([ben_deltas, mal_deltas])
        atk_h0 = _h0(srv._analyser, alld)
        w_inf = persim.bottleneck(ref_h0, atk_h0)

        ids = list(range(len(alld)))
        med = np.median(alld, axis=0)
        dists = srv._bias_distances_from_median(ids, dict(zip(ids, alld)), med)
        n_ben = len(ben_deltas)
        d_ben = np.mean([dists[i] for i in range(n_ben)])
        d_mal = np.mean([dists[i] for i in range(n_ben, len(alld))])

        print(f"{E:>10} {np.mean(norms):>13.3f} {w_inf:>8.4f} "
              f"{'YES' if w_inf > tau else 'no':>6} {d_mal:>8.4f} {d_ben:>8.4f} "
              f"{abs(d_ben - d_mal):>8.4f}")
    print(f"\n(femnist alarm threshold τ_min = {tau}; d_i sep = |benign - malicious|)")


if __name__ == "__main__":
    main()
