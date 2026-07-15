"""Fast single-round probe: how do W_inf (bottleneck alarm) and d_i (filter)
scale with the IBA trigger budget ε on femnist?

Instead of a full 3h analysis run per ε, this reuses TopoSentinel's own bias
extractor + persistence analyzer + median-distance to compute, on a converged
femnist model, the detectability quantities for one round at several ε:

  * benign reference H0 diagram (from B benign clients' bias deltas)
  * for each ε: the attack H0 diagram (benign + M malicious@ε), and
        W_inf   = persim.bottleneck(benign_ref_H0, attack_H0)
        d_i     = TopoSentinel median-distance, malicious vs benign
  * malicious update L2 norm (raw effect of ε on the bias footprint)

Caveat: single-round snapshot implanting into a CLEAN converged model (not the
full multi-round consolidation), so it measures the *implantation-update*
signature vs ε — exactly what the "bigger ε → bigger W_inf" claim is about.
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

EPS_SWEEP = [8 / 255, 0.1, 0.2, 0.3]


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
    cfg.num_clients = 20                     # fewer clients -> faster build
    cfg.attack.num_malicious = 8
    ad = build_adapter(cfg)
    mc = ModelConfig.from_adapter("lenet5", ad)
    mal_ids, clients = build_clients(cfg, ad, mc, dev)
    g = torch.load("results/femnist_benign_iid/final_model.pt",
                   map_location="cpu", weights_only=True)

    # Server purely to reuse the exact bias extractor / analyzer / median-distance.
    srv = TopoSentinelServer(model=get_model(mc), device=dev, analysis_mode=True,
                             min_clients_for_defense=3,
                             **_BOTTLENECK_THRESHOLDS["femnist"])
    gbias = srv._extract_bias_vector(g)

    def delta(state):
        return srv._extract_bias_vector(state) - gbias

    # ---- benign reference cohort (computed once) ----
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

    print(f"\n{'eps':>7} {'mal_upd_norm':>13} {'W_inf':>8} {'>tau?':>6} "
          f"{'d_i_mal':>8} {'d_i_ben':>8} {'sep(ben-mal)':>13}")
    for eps in EPS_SWEEP:
        mal_deltas, norms = [], []
        for c in mal_cohort:
            t = clients[c].config.trigger
            t.eps_0 = eps; t.eps_hat = eps; t.epsilon = eps
            clients[c].config.delta = None      # disable the L2 cap so the
            #                                     update magnitude reflects ε
            clients[c].set_params(g)
            r = clients[c].local_train(epochs=cfg.local_epochs, round_idx=50)
            mal_deltas.append(delta(r.weights))
            norms.append(r.metadata.get("update_norm_projected", float("nan")))
        mal_deltas = np.vstack(mal_deltas)

        alld = np.vstack([ben_deltas, mal_deltas])
        atk_h0 = _h0(srv._analyser, alld)
        w_inf = persim.bottleneck(ref_h0, atk_h0)

        ids = list(range(len(alld)))
        cdel = dict(zip(ids, alld))
        med = np.median(alld, axis=0)
        dists = srv._bias_distances_from_median(ids, cdel, med)
        n_ben = len(ben_deltas)
        d_ben = np.mean([dists[i] for i in range(n_ben)])
        d_mal = np.mean([dists[i] for i in range(n_ben, len(alld))])

        print(f"{eps:>7.4f} {np.mean(norms):>13.3f} {w_inf:>8.4f} "
              f"{'YES' if w_inf > tau else 'no':>6} {d_mal:>8.4f} {d_ben:>8.4f} "
              f"{d_ben - d_mal:>13.4f}")
    print(f"\n(femnist alarm threshold τ_min = {tau})")


if __name__ == "__main__":
    main()
