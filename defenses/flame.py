"""FLAME defense server.

Reference
---------
Nguyen, T. D., Rieger, P., De Viti, R., Chen, H., Brandenburg, B. B.,
Yalame, H., ... & Sadeghi, A. R.
"FLAME: Taming Backdoors in Federated Learning."
USENIX Security 2022.

How it works
-----------
FLAME combines three mechanisms to mitigate backdoor attacks:

1. **Clustering-based detection** — clients are clustered by the cosine
   similarity of their last-layer absolute weights using HDBSCAN.  The
   assumption is that backdoor clients share a common directional bias
   toward the target class, so they form a minority cluster that can be
   separated from the (larger) benign majority cluster.

2. **Adaptive norm clipping** — the clip threshold is set to the *median*
   L2 distance (full update delta) of the benign cluster, so it adapts to
   the current round's legitimate update magnitudes.

3. **Adaptive Gaussian noise** — after aggregation, per-parameter noise
   ``N(0, (λ · clip_norm)²)`` is added.  This provides certified
   differential-privacy-style protection and further suppresses stealthy
   attacks that survive detection.

Design in this framework
------------------------
FLAME does not fit the pure-filtering pattern (like MKrum or DeepSight)
because its aggregation differs from FedAvg: it uses equal client weights
and adds noise.  It therefore implements *both* extension points:

* ``filter_updates(true_malicious)``
  Runs HDBSCAN, identifies the benign majority cluster, stores the median
  clip norm, removes the minority cluster from ``_received_updates``, and
  returns a :class:`~experiment.utils.DetectionResult`.

* ``aggregate()``  — **overrides** FedAvgAggregator
  Computes an equal-weight average of clipped update deltas and injects
  adaptive Gaussian noise into the result.

``reset()`` — inherited unchanged.
"""

from __future__ import annotations

import logging
from typing import FrozenSet, List, Optional, Tuple

import numpy as np
import torch

from fl.server import AggregationResult, FedAvgAggregator
from experiment.utils import DetectionResult

logger = logging.getLogger(__name__)


class FlameServer(FedAvgAggregator):
    """FLAME Byzantine-robust aggregator.

    Args:
        model:   Global model.
        device:  Torch device.
        lamda:   Noise multiplier λ.  Standard deviation of the adaptive
                 Gaussian noise is ``λ · clip_norm``.  Default ``0.001``.
        eta:     Global aggregation learning rate applied to the mean clipped
                 delta before noise.  Keep at ``1.0`` unless you need to damp
                 updates.
        **kwargs: Forwarded to :class:`~fl.server.FedAvgAggregator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        lamda: float = 0.001,
        eta: float = 1.0,
        min_cluster_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model=model, device=device, **kwargs)
        self.lamda            = lamda
        self.eta              = eta
        self.min_cluster_size = min_cluster_size  # None → dynamic n//2+1 each round
        # Set by filter_updates each round; used by aggregate().
        self._clip_norm: float = 1.0

        logger.info(
            "FlameServer — λ=%.4f  η=%.2f  min_cluster_size=%s",
            lamda, eta,
            min_cluster_size if min_cluster_size is not None else "dynamic (n//2+1)",
        )

    # ------------------------------------------------------------------
    # Detection interface (called by runner before aggregate)
    # ------------------------------------------------------------------

    def filter_updates(self, true_malicious: FrozenSet[int]) -> DetectionResult:
        """Cluster clients and remove the minority (suspected malicious) group.

        1. Extracts the last-layer absolute weights for cosine-based HDBSCAN.
        2. Assigns the largest cluster to benign; everything else is rejected.
        3. Computes the median L2 delta norm of the benign cluster and stores
           it as ``self._clip_norm`` for use in ``aggregate()``.
        4. Removes rejected clients from ``_received_updates``.

        Args:
            true_malicious: Ground-truth malicious IDs this round — used only
                            for TPR/FPR bookkeeping, not for filtering.

        Returns:
            :class:`~experiment.utils.DetectionResult`.
        """
        client_ids: List[int] = list(self._received_updates.keys())
        n = len(client_ids)

        if n < 2:
            logger.warning("FLAME: fewer than 2 updates — detection skipped.")
            return DetectionResult(rejected_ids=frozenset(), true_malicious=true_malicious)

        global_params = self.get_params()
        last_keys     = self._get_last_layer_keys()

        # ---- Per-client feature vectors for clustering and distances -----
        cluster_features: List[np.ndarray] = []
        l2_norms: dict[int, float]         = {}

        for cid in client_ids:
            local = self._received_updates[cid]["params"]

            # Absolute last-layer weights → cosine clustering
            feat = torch.cat([
                local[k].float().flatten() for k in last_keys if k in local
            ])
            cluster_features.append(feat.cpu().numpy().astype(np.float64))

            # L2 norm of the full update delta (weight + bias params only)
            delta_parts = [
                (local[k].float() - global_params[k].float()).flatten()
                for k in local
                if k in global_params and local[k].is_floating_point()
                and ("weight" in k or "bias" in k)
            ]
            delta_vec = torch.cat(delta_parts) if delta_parts else torch.zeros(1)
            l2_norms[cid] = delta_vec.norm(p=2).item()

        # ---- HDBSCAN clustering on last-layer cosine similarity ----------
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            logger.error(
                "FLAME requires the 'hdbscan' package. "
                "Install it with: pip install hdbscan"
            )
            return DetectionResult(rejected_ids=frozenset(), true_malicious=true_malicious)

        features_arr = np.array(cluster_features, dtype=np.float64)
        mcs = self.min_cluster_size if self.min_cluster_size is not None else max(2, n // 2 + 1)
        clusterer = hdbscan_lib.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=mcs,
            allow_single_cluster=True,
        )
        labels: np.ndarray = clusterer.fit_predict(features_arr)

        # ---- Largest cluster = benign ------------------------------------
        benign_ids, rejected_list = self._select_benign_cluster(client_ids, labels)

        # ---- Clip norm: median L2 distance of benign clients -------------
        benign_norms = [l2_norms[cid] for cid in benign_ids]
        self._clip_norm = float(np.median(benign_norms)) if benign_norms else 1.0
        self._clip_norm = max(self._clip_norm, 1e-6)   # guard against zero

        # ---- Remove rejected updates ------------------------------------
        rejected_ids: FrozenSet[int] = frozenset(rejected_list)
        for cid in rejected_ids:
            del self._received_updates[cid]

        id_label = {cid: ("MAL" if cid in true_malicious else "ben") for cid in client_ids}
        logger.info(
            "FLAME: %d / %d flagged  |  clip_norm=%.4f  "
            "L2 norms [%s]  HDBSCAN labels %s",
            len(rejected_ids), n,
            self._clip_norm,
            ", ".join(f"{id_label[c]}:{l2_norms[c]:.3f}" for c in client_ids),
            labels.tolist(),
        )

        return DetectionResult(rejected_ids=rejected_ids, true_malicious=true_malicious)

    # ------------------------------------------------------------------
    # Aggregation override: equal weights + clipping + adaptive noise
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregationResult:
        """Equal-weight FedAvg with adaptive norm clipping and Gaussian noise.

        Each surviving (benign) client's delta is clipped to
        ``self._clip_norm``; the mean clipped delta is scaled by ``eta``
        and added to the global model; finally ``N(0, (λ·clip_norm)²)``
        noise is injected into every weight/bias parameter.

        Returns:
            :class:`~fl.server.AggregationResult` — compatible with the
            runner's existing handling.
        """
        if not self._received_updates:
            # Conservative fallback: return current global model unchanged.
            logger.warning("FLAME.aggregate: no updates in buffer — model unchanged.")
            params = self.get_params()
            return AggregationResult(
                aggregated_params=params,
                num_clients=0,
                total_samples=0,
            )

        global_params  = self.get_params()
        client_ids     = list(self._received_updates.keys())
        n              = len(client_ids)
        total_samples  = sum(d["length"] for d in self._received_updates.values())

        # Initialise delta accumulator (float32)
        accum = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_params.items()
        }

        for cid in client_ids:
            local = self._received_updates[cid]["params"]

            # Full L2 norm of this client's weight+bias delta (for clipping)
            delta_parts = [
                (local[k].float() - global_params[k].float()).flatten()
                for k in local
                if k in global_params and local[k].is_floating_point()
                and ("weight" in k or "bias" in k)
            ]
            if delta_parts:
                client_norm = torch.cat(delta_parts).norm(p=2).item()
            else:
                client_norm = 0.0

            scale = (
                min(1.0, self._clip_norm / client_norm)
                if client_norm > 1e-9 else 1.0
            )

            for k, global_v in global_params.items():
                if k not in local or not global_v.is_floating_point():
                    continue
                delta = (local[k].float() - global_v.float()) * scale
                accum[k] += delta / n   # equal weight

        # Apply aggregated delta + adaptive noise
        std_dev = self.lamda * self._clip_norm
        new_params: dict = {}

        for k, global_v in global_params.items():
            if not global_v.is_floating_point():
                # Non-float (e.g. num_batches_tracked): majority vote
                majority = max(client_ids,
                               key=lambda c: self._received_updates[c]["length"])
                new_params[k] = self._received_updates[majority]["params"][k].clone()
                continue

            updated = global_v.float() + self.eta * accum[k]

            if ("weight" in k or "bias" in k) and std_dev > 0.0:
                noise = torch.randn_like(updated) * std_dev
                updated = updated + noise

            new_params[k] = updated.to(global_v.dtype)

        self.set_params({k: v.to(self.device) for k, v in new_params.items()})

        return AggregationResult(
            aggregated_params={k: v.cpu().clone() for k, v in new_params.items()},
            num_clients=n,
            total_samples=total_samples,
            client_weights={cid: 1.0 / n for cid in client_ids},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_benign_cluster(
        client_ids: List[int],
        labels: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Return ``(benign_ids, malicious_ids)`` from HDBSCAN output.

        Strategy:
        * All noise (``-1``) or a single cluster → all clients are benign
          (conservative fallback).
        * Multiple clusters → the *largest* cluster is benign; all others
          (including HDBSCAN noise points ``-1``) are malicious.
        """
        index_to_id = {i: cid for i, cid in enumerate(client_ids)}
        unique_non_noise = set(labels.tolist()) - {-1}

        # Conservative cases: no meaningful separation found
        if len(unique_non_noise) <= 1:
            return client_ids, []

        # Multiple clusters: keep only the largest
        counts = {lbl: int((labels == lbl).sum()) for lbl in unique_non_noise}
        largest = max(counts, key=counts.__getitem__)
        benign_ids   = [index_to_id[i] for i, lbl in enumerate(labels.tolist()) if lbl == largest]
        malicious_ids = [cid for cid in client_ids if cid not in set(benign_ids)]

        return benign_ids, malicious_ids

    def _get_last_layer_keys(self) -> List[str]:
        """Return the state-dict keys for the last two parameter tensors
        (weight + bias of the output layer)."""
        params = self.get_params()
        param_keys = [k for k in params if "weight" in k or "bias" in k]
        return param_keys[-2:]
