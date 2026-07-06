"""TopoSentinel defense server.

A hybrid TDA-based defense that combines:

* **Inter-round anomaly detection** — compares consecutive rounds' H0
  persistence diagrams (computed on client bias-delta vectors) using the
  bottleneck distance.  A decaying threshold triggers intra-round filtering
  when the topological structure changes significantly.

* **Intra-round bias filtering** — when triggered, each client's distance
  from the median bias vector is checked against an adaptively-learned
  benign interval.  Outliers are rejected.

* **Adaptive median-norm clipping** — surviving updates are aggregated with
  sample-count weights; each delta is clipped to the median L2 norm of the
  surviving clients, limiting the per-round damage of any undetected update.

Design in this framework
------------------------
Follows the two-step detection + aggregation pattern:

* ``filter_updates(true_malicious)``
  Runs TDA analysis and (conditionally) bias filtering; removes rejected
  clients from ``_received_updates``; stores ``_clip_norm`` for use in
  ``aggregate()``; returns a :class:`~experiment.utils.DetectionResult`.

* ``aggregate()``  — **overrides** FedAvgAggregator
  Sample-weighted FedAvg with per-client delta clipping.

* ``reset()`` — inherited.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import FrozenSet, List, Optional

import numpy as np
import torch

from fl.server import AggregationResult, FedAvgAggregator
from experiment.utils import DetectionResult
from persistent_homology.analyzer import TopologicalAnalyser
from persistent_homology.metrics import magnitude_cosine_distance

logger = logging.getLogger(__name__)


class TopoSentinelServer(FedAvgAggregator):
    """TDA-based hybrid defense aggregator.

    Args:
        model:                        Global model.
        device:                       Torch device.
        bias_metric:                  Distance metric used for both TDA and
                                      intra-round filtering.  One of
                                      ``"euclidean"`` (default), ``"cosine"``,
                                      ``"magnitude_cosine"``.
        bottleneck_initial_threshold: Starting bottleneck-distance threshold
                                      for the TDA change detector.
        bottleneck_decay_rate:        Multiplicative decay applied each round
                                      (``threshold_t = initial × decay^t``).
        bottleneck_min_threshold:     Floor value for the decaying threshold.
        bias_history_window:          Controls the ``deque`` size for the
                                      benign bias-distance history
                                      (capacity ≈ ``window × min_clients × 2.5``).
        filter_mode:                  ``"dkw"`` (default) for DKW-calibrated
                                      accept interval; ``"fixed"`` for the
                                      original p5/p95 percentile baseline
                                      (ablation mode).
        target_fpr:                   Target false-positive rate *delta*
                                      (probability budget for rejecting a
                                      benign client).  Used in ``"dkw"`` mode.
        dkw_confidence:               DKW confidence level ``1 - alpha``.
                                      Larger values widen the interval to
                                      provide stronger FPR guarantees.
        bias_interval_margin:         Additive margin applied to both ends of
                                      the interval in ``"fixed"`` mode only.
        bias_fallback_interval:       ``[lo, hi]`` interval used when the
                                      benign history is too small (``"fixed"``
                                      mode only).
        min_clients_for_defense:      Minimum number of clients required to
                                      run the defense; fewer → detection skipped.
        min_bias_history_size:        Minimum history entries before the
                                      learned interval is used (``"fixed"``
                                      mode only; ``"dkw"`` mode requires L ≥ 2).
        **kwargs:                     Forwarded to
                                      :class:`~fl.server.FedAvgAggregator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        bias_metric: str = "euclidean",
        bottleneck_initial_threshold: float = 0.8,
        bottleneck_decay_rate: float = 0.99,
        bottleneck_min_threshold: float = 0.01,
        bias_history_window: int = 20,
        filter_mode: str = "dkw",
        target_fpr: float = 0.05,
        dkw_confidence: float = 0.95,
        bias_interval_margin: float = 0.01,
        bias_fallback_interval: Optional[List[float]] = None,
        min_clients_for_defense: int = 3,
        min_bias_history_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model=model, device=device, **kwargs)

        self.bias_metric                  = bias_metric.lower()
        self.bottleneck_initial_threshold = bottleneck_initial_threshold
        self.bottleneck_decay_rate        = bottleneck_decay_rate
        self.bottleneck_min_threshold     = bottleneck_min_threshold
        self.bias_history_window          = bias_history_window
        self.filter_mode                  = filter_mode.lower()
        self.target_fpr                   = float(target_fpr)
        self.dkw_confidence               = float(dkw_confidence)
        self.bias_interval_margin         = bias_interval_margin
        self.bias_fallback_interval       = bias_fallback_interval or [0.0, 0.5]
        self.min_clients_for_defense      = min_clients_for_defense
        self.min_bias_history_size        = (
            min_bias_history_size
            if min_bias_history_size is not None
            else max(50, min_clients_for_defense * 3)
        )

        # Build TDA metric
        if self.bias_metric == "magnitude_cosine":
            tda_metric        = magnitude_cosine_distance
            tda_metric_params = {"alpha": 0.5}
        elif self.bias_metric in ("cosine", "euclidean"):
            tda_metric        = self.bias_metric
            tda_metric_params = {}
        else:
            logger.warning(
                "TopoSentinel: unknown bias_metric '%s', defaulting to 'euclidean'.",
                bias_metric,
            )
            tda_metric        = "euclidean"
            tda_metric_params = {}

        self._analyser = TopologicalAnalyser(
            homology_dimensions=(0,),
            metric=tda_metric,
            metric_params=tda_metric_params,
        )

        # Persistent state across rounds
        self._round: int                       = 0
        self._prev_h0: Optional[np.ndarray]   = None   # reference H0 diagram
        self._bias_history: deque              = deque(
            maxlen=int(bias_history_window * min_clients_for_defense * 2.5)
        )
        self._clip_norm: float                 = 1.0   # set by filter_updates
        # Populated by _run_defense; surfaced in DetectionResult for AUPRC.
        self._last_client_scores: Optional[dict] = None

        logger.info(
            "TopoSentinelServer — metric=%s  threshold=%.3f→%.3f (decay=%.4f)  "
            "history_window=%d  filter_mode=%s  target_fpr=%.3f  dkw_confidence=%.3f",
            self.bias_metric,
            bottleneck_initial_threshold, bottleneck_min_threshold,
            bottleneck_decay_rate, bias_history_window,
            self.filter_mode, self.target_fpr, self.dkw_confidence,
        )

    # ------------------------------------------------------------------
    # Detection interface
    # ------------------------------------------------------------------

    def filter_updates(self, true_malicious: FrozenSet[int]) -> DetectionResult:
        """Run TDA inter-round analysis and (conditionally) bias filtering.

        1. Extracts bias-delta vectors for all clients.
        2. Computes the H0 bottleneck distance against the previous round's
           diagram.  If it exceeds the decaying threshold, intra-round
           filtering is triggered.
        3. When triggered, clients whose bias-to-median distance falls outside
           the adaptively-learned benign interval are rejected.
        4. When not triggered, per-client distances are added to the benign
           history and the reference diagram is updated.
        5. Surviving clients' median delta-norm is stored as ``_clip_norm``
           for the subsequent ``aggregate()`` call.

        Args:
            true_malicious: Ground-truth IDs — used only for TPR/FPR metrics.

        Returns:
            :class:`~experiment.utils.DetectionResult`.
        """
        client_ids: List[int] = list(self._received_updates.keys())
        n = len(client_ids)

        # Decaying bottleneck threshold for this round
        threshold = max(
            self.bottleneck_min_threshold,
            self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self._round),
        )

        rejected: set[int] = set()

        try:
            if n < self.min_clients_for_defense:
                logger.info(
                    "TopoSentinel round %d: n=%d < min=%d — detection skipped.",
                    self._round, n, self.min_clients_for_defense,
                )
            else:
                rejected = self._run_defense(client_ids, threshold)

        except Exception as exc:
            logger.warning(
                "TopoSentinel round %d: defense error (%s) — no clients rejected.",
                self._round, exc,
            )
            rejected.clear()

        # Prune buffer
        rejected_ids: FrozenSet[int] = frozenset(rejected)
        for cid in rejected_ids:
            del self._received_updates[cid]

        # Clip norm for aggregate() — computed from survivors
        self._clip_norm = self._compute_clip_norm()
        self._round    += 1

        logger.info(
            "TopoSentinel round %d: %d / %d rejected  clip_norm=%.4f  "
            "threshold=%.4f  history=%d",
            self._round - 1, len(rejected_ids), n,
            self._clip_norm, threshold, len(self._bias_history),
        )

        return DetectionResult(
            rejected_ids=rejected_ids,
            true_malicious=true_malicious,
            client_scores=self._last_client_scores,
        )

    # ------------------------------------------------------------------
    # Aggregation override: sample-weighted FedAvg + median clipping
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregationResult:
        """Sample-weighted FedAvg where each delta is clipped to ``_clip_norm``.

        Returns:
            :class:`~fl.server.AggregationResult`.
        """
        if not self._received_updates:
            logger.warning("TopoSentinel.aggregate: empty buffer — model unchanged.")
            return AggregationResult(
                aggregated_params=self.get_params(), num_clients=0, total_samples=0
            )

        global_params = self.get_params()
        client_ids    = list(self._received_updates.keys())
        total_samples = sum(d["length"] for d in self._received_updates.values())

        if total_samples == 0:
            return AggregationResult(
                aggregated_params=global_params,
                num_clients=len(client_ids),
                total_samples=0,
            )

        clip_norm = max(self._clip_norm, 1e-6)
        accum     = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_params.items()}

        for cid in client_ids:
            local    = self._received_updates[cid]["params"]
            weight   = self._received_updates[cid]["length"] / total_samples

            # L2 norm of this client's full delta (float params only)
            delta_parts = [
                (local[k].float() - global_params[k].float()).flatten()
                for k in local
                if k in global_params and global_params[k].is_floating_point()
            ]
            client_norm = torch.cat(delta_parts).norm(p=2).item() if delta_parts else 0.0
            scale = min(1.0, clip_norm / client_norm) if client_norm > 1e-9 else 1.0

            for k, global_v in global_params.items():
                if k not in local or not global_v.is_floating_point():
                    continue
                accum[k] += (local[k].float() - global_v.float()) * scale * weight

        # Apply accumulated (clipped, weighted) delta
        new_params = {}
        for k, global_v in global_params.items():
            if global_v.is_floating_point():
                new_params[k] = (global_v.float() + accum[k]).to(global_v.dtype)
            else:
                majority = max(client_ids, key=lambda c: self._received_updates[c]["length"])
                new_params[k] = self._received_updates[majority]["params"][k].clone()

        self.set_params({k: v.to(self.device) for k, v in new_params.items()})

        return AggregationResult(
            aggregated_params={k: v.cpu().clone() for k, v in new_params.items()},
            num_clients=len(client_ids),
            total_samples=total_samples,
            client_weights={
                cid: self._received_updates[cid]["length"] / total_samples
                for cid in client_ids
            },
        )

    # ------------------------------------------------------------------
    # Core defense logic (extracted for testability)
    # ------------------------------------------------------------------

    def _run_defense(
        self,
        client_ids: List[int],
        threshold: float,
    ) -> set[int]:
        """Run TDA + bias filtering; return the set of clients to reject."""
        global_params = self.get_params()
        global_bias   = self._extract_bias_vector(global_params)
        if global_bias is None:
            raise RuntimeError("Cannot extract bias vector from global model.")

        # Per-client bias vectors
        client_biases: dict[int, np.ndarray] = {}
        for cid in client_ids:
            v = self._extract_bias_vector(self._received_updates[cid]["params"])
            if v is not None:
                client_biases[cid] = v

        valid_ids = [cid for cid in client_ids if cid in client_biases]
        if len(valid_ids) < self.min_clients_for_defense:
            raise RuntimeError(
                f"Only {len(valid_ids)} clients have extractable bias vectors "
                f"(need ≥ {self.min_clients_for_defense})."
            )

        # ---- Inter-round TDA (on bias *deltas*) -------------------------
        bias_deltas      = np.vstack([client_biases[c] for c in valid_ids]) - global_bias
        current_diagram  = self._analyser.compute_diagram(bias_deltas)
        trigger_filtering = False

        if current_diagram is not None and len(current_diagram) > 0:
            h0_curr = current_diagram[current_diagram[:, 2] == 0]

            if self._prev_h0 is not None and len(self._prev_h0) > 0 and len(h0_curr) > 0:
                try:
                    import persim
                    finite_prev = self._prev_h0[np.isfinite(self._prev_h0[:, 1])]
                    finite_curr = h0_curr[np.isfinite(h0_curr[:, 1])]
                    if len(finite_prev) > 0 and len(finite_curr) > 0:
                        bd = persim.bottleneck(finite_prev[:, :2], finite_curr[:, :2])
                        logger.info(
                            "TopoSentinel: bottleneck=%.4f  threshold=%.4f", bd, threshold
                        )
                        if bd > threshold:
                            trigger_filtering = True
                except Exception as exc:
                    logger.warning("TopoSentinel: bottleneck computation failed (%s).", exc)

            # Only advance the reference diagram on benign rounds
            if not trigger_filtering:
                self._prev_h0 = h0_curr

        # ---- Per-client distance from median bias vector ----------------
        bias_abs     = np.vstack([client_biases[c] for c in valid_ids])
        median_bias  = np.median(bias_abs, axis=0)
        dists        = self._bias_distances_from_median(valid_ids, client_biases, median_bias)
        # Store bias distances so filter_updates can expose them for AUPRC.
        self._last_client_scores = {cid: dists.get(cid, float("inf")) for cid in valid_ids}

        if not trigger_filtering:
            # Benign round: accumulate distances for learning
            self._bias_history.extend(dists.values())
            logger.debug(
                "TopoSentinel: benign round — added %d distances to history (total=%d).",
                len(dists), len(self._bias_history),
            )
            return set()

        # ---- Intra-round filtering via adaptive learned interval ---------
        logger.info("TopoSentinel: TDA change detected — applying bias filter.")
        lo, hi, hist_L, eps_val = self._get_benign_interval()

        rejected: set[int] = set()
        for cid in valid_ids:
            d = dists.get(cid, np.inf)
            if not (lo <= d <= hi):
                rejected.add(cid)

        # Safety: never reject everyone
        if len(rejected) >= len(client_ids):
            logger.warning(
                "TopoSentinel: bias filter flagged all %d clients — reverting to no rejection.",
                len(client_ids),
            )
            rejected.clear()

        # CSV-parseable log for adaptive-interval paper plots
        n_accepted = len(valid_ids) - len(rejected)
        eps_str = f"{eps_val:.6f}" if not np.isnan(eps_val) else "nan"
        logger.info(
            "TopoSentinel.filter round=%d L=%d eps=%s theta_min=%.6f theta_max=%.6f accepted=%d rejected=%d",
            self._round, hist_L, eps_str, lo, hi, n_accepted, len(rejected),
        )
        return rejected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bias_distances_from_median(
        self,
        valid_ids: List[int],
        client_biases: dict,
        median_bias: np.ndarray,
    ) -> dict[int, float]:
        """Compute each client's distance from the median bias vector."""
        if self.bias_metric == "cosine":
            from scipy.spatial.distance import cosine as scipy_cosine
            med_norm = float(np.linalg.norm(median_bias))
            dists    = {}
            for cid in valid_ids:
                v      = client_biases[cid]
                v_norm = float(np.linalg.norm(v))
                dists[cid] = (
                    1.0 if v_norm < 1e-9 or med_norm < 1e-9
                    else float(scipy_cosine(v, median_bias))
                )
        else:
            bias_mat = np.vstack([client_biases[c] for c in valid_ids])
            arr      = np.linalg.norm(bias_mat - median_bias, axis=1)
            dists    = {cid: float(d) for cid, d in zip(valid_ids, arr)}
        return dists

    def _get_benign_interval(self) -> tuple[float, float, int, float]:
        """Compute the accept interval and calibration metadata.

        Returns:
            ``(theta_min, theta_max, L, eps)`` where ``L`` is the current
            history buffer length and ``eps`` is the DKW correction term
            (``float('nan')`` when ``filter_mode='fixed'``).
        """
        H = np.array(list(self._bias_history))
        L = len(H)

        # ---- DKW-calibrated mode -------------------------------------------
        if self.filter_mode == "dkw":
            if L < 2:
                logger.info(
                    "TopoSentinel: DKW calibration skipped (L=%d < 2) — "
                    "accepting all clients this round.",
                    L,
                )
                return (0.0, float("inf"), L, float("nan"))

            alpha = 1.0 - self.dkw_confidence
            eps   = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * L)))

            p_lo = self.target_fpr / 2.0 - eps
            p_hi = 1.0 - self.target_fpr / 2.0 + eps

            if p_lo <= 0.0:
                theta_min = 0.0
            else:
                theta_min = float(np.quantile(H, float(np.clip(p_lo, 0.0, 1.0))))

            theta_max = float(np.quantile(H, float(np.clip(p_hi, 0.0, 1.0))))
            theta_min = max(0.0, theta_min)

            return (theta_min, theta_max, L, eps)

        # ---- Fixed-percentile mode (ablation baseline) ---------------------
        if L < self.min_bias_history_size:
            logger.info(
                "TopoSentinel: history too small (%d < %d) — using fallback interval %s.",
                L, self.min_bias_history_size, self.bias_fallback_interval,
            )
            lo, hi = self.bias_fallback_interval
            return (float(lo), float(hi), L, float("nan"))

        lo  = float(np.percentile(H, 5.0))
        hi  = float(np.percentile(H, 95.0))
        lo_m = max(0.0, lo - self.bias_interval_margin)
        hi_m = hi + self.bias_interval_margin
        if hi_m <= lo_m:
            hi_m = lo_m + 1e-6

        logger.debug(
            "TopoSentinel: fixed interval [%.4f, %.4f] from %d history points.",
            lo_m, hi_m, L,
        )
        return (lo_m, hi_m, L, float("nan"))

    def _compute_clip_norm(self) -> float:
        """Median L2 delta norm of the clients currently in the buffer."""
        global_params = self.get_params()
        norms = []
        for data in self._received_updates.values():
            local = data["params"]
            parts = [
                (local[k].float() - global_params[k].float()).flatten()
                for k in local
                if k in global_params and global_params[k].is_floating_point()
            ]
            if parts:
                norms.append(torch.cat(parts).norm(p=2).item())
        nonzero = [v for v in norms if v > 0]
        return float(np.median(nonzero)) if nonzero else 1.0

    @staticmethod
    def _extract_bias_vector(params: dict) -> Optional[np.ndarray]:
        """Flatten all bias tensors from a state dict into one float64 vector."""
        tensors = [
            p.detach().cpu().float().flatten()
            for name, p in params.items()
            if "bias" in name and isinstance(p, torch.Tensor)
        ]
        if not tensors:
            return None
        return torch.cat(tensors).numpy().astype(np.float64)
