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

import csv
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
        bias_interval_margin:         Additive margin applied to both ends of
                                      the accept interval (5th/95th percentile
                                      of the benign bias-delta history).
        bias_fallback_interval:       ``[lo, hi]`` interval used when the
                                      benign history is too small.
        min_clients_for_defense:      Minimum number of clients required to
                                      run the defense; fewer → detection skipped.
        min_bias_history_size:        Minimum history entries before the
                                      learned interval is used; below this,
                                      ``bias_fallback_interval`` is used instead.
        analysis_mode:                Pure-observability switch (default
                                      ``False``).  When ``True``, no client is
                                      ever rejected — every detection quantity
                                      (bottleneck distance, decay threshold,
                                      accept interval, per-client distances)
                                      is still computed and logged to
                                      ``_alarm_log`` / ``_filter_log`` (see
                                      :meth:`write_analysis_logs`), but nothing
                                      acts on it.  Production behaviour when
                                      ``False`` is unchanged.
        attack_pattern:               ``{"period": P, "duty": D, "warmup": W}``
                                      sporadic ground-truth attack schedule
                                      used only for analysis-mode logging.
                                      Round ``r`` is an "attack round" iff
                                      ``r >= W and (r - W) % P < D``, i.e. the
                                      first ``W`` rounds are always quiet, so
                                      the benign bias-history / reference H0
                                      diagram is seeded from clean rounds
                                      before any attack begins. Default
                                      ``{"period": 10, "duty": 1, "warmup": 0}``.
        console_monitor:              When ``True`` (production path only,
                                      i.e. ``analysis_mode=False``), print one
                                      line to stdout every round: selected
                                      clients, the malicious ones among them,
                                      the bottleneck distance and threshold,
                                      whether filtering triggered, rejected
                                      clients, and TPR/FPR — for interactive
                                      threshold tuning. Never changes what is
                                      rejected; observability only. Default
                                      ``False``.
        exclude_bn_bias:               When ``True``, BatchNorm bias (beta)
                                      parameters are excluded from bias-space
                                      extraction -- only Conv/Linear additive
                                      bias is used for both the TDA alarm and
                                      the intra-round filter. Default
                                      ``False`` (BatchNorm bias included, as
                                      documented in ``_compute_bias_param_names``).
        clip_mode:                     Post-filter aggregation clipping
                                      strategy applied to each surviving
                                      client's delta before weighted
                                      averaging:
                                      ``"global"`` (default) -- one scalar
                                      scale per client, from the L2 norm of
                                      its *full* flattened delta against the
                                      median survivor norm (original
                                      behaviour); ``"layerwise"`` -- an
                                      independent scale per parameter tensor,
                                      each against that tensor's own median
                                      survivor norm, so a client cannot evade
                                      clipping by concentrating a large
                                      perturbation in one layer while keeping
                                      its overall norm modest; ``"none"`` --
                                      no clipping (deltas pass through
                                      unscaled), for measuring how much
                                      clipping alone suppresses an attack.
        **kwargs:                     Forwarded to
                                      :class:`~fl.server.FedAvgAggregator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        bias_metric: str = "euclidean",
        bottleneck_initial_threshold: float = 0.3,
        bottleneck_decay_rate: float = 0.99,
        bottleneck_min_threshold: float = 0.01,
        bias_history_window: int = 20,
        bias_interval_margin: float = 0.01,
        bias_fallback_interval: Optional[List[float]] = None,
        min_clients_for_defense: int = 3,
        min_bias_history_size: Optional[int] = None,
        analysis_mode: bool = False,
        attack_pattern: Optional[dict] = None,
        console_monitor: bool = False,
        exclude_bn_bias: bool = False,
        clip_mode: str = "global",
        **kwargs,
    ):
        super().__init__(model=model, device=device, **kwargs)

        self.bias_metric                  = bias_metric.lower()
        self.bottleneck_initial_threshold = bottleneck_initial_threshold
        self.bottleneck_decay_rate        = bottleneck_decay_rate
        self.bottleneck_min_threshold     = bottleneck_min_threshold
        self.bias_history_window          = bias_history_window
        self.bias_interval_margin         = bias_interval_margin
        self.bias_fallback_interval       = bias_fallback_interval or [0.0, 0.5]
        self.min_clients_for_defense      = min_clients_for_defense
        self.min_bias_history_size        = (
            min_bias_history_size
            if min_bias_history_size is not None
            else max(50, min_clients_for_defense * 3)
        )

        # ---- Analysis (observability-only) mode ---------------------------
        self.analysis_mode = bool(analysis_mode)
        pattern            = attack_pattern or {}
        self._attack_period = max(1, int(pattern.get("period", 10)))
        self._attack_duty   = max(0, min(int(pattern.get("duty", 1)), self._attack_period))
        self._attack_warmup = max(0, int(pattern.get("warmup", 0)))
        self._alarm_log: List[dict]  = []
        self._filter_log: List[dict] = []

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
        self._clip_norm: float                 = 1.0   # set by filter_updates ("global"/"none" modes)
        self._clip_norms_layerwise: Optional[dict] = None  # set by filter_updates ("layerwise" mode)
        self.clip_mode = clip_mode.lower()
        if self.clip_mode not in ("global", "layerwise", "none"):
            logger.warning(
                "TopoSentinel: unknown clip_mode '%s', defaulting to 'global'.", clip_mode,
            )
            self.clip_mode = "global"
        # Populated by _run_defense; surfaced in DetectionResult for AUPRC.
        self._last_client_scores: Optional[dict] = None
        # Populated by _run_defense; surfaced in filter_updates()'s optional
        # console_monitor summary.
        self._last_bottleneck_distance: Optional[float] = None
        self._last_triggered: Optional[bool]             = None
        self.console_monitor = bool(console_monitor)
        self.exclude_bn_bias = bool(exclude_bn_bias)

        # ---- Bias-parameter selection (module-type based) ------------------
        # Computed once from the live model's module tree, not by matching
        # "bias" as a name substring -- see _compute_bias_param_names().
        self._bias_param_names: List[str] = self._compute_bias_param_names()
        logger.info(
            "TopoSentinelServer: %d additive-bias parameters selected: %s",
            len(self._bias_param_names), self._bias_param_names,
        )

        logger.info(
            "TopoSentinelServer — metric=%s  threshold=%.3f→%.3f (decay=%.4f)  "
            "history_window=%d",
            self.bias_metric,
            bottleneck_initial_threshold, bottleneck_min_threshold,
            bottleneck_decay_rate, bias_history_window,
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

        if self.analysis_mode:
            # Pure observability: compute and log every quantity, but never
            # reject a client — every client is aggregated normally.
            is_attack_round = self._is_attack_round(self._round)
            try:
                self._run_defense_analysis(
                    client_ids, threshold, is_attack_round, true_malicious
                )
            except Exception as exc:
                logger.warning(
                    "TopoSentinel[analysis] round %d: logging failed (%s).",
                    self._round, exc,
                )
        else:
            # Reset so a skipped/errored round doesn't show stale values from
            # a previous round in the console_monitor summary below.
            self._last_bottleneck_distance = None
            self._last_triggered           = None
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

        # Clip norm(s) for aggregate() — computed from survivors
        if self.clip_mode == "layerwise":
            self._clip_norms_layerwise = self._compute_clip_norms_layerwise()
        else:
            self._clip_norm = self._compute_clip_norm()
        self._round    += 1

        logger.info(
            "TopoSentinel round %d: %d / %d rejected  clip_norm=%.4f  "
            "threshold=%.4f  history=%d",
            self._round - 1, len(rejected_ids), n,
            self._clip_norm, threshold, len(self._bias_history),
        )

        if self.console_monitor and not self.analysis_mode:
            self._print_console_monitor_line(
                round_idx=self._round - 1,
                client_ids=client_ids,
                true_malicious=true_malicious,
                rejected_ids=rejected_ids,
                threshold=threshold,
            )

        return DetectionResult(
            rejected_ids=rejected_ids,
            true_malicious=true_malicious,
            client_scores=self._last_client_scores,
        )

    def _print_console_monitor_line(
        self,
        round_idx: int,
        client_ids: List[int],
        true_malicious: FrozenSet[int],
        rejected_ids: FrozenSet[int],
        threshold: float,
    ) -> None:
        """Print one human-readable line to stdout summarizing this round's
        detection decision, for interactive threshold tuning.

        Observability only -- never changes what gets rejected. Gated by
        ``console_monitor`` (production path only; analysis_mode has its own
        CSV-based logging).
        """
        malicious_selected = sorted(set(client_ids) & set(true_malicious))
        benign_selected     = sorted(set(client_ids) - set(true_malicious))

        tp  = len(rejected_ids & set(malicious_selected))
        fp  = len(rejected_ids & set(benign_selected))
        tpr = tp / len(malicious_selected) if malicious_selected else float("nan")
        fpr = fp / len(benign_selected) if benign_selected else float("nan")

        bd_str   = (
            f"{self._last_bottleneck_distance:.4f}"
            if self._last_bottleneck_distance is not None else "n/a"
        )
        trig_str = (
            str(self._last_triggered) if self._last_triggered is not None else "n/a"
        )

        print(
            f"[TopoSentinel] round={round_idx:4d}  "
            f"clients={sorted(client_ids)}  malicious={malicious_selected}  "
            f"bottleneck={bd_str}  threshold={threshold:.4f}  triggered={trig_str}  "
            f"rejected={sorted(rejected_ids)}  TPR={tpr:.3f}  FPR={fpr:.3f}"
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

            if self.clip_mode == "layerwise":
                # Independent scale per parameter tensor: a client cannot
                # evade clipping here by concentrating a large perturbation
                # in one layer while keeping its overall (global) norm modest.
                layer_norms = self._clip_norms_layerwise or {}
                for k, global_v in global_params.items():
                    if k not in local or not global_v.is_floating_point():
                        continue
                    delta_k     = local[k].float() - global_v.float()
                    layer_norm  = delta_k.norm(p=2).item()
                    layer_clip  = max(layer_norms.get(k, 1.0), 1e-6)
                    scale_k     = min(1.0, layer_clip / layer_norm) if layer_norm > 1e-9 else 1.0
                    accum[k]   += delta_k * scale_k * weight
                continue

            if self.clip_mode == "none":
                for k, global_v in global_params.items():
                    if k not in local or not global_v.is_floating_point():
                        continue
                    accum[k] += (local[k].float() - global_v.float()) * weight
                continue

            # "global" (default): one scalar scale per client, from the L2
            # norm of its full flattened delta (float params only).
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
        bd: Optional[float] = None

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

        # Surfaced for filter_updates()'s optional console_monitor summary.
        self._last_bottleneck_distance = bd
        self._last_triggered           = trigger_filtering

        # ---- Per-client distance from median bias DELTA ------------------
        # Consistent with the alarm above: both operate on deltas, not on
        # the raw absolute bias vectors (bias_deltas is already client_bias
        # - global_bias, aligned row-for-row with valid_ids).
        client_deltas = dict(zip(valid_ids, bias_deltas))
        median_delta  = np.median(bias_deltas, axis=0)
        dists         = self._bias_distances_from_median(valid_ids, client_deltas, median_delta)
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
        lo, hi, hist_L = self._get_benign_interval()

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
        logger.info(
            "TopoSentinel.filter round=%d L=%d theta_min=%.6f theta_max=%.6f accepted=%d rejected=%d",
            self._round, hist_L, lo, hi, n_accepted, len(rejected),
        )
        return rejected

    # ------------------------------------------------------------------
    # Analysis (observability-only) mode
    # ------------------------------------------------------------------

    def _is_attack_round(self, round_idx: int) -> bool:
        """Sporadic ground-truth schedule: round ``r`` is an attack round iff
        ``r >= warmup and (r - warmup) % period < duty``. The ``warmup``
        rounds are always quiet, so the benign reference (bias history /
        H0 diagram) is seeded before any attack begins. Used only for
        analysis-mode logging."""
        if round_idx < self._attack_warmup:
            return False
        return ((round_idx - self._attack_warmup) % self._attack_period) < self._attack_duty

    def _run_defense_analysis(
        self,
        client_ids: List[int],
        threshold: float,
        is_attack_round: bool,
        true_malicious: FrozenSet[int],
    ) -> None:
        """Compute and log every detection quantity without rejecting anyone.

        Mirrors ``_run_defense``'s computations (same bottleneck distance,
        same ``_get_benign_interval`` accept interval) but the benign
        baseline (``_prev_h0`` / ``_bias_history``) is advanced on
        ground-truth quiet rounds (``is_attack_round == False``) rather than
        on the computed trigger decision, since no rejection ever happens
        here to keep the trigger meaningful.
        """
        round_idx = self._round
        n         = len(client_ids)

        num_malicious_present = sum(
            1 for cid in client_ids if is_attack_round and cid in true_malicious
        )

        w_inf             = float("nan")
        s_t               = float("nan")
        r_ratio           = float("nan")
        would_alarm_decay = float("nan")

        global_params = self.get_params()
        global_bias   = self._extract_bias_vector(global_params)

        client_biases: dict[int, np.ndarray] = {}
        if global_bias is not None:
            for cid in client_ids:
                v = self._extract_bias_vector(self._received_updates[cid]["params"])
                if v is not None:
                    client_biases[cid] = v

        valid_ids = [cid for cid in client_ids if cid in client_biases]
        h0_curr   = None

        if global_bias is not None and len(valid_ids) >= 2:
            bias_deltas = np.vstack([client_biases[c] for c in valid_ids]) - global_bias
            delta_norms = np.linalg.norm(bias_deltas, axis=1)
            s_t         = float(np.median(np.abs(delta_norms - np.median(delta_norms))))
            tau_min     = max(0.0, self.bottleneck_min_threshold)

            current_diagram = self._analyser.compute_diagram(bias_deltas)
            if current_diagram is not None and len(current_diagram) > 0:
                h0_curr = current_diagram[current_diagram[:, 2] == 0]

            if (
                h0_curr is not None and len(h0_curr) > 0
                and self._prev_h0 is not None and len(self._prev_h0) > 0
            ):
                try:
                    import persim
                    finite_prev = self._prev_h0[np.isfinite(self._prev_h0[:, 1])]
                    finite_curr = h0_curr[np.isfinite(h0_curr[:, 1])]
                    if len(finite_prev) > 0 and len(finite_curr) > 0:
                        w_inf = float(persim.bottleneck(finite_prev[:, :2], finite_curr[:, :2]))
                except Exception as exc:
                    logger.warning(
                        "TopoSentinel[analysis]: bottleneck computation failed (%s).", exc
                    )

            if not np.isnan(w_inf):
                would_alarm_decay = float(w_inf > threshold)
                r_ratio           = w_inf / (s_t + tau_min)

        # ---- Per-client distance to median bias DELTA; interval that -------
        # ---- WOULD be used this round (logging only, never gates). --------
        # Consistent with the alarm: both operate on deltas, not on the raw
        # absolute bias vectors. Computed independently of the TDA block
        # above since that one is gated on len(valid_ids) >= 2, while a
        # single client (len(valid_ids) == 1) can still be logged here.
        if valid_ids:
            client_deltas = {cid: client_biases[cid] - global_bias for cid in valid_ids}
            delta_mat     = np.vstack([client_deltas[c] for c in valid_ids])
            median_delta  = np.median(delta_mat, axis=0)
            dists         = self._bias_distances_from_median(valid_ids, client_deltas, median_delta)
            lo, hi, _     = self._get_benign_interval()

            for cid in valid_ids:
                d               = dists[cid]
                is_mal_present  = is_attack_round and (cid in true_malicious)
                self._filter_log.append({
                    "round": round_idx,
                    "is_attack_round": int(is_attack_round),
                    "client_id": cid,
                    "is_malicious_present": int(is_mal_present),
                    "d_i": d,
                    "theta_min": lo,
                    "theta_max": hi,
                    "inside_interval": int(lo <= d <= hi),
                })

            # Ground-truth-gated baseline update: quiet rounds only. Never
            # let a computed threshold decide what enters the baseline.
            if not is_attack_round:
                if h0_curr is not None:
                    self._prev_h0 = h0_curr
                self._bias_history.extend(dists.values())

        self._alarm_log.append({
            "round": round_idx,
            "is_attack_round": int(is_attack_round),
            "num_clients": n,
            "num_malicious_present": num_malicious_present,
            "W_inf": w_inf,
            "tau_decay": threshold,
            "s_t": s_t,
            "R_ratio": r_ratio,
            "would_alarm_decay": would_alarm_decay,
        })

    def write_analysis_logs(self, alarm_csv_path: str, filter_csv_path: str) -> None:
        """Write the accumulated analysis-mode logs to CSV (no-op if empty).

        Args:
            alarm_csv_path:  Destination for the one-row-per-round alarm log.
            filter_csv_path: Destination for the one-row-per-client-per-round
                             filter log.
        """
        if self._alarm_log:
            with open(alarm_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._alarm_log[0].keys()))
                writer.writeheader()
                writer.writerows(self._alarm_log)

        if self._filter_log:
            with open(filter_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._filter_log[0].keys()))
                writer.writeheader()
                writer.writerows(self._filter_log)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bias_distances_from_median(
        self,
        valid_ids: List[int],
        client_deltas: dict,
        median_delta: np.ndarray,
    ) -> dict[int, float]:
        """Compute each client's distance from the median bias DELTA.

        Both callers pass bias-delta vectors (``client_bias - global_bias``),
        never raw absolute bias vectors, so this always operates in delta
        space -- consistent with the inter-round TDA alarm.
        """
        if self.bias_metric == "cosine":
            from scipy.spatial.distance import cosine as scipy_cosine
            med_norm = float(np.linalg.norm(median_delta))
            dists    = {}
            for cid in valid_ids:
                v      = client_deltas[cid]
                v_norm = float(np.linalg.norm(v))
                dists[cid] = (
                    1.0 if v_norm < 1e-9 or med_norm < 1e-9
                    else float(scipy_cosine(v, median_delta))
                )
        else:
            delta_mat = np.vstack([client_deltas[c] for c in valid_ids])
            arr       = np.linalg.norm(delta_mat - median_delta, axis=1)
            dists     = {cid: float(d) for cid, d in zip(valid_ids, arr)}
        return dists

    def _get_benign_interval(self) -> tuple[float, float, int]:
        """Compute the accept interval from the 5th/95th percentiles of the
        benign bias-delta history, with an additive margin on both ends.

        Returns:
            ``(theta_min, theta_max, L)`` where ``L`` is the current history
            buffer length.
        """
        H = np.array(list(self._bias_history))
        L = len(H)

        if L < self.min_bias_history_size:
            logger.info(
                "TopoSentinel: history too small (%d < %d) — using fallback interval %s.",
                L, self.min_bias_history_size, self.bias_fallback_interval,
            )
            lo, hi = self.bias_fallback_interval
            return (float(lo), float(hi), L)

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
        return (lo_m, hi_m, L)

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

    def _compute_clip_norms_layerwise(self) -> dict:
        """Per-parameter-tensor median L2 delta norm of the clients currently
        in the buffer (used by ``clip_mode="layerwise"``)."""
        global_params = self.get_params()
        norms_per_key: dict = {
            k: [] for k, v in global_params.items() if v.is_floating_point()
        }
        for data in self._received_updates.values():
            local = data["params"]
            for k in norms_per_key:
                if k in local:
                    delta = local[k].float() - global_params[k].float()
                    norms_per_key[k].append(delta.norm(p=2).item())
        result = {}
        for k, norms in norms_per_key.items():
            nonzero = [v for v in norms if v > 0]
            result[k] = float(np.median(nonzero)) if nonzero else 1.0
        return result

    def _compute_bias_param_names(self) -> List[str]:
        """Enumerate exact additive-bias parameter names by walking the
        model's module tree, selecting each module's ``.bias`` when it is a
        learnable :class:`~torch.nn.Parameter`.

        This is module-type driven, not name-substring matching: it
        naturally includes Conv/Linear bias AND BatchNorm's bias (beta),
        while excluding ``.weight`` unambiguously (including BatchNorm's
        gamma), because PyTorch consistently names every layer type's
        additive term ``.bias`` and its multiplicative term ``.weight``
        regardless of module type. Computed once at construction from
        ``self.model`` -- every client's state dict shares the same
        architecture and therefore the same parameter names.

        When ``self.exclude_bn_bias`` is ``True``, BatchNorm modules
        (:class:`~torch.nn.modules.batchnorm._BatchNorm` -- covers
        BatchNorm1d/2d/3d and SyncBatchNorm) are skipped entirely, so only
        Conv/Linear-style additive bias contributes to the bias-space
        distances and TDA diagram.
        """
        names: List[str] = []
        for module_name, module in self.model.named_modules():
            if self.exclude_bn_bias and isinstance(
                module, torch.nn.modules.batchnorm._BatchNorm
            ):
                continue
            bias = getattr(module, "bias", None)
            if isinstance(bias, torch.nn.Parameter):
                names.append(f"{module_name}.bias" if module_name else "bias")
        return names

    def _extract_bias_vector(self, params: dict) -> Optional[np.ndarray]:
        """Flatten this model's additive-bias parameters (see
        ``_bias_param_names``) from a state dict into one float64 vector."""
        tensors = [
            params[name].detach().cpu().float().flatten()
            for name in self._bias_param_names
            if name in params and isinstance(params[name], torch.Tensor)
        ]
        if not tensors:
            return None
        return torch.cat(tensors).numpy().astype(np.float64)
