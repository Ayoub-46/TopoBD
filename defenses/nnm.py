"""Nearest Neighbor Mixing (NNM) + robust base-rule aggregation.

Reference
---------
Allouah, Y., Koloskova, A., Safaryan, M., Jaggi, M., & Guerraoui, R.
"Fixing by Mixing: A Recipe for Optimal Byzantine Fault-Tolerant Aggregation
Rules." AISTATS 2023.

How it works
-----------
NNM is a **pre-aggregation transform**, not a standalone rule.  The full
pipeline is::

    mixed_deltas = NNM(deltas, f)   # step 1 — mixing
    output       = BaseRule(mixed_deltas, f)   # step 2 — robust aggregation

**Step 1 — Nearest Neighbor Mixing**

For each client i, replace its delta Δᵢ with the mean of itself and its f
nearest neighbors (by L2 distance on the full flattened delta vector)::

    Δ̃ᵢ = (Δᵢ + Σⱼ∈Nf(i) Δⱼ) / (f + 1)

**Step 2 — Base aggregation rule** (configurable via ``base_rule``)

The paper evaluates three base rules on the mixed deltas:

* ``"cwtm"``  Coordinate-Wise Trimmed Mean — trim the f smallest and f
              largest values per coordinate, average the rest.
              Recommended for most settings.
* ``"cwmed"`` Coordinate-Wise Median — take the per-coordinate median.
              More robust than CWTM but lower statistical efficiency.
* ``"krum"``  Multi-Krum — select the (n − f) updates with the lowest
              sum-of-distances-to-nearest-neighbours score, then average.
* ``"fedavg"``Simple equal-weight mean.  Keeps NNM's dilution benefit but
              applies no additional filtering.

Design in this framework
------------------------
NNM is a **pure aggregation rule** (no ``filter_updates``), so the runner
will not compute TPR / FPR — those columns remain ``NaN``.  This is correct:
NNM works by diluting Byzantine updates rather than detecting them.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

from fl.server import AggregationResult, FedAvgAggregator

logger = logging.getLogger(__name__)

BaseRule = Literal["cwtm", "cwmed", "krum", "fedavg"]


class NNMServer(FedAvgAggregator):
    """NNM pre-mixing followed by a configurable robust base-aggregation rule.

    Args:
        model:         Global model.
        device:        Torch device.
        num_byzantine: Assumed upper bound on Byzantine clients per round
                       (``f``).  Must satisfy ``f < n/2``.
        base_rule:     Aggregation rule applied to the NNM-mixed deltas.
                       One of ``"cwtm"`` (default), ``"cwmed"``, ``"krum"``,
                       ``"fedavg"``.
        **kwargs:      Forwarded to :class:`~fl.server.FedAvgAggregator`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_byzantine: int = 1,
        base_rule: BaseRule = "cwtm",
        **kwargs,
    ):
        if num_byzantine < 0:
            raise ValueError(f"num_byzantine must be ≥ 0, got {num_byzantine}.")
        valid_rules = {"cwtm", "cwmed", "krum", "fedavg"}
        if base_rule not in valid_rules:
            raise ValueError(f"base_rule must be one of {valid_rules}, got '{base_rule}'.")

        super().__init__(model=model, device=device, **kwargs)
        self.num_byzantine = num_byzantine
        self.base_rule     = base_rule

        logger.info("NNMServer — f=%d  base_rule=%s", num_byzantine, base_rule)

    # ------------------------------------------------------------------
    # Aggregation override
    # ------------------------------------------------------------------

    def aggregate(self) -> AggregationResult:
        """NNM mixing followed by the configured base aggregation rule.

        Returns:
            :class:`~fl.server.AggregationResult`.
        """
        if not self._received_updates:
            raise RuntimeError(
                "NNMServer.aggregate() called with no buffered updates."
            )

        global_params = self.get_params()
        client_ids    = list(self._received_updates.keys())
        n             = len(client_ids)
        total_samples = sum(d["length"] for d in self._received_updates.values())

        float_keys    = [k for k, v in global_params.items() if v.is_floating_point()]
        nonfloat_keys = [k for k, v in global_params.items() if not v.is_floating_point()]

        # ---- Build flat delta matrix (n, d) ------------------------------
        flat_deltas = []
        for cid in client_ids:
            local = self._received_updates[cid]["params"]
            delta = torch.cat([
                (local[k].float() - global_params[k].float()).flatten()
                for k in float_keys
            ])
            flat_deltas.append(delta)

        delta_mat = torch.stack(flat_deltas)   # (n, d)

        # ---- Step 1: NNM mixing ------------------------------------------
        mixed = self._apply_nnm(delta_mat, self.num_byzantine)   # (n, d)

        # ---- Step 2: base aggregation rule on mixed deltas ---------------
        mean_delta = self._apply_base_rule(mixed, self.num_byzantine)   # (d,)

        # ---- Unflatten mean delta → parameter dict -----------------------
        new_params = {}
        offset = 0
        for k, global_v in global_params.items():
            if not global_v.is_floating_point():
                continue
            numel = global_v.numel()
            chunk = mean_delta[offset : offset + numel].reshape(global_v.shape)
            new_params[k] = (global_v.float() + chunk).to(global_v.dtype)
            offset += numel

        if nonfloat_keys:
            majority = max(client_ids, key=lambda c: self._received_updates[c]["length"])
            for k in nonfloat_keys:
                new_params[k] = self._received_updates[majority]["params"][k].clone()

        self.set_params({k: v.to(self.device) for k, v in new_params.items()})

        return AggregationResult(
            aggregated_params={k: v.cpu().clone() for k, v in new_params.items()},
            num_clients=n,
            total_samples=total_samples,
            client_weights={cid: 1.0 / n for cid in client_ids},
        )

    # ------------------------------------------------------------------
    # NNM mixing
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_nnm(delta_mat: torch.Tensor, f: int) -> torch.Tensor:
        """Replace each delta with the mean of itself and its f nearest neighbors.

        Args:
            delta_mat: ``(n, d)`` matrix of client update deltas.
            f:         Number of nearest neighbors to mix with.

        Returns:
            ``(n, d)`` matrix of mixed deltas.
        """
        n = delta_mat.shape[0]
        f = min(f, n - 1)

        if f == 0:
            return delta_mat.clone()

        dist_mat = torch.cdist(delta_mat, delta_mat, p=2)   # (n, n)
        dist_mat.fill_diagonal_(float("inf"))

        _, nn_idx = torch.topk(dist_mat, k=f, dim=1, largest=False)  # (n, f)
        neighbor_deltas = delta_mat[nn_idx]                           # (n, f, d)

        return (delta_mat + neighbor_deltas.sum(dim=1)) / (f + 1)    # (n, d)

    # ------------------------------------------------------------------
    # Base aggregation rules
    # ------------------------------------------------------------------

    def _apply_base_rule(self, mixed: torch.Tensor, f: int) -> torch.Tensor:
        """Dispatch to the configured base aggregation rule.

        Args:
            mixed: ``(n, d)`` NNM-mixed delta matrix.
            f:     Byzantine budget parameter forwarded to the rule.

        Returns:
            ``(d,)`` aggregated delta vector.
        """
        if self.base_rule == "cwtm":
            return self._cwtm(mixed, f)
        if self.base_rule == "cwmed":
            return self._cwmed(mixed)
        if self.base_rule == "krum":
            return self._krum(mixed, f)
        # fedavg
        return mixed.mean(dim=0)

    @staticmethod
    def _cwtm(delta_mat: torch.Tensor, f: int) -> torch.Tensor:
        """Coordinate-Wise Trimmed Mean.

        Sorts each coordinate across clients, removes the ``f`` smallest and
        ``f`` largest values, and averages the remaining ``n - 2f`` values.

        Args:
            delta_mat: ``(n, d)``.
            f:         Number of values to trim from each end per coordinate.

        Returns:
            ``(d,)`` trimmed mean.
        """
        n = delta_mat.shape[0]
        f = min(f, (n - 1) // 2)    # ensure at least 1 value survives trimming
        sorted_mat, _ = delta_mat.sort(dim=0)   # (n, d)
        trimmed = sorted_mat[f : n - f]         # (n-2f, d)
        return trimmed.mean(dim=0)

    @staticmethod
    def _cwmed(delta_mat: torch.Tensor) -> torch.Tensor:
        """Coordinate-Wise Median.

        Args:
            delta_mat: ``(n, d)``.

        Returns:
            ``(d,)`` per-coordinate median.
        """
        return delta_mat.median(dim=0).values

    @staticmethod
    def _krum(delta_mat: torch.Tensor, f: int) -> torch.Tensor:
        """Multi-Krum selection: average the (n − f) lowest-scoring updates.

        Krum score for client i = sum of squared L2 distances to its
        (n − f − 2) nearest neighbours.  The (n − f) clients with the
        lowest scores are selected and averaged.

        Args:
            delta_mat: ``(n, d)``.
            f:         Byzantine budget.

        Returns:
            ``(d,)`` mean of the selected updates.
        """
        n = delta_mat.shape[0]
        k = max(1, n - f - 2)    # neighbours per client for the Krum score
        m = max(1, n - f)        # number of updates to select

        dist_sq = torch.cdist(delta_mat, delta_mat, p=2) ** 2   # (n, n)
        dist_sq.fill_diagonal_(float("inf"))

        scores = torch.stack([
            dist_sq[i].topk(k, largest=False).values.sum()
            for i in range(n)
        ])                                                        # (n,)

        selected = scores.topk(m, largest=False).indices         # (m,)
        return delta_mat[selected].mean(dim=0)
