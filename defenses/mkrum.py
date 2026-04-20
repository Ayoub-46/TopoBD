"""Multi-Krum defense server.

Reference
---------
Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent", NeurIPS 2017.

Design in this framework
------------------------
Krum is a **filtering** defense: it scores every received update, selects the
``m`` most trustworthy ones, and discards the rest.  After filtering, standard
FedAvg is run on the survivors.

This maps cleanly to the framework's two-step server protocol:

1. ``filter_updates(true_malicious)``
   Computes Krum scores, marks ``m`` survivors, **removes** rejected updates
   from ``self._received_updates``, and returns a :class:`DetectionResult`.

2. ``aggregate()`` — **inherited unchanged from FedAvgAggregator**
   Runs weighted-average on whatever remains in ``self._received_updates``.

3. ``reset()`` — inherited.

The runner calls them in exactly that order, so no override of ``aggregate()``
is required and the return type contract (→ ``AggregationResult``) is satisfied
automatically.
"""

from __future__ import annotations

import logging
from typing import FrozenSet

import numpy as np
import torch

from fl.server import FedAvgAggregator
from experiment.utils import DetectionResult

logger = logging.getLogger(__name__)


class MKrumServer(FedAvgAggregator):
    """Multi-Krum Byzantine-robust aggregator.

    Krum scores each client update by the sum of its squared-L2 distances to
    its ``n - f - 2`` nearest neighbours, where ``n`` is the number of
    updates received and ``f`` is the assumed number of Byzantine clients.
    The ``m`` updates with the lowest scores are selected; the rest are
    discarded before aggregation.

    Args:
        model:          Global model (same as FedAvgAggregator).
        num_byzantine:  Assumed upper bound on Byzantine clients per round
                        (the ``f`` parameter).  Krum provides its guarantees
                        when ``n > 2f + 2``.
        num_to_select:  Number of updates to keep for aggregation (the ``m``
                        parameter).  ``m = 1`` recovers standard (single)
                        Krum.  ``m = n - f`` is a common choice for
                        Multi-Krum.
        testloader:     Optional evaluation DataLoader (forwarded to parent).
        device:         Torch device (forwarded to parent).
        **kwargs:       Forwarded to :class:`~fl.server.FedAvgAggregator`.

    Raises:
        ValueError: if ``num_byzantine < 0`` or ``num_to_select < 1``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_byzantine: int,
        num_to_select: int,
        **kwargs,
    ):
        if num_byzantine < 0:
            raise ValueError(f"num_byzantine must be ≥ 0, got {num_byzantine}.")
        if num_to_select < 1:
            raise ValueError(f"num_to_select must be ≥ 1, got {num_to_select}.")

        super().__init__(model=model, **kwargs)
        self.num_byzantine = num_byzantine
        self.num_to_select = num_to_select

        logger.info(
            "MKrumServer initialised — f=%d Byzantine, selecting m=%d updates.",
            num_byzantine, num_to_select,
        )

    # ------------------------------------------------------------------
    # Detection interface (called by runner before aggregate)
    # ------------------------------------------------------------------

    def filter_updates(self, true_malicious: FrozenSet[int]) -> DetectionResult:
        """Run Multi-Krum selection and prune the update buffer.

        Computes pairwise squared-L2 distances between update **deltas**
        (local params − global params), scores each client, selects the
        ``m`` lowest-scoring clients, and removes the rest from
        ``self._received_updates`` so that the inherited ``aggregate()``
        only sees the selected subset.

        If the feasibility condition ``n > 2f + 2`` is not met, Krum's
        guarantees do not hold.  In that case a warning is logged and no
        clients are filtered (all updates proceed to aggregation).

        Args:
            true_malicious: Ground-truth malicious IDs selected this round.
                            Used solely to compute TPR/FPR — not used to
                            make filtering decisions.

        Returns:
            :class:`~experiment.utils.DetectionResult` with the rejected IDs
            and ground-truth malicious IDs so the runner can compute TPR/FPR.
        """
        # FIX: build client list first so the feasibility check can reference it
        client_ids: list[int] = list(self._received_updates.keys())
        n = len(client_ids)

        if n == 0:
            logger.warning("MKrumServer.filter_updates: no updates in buffer.")
            return DetectionResult(
                rejected_ids=frozenset(),
                true_malicious=true_malicious,
            )

        # ---- Feasibility check ------------------------------------------
        if n <= 2 * self.num_byzantine + 2:
            logger.warning(
                "MKrum feasibility condition not met (n=%d, f=%d). "
                "Falling back to FedAvg on all %d updates.",
                n, self.num_byzantine, n,
            )
            return DetectionResult(
                rejected_ids=frozenset(),
                true_malicious=true_malicious,
            )

        # ---- Compute update deltas (local − global) ----------------------
        global_params = self.get_params()   # CPU tensors
        flat_deltas: list[torch.Tensor] = []

        for cid in client_ids:
            local = self._received_updates[cid]["params"]
            delta_vec = torch.cat([
                (local[k] - global_params[k]).flatten().float()
                for k in local
            ])
            flat_deltas.append(delta_vec)

        # ---- Pairwise squared-L2 distances --------------------------------
        # Stack into (n, d) matrix; use broadcasting for vectorised distance.
        delta_mat = torch.stack(flat_deltas)            # (n, d)
        # squared L2: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
        norms_sq = (delta_mat ** 2).sum(dim=1)          # (n,)
        dist_sq = (
            norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1)
            - 2.0 * delta_mat @ delta_mat.T
        ).clamp(min=0.0)                                # (n, n), numerical guard

        # ---- Krum scores --------------------------------------------------
        # Each client's score = sum of distances to its k nearest neighbours
        # (excluding itself), where k = n - f - 2.
        k = n - self.num_byzantine - 2
        scores: list[float] = []
        for i in range(n):
            row = dist_sq[i].clone()
            row[i] = float("inf")           # exclude self
            sorted_dists, _ = torch.sort(row)
            scores.append(sorted_dists[:k].sum().item())

        # ---- Select m lowest-scoring clients ------------------------------
        sorted_idx = np.argsort(scores)
        selected_idx = set(sorted_idx[:self.num_to_select].tolist())
        selected_ids: set[int] = {client_ids[i] for i in selected_idx}
        rejected_ids: FrozenSet[int] = frozenset(
            cid for cid in client_ids if cid not in selected_ids
        )

        logger.info(
            "MKrum selected %d / %d clients: %s  |  rejected: %s",
            len(selected_ids), n,
            sorted(selected_ids),
            sorted(rejected_ids),
        )

        # ---- Prune buffer so aggregate() only sees selected updates ------
        for cid in rejected_ids:
            del self._received_updates[cid]

        return DetectionResult(
            rejected_ids=rejected_ids,
            true_malicious=true_malicious,
        )