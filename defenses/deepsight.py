"""DeepSight defense server.

Reference
---------
Rieger, P., Nguyen, T. D., Miettinen, M., & Sadeghi, A. R.
"DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep
Model Inspection." NDSS 2022.

How it works
-----------
DeepSight inspects the final linear layer of each submitted update to detect
poisoned clients via three complementary signals:

1. **NEUPs** (Normalised Update Projections per class): for each class *c*,
   compute the unit vector of the last-layer weight delta; count the fraction
   of elements exceeding threshold ``neup_tau`` → TE (Threshold Exceeding)
   score per client.

2. **DDifs** (Distribution Differences): pass random-noise images through
   both the global model and each locally-updated model; measure the mean
   KL-divergence of their output distributions.

3. **Cosine distances**: pairwise cosine distances between last-layer bias
   deltas across clients.

The three normalised component matrices are averaged into a single distance
matrix on which HDBSCAN clusters clients.  Clusters whose mean TE exceeds
``cluster_tau`` are labelled malicious — their updates are norm-clipped and
removed from ``self._received_updates`` before the inherited FedAvg
``aggregate()`` runs on the survivors.

Design in this framework
------------------------
Follows the MKrum pattern exactly:
  ``filter_updates(true_malicious)`` prunes ``_received_updates`` and returns
  a :class:`~experiment.utils.DetectionResult`.
  ``aggregate()`` — inherited, runs FedAvg on survivors.
  ``reset()``     — inherited.
"""

from __future__ import annotations

import copy
import logging
from typing import FrozenSet, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fl.server import FedAvgAggregator
from experiment.utils import DetectionResult

logger = logging.getLogger(__name__)


class DeepSightServer(FedAvgAggregator):
    """DeepSight Byzantine-robust aggregator.

    Args:
        model:             Global model.
        device:            Torch device.
        input_shape:       Shape ``(C, H, W)`` of one input image used to
                           generate random noise for DDif computation.
        num_noise_batches: Number of random-noise images forwarded through
                           each client model to estimate DDif.
        neup_tau:          Per-element threshold for NEUP activation; higher →
                           fewer neurons count towards TE.  Range ``(0, 1)``.
        cluster_tau:       Mean-TE threshold above which a cluster is labelled
                           malicious.  Range ``(0, 1)``.
        clip_norm:         L2-norm bound applied to detected malicious updates
                           before they are removed.
        min_cluster_size:  ``min_cluster_size`` passed to HDBSCAN.
        **kwargs:          Forwarded to :class:`~fl.server.FedAvgAggregator`.

    Raises:
        RuntimeError: if the model contains no ``nn.Linear`` layer.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        num_noise_batches: int = 64,
        neup_tau: float = 0.2,
        cluster_tau: float = 0.5,
        clip_norm: float = 2.0,
        min_cluster_size: int = 2,
        **kwargs,
    ):
        super().__init__(model=model, device=device, **kwargs)
        self.input_shape      = input_shape
        self.num_noise_batches = num_noise_batches
        self.neup_tau         = neup_tau
        self.cluster_tau      = cluster_tau
        self.clip_norm        = clip_norm
        self.min_cluster_size = min_cluster_size

        self._last_w_key, self._last_b_key = self._find_last_linear_keys()

        logger.info(
            "DeepSightServer — output layer: '%s' | "
            "neup_tau=%.2f  cluster_tau=%.2f  clip_norm=%.1f",
            self._last_w_key, neup_tau, cluster_tau, clip_norm,
        )

    # ------------------------------------------------------------------
    # Detection interface (called by runner before aggregate)
    # ------------------------------------------------------------------

    def filter_updates(self, true_malicious: FrozenSet[int]) -> DetectionResult:
        """Run DeepSight detection and prune the update buffer.

        Computes NEUPs/TEs, DDifs, and cosine distances; clusters via HDBSCAN;
        labels malicious clusters; clips their updates; removes them from
        ``_received_updates`` so ``aggregate()`` runs on survivors.

        Args:
            true_malicious: Ground-truth malicious IDs this round — used only
                            for TPR/FPR bookkeeping, not for filtering.

        Returns:
            :class:`~experiment.utils.DetectionResult`.
        """
        client_ids: List[int] = list(self._received_updates.keys())
        n = len(client_ids)

        if n < 2:
            logger.warning("DeepSight: fewer than 2 updates — detection skipped.")
            return DetectionResult(rejected_ids=frozenset(), true_malicious=true_malicious)

        global_params = self.get_params()

        # ---- Feature extraction -------------------------------------------
        neup_mat, te_scores = self._calculate_neups(client_ids, global_params)
        ddif_scores         = self._calculate_ddifs(client_ids, global_params)
        cos_dist_mat        = self._calculate_cosine_distances(client_ids, global_params)

        # ---- Combined distance matrix -------------------------------------
        combined = self._build_distance_matrix(neup_mat, ddif_scores, cos_dist_mat)

        # ---- HDBSCAN clustering -------------------------------------------
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            logger.error(
                "DeepSight requires the 'hdbscan' package. "
                "Install it with: pip install hdbscan"
            )
            return DetectionResult(rejected_ids=frozenset(), true_malicious=true_malicious)

        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=max(2, self.min_cluster_size),
            metric="precomputed",
            allow_single_cluster=True,
        )
        labels: np.ndarray = clusterer.fit_predict(combined)  # (n,), -1 = noise

        # ---- Label malicious clusters ------------------------------------
        malicious_set: set[int] = set()
        for cluster_id in set(labels.tolist()) - {-1}:
            mask = labels == cluster_id
            mean_te = float(te_scores[mask].mean())
            flagged = mean_te > self.cluster_tau
            logger.debug(
                "DeepSight cluster %d: size=%d  mean_TE=%.3f  → %s",
                cluster_id, int(mask.sum()), mean_te,
                "MALICIOUS" if flagged else "benign",
            )
            if flagged:
                for i, in_cluster in enumerate(mask.tolist()):
                    if in_cluster:
                        malicious_set.add(client_ids[i])

        id_label = {cid: ("MAL" if cid in true_malicious else "ben") for cid in client_ids}
        logger.info(
            "DeepSight: %d / %d flagged  |  "
            "TE [%s]  DDif [%s]  labels %s",
            len(malicious_set), n,
            ", ".join(
                f"{id_label[cid]}:{t:.3f}"
                for cid, t in zip(client_ids, te_scores.tolist())
            ),
            ", ".join(
                f"{id_label[cid]}:{d:.3f}"
                for cid, d in zip(client_ids, ddif_scores.tolist())
            ),
            labels.tolist(),
        )

        # ---- Clip then remove detected malicious updates -----------------
        for cid in malicious_set:
            self._clip_update(cid)

        rejected_ids: FrozenSet[int] = frozenset(malicious_set)
        for cid in rejected_ids:
            del self._received_updates[cid]

        return DetectionResult(rejected_ids=rejected_ids, true_malicious=true_malicious)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _calculate_neups(
        self,
        client_ids: List[int],
        global_params: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute NEUP feature matrix and per-client TE scores.

        Returns:
            ``neup_mat`` — ``(n, num_classes * hidden_dim)`` float32 array.
            ``te_scores`` — ``(n,)`` float32 array, max-TE per client.
        """
        global_w = global_params[self._last_w_key].float()  # (C, D)
        neup_rows: List[np.ndarray] = []
        te_vals:   List[float]      = []

        for cid in client_ids:
            local_w = self._received_updates[cid]["params"][self._last_w_key].float()
            delta   = local_w - global_w                          # (C, D)

            norms = delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            neup  = delta / norms                                 # (C, D) unit vecs

            te_per_class = (neup.abs() > self.neup_tau).float().mean(dim=1)  # (C,)
            te_vals.append(te_per_class.max().item())
            neup_rows.append(neup.flatten().cpu().numpy())

        return np.stack(neup_rows, dtype=np.float32), np.array(te_vals, dtype=np.float32)

    def _calculate_ddifs(
        self,
        client_ids: List[int],
        global_params: dict,
    ) -> np.ndarray:
        """Compute mean KL-divergence (global || local) on random noise inputs.

        Returns:
            ``ddif_scores`` — ``(n,)`` float32 array.
        """
        C, H, W = self.input_shape
        dev = self.device

        # Build reproducible noise batch (fixed seeds)
        noise_tensors = []
        for seed in range(self.num_noise_batches):
            g = torch.Generator()
            g.manual_seed(seed)
            noise_tensors.append(torch.randn(1, C, H, W, generator=g))
        noise_batch = torch.cat(noise_tensors, dim=0).to(dev)  # (N, C, H, W)

        # Global model reference output
        gm = copy.deepcopy(self.model)
        gm.load_state_dict({k: v.to(dev) for k, v in global_params.items()})
        gm.eval()
        with torch.no_grad():
            g_probs = torch.softmax(gm(noise_batch), dim=-1)     # (N, num_classes)
        del gm

        eps = 1e-10
        ddif_scores: List[float] = []

        for cid in client_ids:
            lm = copy.deepcopy(self.model)
            lm.load_state_dict(
                {k: v.to(dev) for k, v in self._received_updates[cid]["params"].items()}
            )
            lm.eval()
            with torch.no_grad():
                l_probs = torch.softmax(lm(noise_batch), dim=-1)  # (N, num_classes)
            del lm

            kl = (
                g_probs * (g_probs.clamp(min=eps).log() - l_probs.clamp(min=eps).log())
            ).sum(dim=-1).mean().item()
            ddif_scores.append(max(0.0, kl))                      # KL ≥ 0 numerically

        return np.array(ddif_scores, dtype=np.float32)

    def _calculate_cosine_distances(
        self,
        client_ids: List[int],
        global_params: dict,
    ) -> np.ndarray:
        """Pairwise cosine distances between last-layer bias deltas.

        Returns:
            ``(n, n)`` float32 distance matrix with values in ``[0, 1]``.
        """
        n = len(client_ids)

        if self._last_b_key not in global_params:
            return np.zeros((n, n), dtype=np.float32)

        global_b = global_params[self._last_b_key].float()
        bias_deltas: List[torch.Tensor] = []

        for cid in client_ids:
            local_b = self._received_updates[cid]["params"].get(self._last_b_key)
            delta = (local_b.float() - global_b) if local_b is not None else torch.zeros_like(global_b)
            bias_deltas.append(delta.cpu())

        B = torch.stack(bias_deltas)                              # (n, C)
        norms = B.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        B_norm = B / norms
        cos_sim = (B_norm @ B_norm.T).clamp(-1.0, 1.0)           # (n, n)
        dist_mat = ((1.0 - cos_sim) / 2.0).numpy()
        return dist_mat.astype(np.float32)

    # ------------------------------------------------------------------
    # Distance matrix construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_distance_matrix(
        neup_mat: np.ndarray,
        ddif_scores: np.ndarray,
        cos_dist_mat: np.ndarray,
    ) -> np.ndarray:
        """Average three normalised component matrices into one distance matrix.

        Each component is min-max normalised to ``[0, 1]`` before averaging
        so no single signal dominates.

        Returns:
            Symmetric ``(n, n)`` float64 matrix with zero diagonal.
        """
        from scipy.spatial.distance import cdist

        neup_dists = cdist(neup_mat, neup_mat, metric="euclidean").astype(np.float32)

        d = ddif_scores.reshape(-1, 1)
        ddif_dists = np.abs(d - d.T).astype(np.float32)

        def _norm01(m: np.ndarray) -> np.ndarray:
            mx = float(m.max())
            return (m / mx).astype(np.float32) if mx > 1e-10 else m

        combined = (_norm01(neup_dists) + _norm01(ddif_dists) + _norm01(cos_dist_mat)) / 3.0

        # Enforce strict symmetry and zero diagonal
        combined = (combined + combined.T) / 2.0
        np.fill_diagonal(combined, 0.0)
        return combined.astype(np.float64)

    # ------------------------------------------------------------------
    # Clipping and architecture helpers
    # ------------------------------------------------------------------

    def _clip_update(self, cid: int) -> None:
        """Clip client *cid*'s update to ``self.clip_norm`` L2 norm in place."""
        params = self._received_updates[cid]["params"]
        flat   = torch.cat([v.flatten().float() for v in params.values()])
        norm   = flat.norm(p=2).item()
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            self._received_updates[cid]["params"] = {
                k: (v.float() * scale if v.is_floating_point() else v)
                for k, v in params.items()
            }

    def _find_last_linear_keys(self) -> Tuple[str, str]:
        """Return the state-dict key pair ``(weight_key, bias_key)`` for the
        last ``nn.Linear`` layer in the model.

        Raises:
            RuntimeError: if no ``nn.Linear`` is found.
        """
        last_name: Optional[str] = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_name = name

        if last_name is None:
            raise RuntimeError(
                "DeepSightServer: no nn.Linear found in the model. "
                "DeepSight requires a linear output head."
            )

        return f"{last_name}.weight", f"{last_name}.bias"
