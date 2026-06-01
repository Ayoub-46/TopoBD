"""Topological analyser: persistent-homology diagram computation.

Wraps ``ripser`` to compute Vietoris-Rips persistence diagrams from a point
cloud or a precomputed distance matrix.  Returns diagrams in a unified
``(n_features, 3)`` array with columns ``[birth, death, dimension]`` so that
callers can filter by dimension without knowing ripser's internal layout.

Supported metrics
-----------------
* String metrics forwarded directly to ripser (``"euclidean"``, ``"cosine"``).
* Callable metrics — the pairwise distance matrix is computed first via
  ``scipy.spatial.distance.pdist`` and passed to ripser as ``"precomputed"``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TopologicalAnalyser:
    """Compute Vietoris-Rips persistent homology diagrams.

    Args:
        homology_dimensions: Sequence of homology degrees to compute.
                             ``(0,)`` computes connected components (H0) only.
        metric:              Either a string accepted by ripser / scipy (e.g.
                             ``"euclidean"``, ``"cosine"``) or a callable
                             ``f(u, v, **metric_params) -> float``.
        metric_params:       Extra keyword arguments forwarded to a callable
                             metric.  Ignored for string metrics.
    """

    def __init__(
        self,
        homology_dimensions: Tuple[int, ...] = (0,),
        metric: Union[str, Callable] = "euclidean",
        metric_params: Optional[dict] = None,
    ):
        self.homology_dimensions = homology_dimensions
        self.metric              = metric
        self.metric_params       = metric_params or {}
        self._max_dim            = max(homology_dimensions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_diagram(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Compute the persistence diagram of a point cloud.

        Args:
            data: ``(n_points, n_features)`` float array.

        Returns:
            ``(n_features, 3)`` array with columns ``[birth, death, dimension]``,
            or ``None`` if computation fails (ripser not installed, empty input,
            etc.).  The last connected component in H0 has ``death = inf``.
        """
        if data.ndim != 2 or data.shape[0] < 2:
            logger.warning("TopologicalAnalyser: need ≥ 2 points, got shape %s.", data.shape)
            return None

        try:
            from ripser import ripser
        except ImportError:
            logger.error(
                "TopologicalAnalyser requires 'ripser'. Install: pip install ripser"
            )
            return None

        try:
            # Always use a precomputed distance matrix so ripser never sees a
            # square input that it might misinterpret as a distance matrix.
            dist_matrix = self._pairwise(data)
            result = ripser(dist_matrix, maxdim=self._max_dim,
                            metric="precomputed", distance_matrix=True)

            parts = []
            for dim in self.homology_dimensions:
                if dim >= len(result["dgms"]):
                    continue
                dgm     = result["dgms"][dim]            # (k, 2)
                dim_col = np.full((len(dgm), 1), float(dim))
                parts.append(np.hstack([dgm, dim_col]))  # (k, 3)

            return np.vstack(parts) if parts else None

        except Exception as exc:
            logger.warning("TopologicalAnalyser: ripser failed (%s).", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pairwise(self, data: np.ndarray) -> np.ndarray:
        """Compute the full pairwise distance matrix for any metric."""
        from scipy.spatial.distance import pdist, squareform

        if callable(self.metric):
            params = self.metric_params
            def _dist(u: np.ndarray, v: np.ndarray) -> float:
                return self.metric(u, v, **params)   # type: ignore[operator]
            return squareform(pdist(data, metric=_dist))

        # String metric (e.g. "euclidean", "cosine") — scipy handles it natively
        return squareform(pdist(data, metric=self.metric))
