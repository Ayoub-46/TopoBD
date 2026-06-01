"""Custom distance metrics for topological analysis of FL update vectors."""

from __future__ import annotations

import numpy as np


def magnitude_cosine_distance(
    u: np.ndarray,
    v: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """Weighted combination of cosine distance and relative magnitude difference.

    Captures both the *direction* divergence (cosine) and the *scale*
    divergence (magnitude ratio) between two vectors::

        d(u, v) = α · cos_dist(u, v)
                + (1−α) · |‖u‖ − ‖v‖| / (‖u‖ + ‖v‖ + ε)

    Args:
        u, v:  1-D float arrays of equal length.
        alpha: Weight in ``[0, 1]``.  ``1.0`` → pure cosine distance;
               ``0.0`` → pure magnitude ratio.  Default ``0.5``.

    Returns:
        Scalar distance value in ``[0, 1]``.
    """
    eps = 1e-10
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u < eps or norm_v < eps:
        cos_dist = 1.0
    else:
        cos_sim  = float(np.dot(u, v) / (norm_u * norm_v))
        cos_dist = float(np.clip(1.0 - cos_sim, 0.0, 2.0))

    mag_dist = abs(norm_u - norm_v) / (norm_u + norm_v + eps)

    return alpha * cos_dist + (1.0 - alpha) * mag_dist
