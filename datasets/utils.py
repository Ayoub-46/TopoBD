"""Shared low-level utilities for the datasets package.

Keeping these here avoids duplicating logic across ``adapter.py`` and
``backdoor.py``, and gives a single place to extend label-extraction
heuristics when new dataset types are added.
"""

import warnings

import numpy as np
from torch.utils.data import Dataset, Subset


def extract_labels(ds: Dataset) -> np.ndarray:
    """Extract all integer class labels from a dataset as a 1-D numpy array.

    Resolution order:
    1. ``.targets`` / ``.labels``   — torchvision standard (CIFAR, MNIST, …).
    2. ``.samples`` / ``._samples`` — ``ImageFolder`` / GTSRB style
                                      (list of ``(path, label)`` tuples).
    3. ``Subset``                   — delegates to the inner dataset, then
                                      slices by index.
    4. Slow O(n) iteration fallback with a ``RuntimeWarning``.

    Args:
        ds: Any ``torch.utils.data.Dataset`` instance.

    Returns:
        1-D ``np.ndarray`` of ``int`` labels with length ``len(ds)``.
    """
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))

    for attr in ("samples", "_samples"):
        if hasattr(ds, attr):
            return np.asarray([s[1] for s in getattr(ds, attr)])

    if isinstance(ds, Subset):
        inner_labels = extract_labels(ds.dataset)   # type: ignore[arg-type]
        return inner_labels[np.asarray(ds.indices)]

    warnings.warn(
        f"Dataset {type(ds).__name__} has no recognised label attribute. "
        "Falling back to slow O(n) label extraction. "
        "Consider adding a 'targets' property to your dataset class.",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.asarray([int(ds[i][1]) for i in range(len(ds))])  # type: ignore[arg-type]