"""BackdoorDataset — injects a trigger into a configurable fraction of samples.

Value-range contract
--------------------
``trigger_fn`` receives tensors in **raw [0, 1] pixel space** (post-ToTensor,
pre-Normalize).  The ``post_trigger_transform`` (typically the dataset's
``Normalize``) is applied to every sample — clean and poisoned alike — after
the trigger step.

``DatasetAdapter`` is responsible for constructing ``BackdoorDataset`` with a
dataset that has NOT yet been normalised, and for supplying the corresponding
``normalize_transform`` as ``post_trigger_transform``.
"""

import logging
import warnings
from typing import Callable, FrozenSet, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

from .utils import extract_labels

logger = logging.getLogger(__name__)


class BackdoorDataset(Dataset):
    """Dataset wrapper that backdoors a subset of samples.

    Args:
        original_dataset:       Dataset returning ``(tensor [0,1], label)`` —
                                ToTensor applied, Normalize NOT yet applied.
        trigger_fn:             ``(image: Tensor[0,1]) -> Tensor[0,1]``.
                                Must be deterministic and CPU-only.
        target_label:           Label assigned to every poisoned sample.
        post_trigger_transform: Applied to every sample after the (optional)
                                trigger — pass the dataset's ``Normalize``
                                transform here.
        poison_fraction:        Fraction of eligible samples to poison.
        seed:                   RNG seed for reproducible index selection.
        poison_exclude_target:  When ``True`` (default) samples already
                                labelled ``target_label`` are not eligible
                                for poisoning.
        cache:                  Pre-materialise all samples into RAM.
                                Suitable for CIFAR-scale datasets.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        trigger_fn: Callable,
        target_label: int,
        post_trigger_transform: Optional[Callable] = None,
        poison_fraction: float = 1.0,
        seed: int = 0,
        poison_exclude_target: bool = True,
        cache: bool = False,
    ):
        if not (0.0 <= poison_fraction <= 1.0):
            raise ValueError(
                f"poison_fraction must be in [0, 1], got {poison_fraction}."
            )

        self.original_dataset = original_dataset
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.post_trigger_transform = post_trigger_transform
        self.poison_fraction = poison_fraction

        dataset_size = len(self.original_dataset)  # type: ignore[arg-type]
        all_indices = np.arange(dataset_size)

        # ------------------------------------------------------------------
        # Eligible sample selection
        # ------------------------------------------------------------------
        if poison_exclude_target:
            labels = extract_labels(original_dataset)
            eligible_indices = all_indices[labels != target_label]
        else:
            eligible_indices = all_indices

        num_poisoned = int(len(eligible_indices) * poison_fraction)
        rng = np.random.RandomState(seed)
        poisoned_array = (
            rng.choice(eligible_indices, num_poisoned, replace=False)
            if num_poisoned > 0
            else np.array([], dtype=np.int64)
        )
        self._poisoned_indices: FrozenSet[int] = frozenset(poisoned_array.tolist())

        logger.info(
            "BackdoorDataset: %d / %d samples poisoned "
            "(target=%d, fraction=%.3f, seed=%d).",
            num_poisoned, dataset_size, target_label, poison_fraction, seed,
        )

        # ------------------------------------------------------------------
        # Optional eager cache
        # ------------------------------------------------------------------
        self._cached_data: Optional[List] = None
        self._cached_labels: Optional[List] = None
        if cache:
            self._build_cache(dataset_size)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def poisoned_indices(self) -> FrozenSet[int]:
        """Immutable set of local indices whose input carries a trigger."""
        return self._poisoned_indices

    @property
    def poison_rate(self) -> float:
        """Fraction of *this* dataset's samples that are poisoned.

        Note: relative to the wrapped dataset size, not the full original.
        """
        n = len(self.original_dataset)  # type: ignore[arg-type]
        return len(self._poisoned_indices) / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._cached_data is not None:
            return len(self._cached_data)
        return len(self.original_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Tuple:
        # Fast path — post_trigger_transform already baked in during caching
        if self._cached_data is not None:
            return self._cached_data[index], self._cached_labels[index]  # type: ignore[index]

        data, label = self.original_dataset[index]

        # Apply trigger in [0, 1] pre-normalisation space
        if index in self._poisoned_indices:
            data = self.trigger_fn(data)
            label = self.target_label

        # Apply normalisation (and any other post-trigger ops)
        if self.post_trigger_transform is not None:
            data = self.post_trigger_transform(data)

        return data, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache(self, dataset_size: int) -> None:
        cached_data, cached_labels = [], []
        for i in range(dataset_size):
            data, label = self.original_dataset[i]
            if i in self._poisoned_indices:
                data = self.trigger_fn(data)
                label = self.target_label
            if self.post_trigger_transform is not None:
                data = self.post_trigger_transform(data)
            cached_data.append(data)
            cached_labels.append(label)
        self._cached_data = cached_data
        self._cached_labels = cached_labels
        logger.info("BackdoorDataset: cache built (%d samples).", dataset_size)