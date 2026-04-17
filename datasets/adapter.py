"""Abstract dataset adapter for FL backdoor research.

Transform pipeline
------------------
The adapter splits transforms into two stages so that ``BackdoorDataset``
can inject triggers in raw ``[0, 1]`` pixel space:

1. ``pre_transform``  — augmentation + ``ToTensor``  (no normalisation).
2. ``normalize_transform`` — ``Normalize`` only.

Full transforms used for clean training/test datasets::

    train_transform = Compose([train_pre_transform, normalize_transform])
    test_transform  = Compose([test_pre_transform,  normalize_transform])

For backdoor datasets the adapter creates a dataset with ``pre_transform``
only, passes the trigger, then passes ``normalize_transform`` as
``BackdoorDataset.post_trigger_transform``.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from .backdoor import BackdoorDataset
from .utils import extract_labels


class DatasetAdapter(ABC):
    """Abstract adapter for FL datasets.

    Subclasses must implement:

    * :meth:`load_datasets`
    * :attr:`num_classes`
    * :attr:`input_shape`
    * :attr:`normalize_transform`

    Args:
        root:                  Directory for dataset files.
        download:              Whether to download if not present.
        train_pre_transform:   Augmentation + ToTensor (no Normalize).
        test_pre_transform:    ToTensor only (no augmentation, no Normalize).
        normalize_transform:   Normalize transform (applied after trigger).
    """

    def __init__(
        self,
        root: str = "data",
        download: bool = True,
        train_pre_transform: Optional[Callable] = None,
        test_pre_transform: Optional[Callable] = None,
        normalize_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.download = download
        self.train_pre_transform = train_pre_transform
        self.test_pre_transform = test_pre_transform
        self.normalize_transform = normalize_transform

        # Full transforms: pre + normalize, used for clean datasets
        self.train_transform = self._compose(train_pre_transform, normalize_transform)
        self.test_transform = self._compose(test_pre_transform, normalize_transform)

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        # Pre-normalisation versions — created lazily, used for backdoor datasets
        self._train_pre_dataset: Optional[Dataset] = None
        self._test_pre_dataset: Optional[Dataset] = None

    @staticmethod
    def _compose(*transforms: Optional[Callable]) -> Optional[Callable]:
        """Chain non-None transforms into a single Compose, or return None."""
        valid = [t for t in transforms if t is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return T.Compose(valid)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_datasets(self) -> None:
        """Populate ``_train_dataset``, ``_test_dataset``,
        ``_train_pre_dataset``, and ``_test_pre_dataset``.

        The ``*_pre_dataset`` variants must use only ``pre_transform``
        (no normalisation) so that triggers can be applied in [0, 1] space.
        """

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """CHW shape of a single input sample, e.g. ``(3, 32, 32)``."""

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load all dataset variants if not already loaded."""
        if self._train_dataset is None or self._test_dataset is None:
            self.load_datasets()

    @property
    def train_dataset(self) -> Dataset:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def test_dataset(self) -> Dataset:
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    @property
    def train_pre_dataset(self) -> Dataset:
        """Training dataset with pre-transform only (no Normalize)."""
        if self._train_pre_dataset is None:
            raise RuntimeError("Call setup() before accessing train_pre_dataset.")
        return self._train_pre_dataset

    @property
    def test_pre_dataset(self) -> Dataset:
        """Test dataset with pre-transform only (no Normalize)."""
        if self._test_pre_dataset is None:
            raise RuntimeError("Call setup() before accessing test_pre_dataset.")
        return self._test_pre_dataset

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def get_test_loader(
        self,
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """DataLoader over the full normalised test set."""
        self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_backdoor_test_loader(
        self,
        trigger_fn: Callable,
        target_label: int,
        batch_size: int = 256,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """DataLoader for ASR measurement.

        Uses the **pre-normalisation** test dataset so that ``trigger_fn``
        operates in [0, 1] pixel space.  Normalisation is applied via
        ``BackdoorDataset.post_trigger_transform``.

        Only non-target-class samples are included (correct ASR denominator).
        """
        self.setup()
        all_labels = extract_labels(self.test_pre_dataset)
        non_target_indices = [i for i, l in enumerate(all_labels) if l != target_label]

        if not non_target_indices:
            raise ValueError(
                f"No non-target samples for target_label={target_label}."
            )

        source = Subset(self.test_pre_dataset, non_target_indices)
        backdoor_ds = BackdoorDataset(
            original_dataset=source,
            trigger_fn=trigger_fn,
            target_label=target_label,
            post_trigger_transform=self.normalize_transform,
            poison_fraction=1.0,
            poison_exclude_target=False,  # already filtered above
        )
        return DataLoader(
            backdoor_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_client_loaders(
        self,
        num_clients: int,
        batch_size: int = 64,
        strategy: str = "iid",
        seed: int = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        **strategy_args,
    ) -> Dict[int, DataLoader]:
        """Partition the normalised training dataset among clients.

        Returns DataLoaders over the fully-normalised training data.
        Attack clients that need pre-normalisation data should call
        :meth:`get_client_pre_loaders` instead.

        Args:
            num_clients:      Number of clients.
            batch_size:       Mini-batch size.
            strategy:         ``"iid"`` or ``"dirichlet"``.
            seed:             RNG seed.
            num_workers:      DataLoader workers.
            pin_memory:       Pin host memory.
            **strategy_args:  ``alpha`` for Dirichlet (default 0.5).

        Returns:
            ``{client_id: DataLoader}``
        """
        self.setup()
        partitions = self._make_partitions(num_clients, strategy, seed, **strategy_args)
        return self._build_loaders(
            self.train_dataset, partitions, batch_size, num_workers, pin_memory,
            shuffle=True, strategy=strategy,
        )

    def get_client_pre_loaders(
        self,
        num_clients: int,
        batch_size: int = 64,
        strategy: str = "iid",
        seed: int = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        **strategy_args,
    ) -> Dict[int, DataLoader]:
        """Partition the **pre-normalisation** training dataset among clients.

        Attack clients use this to build ``BackdoorDataset`` wrappers with
        the trigger in [0, 1] space.  The partitioning is seeded identically
        to :meth:`get_client_loaders` so that clean and attack client
        assignments are consistent.
        """
        self.setup()
        partitions = self._make_partitions(num_clients, strategy, seed, **strategy_args)
        return self._build_loaders(
            self.train_pre_dataset, partitions, batch_size, num_workers, pin_memory,
            shuffle=True, strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Partition analysis
    # ------------------------------------------------------------------

    def get_partition_stats(
        self,
        partitions: Dict[int, List[int]],
    ) -> Dict[int, Dict[int, int]]:
        """Return ``{client_id: {class_label: count}}`` for a partition.

        Useful for logging non-IID degree before running an experiment.
        """
        self.setup()
        all_labels = extract_labels(self.train_dataset)
        stats: Dict[int, Dict[int, int]] = {}
        for cid, indices in partitions.items():
            counts: Dict[int, int] = {}
            for idx in indices:
                lbl = int(all_labels[idx])
                counts[lbl] = counts.get(lbl, 0) + 1
            stats[cid] = counts
        return stats

    # ------------------------------------------------------------------
    # Static partitioning methods
    # ------------------------------------------------------------------

    @staticmethod
    def partition_iid(
        dataset_size: int,
        num_clients: int,
        seed: int = 0,
    ) -> Dict[int, List[int]]:
        """Uniformly random partition."""
        rng = np.random.RandomState(seed)
        indices = np.arange(dataset_size)
        rng.shuffle(indices)
        return {i: s.tolist() for i, s in enumerate(np.array_split(indices, num_clients))}

    @staticmethod
    def partition_dirichlet(
        labels: np.ndarray,
        num_clients: int,
        alpha: float,
        seed: int,
    ) -> Dict[int, List[int]]:
        """Non-IID partition via Dirichlet distribution.

        Smaller ``alpha`` → more heterogeneous distributions.
        Guarantees that every sample is assigned to exactly one client using
        the largest-remainder method for integer rounding.
        """
        rng = np.random.RandomState(seed)
        num_classes = int(labels.max()) + 1
        label_distribution = rng.dirichlet([alpha] * num_classes, num_clients)

        class_indices = [np.where(labels == c)[0].copy() for c in range(num_classes)]
        for arr in class_indices:
            rng.shuffle(arr)

        client_partitions: List[List[int]] = [[] for _ in range(num_clients)]
        for c_idx, indices in enumerate(class_indices):
            n = len(indices)
            if n == 0:
                continue
            raw = n * label_distribution[:, c_idx] / label_distribution[:, c_idx].sum()
            counts = np.floor(raw).astype(int)
            shortfall = n - counts.sum()
            if shortfall > 0:
                order = np.argsort(raw - counts)[::-1]
                for i in range(shortfall):
                    counts[order[i % num_clients]] += 1
            start = 0
            for cid, count in enumerate(counts):
                client_partitions[cid].extend(indices[start: start + count].tolist())
                start += count

        for i in range(num_clients):
            rng.shuffle(client_partitions[i])

        return {i: client_partitions[i] for i in range(num_clients)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_partitions(
        self,
        num_clients: int,
        strategy: str,
        seed: int,
        **strategy_args,
    ) -> Dict[int, List[int]]:
        labels = extract_labels(self.train_dataset)
        s = strategy.lower()
        if s == "iid":
            return self.partition_iid(len(self.train_dataset), num_clients, seed)
        elif s == "dirichlet":
            alpha = float(strategy_args.get("alpha", 0.5))
            return self.partition_dirichlet(labels, num_clients, alpha, seed)
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid: 'iid', 'dirichlet'."
        )

    @staticmethod
    def _build_loaders(
        dataset: Dataset,
        partitions: Dict[int, List[int]],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        strategy: str,
    ) -> Dict[int, DataLoader]:
        loaders: Dict[int, DataLoader] = {}
        for cid, indices in partitions.items():
            if not indices:
                warnings.warn(
                    f"Client {cid} received 0 samples (strategy='{strategy}') "
                    "and was excluded from the returned loaders.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            loaders[cid] = DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return loaders