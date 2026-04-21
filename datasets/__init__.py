"""The ``datasets`` package.

Public API
----------
:class:`DatasetAdapter`
    Abstract base class.  Subclass this to add new datasets.

:class:`BackdoorDataset`
    Dataset wrapper that injects a trigger into a configurable fraction of
    samples and re-labels them to a target class.

:class:`CIFAR10Dataset`
    Ready-to-use adapter for CIFAR-10.

:func:`extract_labels`
    Standalone utility for extracting a label array from any
    ``torch.utils.data.Dataset``, including ``Subset`` wrappers.

Typical usage::

    from datasets import CIFAR10Dataset

    adapter = CIFAR10Dataset(root="data", download=True)
    adapter.setup()

    # Centralised test loader
    test_loader = adapter.get_test_loader(batch_size=256)

    # ASR test loader (non-target samples only, all triggered)
    asr_loader = adapter.get_backdoor_test_loader(
        trigger_fn=my_trigger, target_label=0
    )

    # Partitioned client loaders (non-IID Dirichlet split)
    client_loaders = adapter.get_client_loaders(
        num_clients=100, strategy="dirichlet", alpha=0.5, seed=42
    )
"""

from .adapter import DatasetAdapter
from .backdoor import BackdoorDataset
from .cifar10 import CIFAR10Dataset
from .femnist import FEMNISTDataset
from .gtsrb import GTSRBDataset
from .mnist import MNISTDataset
from .utils import extract_labels

__all__ = [
    "DatasetAdapter",
    "BackdoorDataset",
    "CIFAR10Dataset",
    "FEMNISTDataset",
    "GTSRBDataset",
    "MNISTDataset",
    "extract_labels",
]