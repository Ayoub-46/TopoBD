from typing import List, Tuple

from torchvision import transforms as T
from torchvision.datasets import CIFAR10

from .adapter import DatasetAdapter


class CIFAR10Dataset(DatasetAdapter):
    """DatasetAdapter for CIFAR-10.

    Transform pipeline
    ------------------
    ``train_pre_transform``:
        RandomCrop(32, pad=4) → RandomHorizontalFlip → ToTensor
    ``test_pre_transform``:
        ToTensor
    ``normalize_transform``:
        Normalize(CIFAR10_MEAN, CIFAR10_STD)

    Four dataset variants are loaded by :meth:`load_datasets`:

    * ``_train_dataset``     — full transform (clean training)
    * ``_test_dataset``      — full transform (clean evaluation)
    * ``_train_pre_dataset`` — pre-transform only (attack client training)
    * ``_test_pre_dataset``  — pre-transform only (ASR evaluation)
    """

    _MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    _STD:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    CLASS_NAMES: List[str] = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(
            root=root,
            download=download,
            train_pre_transform=T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]),
            test_pre_transform=T.ToTensor(),
            normalize_transform=T.Normalize(self._MEAN, self._STD),
        )

    # ------------------------------------------------------------------
    # DatasetAdapter abstract interface
    # ------------------------------------------------------------------

    def load_datasets(self) -> None:
        # Full-transform variants for clean training and evaluation
        self._train_dataset = CIFAR10(
            root=self.root, train=True,
            transform=self.train_transform, download=self.download,
        )
        self._test_dataset = CIFAR10(
            root=self.root, train=False,
            transform=self.test_transform, download=self.download,
        )
        # Pre-normalisation variants for backdoor data construction
        self._train_pre_dataset = CIFAR10(
            root=self.root, train=True,
            transform=self.train_pre_transform, download=self.download,
        )
        self._test_pre_dataset = CIFAR10(
            root=self.root, train=False,
            transform=self.test_pre_transform, download=self.download,
        )

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)