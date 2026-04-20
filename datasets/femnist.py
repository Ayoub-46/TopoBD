from typing import Tuple

from torchvision import transforms as T
from torchvision.datasets import EMNIST

from .adapter import DatasetAdapter


class FEMNISTDataset(DatasetAdapter):
    """DatasetAdapter for FEMNIST (EMNIST 'byclass' split).

    62 classes (digits 0-9, uppercase A-Z, lowercase a-z), 28×28 grayscale.
    Uses EMNIST 'byclass' as the canonical FEMNIST source — the natural
    writer-based partitioning is replaced by IID / Dirichlet splits.
    """

    _MEAN: Tuple[float, ...] = (0.1751,)
    _STD:  Tuple[float, ...] = (0.3332,)
    _SPLIT: str = "byclass"

    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(
            root=root,
            download=download,
            train_pre_transform=T.Compose([
                T.RandomCrop(28, padding=2),
                T.ToTensor(),
            ]),
            test_pre_transform=T.ToTensor(),
            normalize_transform=T.Normalize(self._MEAN, self._STD),
        )

    def load_datasets(self) -> None:
        for train, attr_full, attr_pre in [
            (True,  "_train_dataset",     "_train_pre_dataset"),
            (False, "_test_dataset",      "_test_pre_dataset"),
        ]:
            full_t = self.train_transform if train else self.test_transform
            pre_t  = self.train_pre_transform if train else self.test_pre_transform
            setattr(self, attr_full, EMNIST(
                root=self.root, split=self._SPLIT, train=train,
                transform=full_t, download=self.download,
            ))
            setattr(self, attr_pre, EMNIST(
                root=self.root, split=self._SPLIT, train=train,
                transform=pre_t, download=self.download,
            ))

    @property
    def num_classes(self) -> int:
        return 62

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)
