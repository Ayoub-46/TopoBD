from typing import Tuple

from torchvision import transforms as T
from torchvision.datasets import MNIST

from .adapter import DatasetAdapter


class MNISTDataset(DatasetAdapter):
    """DatasetAdapter for MNIST. 10 classes, 1×28×28 grayscale."""

    _MEAN: Tuple[float, ...] = (0.1307,)
    _STD:  Tuple[float, ...] = (0.3081,)

    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(
            root=root,
            download=download,
            train_pre_transform=T.ToTensor(),
            test_pre_transform=T.ToTensor(),
            normalize_transform=T.Normalize(self._MEAN, self._STD),
        )

    def load_datasets(self) -> None:
        for train, attr_full, attr_pre in [
            (True,  "_train_dataset",  "_train_pre_dataset"),
            (False, "_test_dataset",   "_test_pre_dataset"),
        ]:
            full_t = self.train_transform if train else self.test_transform
            pre_t  = self.train_pre_transform if train else self.test_pre_transform
            setattr(self, attr_full, MNIST(
                root=self.root, train=train,
                transform=full_t, download=self.download,
            ))
            setattr(self, attr_pre, MNIST(
                root=self.root, train=train,
                transform=pre_t, download=self.download,
            ))

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)
