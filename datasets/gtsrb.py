from typing import Tuple

from torchvision import transforms as T
from torchvision.datasets import GTSRB

from .adapter import DatasetAdapter


class GTSRBDataset(DatasetAdapter):
    """DatasetAdapter for GTSRB (German Traffic Sign Recognition Benchmark).

    Images are resized to 32×32 to match the CIFAR-10 input convention.
    43 traffic-sign classes.
    """

    _MEAN: Tuple[float, float, float] = (0.3403, 0.3121, 0.3214)
    _STD:  Tuple[float, float, float] = (0.2724, 0.2608, 0.2669)

    def __init__(self, root: str = "data", download: bool = True):
        super().__init__(
            root=root,
            download=download,
            train_pre_transform=T.Compose([
                T.Resize((32, 32)),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
            ]),
            test_pre_transform=T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
            ]),
            normalize_transform=T.Normalize(self._MEAN, self._STD),
        )

    def load_datasets(self) -> None:
        for split, attr_full, attr_pre in [
            ("train", "_train_dataset", "_train_pre_dataset"),
            ("test",  "_test_dataset",  "_test_pre_dataset"),
        ]:
            full_t = self.train_transform if split == "train" else self.test_transform
            pre_t  = self.train_pre_transform if split == "train" else self.test_pre_transform
            setattr(self, attr_full, GTSRB(
                root=self.root, split=split, transform=full_t, download=self.download,
            ))
            setattr(self, attr_pre, GTSRB(
                root=self.root, split=split, transform=pre_t, download=self.download,
            ))

    @property
    def num_classes(self) -> int:
        return 43

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
