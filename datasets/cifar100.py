from typing import List, Tuple

from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from .adapter import DatasetAdapter


class CIFAR100Dataset(DatasetAdapter):
    """DatasetAdapter for CIFAR-100.

    Transform pipeline
    ------------------
    ``train_pre_transform``:
        RandomCrop(32, pad=4) → RandomHorizontalFlip → ToTensor
    ``test_pre_transform``:
        ToTensor
    ``normalize_transform``:
        Normalize(CIFAR100_MEAN, CIFAR100_STD)

    Four dataset variants are loaded by :meth:`load_datasets`:

    * ``_train_dataset``     — full transform (clean training)
    * ``_test_dataset``      — full transform (clean evaluation)
    * ``_train_pre_dataset`` — pre-transform only (attack client training)
    * ``_test_pre_dataset``  — pre-transform only (ASR evaluation)
    """

    _MEAN: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    _STD:  Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)

    CLASS_NAMES: List[str] = [
        "apple", "aquarium_fish", "baby", "bear", "beaver",
        "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly",
        "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach",
        "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox",
        "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard",
        "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid",
        "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum",
        "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew",
        "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe",
        "whale", "willow_tree", "wolf", "woman", "worm",
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
        self._train_dataset = CIFAR100(
            root=self.root, train=True,
            transform=self.train_transform, download=self.download,
        )
        self._test_dataset = CIFAR100(
            root=self.root, train=False,
            transform=self.test_transform, download=self.download,
        )
        self._train_pre_dataset = CIFAR100(
            root=self.root, train=True,
            transform=self.train_pre_transform, download=self.download,
        )
        self._test_pre_dataset = CIFAR100(
            root=self.root, train=False,
            transform=self.test_pre_transform, download=self.download,
        )

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
