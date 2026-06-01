"""Tiny-ImageNet dataset adapter.

Tiny-ImageNet is a subset of ILSVRC with 200 classes and 64×64 RGB images:
  * train : 100,000 images (500/class)
  * val   :  10,000 images (50/class) — used as the test set (labels are public)

Download
--------
The zip (~237 MB) is fetched from the Stanford CS231n mirror on first use.
Set ``download=False`` if you pre-placed the extracted directory at
``{root}/tiny-imagenet-200/``.

Directory layout expected after extraction
------------------------------------------
    {root}/tiny-imagenet-200/
        wnids.txt
        train/{wnid}/images/*.JPEG
        val/images/*.JPEG
        val/val_annotations.txt
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .adapter import DatasetAdapter

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_ZIP_NAME = "tiny-imagenet-200.zip"
_DIR_NAME = "tiny-imagenet-200"

# Per-dataset normalisation statistics (computed over the training split).
_MEAN: Tuple[float, float, float] = (0.4802, 0.4481, 0.3975)
_STD:  Tuple[float, float, float] = (0.2302, 0.2265, 0.2262)


# ---------------------------------------------------------------------------
# Internal Dataset class
# ---------------------------------------------------------------------------

class _TinyImageNet(Dataset):
    """PyTorch Dataset for Tiny-ImageNet train or val split.

    Args:
        root:      Directory that contains the ``tiny-imagenet-200/`` folder.
        train:     If True, loads the training split; otherwise the val split.
        transform: Optional callable applied to each PIL Image.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        data_dir = os.path.join(root, _DIR_NAME)

        wnids_path = os.path.join(data_dir, "wnids.txt")
        if not os.path.isfile(wnids_path):
            raise FileNotFoundError(
                f"wnids.txt not found at {wnids_path}. "
                "Pass download=True or extract the dataset manually."
            )
        with open(wnids_path) as f:
            classes: List[str] = sorted(f.read().splitlines())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        if train:
            self.samples = self._load_train(data_dir)
        else:
            self.samples = self._load_val(data_dir)

    def _load_train(self, data_dir: str) -> List[Tuple[str, int]]:
        samples = []
        train_dir = os.path.join(data_dir, "train")
        for wnid, label in self.class_to_idx.items():
            img_dir = os.path.join(train_dir, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    samples.append((os.path.join(img_dir, fname), label))
        return samples

    def _load_val(self, data_dir: str) -> List[Tuple[str, int]]:
        ann_path = os.path.join(data_dir, "val", "val_annotations.txt")
        img_dir  = os.path.join(data_dir, "val", "images")
        samples  = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                if wnid in self.class_to_idx:
                    samples.append(
                        (os.path.join(img_dir, fname), self.class_to_idx[wnid])
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------

class TinyImageNetDataset(DatasetAdapter):
    """DatasetAdapter for Tiny-ImageNet (200 classes, 64×64 RGB).

    Transform pipeline
    ------------------
    ``train_pre_transform``:
        RandomCrop(64, pad=8) → RandomHorizontalFlip → ToTensor
    ``test_pre_transform``:
        ToTensor
    ``normalize_transform``:
        Normalize(_MEAN, _STD)

    The val split is used as both validation and test set (labels are public;
    there is no labelled test set in the public release).
    """

    _MEAN = _MEAN
    _STD  = _STD

    def __init__(self, root: str = "data", download: bool = True) -> None:
        super().__init__(
            root=root,
            download=download,
            train_pre_transform=T.Compose([
                T.RandomCrop(64, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]),
            test_pre_transform=T.ToTensor(),
            normalize_transform=T.Normalize(_MEAN, _STD),
        )

    # ------------------------------------------------------------------
    # DatasetAdapter interface
    # ------------------------------------------------------------------

    def load_datasets(self) -> None:
        if self.download:
            _download(self.root)

        for train, attr_full, attr_pre in [
            (True,  "_train_dataset",     "_train_pre_dataset"),
            (False, "_test_dataset",      "_test_pre_dataset"),
        ]:
            full_t = self.train_transform if train else self.test_transform
            pre_t  = self.train_pre_transform if train else self.test_pre_transform
            setattr(self, attr_full, _TinyImageNet(self.root, train=train, transform=full_t))
            setattr(self, attr_pre,  _TinyImageNet(self.root, train=train, transform=pre_t))

    @property
    def num_classes(self) -> int:
        return 200

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 64, 64)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download(root: str) -> None:
    """Download and extract Tiny-ImageNet if not already present."""
    data_dir = os.path.join(root, _DIR_NAME)
    if os.path.isdir(data_dir) and os.path.isfile(os.path.join(data_dir, "wnids.txt")):
        return  # already extracted

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, _ZIP_NAME)

    if not os.path.isfile(zip_path):
        print(f"Downloading Tiny-ImageNet (~237 MB) from {_URL} …")
        _urlretrieve_with_progress(_URL, zip_path)
        print()

    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    os.remove(zip_path)
    print(f"Tiny-ImageNet ready at {data_dir}")


def _urlretrieve_with_progress(url: str, dest: str) -> None:
    """Download url → dest, printing a simple progress indicator."""
    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(count * block_size, total_size)
        pct = 100 * done // total_size
        bar = "=" * (pct // 2) + " " * (50 - pct // 2)
        print(f"\r  [{bar}] {pct:3d}%  {done // 1_048_576} / {total_size // 1_048_576} MB",
              end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
