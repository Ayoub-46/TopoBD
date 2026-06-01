"""FEMNIST dataset with LEAF natural writer-based partitioning.

Each FL client represents one human writer — exactly the natural heterogeneity
in the plan (§1: "Natural (+ optional synthetic α=0.3)").

Data preparation (one-time, offline)
-------------------------------------
The LEAF framework preprocessing scripts must be run once before this adapter
can be used.  Two approaches:

**Option A — via the LEAF repo:**

    git clone https://github.com/TalwalkarLab/leaf.git leaf_repo
    cd leaf_repo/data/femnist
    # Install dependencies (pillow, numpy, scipy)
    pip install pillow numpy scipy
    # Preprocess: sample ~200 writers with at least 50 images each
    ./preprocess.sh -s niid --sf 0.1 -k 50 -t sample
    # Copy output to your data directory
    cp -r data/ /path/to/repo/data/femnist_leaf/

**Option B — download preprocessed JSON from a mirror:**

    # A preprocessed archive (sf=0.1, k=50, ~3,500 writers) is available
    # from the LEAF authors; see https://leaf.cmu.edu/ for the latest URL.

Expected directory layout after setup::

    data/femnist_leaf/
        train/
            all_data_0_niid_0_keep_50_train_9.json
            ...
        test/
            all_data_0_niid_0_keep_50_test_9.json
            ...

Each JSON has the schema::

    {
      "users": ["f0000_14", "f0001_41", ...],
      "num_samples": [120, 87, ...],
      "user_data": {
        "f0000_14": {
          "x": [[pixel_values, ...], ...],   # 784 floats in [0, 1]
          "y": [label_int, ...]
        }
      }
    }

62 classes: 0–9 digits, 10–35 uppercase A–Z, 36–61 lowercase a–z.
Images are 28×28 greyscale, stored flattened as 784 floats.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from .adapter import DatasetAdapter


# FEMNIST statistics (white background, dark strokes)
_MEAN = (0.9641,)
_STD  = (0.1592,)


# ---------------------------------------------------------------------------
# Internal flat Dataset backed by in-memory arrays
# ---------------------------------------------------------------------------

class _LEAFFlatDataset(Dataset):
    """Flat Dataset over raw LEAF image arrays.

    Args:
        images:    (N, 784) float32 array, values in [0, 1].
        labels:    (N,) int64 array.
        transform: Optional torchvision transform.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        img_arr = (self.images[idx].reshape(28, 28) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ---------------------------------------------------------------------------
# LEAF JSON loader
# ---------------------------------------------------------------------------

def _load_leaf_json_dir(json_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """Parse all LEAF JSON files in *json_dir*.

    Returns
    -------
    images          (N, 784) float32
    labels          (N,) int64
    writer_ids      list of unique writer IDs (in order of first appearance)
    writer_to_idxs  {writer_id: [sample_indices]}
    """
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"No LEAF JSON files found in {json_dir}. "
            "See datasets/leaf_femnist.py docstring for setup instructions."
        )

    all_images:  List[np.ndarray] = []
    all_labels:  List[int]        = []
    writer_ids:  List[str]        = []
    writer_to_idxs: Dict[str, List[int]] = {}

    offset = 0
    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        for user in data["users"]:
            if user not in writer_to_idxs:
                writer_ids.append(user)
                writer_to_idxs[user] = []
            udata = data["user_data"][user]
            imgs   = np.array(udata["x"], dtype=np.float32)   # (n, 784)
            labels = np.array(udata["y"], dtype=np.int64)      # (n,)
            n = len(labels)
            all_images.append(imgs)
            all_labels.append(labels)
            writer_to_idxs[user].extend(range(offset, offset + n))
            offset += n

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return images, labels, writer_ids, writer_to_idxs


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class LEAFFEMNISTDataset(DatasetAdapter):
    """FEMNIST adapter with LEAF natural writer-based client partitioning.

    Each call to ``get_client_loaders()`` / ``get_client_pre_loaders()``
    assigns one writer per client (sampling *num_clients* writers deterministically
    from all available writers, seeded by the *seed* argument).

    Args:
        root:         Directory that contains the ``femnist_leaf/`` sub-directory.
        download:     Ignored (LEAF data requires offline preprocessing).
        max_writers:  Cap on the number of writers loaded.  ``None`` = all.
    """

    _MEAN = _MEAN
    _STD  = _STD

    def __init__(
        self,
        root: str = "data",
        download: bool = False,
        max_writers: Optional[int] = None,
    ) -> None:
        super().__init__(
            root=root,
            download=False,
            train_pre_transform=T.ToTensor(),
            test_pre_transform=T.ToTensor(),
            normalize_transform=T.Normalize(_MEAN, _STD),
        )
        self.max_writers = max_writers
        self._train_images: Optional[np.ndarray] = None
        self._test_images:  Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_labels:  Optional[np.ndarray] = None
        self._train_writer_ids:   Optional[List[str]] = None
        self._test_writer_ids:    Optional[List[str]] = None
        self._train_writer_idxs:  Optional[Dict[str, List[int]]] = None
        self._test_writer_idxs:   Optional[Dict[str, List[int]]] = None

    # ------------------------------------------------------------------
    # DatasetAdapter interface
    # ------------------------------------------------------------------

    def load_datasets(self) -> None:
        leaf_dir = os.path.join(self.root, "femnist_leaf")

        (self._train_images, self._train_labels,
         self._train_writer_ids, self._train_writer_idxs) = \
            _load_leaf_json_dir(os.path.join(leaf_dir, "train"))

        (self._test_images, self._test_labels,
         self._test_writer_ids, self._test_writer_idxs) = \
            _load_leaf_json_dir(os.path.join(leaf_dir, "test"))

        if self.max_writers is not None:
            self._train_writer_ids = self._train_writer_ids[:self.max_writers]
            self._test_writer_ids  = self._test_writer_ids[:self.max_writers]

        self._train_dataset = _LEAFFlatDataset(
            self._train_images, self._train_labels, self.train_transform
        )
        self._test_dataset = _LEAFFlatDataset(
            self._test_images, self._test_labels, self.test_transform
        )
        self._train_pre_dataset = _LEAFFlatDataset(
            self._train_images, self._train_labels, self.train_pre_transform
        )
        self._test_pre_dataset = _LEAFFlatDataset(
            self._test_images, self._test_labels, self.test_pre_transform
        )

    @property
    def num_classes(self) -> int:
        return 62

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)

    # ------------------------------------------------------------------
    # Natural client partitioning (override base class)
    # ------------------------------------------------------------------

    def get_client_loaders(
        self,
        num_clients: int,
        batch_size: int = 64,
        strategy: str = "natural_femnist",   # accepted but ignored — always natural
        seed: int = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        **_,
    ) -> Dict[int, DataLoader]:
        self.setup()
        writers = self._sample_writers(
            self._train_writer_ids, num_clients, seed
        )
        loaders: Dict[int, DataLoader] = {}
        for cid, writer in enumerate(writers):
            indices = self._train_writer_idxs[writer]
            if not indices:
                continue
            loaders[cid] = DataLoader(
                Subset(self._train_dataset, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return loaders

    def get_client_pre_loaders(
        self,
        num_clients: int,
        batch_size: int = 64,
        strategy: str = "natural_femnist",
        seed: int = 0,
        num_workers: int = 2,
        pin_memory: bool = True,
        **_,
    ) -> Dict[int, DataLoader]:
        self.setup()
        writers = self._sample_writers(
            self._train_writer_ids, num_clients, seed
        )
        loaders: Dict[int, DataLoader] = {}
        for cid, writer in enumerate(writers):
            indices = self._train_writer_idxs[writer]
            if not indices:
                continue
            loaders[cid] = DataLoader(
                Subset(self._train_pre_dataset, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return loaders

    # ------------------------------------------------------------------
    # Override _make_partitions for atypicality computation
    # ------------------------------------------------------------------

    def _make_partitions(self, num_clients, strategy, seed, **_):
        """Return {client_id: [indices]} using natural writer assignment."""
        self.setup()
        writers = self._sample_writers(self._train_writer_ids, num_clients, seed)
        return {
            cid: list(self._train_writer_idxs[writer])
            for cid, writer in enumerate(writers)
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_writers(
        all_writers: List[str], num_clients: int, seed: int
    ) -> List[str]:
        """Deterministically sample *num_clients* writers."""
        rng = random.Random(seed)
        writers = list(all_writers)
        if len(writers) > num_clients:
            writers = sorted(rng.sample(writers, num_clients))
        return writers
