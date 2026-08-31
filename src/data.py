"""Oxford-IIIT Pet data loading, splitting and preprocessing.

Trimap convention: 1 = foreground, 2 = background, 3 = not classified (boundary).
Per the project guidelines, boundary pixels are merged into the foreground, so the
binary mask is: 0 where trimap == 2 (background), 1 everywhere else.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

NUM_CLASSES = 37

KAGGLE_DATA_CANDIDATES = [
    "/kaggle/input/oxford-iiit-pet",
    "/kaggle/input/oxford-iiit-pets",
]


def resolve_data_root(configured=None) -> str:
    if configured:
        return configured
    env = os.environ.get("CSE428_DATA_ROOT")
    if env and os.path.isdir(env):
        return env
    for cand in KAGGLE_DATA_CANDIDATES:
        if os.path.isdir(os.path.join(cand, "oxford-iiit-pet")):
            return cand
        if os.path.isdir(os.path.join(cand, "images")) and os.path.isdir(
            os.path.join(cand, "annotations")
        ):
            return os.path.dirname(cand)
    return "./data"


def make_splits(n_total: int, val_frac: float = 0.1, seed: int = 42):
    """Deterministic train/val index split - identical on every machine/session."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    n_val = int(round(n_total * val_frac))
    return perm[n_val:], perm[:n_val]


class PetSegDataset(Dataset):
    """Wraps torchvision's OxfordIIITPet to return image, binary mask and breed label.

    The raw trimap stays accessible via return_trimap=True (needed for the
    3-class bonus task).
    """

    def __init__(
        self,
        root,
        split="trainval",
        indices=None,
        img_size=256,
        augment=False,
        download=False,
        return_trimap=False,
    ):
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types=("category", "segmentation"),
            download=download,
        )
        self.split = split
        self.indices = (
            np.arange(len(self.base)) if indices is None else np.asarray(indices)
        )
        self.img_size = img_size
        self.augment = augment
        self.return_trimap = return_trimap
        self.classes = list(self.base.classes)
        self.jitter = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def label_name(self, idx: int) -> str:
        return self.classes[int(idx)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, (label, trimap) = self.base[int(self.indices[i])]
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        trimap = TF.resize(
            trimap,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        if self.augment:
            if torch.rand(()) < 0.5:
                img = TF.hflip(img)
                trimap = TF.hflip(trimap)
            img = self.jitter(img)
        mask = (np.array(trimap, dtype=np.uint8) != 2).astype(np.int64)
        sample = {
            "image": TF.to_tensor(img),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(int(label), dtype=torch.long),
        }
        if self.return_trimap:
            sample["trimap"] = torch.from_numpy(np.array(trimap, dtype=np.int64))
        return sample


def get_datasets(
    root=None,
    img_size=256,
    val_frac=0.1,
    seed=42,
    augment=True,
    download=False,
    return_trimap=False,
):
    root = resolve_data_root(root)
    probe = PetSegDataset(root, split="trainval", img_size=img_size, download=download)
    train_idx, val_idx = make_splits(len(probe), val_frac=val_frac, seed=seed)
    train_ds = PetSegDataset(
        root, "trainval", indices=train_idx, img_size=img_size,
        augment=augment, return_trimap=return_trimap,
    )
    val_ds = PetSegDataset(
        root, "trainval", indices=val_idx, img_size=img_size,
        augment=False, return_trimap=return_trimap,
    )
    test_ds = PetSegDataset(
        root, "test", img_size=img_size, augment=False, return_trimap=return_trimap
    )
    return train_ds, val_ds, test_ds


def get_loaders(
    root=None,
    img_size=256,
    val_frac=0.1,
    seed=42,
    augment=True,
    batch_size=16,
    num_workers=2,
    download=False,
    return_trimap=False,
):
    train_ds, val_ds, test_ds = get_datasets(
        root=root,
        img_size=img_size,
        val_frac=val_frac,
        seed=seed,
        augment=augment,
        download=download,
        return_trimap=return_trimap,
    )
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common
        ),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    }
    return datasets, loaders
