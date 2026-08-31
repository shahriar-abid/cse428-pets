"""Fast local sanity checks: device, dataset, shapes, mask values, EDA grid.

Runs on CPU/MPS in under a minute - no Kaggle GPU needed.
"""

import os
import sys
import time
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import seed_everything, get_device
from src.data import get_loaders
from src.viz import plot_overlay_grid

SEED = 42
IMG_SIZE = 128


def main():
    seed_everything(SEED)

    device = get_device()
    print(f"[1] device: {device}")
    a = torch.randn(256, 256, device=device)
    b = a @ a
    print(f"[2] {device.type} matmul ok, sum={b.sum().item():.2f}")

    t0 = time.time()
    ds, loaders = get_loaders(
        root=os.path.join(ROOT, "data"),
        img_size=IMG_SIZE,
        seed=SEED,
        augment=True,
        batch_size=8,
        num_workers=0,
        download=False,
    )
    print(
        f"[3] datasets ready in {time.time() - t0:.1f}s: "
        f"train={len(ds['train'])}, val={len(ds['val'])}, test={len(ds['test'])}"
    )
    assert len(ds["train"]) + len(ds["val"]) == 3680, "unexpected trainval size"
    assert len(ds["test"]) == 3669, "unexpected test size"

    sample = ds["train"][0]
    img, mask, label = sample["image"], sample["mask"], sample["label"]
    name = ds["train"].label_name(label.item())
    print(f"[4] sample: image {tuple(img.shape)}, mask {tuple(mask.shape)}, "
          f"label {label.item()} ({name})")
    assert img.shape == (3, IMG_SIZE, IMG_SIZE) and img.dtype == torch.float32
    assert mask.shape == (IMG_SIZE, IMG_SIZE)
    unique = set(mask.unique().tolist())
    assert unique <= {0, 1}, f"mask values not binary: {unique}"
    assert 0 <= label.item() < 37

    t0 = time.time()
    batch = next(iter(loaders["train"]))
    print(f"[5] first batch in {time.time() - t0:.2f}s: "
          f"images {tuple(batch['image'].shape)}, masks {tuple(batch['mask'].shape)}, "
          f"labels {tuple(batch['label'].shape)}")

    base_labels = getattr(ds["train"].base, "_labels", None)
    if base_labels is not None:
        train_labels = [base_labels[i] for i in ds["train"].indices]
        counts = collections.Counter(train_labels)
        print(f"[6] {len(ds['train'].classes)} classes; "
              f"train top-5: {counts.most_common(5)}")

    rep = os.path.join(ROOT, "reports")
    os.makedirs(rep, exist_ok=True)
    fig = plot_overlay_grid(
        ds["val"], figsize=(12, 12), title="EDA: random val samples with mask overlay"
    )
    out = os.path.join(rep, "eda_overlay_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[7] saved {out}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
