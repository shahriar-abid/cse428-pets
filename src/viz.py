"""Visualization helpers: mask overlays and the required 3x3 exploration grid."""

import numpy as np
import matplotlib.pyplot as plt

FOREGROUND_COLOR = (1.0, 0.0, 0.0)


def overlay_mask(image, mask, alpha=0.5, color=FOREGROUND_COLOR):
    """Blend the foreground mask over an image (HxWx3 float in [0, 1])."""
    image = np.asarray(image, dtype=np.float32).copy()
    mask = np.asarray(mask).astype(bool)
    color = np.asarray(color, dtype=np.float32)
    if mask.any():
        image[mask] = (1.0 - alpha) * image[mask] + alpha * color
    return np.clip(image, 0.0, 1.0)


def plot_overlay_grid(
    dataset,
    indices=None,
    alpha=0.5,
    nrows=3,
    ncols=3,
    figsize=(12, 12),
    title=None,
    save_path=None,
):
    """3x3 grid of images with mask overlays (project requirement)."""
    n = nrows * ncols
    if indices is None:
        indices = np.random.choice(len(dataset), size=n, replace=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for ax, idx in zip(np.ravel(axes), indices):
        sample = dataset[int(idx)]
        img = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()
        ax.imshow(overlay_mask(img, mask, alpha=alpha))
        ax.set_title(
            f"idx {int(idx)} | {dataset.label_name(sample['label'].item())}",
            fontsize=9,
        )
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
