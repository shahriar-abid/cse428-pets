"""Visualization: mask overlays, 3x3 exploration grid, prediction grid, training curves."""

import numpy as np
import torch
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


@torch.no_grad()
def plot_prediction_grid(
    dataset,
    model,
    device,
    indices=None,
    nrows=3,
    threshold=0.5,
    figsize=(13, 13),
    title=None,
    save_path=None,
):
    """Required prediction format: image | ground-truth overlay | prediction overlay."""
    model.eval()
    if indices is None:
        indices = np.random.choice(len(dataset), size=nrows, replace=False)
    fig, axes = plt.subplots(nrows, 3, figsize=figsize, squeeze=False)
    for r, idx in enumerate(indices):
        sample = dataset[int(idx)]
        out = model(sample["image"].unsqueeze(0).to(device))
        if not isinstance(out, dict):
            out = {"cls_logits": out}
        prob = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        pred = prob > threshold
        probs = torch.softmax(out["cls_logits"], dim=1)[0]
        conf, pred_label = probs.max(0)
        pred_name = dataset.label_name(pred_label.item())
        true_name = dataset.label_name(sample["label"].item())
        gt = sample["mask"].numpy().astype(bool)
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = inter / max(union, 1)
        img = sample["image"].permute(1, 2, 0).numpy()
        cols = [
            (f"idx {int(idx)} | input", img, None),
            (f"ground truth | {true_name}", img, gt),
            (f"prediction | {pred_name} {conf:.0%} | IoU {iou:.2f}", img, pred),
        ]
        for c, (col_title, im, mask) in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(overlay_mask(im, mask) if mask is not None else im)
            ax.set_title(col_title, fontsize=9)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_history(history, save_path=None):
    """Loss curves + per-epoch validation metrics (project requirement)."""
    if not history:
        raise ValueError("history is empty")
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="total")
    axes[0].plot(epochs, [h["train_seg_loss"] for h in history], label="segmentation")
    axes[0].plot(
        epochs, [h["train_cls_loss"] for h in history], label="classification"
    )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, [h["val_miou"] for h in history], label="mIoU")
    axes[1].plot(epochs, [h["val_dice"] for h in history], label="Dice")
    axes[1].plot(epochs, [h["val_pixel_acc"] for h in history], label="pixel acc")
    axes[1].set_title("Validation — segmentation")
    axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, [h["val_acc"] for h in history], label="accuracy")
    axes[2].plot(epochs, [h["val_precision"] for h in history], label="precision")
    axes[2].plot(epochs, [h["val_recall"] for h in history], label="recall")
    axes[2].plot(epochs, [h["val_f1"] for h in history], label="F1")
    axes[2].set_title("Validation — classification")
    axes[2].set_xlabel("epoch")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
