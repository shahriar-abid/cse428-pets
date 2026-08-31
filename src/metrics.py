"""Metrics: segmentation (mIoU, Dice, pixel accuracy) and classification (acc/precision/recall/F1)."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class SegMetricAccumulator:
    """Streaming per-class intersection/union counts for binary segmentation."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.num_classes, dtype=torch.float64)
        self.union = torch.zeros(self.num_classes, dtype=torch.float64)
        self.correct = 0.0
        self.total = 0.0

    @torch.no_grad()
    def update(self, preds, targets):
        preds = preds.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)
        for c in range(self.num_classes):
            p, t = preds == c, targets == c
            self.inter[c] += (p & t).sum().item()
            self.union[c] += (p | t).sum().item()
        self.correct += (preds == targets).sum().item()
        self.total += preds.numel()

    def compute(self):
        iou = self.inter / self.union.clamp(min=1.0)
        dice = 2.0 * self.inter / (self.inter + self.union).clamp(min=1.0)
        present = self.union > 0
        if present.any():
            miou = iou[present].mean().item()
            mdice = dice[present].mean().item()
        else:
            miou = mdice = 1.0
        return {
            "miou": miou,
            "dice": mdice,
            "iou_fg": iou[1].item(),
            "dice_fg": dice[1].item(),
            "pixel_acc": self.correct / max(self.total, 1.0),
        }


class ClsMetricAccumulator:
    """Collects predictions, reports macro-averaged classification metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    @torch.no_grad()
    def update(self, logits, targets):
        self.preds.extend(logits.argmax(1).detach().cpu().tolist())
        self.targets.extend(targets.detach().cpu().tolist())

    def compute(self):
        p, t = np.asarray(self.preds), np.asarray(self.targets)
        precision, recall, f1, _ = precision_recall_fscore_support(
            t, p, average="macro", zero_division=0
        )
        return {
            "acc": accuracy_score(t, p),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """Run both heads over a loader; returns (seg_metrics, cls_metrics).

    Works for joint models (dict outputs) and classifier-only models (tensor outputs).
    """
    model.eval()
    seg_acc = SegMetricAccumulator()
    cls_acc = ClsMetricAccumulator()
    for batch in loader:
        out = model(batch["image"].to(device))
        if not isinstance(out, dict):
            out = {"cls_logits": out}
        cls_acc.update(out["cls_logits"], batch["label"])
        seg_logits = out.get("seg_logits")
        if seg_logits is not None:
            preds = (torch.sigmoid(seg_logits) > threshold).long().squeeze(1)
            seg_acc.update(preds, batch["mask"])
    return seg_acc.compute(), cls_acc.compute()
