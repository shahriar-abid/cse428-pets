"""Checkpoint-chunked, resumable training for joint segmentation + classification."""

import glob
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from .metrics import evaluate


class DiceBCELoss(nn.Module):
    """BCE-with-logits + soft Dice for binary segmentation logits."""

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, logits.dim()))
        inter = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + self.eps) / (cardinality + self.eps)).mean()
        return bce + dice


def find_resume_checkpoint(explicit=None, out_dir=None, model_name=None):
    """Locate the newest usable checkpoint: local out_dir first, then any
    Kaggle input attached from a previous notebook version (groupmate handoff).
    Candidates whose stored config references a different model are skipped.
    """
    candidates = []
    if out_dir:
        candidates.append(os.path.join(out_dir, "checkpoints", "last.pth"))
    if explicit:
        candidates.append(explicit)
    if os.path.isdir("/kaggle/input"):
        candidates += sorted(
            glob.glob("/kaggle/input/**/checkpoints/last.pth", recursive=True)
        )
        candidates += sorted(
            glob.glob("/kaggle/input/**/outputs/*/checkpoints/last.pth", recursive=True)
        )
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if (
            model_name
            and ckpt.get("cfg", {}).get("model", {}).get("name") != model_name
        ):
            continue
        return path
    return None


class Trainer:
    """Trains epochs start..epochs_total, saving last/best checkpoints each epoch.

    Resume is fully automatic: pass resume=<path to last.pth> (see
    find_resume_checkpoint) and training continues from the next epoch with
    model, optimizer, scheduler, history and best-mIoU restored.
    """

    def __init__(self, model, loaders, device, cfg, out_dir, resume=None):
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.cfg = cfg
        self.out_dir = out_dir
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        train_cfg, model_cfg = cfg["train"], cfg["model"]
        self.epochs_total = train_cfg["epochs_total"]
        self.lambda_cls = model_cfg.get("lambda_cls", 1.0)
        self.amp = train_cfg.get("amp", True) and device.type == "cuda"

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        self.scaler = GradScaler("cuda", enabled=self.amp)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=self.epochs_total
        )
        self.seg_criterion = DiceBCELoss()
        self.cls_criterion = nn.CrossEntropyLoss()

        self.history = []
        self.start_epoch = 1
        self.best_miou = -1.0
        if resume:
            self._load_resume(resume)

    def _load_resume(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scheduler.T_max = self.epochs_total
        self.history = ckpt.get("history", [])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_miou = ckpt.get("best_miou", -1.0)
        print(
            f"resumed from {path} at epoch {ckpt['epoch']} "
            f"(best val mIoU {self.best_miou:.4f}); "
            f"training epochs {self.start_epoch}..{self.epochs_total}"
        )

    def _train_epoch(self):
        self.model.train()
        totals = {"loss": 0.0, "seg": 0.0, "cls": 0.0}
        seen = 0
        for batch in self.loaders["train"]:
            imgs = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].unsqueeze(1).to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=self.amp):
                out = self.model(imgs)
                if not isinstance(out, dict):
                    out = {"cls_logits": out}
                if out.get("seg_logits") is not None:
                    seg_loss = self.seg_criterion(out["seg_logits"], masks)
                else:
                    seg_loss = torch.zeros((), device=self.device)
                cls_loss = self.cls_criterion(out["cls_logits"], labels)
                loss = seg_loss + self.lambda_cls * cls_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            bs = imgs.size(0)
            totals["loss"] += loss.item() * bs
            totals["seg"] += seg_loss.item() * bs
            totals["cls"] += cls_loss.item() * bs
            seen += bs

        self.scheduler.step()
        lr = self.optim.param_groups[0]["lr"]
        return {
            "train_loss": totals["loss"] / seen,
            "train_seg_loss": totals["seg"] / seen,
            "train_cls_loss": totals["cls"] / seen,
            "lr": lr,
        }

    def _snapshot(self):
        return {
            "epoch": self.history[-1]["epoch"] if self.history else 0,
            "model_state": self.model.state_dict(),
            "optim_state": self.optim.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "history": self.history,
            "best_miou": self.best_miou,
            "cfg": self.cfg,
        }

    def fit(self):
        """Train new epochs only; returns the records produced in this run."""
        if self.start_epoch > self.epochs_total:
            print(
                f"all {self.epochs_total} epochs already complete - "
                "bump train.epochs_total to train further"
            )
            return []
        new_records = []
        for epoch in range(self.start_epoch, self.epochs_total + 1):
            t0 = time.time()
            train_stats = self._train_epoch()
            seg_m, cls_m = evaluate(self.model, self.loaders["val"], self.device)
            rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1), **train_stats}
            rec.update({f"val_{k}": v for k, v in seg_m.items()})
            rec.update({f"val_{k}": v for k, v in cls_m.items()})
            self.history.append(rec)
            new_records.append(rec)

            is_best = seg_m["miou"] > self.best_miou
            if is_best:
                self.best_miou = seg_m["miou"]
            torch.save(self._snapshot(), os.path.join(self.ckpt_dir, "last.pth"))
            if is_best:
                torch.save(self._snapshot(), os.path.join(self.ckpt_dir, "best.pth"))

            print(
                f"epoch {epoch}/{self.epochs_total} | "
                f"loss {rec['train_loss']:.4f} "
                f"(seg {rec['train_seg_loss']:.4f} cls {rec['train_cls_loss']:.4f}) | "
                f"val mIoU {seg_m['miou']:.4f} dice {seg_m['dice']:.4f} "
                f"pacc {seg_m['pixel_acc']:.4f} | "
                f"acc {cls_m['acc']:.4f} f1 {cls_m['f1']:.4f} | "
                f"{rec['time_s']:.1f}s"
                + (" *best*" if is_best else "")
            )
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        self.start_epoch = self.epochs_total + 1
        return new_records

    def final_report(self):
        """Load best checkpoint, evaluate on train/val/test, write results.json."""
        best_path = os.path.join(self.ckpt_dir, "best.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            print(
                f"final report with best checkpoint: epoch {ckpt['epoch']}, "
                f"val mIoU {ckpt['best_miou']:.4f}"
            )
        results = {}
        for name in ("train", "val", "test"):
            seg_m, cls_m = evaluate(self.model, self.loaders[name], self.device)
            results[name] = {"seg": seg_m, "cls": cls_m}
            print(
                f"{name:>5}: mIoU {seg_m['miou']:.4f} dice {seg_m['dice']:.4f} "
                f"pacc {seg_m['pixel_acc']:.4f} | "
                f"acc {cls_m['acc']:.4f} prec {cls_m['precision']:.4f} "
                f"rec {cls_m['recall']:.4f} f1 {cls_m['f1']:.4f}"
            )
        with open(os.path.join(self.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results
