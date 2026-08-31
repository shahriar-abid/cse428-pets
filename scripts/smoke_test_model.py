"""Fast local training sanity check: tiny U-Net, tiny subset, CPU/MPS.

Verifies: forward shapes, joint loss, checkpoint save, automatic resume,
metrics sanity, final report, plots - before any Kaggle GPU time is spent.
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import torch
import yaml

from src.data import get_loaders
from src.models import build_model
from src.metrics import SegMetricAccumulator
from src.train import Trainer, find_resume_checkpoint
from src.utils import get_device, seed_everything
from src.viz import plot_history, plot_prediction_grid

OUT = "/tmp/cse428_smoke_model"


def main():
    seed_everything(42)
    device = get_device()
    print(f"[1] device: {device}")

    with open(os.path.join(ROOT, "configs/config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(img_size=64, batch_size=4, num_workers=0)
    cfg["model"].update(name="unet", base_channels=8)
    cfg["train"].update(epochs_total=2, lr=1e-3)

    datasets, loaders = get_loaders(
        root=os.path.join(ROOT, "data"),
        img_size=cfg["data"]["img_size"],
        seed=cfg["seed"],
        augment=True,
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
        download=False,
    )
    datasets["train"].indices = datasets["train"].indices[:32]
    datasets["val"].indices = datasets["val"].indices[:16]
    datasets["test"].indices = datasets["test"].indices[:32]

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] tiny U-Net: {n_params / 1e6:.2f}M params")

    trainer = Trainer(model, loaders, device, cfg, out_dir=OUT)
    recs = trainer.fit()
    assert len(recs) == 2 and recs[-1]["epoch"] == 2, recs
    assert all(torch.isfinite(torch.tensor(r["train_loss"])) for r in recs)
    assert os.path.exists(f"{OUT}/checkpoints/last.pth")
    assert os.path.exists(f"{OUT}/checkpoints/best.pth")
    print("[3] 2-epoch fit ok, checkpoints saved")

    cfg["train"]["epochs_total"] = 3
    resume = find_resume_checkpoint(out_dir=OUT, model_name="unet")
    assert resume == f"{OUT}/checkpoints/last.pth", resume
    trainer = Trainer(model, loaders, device, cfg, out_dir=OUT, resume=resume)
    recs = trainer.fit()
    assert len(recs) == 1 and recs[0]["epoch"] == 3, recs
    print("[4] resume from checkpoint ok (epoch 3)")

    recs = trainer.fit()
    assert recs == [], recs
    print("[5] no-op fit when complete ok")

    acc = SegMetricAccumulator()
    perfect = torch.ones(8, 8)
    acc.update(perfect, perfect)
    m = acc.compute()
    assert m["miou"] == 1.0 and m["dice"] == 1.0 and m["pixel_acc"] == 1.0, m
    print("[6] perfect-prediction metrics = 1.0 ok")

    results = trainer.final_report()
    assert {"train", "val", "test"} <= set(results)
    print("[7] final report ok")

    plot_history(trainer.history, save_path=f"{ROOT}/reports/smoke_history.png")
    fig = plot_prediction_grid(
        datasets["val"], trainer.model, device, indices=[0, 1, 2], nrows=3
    )
    fig.savefig(f"{ROOT}/reports/smoke_predictions.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("[8] history + prediction plots saved to reports/")

    shutil.rmtree(OUT, ignore_errors=True)
    print("\nALL MODEL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
