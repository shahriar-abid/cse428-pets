# CSE428 Project — Pet Segmentation & Breed Classification

Semantic segmentation and breed classification on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset.

- **Models**: U-Net and Attention U-Net (segmentation) with a 37-breed classifier head attached to the encoder bottleneck — trained jointly with `L = L_seg + λ · L_cls`
- **Data**: trimap `1`(foreground) + `3`(boundary) → foreground, `2` → background (per guidelines); deterministic seeded 90/10 train/val split; official test set
- **Metrics**: mIoU, Dice, pixel accuracy (segmentation) · accuracy, precision, recall, F1 (classification) — reported for train/val/test
- **Bonus**: classifier backbone comparison (ResNet18 / MobileNetV3 / EfficientNet-B0)

## Structure

```
configs/config.yaml        single source of truth for all hyperparameters
src/
  data.py                  dataset, binary-mask conversion, splits, loaders
  models/unet.py           U-Net + joint classifier head
  models/attention_unet.py attention-gated skip connections
  models/heads.py          bonus: standalone classifier backbones
  metrics.py               streaming mIoU/Dice/pixel-acc + macro acc/prec/rec/F1
  train.py                 resumable Trainer (checkpoint chunks), Dice+BCE loss
  viz.py                   3x3 overlay grid, prediction grid, training curves
  utils.py                 seeding, device selection
scripts/
  smoke_test.py            data pipeline sanity check (CPU/MPS, <1 min)
  smoke_test_model.py      training/checkpoint/resume sanity check (CPU/MPS)
notebooks/ce428_pets.ipynb Kaggle driver notebook (training + presentation)
```

## Local quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke_test.py       # needs data/ (auto-downloads once)
.venv/bin/python scripts/smoke_test_model.py
```

Dataset: `torchvision.datasets.OxfordIIITPet` downloads to `data/` automatically (~800 MB, gitignored).

## Kaggle workflow

1. New notebook → **File → Import Notebook** → `notebooks/ce428_pets.ipynb`
2. Session options: **Internet ON**, **GPU accelerator: T4 x2** (P100 is NOT supported by Kaggle's current PyTorch — the setup cell raises a clear error if selected)
3. Run — cell 0 clones this repo; everything else comes from the repo

### Checkpoint chunks (never retrain from scratch)

- Set `CFG["train"]["epochs_total"]` (e.g. 15) and **Save Version → Save & Run All**. The run trains epochs `1..15` and writes `outputs/<model>/checkpoints/{last,best}.pth` + `history.json` + `results.json` to the version output.
- Next session (you **or a groupmate**): *Add Input → your notebook → previous version output*. The trainer auto-discovers `/kaggle/input/<slug>/outputs/<model>/checkpoints/last.pth` and resumes (weights + optimizer + scheduler + history + best mIoU). Bump `epochs_total` (e.g. 30) and run again.
- The **final version** re-runs with no new epochs: it just reloads checkpoints, plots curves, prints the train/val/test summary tables and prediction grids — present from that version.
- For the demonstration: `plot_prediction_grid(..., indices=<their random indices>)` loads the best checkpoint and runs any image index.

## Team

- Shahriar Abid — code, training, presentation
- *(groupmate)*
