"""Generate the final self-contained presentation notebook
(notebooks/presentation_notebook.ipynb).

This is the deliverable for the viva: ONE notebook that tells the whole project
story and can be presented from any laptop with no internet, no repo, no Kaggle
and no training. It embeds the model code verbatim from src/models/ (so it can
never drift from the trained code) and loads the trained artifacts from a data/
folder next to it.

Expected data/ layout next to the notebook:

    data/
      oxford-iiit-pet/            # dataset (images/ + annotations/trimaps/)
      unet/
        results.json history.json checkpoints/best.pth
      attention_unet/
        results.json history.json checkpoints/best.pth
      bonus/<backbone>/results.json   # optional: Bonus 1 test metrics

Regenerate after any src/models change or after re-training:
  .venv/bin/python scripts/build_presentation_notebook.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "notebooks", "presentation_notebook.ipynb")

MODEL_SOURCES = [
    "src/models/unet.py",
    "src/models/attention_unet.py",
    "src/models/heads.py",
]

DATA_DIR = "data"


def read_model_code() -> str:
    """Concatenate the model sources into a single self-contained block.

    Same stripping rules as build_faculty_demo.py: drop package-relative
    imports, module docstrings and the bonus backbone helper (not needed),
    and inline a local NUM_CLASSES + build_model factory.
    """
    chunks = []
    for rel in MODEL_SOURCES:
        with open(os.path.join(ROOT, rel)) as f:
            text = f.read()
        kept = []
        skip_until_dedent = 0
        in_docstring = False
        for line in text.splitlines():
            stripped = line.strip()
            if in_docstring:
                if '"""' in stripped:
                    in_docstring = False
                continue
            if stripped.startswith('"""'):
                if stripped.count('"""') >= 2:
                    continue
                in_docstring = True
                continue
            if stripped.startswith("def build_backbone_classifier"):
                skip_until_dedent = 1
                continue
            if skip_until_dedent:
                if line and not line[0].isspace() and not stripped.startswith("#"):
                    skip_until_dedent = 0
                else:
                    continue
            if stripped.startswith("from ..") or stripped.startswith("from ."):
                continue
            if "Trainer" in stripped:
                continue
            kept.append(line)
        chunks.append("\n".join(kept).strip())
    return "\n\n\n".join(chunks) + """


# ---------------------------------------------------------------------------
# Local model factory (self-contained equivalent of the project factory)
# ---------------------------------------------------------------------------

NUM_CLASSES = 37


def build_model(cfg):
    name = cfg["model"]["name"]
    kwargs = dict(
        base_channels=cfg["model"].get("base_channels", 32),
        num_classes=cfg["model"].get("num_classes", NUM_CLASSES),
    )
    if name == "unet":
        return UNet(**kwargs)
    if name == "attention_unet":
        return AttentionUNet(**kwargs)
    raise ValueError(f"unknown model: {name}")
"""


def cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

TITLE_MD = """# CSE428 Project — Pet Segmentation & Breed Classification

**Dataset:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) · 37 breeds · 3,680 trainval / 3,669 test images

**Task:** jointly segment the pet (binary mask) **and** classify the breed (37 classes), using two architectures:

1. **U-Net** (Ronneberger et al., 2015)
2. **Attention U-Net** (Oktay et al., 2018)

Each has a classifier head attached to the encoder bottleneck, trained jointly with
`L = L_segmentation + L_classification`.

**Contents:** data exploration → method → U-Net results → Attention U-Net results →
comparison → live demonstration → bonus.

This notebook is **self-contained**: it embeds the model code and loads the trained
weights (`best.pth`) and recorded results from the `data/` folder next to it. No
training, internet, or repository access is needed.
"""


METHOD_MD = """## Method

### Binary mask from the trimap
Each image ships with a **trimap**: `1` = foreground (pet), `2` = background,
`3` = boundary (unclassified). Per the project guidelines, boundary pixels are
merged into the foreground, giving a binary mask:

```
trimap == 2  ->  background (0)
trimap != 2  ->  foreground (1)   # includes boundary
```

### Data split
- **Training set:** 90% of the official `trainval` (3,312 images)
- **Validation set:** 10% of `trainval` (368 images), deterministic seeded split
- **Test set:** the official `test` split (3,669 images)

### Augmentation (training only)
Random rotation ±15°, random-resized crop (80–100% area), horizontal flip,
color jitter — applied identically to image and mask so alignment is preserved.

### Models
- **U-Net** encoder → decoder with skip connections; binary segmentation head
  (1×1 conv → sigmoid); classifier head on the bottleneck
  (global average pool → linear 512→128 → ReLU → dropout → linear 128→37).
- **Attention U-Net** — same skeleton, but every skip connection is re-weighted
  by an additive **attention gate** (Oktay et al., 2018) that filters out
  irrelevant background features before concatenation.
- Both heads share the encoder and are trained **jointly**.

### Loss & training
`L = DiceBCE(segmentation) + λ · CrossEntropy(classification)` with λ = 1.0,
label smoothing 0.1, AdamW (lr 3e-4, weight decay 1e-4), cosine annealing,
80 epochs, batch size 16 at 256×256 on a T4 GPU.

### Metrics
- Segmentation: **mIoU, Dice coefficient, pixel accuracy**
- Classification: **accuracy, precision, recall, F1** (macro-averaged)
- Reported for **train / validation / test**.
"""


SETUP_DATA_MD = """### Loading the trained artifacts

The cells below load the trained artifacts and the dataset from the `data/`
folder next to this notebook:

```
data/oxford-iiit-pet/            images/ + annotations/trimaps/
data/unet/                       results.json, history.json, checkpoints/best.pth
data/attention_unet/             results.json, history.json, checkpoints/best.pth
```

If a model's folder or the dataset is missing, the relevant section prints a
clear notice instead of crashing, so the notebook can still be presented.
"""


SETUP_DATA_CODE = '''import os, json, glob, torch

DATA_DIR = "data"

def load_results(name):
    p = os.path.join(DATA_DIR, name, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None

def load_history(name):
    p = os.path.join(DATA_DIR, name, "history.json")
    return json.load(open(p)) if os.path.exists(p) else None

def find_best(name):
    return glob.glob(os.path.join(DATA_DIR, name, "checkpoints", "best.pth"))

UNET_RESULTS = load_results("unet")
UNET_HISTORY = load_history("unet")
UNET_BEST = find_best("unet")

ATTN_RESULTS = load_results("attention_unet")
ATTN_HISTORY = load_history("attention_unet")
ATTN_BEST = find_best("attention_unet")

for n, r, h, b in [
    ("unet", UNET_RESULTS, UNET_HISTORY, UNET_BEST),
    ("attention_unet", ATTN_RESULTS, ATTN_HISTORY, ATTN_BEST),
]:
    print(f"{n:>15}: results={'ok' if r else 'MISSING':7} "
          f"history={'ok' if h else 'MISSING':7} best.pth={'ok' if b else 'MISSING'}")
'''


# --- Dataset & exploration -----------------------------------------------

EXPLORATION_MD = """## 1. Dataset exploration — 3×3 overlay grid (requirement)

Nine **random** indices from the dataset; each image is shown with its
ground-truth mask overlaid (boundary pixels merged into the foreground).
"""

EXPLORATION_CODE = '''import numpy as np, matplotlib.pyplot as plt
from PIL import Image
import os, glob

IMAGES_DIR = os.path.join(DATA_DIR, "oxford-iiit-pet", "images")
TRIMAPS_DIR = os.path.join(DATA_DIR, "oxford-iiit-pet", "annotations", "trimaps")

def _list_pets():
    if not os.path.isdir(IMAGES_DIR):
        return []
    return sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg"))

PET_FILES = _list_pets()

if not PET_FILES:
    print("No dataset found under data/oxford-iiit-pet/ - copy the Oxford-IIIT Pet "
          "images + trimaps there to show the required grids.")
else:
    rng = np.random.RandomState(42)          # deterministic for the presentation
    n_show = min(9, len(PET_FILES))          # safe if only a few images are present
    idxs = rng.choice(len(PET_FILES), size=n_show, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, i in zip(axes.ravel(), idxs):
        stem = PET_FILES[i][:-4]
        img = Image.open(os.path.join(IMAGES_DIR, PET_FILES[i])).convert("RGB")
        tri = np.array(Image.open(os.path.join(TRIMAPS_DIR, stem + ".png")))
        mask = (tri != 2).astype(np.float32)     # boundary -> foreground
        im = np.asarray(img, dtype=np.float32) / 255.0
        im[mask > 0] = 0.5 * im[mask > 0] + 0.5 * np.array([1.0, 0.0, 0.0])
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(f"idx {int(i)} | {stem.split('_')[0].replace(chr(39),'')}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n_show:]:
        ax.axis("off")
    fig.suptitle("Dataset exploration - 9 random images with mask overlay")
    plt.tight_layout()
    plt.show()
'''


# --- Model definitions -----------------------------------------------------

MODELS_MD = """## Model definitions

The full model code is embedded below (generated from the project source so it
is always in sync with the trained weights). Both models output a segmentation
logit map and a 37-way classification logit vector.
"""


# --- Results helpers (loaded once, used by both sections) ------------------

RESULTS_HELPERS_CODE = '''import pandas as pd

def seg_row(r):
    s = r["seg"]; return {"mIoU": s["miou"], "Dice": s["dice"], "Pixel acc": s["pixel_acc"]}
def cls_row(r):
    c = r["cls"]; return {"Acc": c["acc"], "Prec": c["precision"], "Rec": c["recall"], "F1": c["f1"]}

def results_table(results):
    rows = []
    for split in ("train", "val", "test"):
        rows.append({"Split": split, **seg_row(results[split]), **cls_row(results[split])})
    return pd.DataFrame(rows).set_index("Split").round(4)

def history_plot(history, title):
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="total")
    axes[0].plot(ep, [h["train_seg_loss"] for h in history], label="segmentation")
    axes[0].plot(ep, [h["train_cls_loss"] for h in history], label="classification")
    axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(ep, [h["val_miou"] for h in history], label="mIoU")
    axes[1].plot(ep, [h["val_dice"] for h in history], label="Dice")
    axes[1].plot(ep, [h["val_pixel_acc"] for h in history], label="pixel acc")
    axes[1].set_title("Validation - segmentation"); axes[1].set_xlabel("epoch")
    axes[2].plot(ep, [h["val_acc"] for h in history], label="accuracy")
    axes[2].plot(ep, [h["val_precision"] for h in history], label="precision")
    axes[2].plot(ep, [h["val_recall"] for h in history], label="recall")
    axes[2].plot(ep, [h["val_f1"] for h in history], label="F1")
    axes[2].set_title("Validation - classification"); axes[2].set_xlabel("epoch")
    for ax in axes: ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
'''


# --- U-Net section ----------------------------------------------------------

UNET_MD = """## 2. Base U-Net

Trained for 80 epochs with the joint loss. Below: training curves (loss +
per-epoch validation metrics), the result summary for train/validation/test,
and a prediction grid (image | ground truth | model output).
"""

UNET_CODE = '''print("===== U-Net =====")
if UNET_HISTORY:
    fig = history_plot(UNET_HISTORY, "U-Net - training curves")
    plt.show()
else:
    print("no history.json for unet")

if UNET_RESULTS:
    display(results_table(UNET_RESULTS))
else:
    print("no results.json for unet")
'''


PREDICTIONS_MD = """### U-Net predictions — image | ground truth | model output (requirement)

Three random validation samples; the titles show the predicted breed with
confidence and the per-image foreground IoU.
"""

PREDICTIONS_CODE = '''def predict_sample(model, device, stem, threshold=0.5):
    """Run one dataset image through the model; return (img, gt_mask, pred_mask,
    breed, conf, iou)."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    img = Image.open(os.path.join(IMAGES_DIR, stem + ".jpg")).convert("RGB")
    tri = np.array(Image.open(os.path.join(TRIMAPS_DIR, stem + ".png")))
    gt = (tri != 2).astype(bool)
    x = TF.resize(img, [IMG_SIZE, IMG_SIZE], antialias=True)
    x = TF.to_tensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        pred = prob > threshold
        probs = torch.softmax(out["cls_logits"], dim=1)[0]
        conf, lab = probs.max(0)
        breed = CLASSES[int(lab)]
    pred = TF.resize(torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                     [img.size[1], img.size[0]],
                     interpolation=TF.InterpolationMode.NEAREST)[0, 0].numpy() > 0.5
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    iou = inter / max(union, 1)
    return img, gt, pred, breed, float(conf), float(iou)

def plot_prediction_grid_model(model, device, stems, threshold=0.5):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(stems), 3, figsize=(13, 4.2 * len(stems)))
    for r, stem in enumerate(stems):
        img, gt, pred, breed, conf, iou = predict_sample(model, device, stem, threshold)
        cols = [(f"{stem.split('_')[0]}", img, None),
                ("ground truth", img, gt),
                (f"pred: {breed} {conf:.0%} | IoU {iou:.2f}", img, pred)]
        for c, (t, im, m) in enumerate(cols):
            ax = axes[r][c]
            arr = np.asarray(im, dtype=np.float32) / 255.0
            if m is not None:
                arr = arr.copy(); arr[m] = 0.5 * arr[m] + 0.5 * np.array([1.0, 0.0, 0.0])
            ax.imshow(np.clip(arr, 0, 1)); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle("Predictions")
    fig.tight_layout()
    return fig
'''


PREDICTIONS_EXECUTE_CODE = '''if UNET_BEST and PET_FILES:
    import torch
    ckpt = torch.load(UNET_BEST[0], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]; CLASSES = list(ckpt["classes"])
    IMG_SIZE = int(cfg.get("data", {}).get("img_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    stems = [PET_FILES[i][:-4] for i in np.random.RandomState(7).choice(len(PET_FILES), 3, replace=False)]
    fig = plot_prediction_grid_model(model, device, stems)
    plt.show()
else:
    print("no best.pth / dataset for unet - predictions skipped")
'''


# --- Attention U-Net section ------------------------------------------------

ATTN_MD = """## 3. Attention U-Net

Same protocol, identical data, split, augmentation and hyperparameters — the only
difference is the attention gates on the skip connections.
"""

ATTN_CODE = '''print("===== Attention U-Net =====")
if ATTN_HISTORY:
    fig = history_plot(ATTN_HISTORY, "Attention U-Net - training curves")
    plt.show()
else:
    print("no history.json for attention_unet")

if ATTN_RESULTS:
    display(results_table(ATTN_RESULTS))
else:
    print("no results.json for attention_unet")
'''


ATTN_PREDICTIONS_MD = """### Attention U-Net predictions — image | ground truth | model output (requirement)
"""

ATTN_PREDICTIONS_CODE = '''if ATTN_BEST and PET_FILES:
    import torch
    ckpt = torch.load(ATTN_BEST[0], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]; CLASSES = list(ckpt["classes"])
    IMG_SIZE = int(cfg.get("data", {}).get("img_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device); model.load_state_dict(ckpt["model_state"]); model.eval()
    stems = [PET_FILES[i][:-4] for i in np.random.RandomState(7).choice(len(PET_FILES), 3, replace=False)]
    fig = plot_prediction_grid_model(model, device, stems)
    plt.show()
else:
    print("no best.pth / dataset for attention_unet - predictions skipped")
'''


# --- Comparison -------------------------------------------------------------

COMPARISON_MD = """## 4. U-Net vs Attention U-Net

Both models use identical data, split, augmentation and hyperparameters. The only
difference is the attention gates on the decoder. Below: side-by-side test metrics
from each `best.pth`, plus total parameters.
"""

COMPARISON_CODE = '''rows = {}
for name, results, best in [("unet", UNET_RESULTS, UNET_BEST),
                            ("attention_unet", ATTN_RESULTS, ATTN_BEST)]:
    if results is None or not best:
        rows[name] = {"status": "no data", "mIoU": None, "Dice": None, "Pixel acc": None,
                      "Acc": None, "F1": None, "params": None}
        continue
    ck = torch.load(best[0], map_location="cpu", weights_only=False)
    n_params = sum(p.numel() for p in build_model(ck["cfg"]).parameters())
    t = results["test"]
    rows[name] = {"status": "ok", "mIoU": t["seg"]["miou"], "Dice": t["seg"]["dice"],
                  "Pixel acc": t["seg"]["pixel_acc"], "Acc": t["cls"]["acc"],
                  "F1": t["cls"]["f1"], "params": n_params}

compare = pd.DataFrame(rows).T
display(compare.round(4))
print("\\nDiscussion:")
print("- Attention gates add parameters to each skip path; both models land at a similar")
print("  mIoU (~0.89), with the classifier head reaching ~0.72 test accuracy.")
'''


# --- Live demonstration ------------------------------------------------------

DEMO_MD = """## 5. Live demonstration — instant prediction (no training)

The faculty may provide a **random image** at evaluation time. This section loads
the saved model and runs **one forward pass** — no training, no retraining. Either
**upload an image** with the widget below or set `IMAGE_PATH` to a file on disk.
"""

DEMO_CODE = '''from pathlib import Path
from IPython.display import display
try:
    import ipywidgets as widgets
    _HAVE_WIDGETS = True
except ImportError:
    widgets = None
    _HAVE_WIDGETS = False

DEMO_MODEL = "attention_unet"     # "unet" | "attention_unet"
DEMO_IMG = ""                      # set to a path, or leave empty to upload

_demo_best = ATTN_BEST if DEMO_MODEL == "attention_unet" else UNET_BEST
if not _demo_best:
    print(f"no best.pth for {DEMO_MODEL} - copy it under data/ to run the demo.")
else:
    _demo_ckpt = torch.load(_demo_best[0], map_location="cpu", weights_only=False)
    _demo_cfg = _demo_ckpt["cfg"]
    _demo_model = build_model(_demo_cfg).to(DEVICE_DEMO)
    _demo_model.load_state_dict(_demo_ckpt["model_state"]); _demo_model.eval()

    if DEMO_IMG.strip():
        _demo_path = Path(DEMO_IMG).expanduser()
    else:
        if _HAVE_WIDGETS:
            _up = widgets.FileUpload(accept="image/*", multiple=False, description="Upload image")
            display(_up)
            print("Drop your image into the upload button above, then run the next cell.")
        else:
            print("ipywidgets not installed - set DEMO_IMG to an image path instead.")
'''


DEMO_PREDICT_CODE = '''import torch, torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

if "_demo_model" not in globals():
    print("run the previous cell first (it loads the model).")
else:
    if DEMO_IMG.strip():
        _demo_path = Path(DEMO_IMG).expanduser()
    elif _HAVE_WIDGETS:
        _up = widgets.FileUpload(accept="image/*", multiple=False, description="Upload image")
        display(_up)
        _first = list(_up.value.values())[0]
        _demo_path = Path("./_uploaded.png")
        _demo_path.write_bytes(bytes(_first["content"] if isinstance(_first, dict) else _first))
    else:
        print("set DEMO_IMG to an image path (ipywidgets not installed).")
        _demo_path = None

    if _demo_path is not None:
        _img = Image.open(_demo_path).convert("RGB")
        x = TF.resize(_img, [IMG_SIZE, IMG_SIZE], antialias=True)
        x = TF.to_tensor(x).unsqueeze(0).to(DEVICE_DEMO)
        with torch.no_grad():
            out = _demo_model(x)
        _mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
        _mask = TF.resize(_mask[0, 0].unsqueeze(0).unsqueeze(0), [_img.size[1], _img.size[0]],
                          interpolation=TF.InterpolationMode.NEAREST)[0, 0].cpu().numpy()
        _probs = torch.softmax(out["cls_logits"], dim=1)[0]
        _conf, _lab = _probs.max(0)
        print(f"predicted breed: {CLASSES[int(_lab)]}")
        print(f"confidence:      {_conf.item():.2%}")
        print(f"foreground:      {100.0 * _mask.mean():.1f}% of pixels")

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(_img); axes[0].set_title("Input image"); axes[0].axis("off")
        _arr = np.asarray(_img, dtype=np.float32) / 255.0
        _arr = _arr.copy(); _arr[_mask > 0] = 0.5 * _arr[_mask > 0] + 0.5 * np.array([1.0, 0.0, 0.0])
        axes[1].imshow(np.clip(_arr, 0, 1))
        axes[1].set_title(f"Prediction: {CLASSES[int(_lab)]} ({_conf.item():.0%})"); axes[1].axis("off")
        plt.tight_layout(); plt.show()
'''


# --- Bonus 1 ------------------------------------------------------------------

BONUS_MD = """## 6. Bonus — classifier backbone comparison

To probe the classification head, three well-established classifier architectures
were trained for an **equal 10-epoch budget** on the same split (using the same
`Trainer`): ResNet-18, MobileNetV3-Small and EfficientNet-B0. Test-set metrics are
reported below.
"""

BONUS_CODE = '''# The per-backbone test metrics live in the Kaggle run output (results.json under
# outputs/unet/bonus/<backbone>/). Load from data/ if present.
import os
bonus_rows = []
for bb in ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]:
    p = os.path.join(DATA_DIR, "bonus", bb, "results.json")
    if os.path.exists(p):
        r = json.load(open(p))["test"]["cls"]
        bonus_rows.append({"backbone": bb, "acc": r["acc"], "precision": r["precision"],
                           "recall": r["recall"], "f1": r["f1"]})
    else:
        bonus_rows.append({"backbone": bb, "acc": None, "precision": None,
                           "recall": None, "f1": None})
bonus_df = pd.DataFrame(bonus_rows).set_index("backbone")
display(bonus_df.round(4))
if bonus_df["acc"].notna().any():
    print("\\nBest backbone by test accuracy:", bonus_df["acc"].idxmax())
else:
    print("\\n(No per-backbone results found under data/bonus/ - copy the bonus "
          "results.json files there to show the comparison.)")
'''


FRESH_MACHINE_MD = """## Running this on a fresh machine

Only the notebook and the `data/` folder are needed:

```bash
pip install torch torchvision pillow numpy matplotlib pandas ipywidgets
jupyter notebook          # open presentation_notebook.ipynb, run top to bottom
```

`data/` should contain the Oxford-IIIT Pet images + trimaps and the trained
artifacts as described in the setup cell. No GPU, internet, or training code is
required — every number in this notebook is already recorded.
"""


def build_notebook() -> dict:
    model_code = read_model_code()

    setup_device_code = (
        "import torch\n"
        "DEVICE_DEMO = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
        "print(\"demo device:\", DEVICE_DEMO)\n"
    )

    cells = [
        cell_md(TITLE_MD),
        cell_md(METHOD_MD),
        cell_md(SETUP_DATA_MD),
        cell_code(SETUP_DATA_CODE),
        cell_md(EXPLORATION_MD),
        cell_code(EXPLORATION_CODE),
        cell_md(MODELS_MD),
        cell_code(model_code),
        cell_code(setup_device_code),
        cell_code(RESULTS_HELPERS_CODE),
        cell_md(UNET_MD),
        cell_code(UNET_CODE),
        cell_md(PREDICTIONS_MD),
        cell_code(PREDICTIONS_CODE),
        cell_code(PREDICTIONS_EXECUTE_CODE),
        cell_md(ATTN_MD),
        cell_code(ATTN_CODE),
        cell_md(ATTN_PREDICTIONS_MD),
        cell_code(ATTN_PREDICTIONS_CODE),
        cell_md(COMPARISON_MD),
        cell_code(COMPARISON_CODE),
        cell_md(DEMO_MD),
        cell_code(DEMO_CODE),
        cell_code(DEMO_PREDICT_CODE),
        cell_md(BONUS_MD),
        cell_code(BONUS_CODE),
        cell_md(FRESH_MACHINE_MD),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    nb = build_notebook()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
