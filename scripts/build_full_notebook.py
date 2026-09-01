"""Generate the ONE full-project notebook (notebooks/cse428_project.ipynb).

This notebook is the complete project: it can TRAIN everything from scratch
(config MODE='train') and it can PRESENT the saved results (config MODE='present').

Layout:
  0. Config cell (MODE, epochs, device, output dir, dataset dir)
  1. Dataset download/load (Oxford-IIIT Pet, trimap->binary, split, augment)
  2. Model definitions (inlined from src/models)
  3. Metrics + training code (inlined from src/metrics, src/train)
  4. Train (MODE='train') OR Load (MODE='present') both models
  5. Present: 3x3 grid, curves, result tables, predictions, comparison,
     live demo, bonus

Data layout (created by training, or provided for presentation):
  <DATA_DIR>/oxford-iiit-pet/            dataset (images + trimaps)
  <DATA_DIR>/unet/                       results.json, history.json, checkpoints/best.pth
  <DATA_DIR>/attention_unet/             results.json, history.json, checkpoints/best.pth

Regenerate: .venv/bin/python scripts/build_full_notebook.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "notebooks", "cse428_project.ipynb")

# Files whose source is injected verbatim (stripped of package-relative imports).
MODEL_SOURCES = ["src/models/unet.py", "src/models/attention_unet.py", "src/models/heads.py"]
DATA_SOURCES = ["src/data.py"]
METRICS_SOURCES = ["src/metrics.py"]
TRAIN_SOURCES = ["src/train.py"]


def strip_module_source(rel_paths, drop_funcs=()):
    """Concatenate source files, dropping package-relative imports, docstrings,
    and any function whose def line starts with one of drop_funcs."""
    chunks = []
    for rel in rel_paths:
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
            if any(stripped.startswith(f"def {fn}") for fn in drop_funcs):
                skip_until_dedent = 1
                continue
            if skip_until_dedent:
                if line and not line[0].isspace() and not stripped.startswith("#"):
                    skip_until_dedent = 0
                else:
                    continue
            if stripped.startswith("from ..") or stripped.startswith("from ."):
                continue
            if "Trainer" in stripped and rel.startswith("src/models/"):
                continue  # trainer refs only appear in model docstrings
            kept.append(line)
        chunks.append("\n".join(kept).strip())
    return "\n\n\n".join(chunks)


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

**Task:** jointly segment the pet (binary mask) **and** classify the breed (37 classes).

**Architectures:**
1. **U-Net** (Ronneberger et al., 2015)
2. **Attention U-Net** (Oktay et al., 2018)

Each has a classifier head on the encoder bottleneck, trained jointly with
`L = L_seg + λ·L_cls`.

**This notebook is the whole project.** Set `MODE` in the config cell below:

- `MODE = "train"` — trains both models from scratch (needs the dataset + GPU),
  saving `data/unet/` and `data/attention_unet/` artifacts.
- `MODE = "present"` — loads the saved artifacts and presents the results
  (no training). This is what you run for the demonstration.
"""


CONFIG_MD = """## 0. Configuration

Choose the mode, then run the cell below.
"""


CONFIG_CODE = '''import torch

# ============ CONFIG ============
MODE = "present"          # "train"  -> train both models from scratch
                          # "present" -> load saved artifacts and present

# Dataset location (auto-downloaded in train mode if missing)
DATA_DIR = "data"

# Output directory for artifacts (train mode) / where to look (present mode)
OUT_DIR = "outputs"

# --- training hyperparameters (train mode) ---
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS_UNET = 80          # set lower (e.g. 2) to smoke-test on CPU
EPOCHS_ATTN = 80
LR = 3.0e-4
WEIGHT_DECAY = 1.0e-4
LAMBDA_CLS = 1.0
SEED = 42
NUM_WORKERS = 2
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

print(f"mode: {MODE} | device: {DEVICE}")
'''


def setup_code() -> str:
    # seeding + dataset download helpers
    return '''import os, random, glob, json, time, shutil
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# --- dataset ---
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

def download_dataset():
    """Download Oxford-IIIT Pet into DATA_DIR if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    # torchvision downloads into DATA_DIR/oxford-iiit-pet
    OxfordIIITPet(root=DATA_DIR, split="trainval", target_types=("category","segmentation"), download=True)
    OxfordIIITPet(root=DATA_DIR, split="test", target_types=("category","segmentation"), download=True)
    print("dataset ready under", os.path.join(DATA_DIR, "oxford-iiit-pet"))

if MODE == "train":
    if not os.path.isdir(os.path.join(DATA_DIR, "oxford-iiit-pet", "images")):
        download_dataset()
    else:
        print("dataset already present")
'''


def dataset_code() -> str:
    # the PetSegDataset + loaders, inlined from src/data.py
    return strip_module_source(DATA_SOURCES)


def models_code() -> str:
    code = strip_module_source(MODEL_SOURCES, drop_funcs=("build_backbone_classifier",))
    return code + """


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


def metrics_code() -> str:
    return strip_module_source(METRICS_SOURCES)


def train_code() -> str:
    return strip_module_source(TRAIN_SOURCES)


def train_or_load_md() -> str:
    return """## 1. Train (MODE='train') or load (MODE='present')

In **train** mode, both models are trained and the artifacts are saved under
`data/`. In **present** mode, the saved artifacts are loaded.
"""


def train_or_load_code() -> str:
    return '''# Artifact discovery (used in both modes: present loads, train saves then loads)
def load_results(name):
    p = os.path.join(DATA_DIR, name, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None

def load_history(name):
    p = os.path.join(DATA_DIR, name, "history.json")
    return json.load(open(p)) if os.path.exists(p) else None

def find_best(name):
    return glob.glob(os.path.join(DATA_DIR, name, "checkpoints", "best.pth"))

def report_artifacts():
    for n in ("unet", "attention_unet"):
        r, h, b = load_results(n), load_history(n), find_best(n)
        print(f"{n:>15}: results={'ok' if r else 'MISSING':7} "
              f"history={'ok' if h else 'MISSING':7} best.pth={'ok' if b else 'MISSING'}")

# Build the datasets + loaders (train mode only)
def make_loaders():
    from torch.utils.data import DataLoader
    ds_train = PetSegDataset(DATA_DIR, split="trainval", img_size=IMG_SIZE, augment=True,
                             download=(MODE == "train"))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ds_train))
    n_val = int(round(len(ds_train) * 0.1))
    train_idx, val_idx = perm[n_val:], perm[:n_val]
    train_ds = PetSegDataset(DATA_DIR, split="trainval", indices=train_idx, img_size=IMG_SIZE,
                             augment=True, download=(MODE == "train"))
    val_ds = PetSegDataset(DATA_DIR, split="trainval", indices=val_idx, img_size=IMG_SIZE,
                           augment=False, download=(MODE == "train"))
    test_ds = PetSegDataset(DATA_DIR, split="test", img_size=IMG_SIZE, augment=False,
                            download=(MODE == "train"))
    common = dict(num_workers=0, pin_memory=False)
    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **common),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **common),
    }
    return {"train": train_ds, "val": val_ds, "test": test_ds}, loaders

if MODE == "train":
    datasets, loaders = make_loaders()
    print("train", len(datasets["train"]), "val", len(datasets["val"]), "test", len(datasets["test"]))
else:
    report_artifacts()
'''


def train_unet_md() -> str:
    return """## 2. Train / load U-Net
"""


def train_unet_code() -> str:
    return '''if MODE == "train":
    cfg = {
        "model": {"name": "unet", "base_channels": 32, "num_classes": 37, "lambda_cls": LAMBDA_CLS},
        "train": {"lr": LR, "weight_decay": WEIGHT_DECAY, "epochs_total": EPOCHS_UNET, "label_smoothing": 0.1},
        "data": {"img_size": IMG_SIZE},
    }
    unet_out = os.path.join(OUT_DIR, "unet")
    model = build_model(cfg).to(DEVICE)
    tr = Trainer(model, loaders, DEVICE, cfg, out_dir=unet_out)
    tr.fit()
    tr.final_report()
    # copy artifacts to DATA_DIR for the present-mode layout
    os.makedirs(os.path.join(DATA_DIR, "unet", "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(unet_out, "results.json"), os.path.join(DATA_DIR, "unet", "results.json"))
    shutil.copy(os.path.join(unet_out, "history.json"), os.path.join(DATA_DIR, "unet", "history.json"))
    shutil.copy(os.path.join(unet_out, "checkpoints", "best.pth"), os.path.join(DATA_DIR, "unet", "checkpoints", "best.pth"))
    shutil.copy(os.path.join(unet_out, "checkpoints", "last.pth"), os.path.join(DATA_DIR, "unet", "checkpoints", "last.pth"))
    print("U-Net artifacts saved to", os.path.join(DATA_DIR, "unet"))

# refresh globals so present/demo cells work in both modes
UNET_RESULTS, UNET_HISTORY, UNET_BEST = load_results("unet"), load_history("unet"), find_best("unet")
'''


def train_attn_md() -> str:
    return """## 3. Train / load Attention U-Net
"""


def train_attn_code() -> str:
    return '''if MODE == "train":
    cfg = {
        "model": {"name": "attention_unet", "base_channels": 32, "num_classes": 37, "lambda_cls": LAMBDA_CLS},
        "train": {"lr": LR, "weight_decay": WEIGHT_DECAY, "epochs_total": EPOCHS_ATTN, "label_smoothing": 0.1},
        "data": {"img_size": IMG_SIZE},
    }
    attn_out = os.path.join(OUT_DIR, "attention_unet")
    model = build_model(cfg).to(DEVICE)
    tr = Trainer(model, loaders, DEVICE, cfg, out_dir=attn_out)
    tr.fit()
    tr.final_report()
    os.makedirs(os.path.join(DATA_DIR, "attention_unet", "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(attn_out, "results.json"), os.path.join(DATA_DIR, "attention_unet", "results.json"))
    shutil.copy(os.path.join(attn_out, "history.json"), os.path.join(DATA_DIR, "attention_unet", "history.json"))
    shutil.copy(os.path.join(attn_out, "checkpoints", "best.pth"), os.path.join(DATA_DIR, "attention_unet", "checkpoints", "best.pth"))
    shutil.copy(os.path.join(attn_out, "checkpoints", "last.pth"), os.path.join(DATA_DIR, "attention_unet", "checkpoints", "last.pth"))
    print("Attention U-Net artifacts saved to", os.path.join(DATA_DIR, "attention_unet"))

# refresh globals so present/demo cells work in both modes
ATTN_RESULTS, ATTN_HISTORY, ATTN_BEST = load_results("attention_unet"), load_history("attention_unet"), find_best("attention_unet")
report_artifacts()
'''


def present_helpers_code() -> str:
    return '''import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
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
    fig.suptitle(title); fig.tight_layout()
    return fig
'''


def present_unet_md() -> str:
    return """## 4. Results — U-Net
"""


def present_unet_code() -> str:
    return '''if UNET_HISTORY:
    fig = history_plot(UNET_HISTORY, "U-Net - training curves")
    plt.show()
if UNET_RESULTS:
    display(results_table(UNET_RESULTS))
else:
    print("no U-Net results found (train first, or set MODE=present with data/)")
'''


def present_attn_md() -> str:
    return """## 5. Results — Attention U-Net
"""


def present_attn_code() -> str:
    return '''if ATTN_HISTORY:
    fig = history_plot(ATTN_HISTORY, "Attention U-Net - training curves")
    plt.show()
if ATTN_RESULTS:
    display(results_table(ATTN_RESULTS))
else:
    print("no Attention U-Net results found (train first, or set MODE=present with data/)")
'''


def comparison_md() -> str:
    return """## 6. Comparison — U-Net vs Attention U-Net
"""


def comparison_code() -> str:
    return '''rows = {}
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


def demo_md() -> str:
    return """## 7. Live demonstration — instant prediction (no training)

Load a saved model and run **one forward pass** on any image (upload or path).
"""


def demo_code() -> str:
    return '''from pathlib import Path
from IPython.display import display
try:
    import ipywidgets as widgets
    _HAVE_WIDGETS = True
except ImportError:
    widgets = None; _HAVE_WIDGETS = False

DEMO_MODEL = "attention_unet"     # "unet" | "attention_unet"
DEMO_IMG = ""                      # path to an image, or leave empty to upload

_best = ATTN_BEST if DEMO_MODEL == "attention_unet" else UNET_BEST
if not _best:
    print(f"no best.pth for {DEMO_MODEL} - copy it under data/ to run the demo.")
else:
    _ck = torch.load(_best[0], map_location="cpu", weights_only=False)
    _model = build_model(_ck["cfg"]).to(DEVICE)
    _model.load_state_dict(_ck["model_state"]); _model.eval()
    _IMGSIZE = int(_ck["cfg"].get("data", {}).get("img_size", 256))
    _CLASSES = list(_ck.get("classes") or [f"class_{i}" for i in range(37)])
    _THRESH = float(_ck["cfg"].get("model", {}).get("seg_threshold", 0.5))
    print(f"loaded {DEMO_MODEL} | img_size {_IMGSIZE} | threshold {_THRESH} | {len(_CLASSES)} breeds")

    if DEMO_IMG.strip():
        _demo_path = Path(DEMO_IMG).expanduser()
    elif _HAVE_WIDGETS:
        _up = widgets.FileUpload(accept="image/*", multiple=False, description="Upload image")
        display(_up)
        print("Drop your image into the upload button above, then run the next cell.")
    else:
        print("set DEMO_IMG to an image path (ipywidgets not installed).")
'''


def demo_predict_code() -> str:
    return '''import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

if "_model" not in globals():
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
        x = TF.resize(_img, [_IMGSIZE, _IMGSIZE], antialias=True)
        x = TF.to_tensor(x).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = _model(x)
        _mask = (torch.sigmoid(out["seg_logits"]) > _THRESH).float()
        _mask = TF.resize(_mask[0, 0].unsqueeze(0).unsqueeze(0), [_img.size[1], _img.size[0]],
                          interpolation=TF.InterpolationMode.NEAREST)[0, 0].cpu().numpy()
        _probs = torch.softmax(out["cls_logits"], dim=1)[0]
        _conf, _lab = _probs.max(0)
        print(f"predicted breed: {_CLASSES[int(_lab)]}")
        print(f"confidence:      {_conf.item():.2%}")
        print(f"foreground:      {100.0 * _mask.mean():.1f}% of pixels")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(_img); axes[0].set_title("Input image"); axes[0].axis("off")
        _arr = np.asarray(_img, dtype=np.float32) / 255.0
        _arr = _arr.copy(); _arr[_mask > 0] = 0.5 * _arr[_mask > 0] + 0.5 * np.array([1.0, 0.0, 0.0])
        axes[1].imshow(np.clip(_arr, 0, 1))
        axes[1].set_title(f"Prediction: {_CLASSES[int(_lab)]} ({_conf.item():.0%})"); axes[1].axis("off")
        plt.tight_layout(); plt.show()
'''


def bonus_md() -> str:
    return """## 8. Bonus — classifier backbone comparison

ResNet-18, MobileNetV3-Small and EfficientNet-B0, each trained for 10 epochs on
the same split. Test metrics below (loads from data/bonus/ if present).
"""


def bonus_code() -> str:
    return '''bonus_rows = []
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

```bash
pip install torch torchvision pillow numpy matplotlib pandas ipywidgets
jupyter notebook          # open cse428_project.ipynb
```

- **present** mode: set `MODE = "present"` and run all cells — needs only the
  `data/` folder with the saved artifacts.
- **train** mode: set `MODE = "train"` and run all cells — needs the dataset
  (auto-downloads) and a GPU; 80 epochs takes a few hours.
"""


def build_notebook() -> dict:
    cells = [
        cell_md(TITLE_MD),
        cell_md(CONFIG_MD),
        cell_code(CONFIG_CODE),
        cell_code(setup_code()),
        cell_md("## Dataset & loaders (train mode) / present-mode loader"),
        cell_code(dataset_code()),
        cell_md("## Model definitions"),
        cell_code(models_code()),
        cell_md("## Metrics"),
        cell_code(metrics_code()),
        cell_md("## Training code"),
        cell_code(train_code()),
        cell_md(train_or_load_md()),
        cell_code(train_or_load_code()),
        cell_md(train_unet_md()),
        cell_code(train_unet_code()),
        cell_md(train_attn_md()),
        cell_code(train_attn_code()),
        cell_code(present_helpers_code()),
        cell_md(present_unet_md()),
        cell_code(present_unet_code()),
        cell_md(present_attn_md()),
        cell_code(present_attn_code()),
        cell_md(comparison_md()),
        cell_code(comparison_code()),
        cell_md(demo_md()),
        cell_code(demo_code()),
        cell_code(demo_predict_code()),
        cell_md(bonus_md()),
        cell_code(bonus_code()),
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
