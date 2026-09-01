"""Generate the self-contained faculty-demo notebook (notebooks/faculty_demo.ipynb).

The demo notebook embeds the model classes (UNet / AttentionUNet / factory) so it
runs standalone on the faculty machine with ONLY the notebook + best.pth — no repo,
no dataset, no training code, no Kaggle paths.

The model source is injected verbatim from src/models/*.py so it cannot drift from
the trained code. Re-run this script after any change to src/models/:

    .venv/bin/python scripts/build_faculty_demo.py

The notebook itself contains no trace of this generator or of the project repo.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "notebooks", "faculty_demo.ipynb")

# Files whose source is injected verbatim into the "model definitions" cell.
MODEL_SOURCES = [
    "src/models/unet.py",
    "src/models/attention_unet.py",
    "src/models/heads.py",
]


def read_model_code() -> str:
    """Concatenate the model sources into a single self-contained code block.

    The project modules use relative imports (from ..data import NUM_CLASSES,
    from .unet import UNet); here we strip import lines that only make sense
    inside the package and inline a local NUM_CLASSES + build_model factory.
    The architecture class bodies are copied byte-for-byte.
    """
    chunks = []
    for rel in MODEL_SOURCES:
        with open(os.path.join(ROOT, rel)) as f:
            text = f.read()
        kept = []
        skip_until_dedent = 0  # >0 while inside a function we drop entirely
        in_docstring = False   # True while inside a module docstring (""") we drop
        for line in text.splitlines():
            stripped = line.strip()
            if in_docstring:
                if '"""' in stripped:
                    in_docstring = False
                continue
            if stripped.startswith('"""'):
                # start of a module docstring; drop it and everything until
                # its closing """ (same line or later)
                if stripped.count('"""') >= 2:
                    continue  # single-line """..."""
                in_docstring = True
                continue
            if stripped.startswith("def build_backbone_classifier"):
                # bonus-task helper is not needed by the demo notebook
                skip_until_dedent = 1
                continue
            if skip_until_dedent:
                # end of the dropped function: first line at indentation 0
                if line and not line[0].isspace() and not stripped.startswith("#"):
                    skip_until_dedent = 0
                else:
                    continue
            if stripped.startswith("from ..") or stripped.startswith("from ."):
                continue  # package-relative imports are replaced by the block below
            if "Trainer" in stripped:
                continue  # dev/training references must not leak into the demo
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
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    model_code = read_model_code()

    cells = [
        # 1. Title & instructions ---------------------------------------------
        cell_md(
            "# Pet Segmentation & Breed Classification — Live Demo\n"
            "\n"
            "This notebook runs a **trained model** on any image you provide. "
            "**No training happens here** — it only loads a saved model "
            "(`best.pth`) and predicts instantly.\n"
            "\n"
            "**How to use:**\n"
            "1. Make sure `best.pth` (the trained model) is in the **same folder** "
            "as this notebook.\n"
            "2. Run all cells from top to bottom.\n"
            "3. When the *Upload* button appears, **drop your image** into it — "
            "or set `IMAGE_PATH` below to a file on disk and skip the upload.\n"
            "4. The last cell shows your image with the predicted pet region "
            "overlaid, plus the predicted **breed** and **confidence**."
        ),
        # 2. Image input -------------------------------------------------------
        cell_code(
            "from pathlib import Path\n"
            "from PIL import Image\n"
            "\n"
            "# Option A: set this to an image path on disk, e.g.\n"
            "#   IMAGE_PATH = \"/home/user/Downloads/my_pet.jpg\"\n"
            "# Leave it empty to use the upload button below instead.\n"
            "IMAGE_PATH = \"\"\n"
            "\n"
            "if IMAGE_PATH.strip():\n"
            "    _img_path = Path(IMAGE_PATH).expanduser()\n"
            "    if not _img_path.is_file():\n"
            "        raise FileNotFoundError(f\"image not found: {_img_path}\")\n"
            "else:\n"
            "    try:\n"
            "        import ipywidgets as widgets\n"
            "        _uploader = widgets.FileUpload(\n"
            "            accept=\"image/*\", multiple=False,\n"
            "            description=\"Upload an image\",\n"
            "        )\n"
            "        display(_uploader)\n"
            "        print(\"\\nDrop your image into the upload button above, \"\n"
            "              \"then re-run this cell and the next one.\")\n"
            "    except ImportError:\n"
            "        from IPython.display import FileUpload as _IPyUpload\n"
            "        _uploader = _IPyUpload(accept=\"image/*\", multiple=False)\n"
            "        display(_uploader)\n"
            "        print(\"\\nDrop your image into the upload widget above, \"\n"
            "              \"then re-run this cell and the next one.\")\n"
            "    if getattr(_uploader, \"value\", None):\n"
            "        _uploaded = _uploader.value\n"
            "        _first = list(_uploaded.values())[0]\n"
            "        _raw = _first[\"content\"] if isinstance(_first, dict) else _first\n"
            "        _img_path = Path(\"./_uploaded_image.png\")\n"
            "        _img_path.write_bytes(bytes(_raw))\n"
            "        print(f\"Loaded uploaded image ({_img_path.stat().st_size} bytes)\")\n"
            "\n"
            "_img = Image.open(_img_path).convert(\"RGB\")\n"
            "print(f\"image: {_img.size[0]} x {_img.size[1]}\")"
        ),
        # 3. Model definitions (injected verbatim from src/models) -------------
        cell_code(model_code),
        # 4. Load the checkpoint ------------------------------------------------
        cell_code(
            "import torch\n"
            "\n"
            "# The trained model file. It stores everything needed to rebuild the\n"
            "# model: architecture name, image size, threshold and the 37 breed\n"
            "# names. Keep it in the same folder as this notebook.\n"
            "CKPT_PATH = \"best.pth\"\n"
            "\n"
            "ckpt = torch.load(CKPT_PATH, map_location=\"cpu\", weights_only=False)\n"
            "cfg = ckpt[\"cfg\"]\n"
            "\n"
            "model = build_model(cfg)\n"
            "model.load_state_dict(ckpt[\"model_state\"])\n"
            "model.eval()\n"
            "\n"
            "CLASSES = list(ckpt.get(\"classes\") or [f\"class_{i}\" for i in range(37)])\n"
            "IMG_SIZE = int(cfg.get(\"data\", {}).get(\"img_size\", 256))\n"
            "THRESHOLD = float(cfg.get(\"model\", {}).get(\"seg_threshold\", 0.5))\n"
            "\n"
            "print(f\"model:        {cfg['model']['name']}\")\n"
            "print(f\"image size:  {IMG_SIZE} x {IMG_SIZE}\")\n"
            "print(f\"threshold:   {THRESHOLD}\")\n"
            "print(f\"breeds:      {len(CLASSES)} classes\")\n"
            "if \"epoch\" in ckpt:\n"
            "    print(f\"best epoch:  {ckpt['epoch']} \"\n"
            "          f\"(val mIoU {ckpt.get('best_miou', float('nan')):.4f})\")\n"
            "print(\"model loaded — ready to predict.\")"
        ),
        # 5. Predict on the image ------------------------------------------------
        cell_code(
            "import torch\n"
            "import torch.nn.functional as F\n"
            "import torchvision.transforms.functional as TF\n"
            "import numpy as np\n"
            "\n"
            "# Apply exactly the same preprocessing as training:\n"
            "# resize to IMG_SIZE x IMG_SIZE, then scale pixels to [0, 1].\n"
            "# (Training used no additional normalization.)\n"
            "x = TF.resize(_img, [IMG_SIZE, IMG_SIZE], antialias=True)\n"
            "x = TF.to_tensor(x).unsqueeze(0)\n"
            "\n"
            "with torch.no_grad():\n"
            "    out = model(x)\n"
            "\n"
            "# Segmentation: sigmoid -> binary foreground mask\n"
            "mask = (torch.sigmoid(out[\"seg_logits\"]) > THRESHOLD).float()\n"
            "mask = TF.resize(\n"
            "    mask[0, 0].unsqueeze(0).unsqueeze(0), [_img.size[1], _img.size[0]],\n"
            "    interpolation=TF.InterpolationMode.NEAREST,\n"
            ")[0, 0].cpu().numpy()\n"
            "\n"
            "# Classification: softmax -> breed + confidence\n"
            "probs = torch.softmax(out[\"cls_logits\"], dim=1)[0]\n"
            "conf, label = probs.max(0)\n"
            "breed = CLASSES[int(label)]\n"
            "\n"
            "print(f\"predicted breed:  {breed}\")\n"
            "print(f\"confidence:       {conf.item():.2%}\")\n"
            "print(f\"foreground:       {100.0 * mask.mean():.1f}% of pixels\")"
        ),
        # 6. Display: image | prediction overlay ---------------------------------
        cell_code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "def _overlay(img, mask, alpha=0.5, color=(1.0, 0.0, 0.0)):\n"
            "    img = np.asarray(img, dtype=np.float32).copy()\n"
            "    m = np.asarray(mask).astype(bool)\n"
            "    if m.any():\n"
            "        img[m] = (1.0 - alpha) * img[m] + alpha * np.asarray(color, dtype=np.float32)\n"
            "    return np.clip(img, 0.0, 1.0)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n"
            "axes[0].imshow(_img)\n"
            "axes[0].set_title(\"Input image\")\n"
            "axes[0].axis(\"off\")\n"
            "axes[1].imshow(_overlay(_img, mask))\n"
            "axes[1].set_title(f\"Prediction: {breed} ({conf.item():.0%})\")\n"
            "axes[1].axis(\"off\")\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        # 7. Fresh-machine note ---------------------------------------------------
        cell_md(
            "### Running this on a fresh machine\n"
            "\n"
            "Only two files are needed: this notebook and `best.pth`.\n"
            "Install the packages with:\n"
            "\n"
            "```bash\n"
            "pip install torch torchvision pillow numpy matplotlib ipywidgets\n"
            "```\n"
            "\n"
            "Then open this notebook (e.g. `jupyter notebook`) and run it top to bottom. "
            "No GPU, dataset, internet, or training code is required."
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
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
    n_cells = len(nb["cells"])
    print(f"wrote {OUT_PATH} ({n_cells} cells)")


if __name__ == "__main__":
    main()
