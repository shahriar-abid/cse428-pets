"""Instant deployable inference: load a checkpoint and segment/classify a random image.

The instructor will give a fresh image at evaluation; no training is needed here.
We only restore the saved model weights, run one forward pass, and save an overlay.

Usage:
    python scripts/predict.py --checkpoint outputs/unet/checkpoints/best.pth \
        --image /path/to/random_pet.jpg [--out /tmp/prediction.png]

The checkpoint stores config (model name, num classes) and class label list, so
nothing but torch is required at inference time.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.models import build_model


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    if "model" not in cfg:
        cfg["model"] = {"name": ckpt.get("model_name", "unet")}
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    classes = ckpt.get("classes") or build_default_classes()
    return model, classes, cfg


def build_default_classes():
    try:
        from src.data import PetSegDataset
        ds = PetSegDataset(os.environ.get("CSE428_DATA_ROOT", "./data"))
        return list(ds.classes)
    except Exception:
        return [f"class_{i}" for i in range(37)]


@torch.no_grad()
def predict(model, image_path, device, img_size=256, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    x = TF.resize(img, [img_size, img_size], antialias=True)
    x = TF.to_tensor(x).unsqueeze(0).to(device)

    out = model(x)
    if not isinstance(out, dict):
        out = {"cls_logits": out}

    seg_prob = torch.sigmoid(out["seg_logits"])[0, 0]
    mask = (seg_prob > threshold).float()

    cls_prob = torch.softmax(out["cls_logits"], dim=1)[0]
    conf, label = cls_prob.max(0)

    mask_img = TF.resize(
        mask.unsqueeze(0).unsqueeze(0), [orig_h, orig_w],
        interpolation=TF.InterpolationMode.NEAREST,
    )[0, 0].cpu().numpy()
    return mask_img, int(label.item()), float(conf.item()), out


def make_overlay(image_path, mask, out_path):
    img = Image.open(image_path).convert("RGB").convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
    import numpy as np
    px = overlay.load()
    m = (mask > 0.5)
    xs, ys = m.nonzero()
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        px[xx, yy] = (255, 0, 0, 120)
    out = Image.alpha_composite(img, overlay).convert("RGB")
    out.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/tmp/prediction.png")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg = load_model(args.checkpoint, device)
    mask, label, conf, _ = predict(
        model, args.image, device, args.img_size, args.threshold
    )
    name = classes[label] if label < len(classes) else f"class_{label}"
    out = make_overlay(args.image, mask, args.out)

    print(f"model:        {cfg.get('model', {}).get('name')}")
    print(f"predicted:    {name} (id {label})")
    print(f"confidence:   {conf:.2%}")
    print(f"foreground %: {100.0 * mask.mean():.1f}")
    print(f"overlay      -> {out}")


if __name__ == "__main__":
    main()
