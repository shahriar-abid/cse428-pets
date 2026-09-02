"""Gradio live demo for CSE428 pet segmentation + breed classification.

Loads a saved checkpoint (self-describing: cfg + classes are embedded), builds
the model from it, and serves an interactive UI with image upload -> mask
overlay + breed + confidence.

Run:
    python gradio_demo.py                      # local UI at http://127.0.0.1:7860
    python gradio_demo.py --share              # + free 72h public link
    python gradio_demo.py --ckpt path/to/best.pth   # pick a specific model
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Reuse the exact model code the notebook uses (no duplication).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model  # noqa: E402


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    img_size = int(ck["cfg"].get("data", {}).get("img_size", 256))
    classes = list(ck.get("classes") or [f"class_{i}" for i in range(37)])
    threshold = float(ck["cfg"].get("model", {}).get("seg_threshold", 0.5))
    name = ck["cfg"]["model"]["name"]
    return model, img_size, classes, threshold, name


def make_predict(model, img_size, classes, threshold, device):
    @torch.no_grad()
    def predict(img):
        if img is None:
            raise gr.Error("Please upload an image first.")
        img = img.convert("RGB")
        x = TF.resize(img, [img_size, img_size], antialias=True)
        x = TF.to_tensor(x).unsqueeze(0).to(device)
        out = model(x)
        mask = (torch.sigmoid(out["seg_logits"]) > threshold)[0, 0].cpu().numpy()
        probs = torch.softmax(out["cls_logits"], dim=1)[0]
        conf, lab = probs.max(0)
        breed = classes[int(lab)]
        # resize the mask back to the original image size
        mask = TF.resize(torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                         [img.size[1], img.size[0]],
                         interpolation=TF.InterpolationMode.NEAREST)[0, 0].numpy() > 0.5
        # overlay the mask in red on the original image
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.copy()
        arr[mask > 0] = 0.5 * arr[mask > 0] + 0.5 * np.array([1.0, 0.0, 0.0])
        return np.clip(arr, 0, 1), f"{breed} ({conf.item():.0%})"
    return predict


def main():
    ap = argparse.ArgumentParser(description="Gradio demo - pet segmentation + breed classification")
    ap.add_argument("--ckpt", default="data/attention_unet/checkpoints/best.pth",
                    help="Path to a saved best.pth (unet or attention_unet)")
    ap.add_argument("--share", action="store_true", help="Create a free 72h public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    ckpt_path = args.ckpt
    if not Path(ckpt_path).is_file():
        sys.exit(f"checkpoint not found: {ckpt_path}\n"
                 "Run the notebook in train mode first, or point --ckpt at a saved best.pth")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                else "cpu"))
    model, img_size, classes, threshold, name = load_model(ckpt_path, device)
    predict = make_predict(model, img_size, classes, threshold, device)
    print(f"loaded {name} | img_size {img_size} | threshold {threshold} | "
          f"{len(classes)} breeds | device {device}")

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a pet image"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation overlay"),
            gr.Label(num_top_classes=3, label="Predicted breed (top 3)"),
        ],
        title="CSE428 - Pet Segmentation & Breed Classification",
        description=(
            f"Model: {name}. Drag in any pet photo - it returns the segmentation "
            "mask overlay and the top-3 predicted breeds with confidence."
        ),
    )
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
